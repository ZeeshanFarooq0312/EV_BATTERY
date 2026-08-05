
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
import re
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))

from curve_utils import (
    NOMINAL_CAPACITY, HEAD_CHECKPOINTS_PCT, SFT_SAMPLE_FRACTIONS,
    compute_soh, extract_and_resample_curve, extract_enhanced_features,
    detect_knee_index,
)
from physics_calibration import calibrate as _pc_calibrate

# SOH is now derived from measured capacity instead of a hardcoded lookup:
#   SOH = (actual_capacity_delivered / NOMINAL_CAPACITY) * 100

# Real packs only span ~86.7%-95.9% SOH. We extrapolate synthetic curves down
# to this floor so the model also learns what a more degraded pack looks like.
SYNTHETIC_SOH_FLOOR = 80.0

# Extra start-of-discharge IR sag (volts) introduced per percentage point of
# SOH degraded *beyond* a template's own SOH. Sourced from
# physics_calibration.py's regression across all real full-curve tests
# (single source of truth -- this used to be a second, independently
# hardcoded copy that drifted out of sync with module_soh_train.py's, which
# had the wrong value and the wrong C-rate ratio direction).
_cal = _pc_calibrate()
EXTRA_IR_DROP_PER_SOH_POINT = {
    bucket: fit['slope'] for bucket, fit in _cal['ir_sag_fit'].items()
}
IR_DROP_DECAY_FRACTION = 0.05  # sag fades out over the first 5% of capacity

SYNTH_CURVES_PER_CRATE = 60

# Real 0.3C/1.0C packs show the discharge knee at ~83-92% of total capacity.
# The original cutoffs (0.20-0.75) always land BEFORE the knee, so the model
# always sees "knee + everything after it" bundled together in the observed
# tail -- it never gets a training row that isolates just the post-knee decay
# the way a short field test that starts late-in-life actually would. Adding
# cutoffs from 0.78 up to 0.90 forces some training rows to start the
# observed segment in/after the knee, giving the model dedicated signal on
# reconstructing the post-knee tail from very little pre-knee context.
CUTOFF_PCTS = [
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.78, 0.80, 0.825, 0.85, 0.875, 0.90
]

# When cutting a synthetic curve's post-knee decay, how much steeper/shallower
# to make it vs. the template it's derived from (see generate_synthetic_curve).
POST_KNEE_DECAY_JITTER = (0.7, 1.5)


def generate_synthetic_curve(template_ah, template_v, template_cell_std, template_soh, c_rate_name,
                              post_knee_jitter=POST_KNEE_DECAY_JITTER):
    """Scale a real curve to a new (lower) SOH and add a C-rate specific extra
    IR sag for the portion of degradation beyond what the template itself shows.
    This keeps synthetic curves shaped like real ones instead of pure noise.

    post_knee_jitter: (low, high) range to randomly re-steepen/flatten the
    post-knee tail, or None to skip this step entirely and just use the
    template's own (already realistic) tail shape, scaled. Defaults to the
    module-level POST_KNEE_DECAY_JITTER for backward compatibility with
    curve_train.py's own pack-mean model, where it's smoothed out across 9
    averaged modules. module_curve_train.py passes None: its single-module
    templates are already the steepest real curve available (the pack's
    weakest module), so re-jittering on top compounds into unrealistically
    steep, kinked synthetic tails (visible as a slope discontinuity right at
    the knee) instead of adding useful shape diversity."""
    target_soh = float(np.random.uniform(SYNTHETIC_SOH_FLOOR, template_soh))
    target_total_cap = (target_soh / 100.0) * NOMINAL_CAPACITY
    template_total_cap = template_ah[-1]
    scale = target_total_cap / template_total_cap

    synth_ah = template_ah * scale
    synth_v = template_v.copy()

    extra_degradation = max(0.0, template_soh - target_soh)
    sag_amplitude = EXTRA_IR_DROP_PER_SOH_POINT[c_rate_name] * extra_degradation
    if sag_amplitude > 0:
        decay_window = IR_DROP_DECAY_FRACTION * target_total_cap
        sag = sag_amplitude * np.exp(-synth_ah / (decay_window + 1e-6))
        synth_v = synth_v - sag

    # Vary the steepness of the post-knee voltage collapse itself, so the
    # model sees a family of end-of-discharge drop-off shapes rather than
    # only ever the one recorded shape from whichever real curve was used as
    # the template -- real data has very few genuinely independent examples
    # of this region.
    if post_knee_jitter is not None:
        knee_idx = detect_knee_index(synth_ah, synth_v)
        if knee_idx < len(synth_v) - 3:
            decay_factor = float(np.random.uniform(*post_knee_jitter))
            knee_v = synth_v[knee_idx]
            synth_v[knee_idx:] = knee_v - (knee_v - synth_v[knee_idx:]) * decay_factor

    noise = np.random.normal(0, 0.004, synth_v.shape)
    synth_v = np.clip(synth_v + noise, 2.0, 4.2)

    synth_cell_std = template_cell_std.copy() if template_cell_std is not None else None
    return synth_ah, synth_v, synth_cell_std, target_soh


def build_rows_from_curve(ah, v, cell_std, true_soh, cutoff_pcts=CUTOFF_PCTS):
    """Slice a full discharge curve at several cutoff points, treating the
    tail after the cutoff as the observed segment (what a short field test
    would see) and the 41 checkpoints across the WHOLE curve as the target -
    i.e. we reconstruct the complete global curve, not just the missing head.

    `cutoff_pcts` defaults to this module's own CUTOFF_PCTS (unchanged
    behavior for existing callers); module_curve_train.py passes a much
    denser, more complete range since it only has 6 real curves total and
    benefits from squeezing more distinct input-slice examples out of each."""
    total_cap = ah[-1]
    X_rows, y_rows = [], []

    for cutoff_pct in cutoff_pcts:
        cutoff_ah = float(total_cap * cutoff_pct)
        cutoff_idx = np.searchsorted(ah, cutoff_ah)
        if cutoff_idx < 15 or cutoff_idx >= len(ah) - 20:
            continue

        sft_ah = ah[cutoff_idx:] - ah[cutoff_idx]
        sft_v = v[cutoff_idx:]
        sft_cell_imbalance = cell_std[cutoff_idx:] if cell_std is not None else None

        target_v = [float(np.interp(pct * total_cap, ah, v)) for pct in HEAD_CHECKPOINTS_PCT]

        # Sample the REAL observed curve (not a straight line between its
        # endpoints) at the non-uniform, knee-dense fractions of the tail.
        sft_sampled_v = [float(np.interp(frac * sft_ah[-1], sft_ah, sft_v)) for frac in SFT_SAMPLE_FRACTIONS]
        enhanced_feats = extract_enhanced_features(sft_v, sft_ah, sft_cell_imbalance)

        features = sft_sampled_v + [
            true_soh,
            enhanced_feats['initial_slope'], enhanced_feats['final_slope'], enhanced_feats['overall_slope'],
            enhanced_feats['mean_curvature'], enhanced_feats['max_curvature'], enhanced_feats['voltage_std'],
            enhanced_feats['voltage_range'], enhanced_feats['plateau_length'], enhanced_feats['end_voltage_drop'],
            enhanced_feats['end_slope'], enhanced_feats['cell_imbalance_mean'], enhanced_feats['cell_imbalance_std'],
            enhanced_feats['tail_knee_pct'], enhanced_feats['tail_knee_slope'],
            float(cutoff_ah)
        ]

        X_rows.append(features)
        y_rows.append(target_v)

    return X_rows, y_rows


def prepare_training_data_by_crate(data_folder):
    real_curves = {'0.3C': [], '1.0C': []}

    for file in os.listdir(data_folder):
        if not file.endswith('.csv'):
            continue
        if 'FFCT' not in file.upper() and 'FFT' not in file.upper():
            continue

        c_rate_match = re.search(r'(\d+(?:\.\d+)?)C', file)
        if not c_rate_match:
            continue
        c_rate = float(c_rate_match.group(1))

        ah, v, cell_std_data = extract_and_resample_curve(os.path.join(data_folder, file))
        if ah is None or len(ah) < 50:
            continue

        true_soh = compute_soh(ah[-1])

        if abs(c_rate - 0.3) < 0.05:
            real_curves['0.3C'].append((ah, v, cell_std_data, true_soh))
        elif c_rate >= 0.9:
            real_curves['1.0C'].append((ah, v, cell_std_data, true_soh))

    data_0_3c = {'X': [], 'y': []}
    data_1_0c = {'X': [], 'y': []}
    bucket_data = {'0.3C': data_0_3c, '1.0C': data_1_0c}

    for c_rate_name, curves in real_curves.items():
        if not curves:
            continue
        sohs = [c[3] for c in curves]
        print(f"{c_rate_name}: {len(curves)} real curves, SOH range {min(sohs):.2f}%-{max(sohs):.2f}%")

    print(f"\nGenerating synthetic curves down to {SYNTHETIC_SOH_FLOOR:.0f}% SOH...")
    for c_rate_name, curves in real_curves.items():
        if not curves:
            continue
        for _ in range(SYNTH_CURVES_PER_CRATE):
            template_ah, template_v, template_cell_std, template_soh = curves[np.random.randint(len(curves))]
            synth_ah, synth_v, synth_cell_std, synth_soh = generate_synthetic_curve(
                template_ah, template_v, template_cell_std, template_soh, c_rate_name
            )
            curves.append((synth_ah, synth_v, synth_cell_std, synth_soh))

    for c_rate_name, curves in real_curves.items():
        data = bucket_data[c_rate_name]
        for ah, v, cell_std_data, true_soh in curves:
            X_rows, y_rows = build_rows_from_curve(ah, v, cell_std_data, true_soh)
            data['X'].extend(X_rows)
            data['y'].extend(y_rows)

    print(f"\n0.3C samples: {len(data_0_3c['X'])}")
    print(f"1.0C samples: {len(data_1_0c['X'])}")
    return data_0_3c, data_1_0c


def train_crate_specific_models(data_folder):
    print("Preparing C-rate specific training data...")
    data_0_3c, data_1_0c = prepare_training_data_by_crate(data_folder)
    models = {}

    for c_rate_name, data in [('0.3C', data_0_3c), ('1.0C', data_1_0c)]:
        if len(data['X']) == 0:
            continue

        lengths = [len(x) for x in data['X']]
        if len(set(lengths)) > 1:
            most_common_len = Counter(lengths).most_common(1)[0][0]
            X_list = [x for x, l in zip(data['X'], lengths) if l == most_common_len]
            y_list = [y for y, l in zip(data['y'], lengths) if l == most_common_len]
        else:
            X_list, y_list = data['X'], data['y']

        X, y = np.array(X_list), np.array(y_list)
        print(f"\nTraining {c_rate_name} model with {len(X)} samples...")
        print(f"  Input features: {X.shape[1]}")
        print(f"  Output checkpoints: {y.shape[1]}")

        model = xgb.XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.015, random_state=42,
                                 reg_alpha=0.15, reg_lambda=2.5, subsample=0.85, colsample_bytree=0.85, min_child_weight=3)
        multi_model = MultiOutputRegressor(model)

        if len(X) > 50:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            multi_model.fit(X_train, y_train)
            mae = mean_absolute_error(y_test, multi_model.predict(X_test))
            print(f"✅ {c_rate_name} Model MAE: {mae*1000:.1f} mV")
        else:
            multi_model.fit(X, y)
        models[c_rate_name] = multi_model
    return models


if __name__ == "__main__":
    from build_module_dataset import DATA_FOLDER
    # legacy/disabled model -- not loaded by app.py by default (see README) --
    # saved into models_legacy/, not models/, so it doesn't mix with the active models
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models_legacy')
    if os.path.exists(DATA_FOLDER):
        models = train_crate_specific_models(DATA_FOLDER)
        for c_rate_name, model in models.items():
            safe_name = c_rate_name.replace('.', '_')
            out_path = os.path.join(out_dir, f'reconstruction_model_{safe_name}_v10_knee.pkl')
            joblib.dump(model, out_path)
            print(f"✅ Saved {out_path}")
