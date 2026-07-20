
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import re
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')

# SOH is now derived from measured capacity instead of a hardcoded lookup:
#   SOH = (actual_capacity_delivered / NOMINAL_CAPACITY) * 100
NOMINAL_CAPACITY = 156.0

# Real packs only span ~86.7%-95.9% SOH. We extrapolate synthetic curves down
# to this floor so the model also learns what a more degraded pack looks like.
SYNTHETIC_SOH_FLOOR = 80.0

# Extra start-of-discharge IR sag (volts) introduced per percentage point of
# SOH degraded *beyond* a template's own SOH, measured from real 0.3C vs 1.0C
# packs (pk1/pk4 FFCT curves): ~30mV drop in the first 1% of capacity at 0.3C
# vs ~130-150mV at 1.0C -> roughly a 4-5x sharper initial drop under load.
EXTRA_IR_DROP_PER_SOH_POINT = {'0.3C': 0.003, '1.0C': 0.015}
IR_DROP_DECAY_FRACTION = 0.05  # sag fades out over the first 5% of capacity

SYNTH_CURVES_PER_CRATE = 60

# Checkpoints as a fraction of the COMPLETE discharge (0-100% of total
# capacity). Keeps the original uniform 2.5% resolution everywhere (that
# density mattered for generalizing across separate SFT/FFCT test sessions,
# not just same-curve slicing) and ADDS extra points at the start (initial
# IR-drop knee) and from 71.25-100% (discharge knee, which real 0.3C/1.0C
# packs show at ~83-92% of total capacity) -- density is added, not moved.
HEAD_CHECKPOINTS_PCT = [
    0.0, 0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075, 0.0875, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325,
    0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.7125, 0.725, 0.7375,
    0.75, 0.7625, 0.775, 0.7875, 0.8, 0.8125, 0.825, 0.8375, 0.85, 0.8625, 0.875, 0.8875, 0.9, 0.9125, 0.925, 0.9375,
    0.95, 0.9625, 0.975, 0.9875, 1.0
]

# Fractions of the OBSERVED TAIL's own Ah span used to sample its real voltage
# curve. Same principle: original ~2.5%-step density kept over the first 65%
# of the tail, doubled density (~1.3% step) added over the last ~34%, where
# the discharge knee almost always falls.
SFT_SAMPLE_FRACTIONS = [
    0.0, 0.026, 0.052, 0.078, 0.104, 0.13, 0.156, 0.182, 0.208, 0.234, 0.26, 0.286, 0.312, 0.338, 0.364, 0.39, 0.416,
    0.442, 0.468, 0.494, 0.52, 0.546, 0.572, 0.598, 0.624, 0.65, 0.6635, 0.6764, 0.6894, 0.7023, 0.7153, 0.7282, 0.7412,
    0.7541, 0.767, 0.78, 0.7929, 0.8059, 0.8188, 0.8318, 0.8447, 0.8576, 0.8706, 0.8835, 0.8965, 0.9094, 0.9223, 0.9353,
    0.9482, 0.9612, 0.9741, 0.9871, 1.0
]
CUTOFF_PCTS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def extract_and_resample_curve(file_path):
    df = pd.read_csv(file_path)
    cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
    df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)

    discharge_df = df[(df['Mean_Cell_Voltage'] < 4.15) & (df['Mean_Cell_Voltage'] > 2.0)].copy()
    if 'AHDischarge' not in discharge_df.columns:
        return None, None, None

    discharge_df = discharge_df.sort_values('AHDischarge')
    discharge_df['Ah_Relative'] = discharge_df['AHDischarge'] - discharge_df['AHDischarge'].iloc[0]

    ah = discharge_df['Ah_Relative'].values
    v = discharge_df['Mean_Cell_Voltage'].values

    if len(cell_cols) > 0:
        mask = (df['Mean_Cell_Voltage'] < 4.15) & (df['Mean_Cell_Voltage'] > 2.0)
        cell_std = df[cell_cols].std(axis=1)[mask].values
    else:
        cell_std = None

    return ah, v, cell_std


def compute_soh(total_capacity_ah):
    return float((total_capacity_ah / NOMINAL_CAPACITY) * 100.0)


def extract_enhanced_features(sft_v, sft_ah, cell_data=None):
    features = {}

    dv_dah = np.gradient(sft_v, sft_ah)
    d2v_dah2 = np.gradient(dv_dah, sft_ah)

    mid_idx = len(sft_v) // 2
    features['initial_slope'] = float((sft_v[mid_idx] - sft_v[0]) / (sft_ah[mid_idx] - sft_ah[0] + 1e-5))
    features['final_slope'] = float((sft_v[-1] - sft_v[mid_idx]) / (sft_ah[-1] - sft_ah[mid_idx] + 1e-5))
    features['overall_slope'] = float((sft_v[-1] - sft_v[0]) / (sft_ah[-1] - sft_ah[0] + 1e-5))

    features['mean_curvature'] = float(np.mean(np.abs(d2v_dah2)))
    features['max_curvature'] = float(np.max(np.abs(d2v_dah2)))
    features['curvature_std'] = float(np.std(d2v_dah2))

    features['voltage_std'] = float(np.std(sft_v))
    features['voltage_range'] = float(sft_v[0] - sft_v[-1])
    features['voltage_mean'] = float(np.mean(sft_v))

    plateau_mask = (sft_v > 3.4) & (sft_v < 3.7)
    features['plateau_length'] = float(np.sum(plateau_mask) / len(sft_v))

    end_idx = int(len(sft_v) * 0.8)
    features['end_slope'] = float((sft_v[-1] - sft_v[end_idx]) / (sft_ah[-1] - sft_ah[end_idx] + 1e-5))
    features['end_voltage_drop'] = float(sft_v[end_idx] - sft_v[-1])

    if cell_data is not None and len(cell_data) > 0:
        features['cell_imbalance_mean'] = float(np.mean(cell_data))
        features['cell_imbalance_std'] = float(np.std(cell_data))
    else:
        features['cell_imbalance_mean'] = 0.0
        features['cell_imbalance_std'] = 0.0

    # Locate the sharpest local bend in the observed tail -- an explicit signal
    # for where the discharge knee sits and how steep it is, on top of the
    # coarser aggregate stats above.
    knee_idx = int(np.argmax(np.abs(dv_dah)))
    features['tail_knee_pct'] = float(sft_ah[knee_idx] / (sft_ah[-1] + 1e-5))
    features['tail_knee_slope'] = float(dv_dah[knee_idx])

    return features


def generate_synthetic_curve(template_ah, template_v, template_cell_std, template_soh, c_rate_name):
    """Scale a real curve to a new (lower) SOH and add a C-rate specific extra
    IR sag for the portion of degradation beyond what the template itself shows.
    This keeps synthetic curves shaped like real ones instead of pure noise."""
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

    noise = np.random.normal(0, 0.004, synth_v.shape)
    synth_v = np.clip(synth_v + noise, 2.0, 4.2)

    synth_cell_std = template_cell_std.copy() if template_cell_std is not None else None
    return synth_ah, synth_v, synth_cell_std, target_soh


def build_rows_from_curve(ah, v, cell_std, true_soh):
    """Slice a full discharge curve at several cutoff points, treating the
    tail after the cutoff as the observed segment (what a short field test
    would see) and the 41 checkpoints across the WHOLE curve as the target -
    i.e. we reconstruct the complete global curve, not just the missing head."""
    total_cap = ah[-1]
    X_rows, y_rows = [], []

    for cutoff_pct in CUTOFF_PCTS:
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
    DATA_FOLDER = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/fft_raw_data"
    if os.path.exists(DATA_FOLDER):
        models = train_crate_specific_models(DATA_FOLDER)
        for c_rate_name, model in models.items():
            safe_name = c_rate_name.replace('.', '_')
            joblib.dump(model, f'reconstruction_model_{safe_name}_v10_knee.pkl')
            print(f"✅ Saved reconstruction_model_{safe_name}_v10_knee.pkl")
