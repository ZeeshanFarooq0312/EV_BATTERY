"""Trains a per-module curve-reconstruction model using the SAME proven
method as curve_train.py's pack-level "Complete Global Curve Reconstruction"
model (which already gives smooth, good results) -- just swapping the data
source from pack-mean voltage to each real file's own WEAKEST module.

The weakest module in a file is identified directly (whichever module's
min-cell-voltage trace ends at the lowest voltage). Since the 9 modules in a
pack are series-connected, the pack-level test stops the instant that one
module crosses the global cutoff -- so that ONE module per file is the only
one with real, complete, uncensored data all the way from ~4.2V down to
~2.1-2.4V. No extrapolation, no per-cell physics simulation needed at all.

Earlier attempts at this file built training curves for ALL 9 modules per
file (8 of which are right-censored, needing tiered extrapolation to fill in
the missing tail) and/or used a physics-based per-cell synthetic generator --
both introduced artifacts (kinks at the real/extrapolated boundary,
checkpoint-level noise from sparse deep-region coverage) that needed several
rounds of debugging (isotonic regression, checkpoint smoothing) in app.py to
paper over. This version sidesteps all of that by only ever training on
real, complete curves, reusing curve_train.py's build_rows_from_curve and
generate_synthetic_curve UNCHANGED -- the exact method already validated to
produce smooth results at the pack level.
"""

import os
import sys
import numpy as np
import xgboost as xgb
import joblib
from collections import Counter
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))

from curve_utils import compute_soh, extract_and_resample_curve, list_full_curve_files
from build_module_dataset import DATA_FOLDER
from curve_train import (
    generate_synthetic_curve, build_rows_from_curve, CUTOFF_PCTS, SYNTH_CURVES_PER_CRATE,
)  # noqa: F401 (CUTOFF_PCTS re-exported for callers)

# curve_train.py's own CUTOFF_PCTS only spans 0.20-0.90 (tuned for its ~54
# real pack-mean curves). With only 6 real weakest-module curves total here,
# every distinct cutoff position is a distinct training example squeezed out
# of the same small real-curve set -- so this covers close to the FULL
# 0-1 range (start to end) at a much finer step, to learn from every
# possible observed-tail starting point a real partial SFT upload could land
# on, not just the middle 70%.
MODULE_CUTOFF_PCTS = [round(p, 3) for p in np.arange(0.03, 0.98, 0.02)]


def _c_rate_bucket(c_rate):
    return '1.0C' if c_rate >= 0.9 else '0.3C'


def prepare_weakest_module_curves(data_folder=DATA_FOLDER):
    """Returns {bucket: [(ah, v, imbalance_proxy, true_soh, pack_id), ...]} --
    exactly one real curve per file: that file's own weakest module."""
    curves = {'0.3C': [], '1.0C': []}
    for entry in list_full_curve_files(data_folder):
        ah, _pack_v, _cell_std, modules = extract_and_resample_curve(entry['path'], want_modules=True)
        if ah is None or len(ah) < 50:
            continue
        weakest_m = min(modules, key=lambda m: modules[m]['min_v'][-1])
        v = modules[weakest_m]['min_v']
        imbalance_proxy = modules[weakest_m]['mean_v'] - v
        true_soh = compute_soh(ah[-1])
        bucket = _c_rate_bucket(entry['c_rate'])
        curves[bucket].append((ah, v, imbalance_proxy, true_soh, entry['pack_id']))
    return curves


_start_voltage_cache = {}


def typical_start_voltage(bucket, data_folder=DATA_FOLDER):
    """Real weakest-module curves' own v[0] (voltage at Ah=0, i.e. full
    charge) is a tight, near-constant cluster across packs -- 4.105-4.138V
    for every real template except pk6 0.3C's known-anomalous 2.277V (see
    app.py's reconstruct_module_curve docstring). The model's own checkpoint-0
    regressor still sometimes predicts 0.1-0.2V below this real range (only 6
    real templates to learn from, and this single point carries almost no
    real signal from the SFT input features, unlike every other checkpoint).
    Median (not mean) is used specifically because it ignores the pk6
    outlier automatically without needing to hand-exclude it."""
    key = (bucket, data_folder)
    if key in _start_voltage_cache:
        return _start_voltage_cache[key]
    curves = prepare_weakest_module_curves(data_folder)[bucket]
    starts = [v[0] for _ah, v, _imb, _soh, _pid in curves]
    result = float(np.median(starts))
    _start_voltage_cache[key] = result
    return result


def build_dataset(data_folder=DATA_FOLDER, synth_per_crate=SYNTH_CURVES_PER_CRATE, seed=None):
    real = prepare_weakest_module_curves(data_folder)
    rng = np.random.default_rng(seed)

    datasets = {}
    for bucket, curves in real.items():
        if not curves:
            continue
        sohs = [c[3] for c in curves]
        print(f"{bucket}: {len(curves)} real weakest-module curves, SOH range {min(sohs):.2f}%-{max(sohs):.2f}%")

        X_rows, y_rows, groups = [], [], []
        for ah, v, imb, true_soh, pack_id in curves:
            Xr, yr = build_rows_from_curve(ah, v, imb, true_soh, cutoff_pcts=MODULE_CUTOFF_PCTS)
            X_rows.extend(Xr); y_rows.extend(yr); groups.extend([pack_id] * len(Xr))

        for _ in range(synth_per_crate):
            idx = int(rng.integers(len(curves)))
            t_ah, t_v, t_imb, t_soh, _pack_id = curves[idx]
            # post_knee_jitter=None: the weakest-module template is already
            # the steepest real tail available (see generate_synthetic_curve's
            # docstring) -- re-jittering it on top produced a visible slope
            # kink and systematically over-steep synthetic tails.
            s_ah, s_v, s_imb, s_soh = generate_synthetic_curve(t_ah, t_v, t_imb, t_soh, bucket, post_knee_jitter=None)
            Xr, yr = build_rows_from_curve(s_ah, s_v, s_imb, s_soh, cutoff_pcts=MODULE_CUTOFF_PCTS)
            X_rows.extend(Xr); y_rows.extend(yr); groups.extend(['synthetic'] * len(Xr))

        print(f"  -> {len(X_rows)} training rows ({synth_per_crate} synthetic curves added)")
        datasets[bucket] = (X_rows, y_rows, groups)
    return datasets


def _make_model(n_jobs=1):
    # n_jobs is pickled into the saved model and also applies to .predict()
    # calls, not just .fit() -- a persistent n_jobs=-1 model spawns a full
    # worker-process pool on every single inference request in app.py,
    # which is how a single Flask process ended up with a runaway tree of
    # hundreds of child processes. Training uses _make_model(n_jobs=-1)
    # explicitly; the FINAL saved model must stay at the n_jobs=1 default.
    base = xgb.XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.015, random_state=42,
                             reg_alpha=0.15, reg_lambda=2.5, subsample=0.85, colsample_bytree=0.85, min_child_weight=3)
    return MultiOutputRegressor(base, n_jobs=n_jobs)


def train_and_validate(data_folder=DATA_FOLDER, synth_per_crate=SYNTH_CURVES_PER_CRATE, seed=None):
    datasets = build_dataset(data_folder, synth_per_crate, seed=seed)
    models = {}

    for bucket, (X_rows, y_rows, groups) in datasets.items():
        lengths = [len(x) for x in X_rows]
        most_common_len = Counter(lengths).most_common(1)[0][0]
        keep = [i for i, l in enumerate(lengths) if l == most_common_len]
        X = np.array([X_rows[i] for i in keep])
        y = np.array([y_rows[i] for i in keep])
        groups = np.array([groups[i] for i in keep])

        print(f"\n{'=' * 60}\nTraining module curve-reconstruction model for {bucket}\n{'=' * 60}")
        print(f"Samples: {len(X)}  Input features: {X.shape[1]}  Output checkpoints: {y.shape[1]}")

        real_packs = sorted(set(g for g in groups if g != 'synthetic'))
        mae_list = []
        for pack in real_packs:
            train_mask, test_mask = groups != pack, groups == pack
            if test_mask.sum() == 0:
                continue
            model = _make_model(n_jobs=-1)  # LOPO fold, never saved -- fine to parallelize
            model.fit(X[train_mask], y[train_mask])
            pred = model.predict(X[test_mask])
            mae = mean_absolute_error(y[test_mask], pred)
            mae_list.append(mae)
            print(f"  Left out {pack} -> MAE: {mae * 1000:.1f} mV  (n={test_mask.sum()})")

        if mae_list:
            print(f">>> AVERAGE LOPO curve-reconstruction MAE for {bucket}: {np.mean(mae_list) * 1000:.1f} mV <<<")

        final_model = _make_model(n_jobs=-1)  # fast fit
        final_model.fit(X, y)
        final_model.n_jobs = 1  # must NOT parallelize at inference time (see _make_model docstring)
        models[bucket] = final_model

    return models


if __name__ == "__main__":
    models = train_and_validate(DATA_FOLDER)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    for bucket, model in models.items():
        safe_name = bucket.replace('.', '_')
        path = os.path.join(out_dir, f'module_curve_reconstruction_model_{safe_name}.pkl')
        joblib.dump(model, path)
        print(f"Saved {path}")
