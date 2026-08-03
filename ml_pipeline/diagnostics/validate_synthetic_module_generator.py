"""Honest validation of synthetic_module_generator.py, mirroring
validate_module_extrapolation.py's style: no assumed trust, only what the
data actually shows.

There is almost no real ground truth to check synthetic CAPACITY labels
against directly (only ~10-11 real Tier-0 points existed before any of this
module-level work, and using them to validate the very generator meant to
reduce dependence on them is somewhat circular). So the load-bearing test
here is #2 below: does a model trained ONLY on synthetic data still predict
reasonably on REAL held-out packs? That is a genuinely harder generalization
test than the old template-copy augmentation could ever face, since old
synthetic rows were trivial near-copies of their own real template.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

_ML_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ML_PIPELINE_DIR, 'core'))
sys.path.insert(0, os.path.join(_ML_PIPELINE_DIR, 'training'))

from module_soh_train import build_real_module_rows, _c_rate_bucket, DATA_FOLDER
from synthetic_module_generator import build_synthetic_module_dataset

FEATURE_COLS_EXTRA = ['voltage_drop_norm', 'delta_Ah_norm']


def _prep(df, target_c):
    if target_c == 0.3:
        sub = df[abs(df['c_rate'] - 0.3) < 0.05].copy()
    else:
        sub = df[df['c_rate'] >= 0.9].copy()
    sub['voltage_drop_norm'] = sub['voltage_drop'] / (target_c + 1e-5)
    sub['delta_Ah_norm'] = sub['delta_Ah'] / (target_c + 1e-5)
    return sub


def _feature_cols(df):
    return [c for c in df.columns if c not in ('true_soh', 'pack_id', 'module_idx', 'sample_weight')]


def distributional_sanity(real_df, synth_df, feature_cols):
    print(f"\n{'='*90}\nDistributional sanity: synthetic vs. real feature ranges (5th-95th percentile)\n{'='*90}")
    print(f"{'feature':<28}{'real range':<28}{'synth range':<28}{'flag'}")
    flags = []
    for col in feature_cols:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        r = real_df[col].dropna()
        s = synth_df[col].dropna()
        if len(r) < 3 or len(s) < 3:
            continue
        r_lo, r_hi = np.percentile(r, 5), np.percentile(r, 95)
        s_lo, s_hi = np.percentile(s, 5), np.percentile(s, 95)

        flag = ''
        if (s_hi - s_lo) < 1e-9:
            flag = 'COLLAPSED VARIANCE'
        elif s_hi < r_lo or s_lo > r_hi:
            flag = 'ZERO OVERLAP'
        if flag:
            flags.append((col, flag))

        print(f"{col:<28}{f'[{r_lo:.4g}, {r_hi:.4g}]':<28}{f'[{s_lo:.4g}, {s_hi:.4g}]':<28}{flag}")

    if flags:
        print(f"\n!! {len(flags)} feature(s) flagged: {flags}")
    else:
        print("\nNo collapsed-variance or zero-overlap features.")
    return flags


def _train_xgb(X, y, w=None):
    model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                              objective='reg:squarederror', random_state=42,
                              reg_alpha=0.1, reg_lambda=1.0)
    model.fit(X, y, sample_weight=w)
    return model


def _ranking_hit_rate(df, y_true, y_pred):
    """Fraction of (pack_id, c_rate, slice_start_pct) groups where the
    predicted weakest module matches the true weakest module."""
    tmp = df.copy()
    tmp['y_true'] = y_true
    tmp['y_pred'] = y_pred
    hits, total = 0, 0
    for _, g in tmp.groupby(['pack_id', 'c_rate', 'slice_start_pct']):
        if len(g) < N_MODULES_EXPECTED:
            continue
        total += 1
        if g.loc[g['y_true'].idxmin(), 'module_idx'] == g.loc[g['y_pred'].idxmin(), 'module_idx']:
            hits += 1
    return (hits / total) if total else float('nan'), total


N_MODULES_EXPECTED = 9


def generalization_gate(real_df, synth_df, target_c):
    real_c = _prep(real_df, target_c)
    synth_c = _prep(synth_df, target_c)
    if len(real_c) == 0 or len(synth_c) == 0:
        print(f"  [skip] no data for {target_c}C")
        return

    feature_cols = [c for c in _feature_cols(synth_c) if c in real_c.columns]

    # 1. Synthetic-only training, evaluated on ALL real rows.
    X_train, y_train = synth_c[feature_cols], synth_c['true_soh'].values
    X_test, y_test = real_c[feature_cols], real_c['true_soh'].values
    model = _train_xgb(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rank_hit, rank_n = _ranking_hit_rate(real_c, y_test, pred)

    # 2. Trivial baseline: linear regression on end_voltage alone.
    baseline = LinearRegression().fit(synth_c[['end_voltage']], y_train)
    baseline_pred = baseline.predict(real_c[['end_voltage']])
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    print(f"\n--- {target_c}C: synthetic-only model vs. ALL real rows ---")
    print(f"  Synthetic-only MAE on real data: {mae:.2f}%  (n_real={len(real_c)})")
    print(f"  Ranking hit-rate (true vs predicted weakest module): {rank_hit:.2%}  (n_groups={rank_n})")
    print(f"  Trivial baseline (linear on end_voltage alone) MAE:  {baseline_mae:.2f}%")
    if mae > baseline_mae:
        print("  !! Synthetic-only model does WORSE than the trivial baseline -- generator needs work.")


def blended_lopo(real_df, synth_df, target_c):
    from sklearn.model_selection import LeaveOneGroupOut
    real_c = _prep(real_df, target_c)
    synth_c = _prep(synth_df, target_c)
    if len(real_c) == 0:
        return
    feature_cols = [c for c in _feature_cols(real_c) if c in synth_c.columns] if len(synth_c) else _feature_cols(real_c)

    real_c = real_c.copy()
    real_c['sample_weight'] = real_c.get('sample_weight', 1.0)
    if len(synth_c):
        synth_c = synth_c.copy()
        synth_c['sample_weight'] = synth_c.get('sample_weight', 0.4)

    scores = []
    logo = LeaveOneGroupOut()
    groups = real_c['pack_id'].values
    for train_idx, test_idx in logo.split(real_c, real_c['true_soh'], groups):
        train_real = real_c.iloc[train_idx]
        train_df = pd.concat([train_real, synth_c], ignore_index=True) if len(synth_c) else train_real
        model = _train_xgb(train_df[feature_cols], train_df['true_soh'].values, w=train_df['sample_weight'].values)
        test_df = real_c.iloc[test_idx]
        pred = model.predict(test_df[feature_cols])
        scores.append(mean_absolute_error(test_df['true_soh'], pred))

    print(f"\n--- {target_c}C: blended (real + synthetic) LOPO-by-real-pack ---")
    print(f"  Mean LOPO MAE: {np.mean(scores):.2f}%  (per-fold: {[f'{s:.2f}' for s in scores]})")


if __name__ == "__main__":
    print(f"Loading real rows from {DATA_FOLDER} ...")
    real_df = build_real_module_rows(DATA_FOLDER)
    print(f"Real rows: {len(real_df)}")

    print("Generating synthetic rows (physics generator) ...")
    synth_df = build_synthetic_module_dataset(n_per_crate=300, seed=123, data_folder=DATA_FOLDER)
    print(f"Synthetic rows: {len(synth_df)}")

    shared_feature_cols = [c for c in _feature_cols(real_df) if c in synth_df.columns]
    distributional_sanity(real_df, synth_df, shared_feature_cols)

    print(f"\n{'='*90}\nGeneralization gate: synthetic-only model vs. real holdout\n{'='*90}")
    for target_c in (0.3, 1.0):
        generalization_gate(real_df, synth_df, target_c)

    print(f"\n{'='*90}\nBlended LOPO (reference: current real+legacy-augmentation benchmark ~2.9%/1.5%)\n{'='*90}")
    for target_c in (0.3, 1.0):
        blended_lopo(real_df, synth_df, target_c)
