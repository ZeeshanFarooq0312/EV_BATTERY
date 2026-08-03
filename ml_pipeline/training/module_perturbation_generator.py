"""Perturbation-based cross-rate synthetic data: instead of synthesizing a
curve from a canonical shape + sampled physics parameters (synthetic_module_
generator.py -- which ran into a deep, unresolved artifact when adapted for
the cross-rate case, see synthetic_cross_rate_generator.py's docstring
history), this ANCHORS every synthetic instance directly on one of the 54
real (pack, module) FFT curve pairs and applies a small random rescale +
noise -- staying close to real by construction, since it IS real data, just
lightly perturbed, rather than reconstructed from scratch.

Validated visually before building this (see module_perturbation_capacity_
scatter.png / module_perturbation_curve_examples.png, TEST_CASE dir):
10800 synthetic points (200/module x 54 real modules) form a tight cloud
directly around each real anchor (deviation ~0 +/- 3.2 Ah), and example
curve overlays hug the real shape closely except a bit of fan-out right at
the knee (expected, physically -- that's where small capacity differences
legitimately look different).

Generation happens at the PACK level, not independently per module: for
each synthetic instance, all 9 of a real pack's modules are perturbed
TOGETHER (one shared pack-level scale + per-module jitter on top), so
add_sibling_features' rank/relative-voltage features stay meaningful --
they compare a module against its own (correlated) synthetic packmates, the
same way real modules are compared against their own real packmates.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))

from curve_utils import (
    get_cell_columns, module_cell_columns, N_MODULES, parse_pack_and_crate,
    extract_module_features_from_slice, add_sibling_features, compute_soh,
    _trim_leading_glitch_row,
)
from build_module_dataset import DATA_FOLDER
from module_soh_train import SLICE_PCTS

PACKS = ['pk1', 'pk2', 'pk3', 'pk4', 'pk5', 'pk6']

N_PER_PACK = 200  # -> 200 synthetic instances per module too, since every instance covers all 9 modules of a pack
PACK_SCALE_JITTER_STD = 0.02   # shared "this instance's overall pack state" jitter
MODULE_JITTER_STD = 0.01       # additional per-module jitter within one instance (mirrors real intra-pack spread)
GAP_JITTER_STD = 0.01          # independent jitter between a module's OWN 1.0C vs 0.3C scale (keeps the real gap from being perfectly rigid)
VOLTAGE_NOISE_STD = 0.003      # V, small per-cell measurement-like noise


def _find_full_file(pack_id, want_c_bucket, data_folder=DATA_FOLDER):
    for fname in sorted(os.listdir(data_folder)):
        if not fname.endswith('.csv'):
            continue
        pid, c_rate = parse_pack_and_crate(fname)
        if pid != pack_id:
            continue
        c_bucket = '1.0C' if c_rate >= 0.9 else '0.3C'
        if c_bucket != want_c_bucket:
            continue
        if 'FFCT' in fname.upper() or 'FFT' in fname.upper():
            return fname
    return None


def _load_module_raw(path, module_idx, cell_cols):
    """Real per-cell (12-col) trace for one module, filtered/sorted/
    relativized the same way extract_and_resample_curve does (including its
    leading-glitch-row trim -- see curve_utils.GLITCH_ROW_DROP_THRESHOLD_V;
    confirmed on pk6's 0.3C FFCT file, whose row 0 must not be used as a
    synthetic-data anchor)."""
    df = pd.read_csv(path)
    df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)
    mask = (df['Mean_Cell_Voltage'] < 4.15) & (df['Mean_Cell_Voltage'] > 2.0)
    d = df[mask].copy().sort_values('AHDischarge')
    d = _trim_leading_glitch_row(d, cell_cols)
    d['Ah_Relative'] = d['AHDischarge'] - d['AHDischarge'].iloc[0]
    mod_cols = module_cell_columns(cell_cols, module_idx)
    return d['Ah_Relative'].values, d[mod_cols].values


def _load_pack_modules(pack_id, data_folder=DATA_FOLDER):
    """Returns (mod_cols_by_module, real_1_0c, real_0_3c) or None if either
    full-curve file is missing. real_1_0c/real_0_3c: {module_idx: (ah, cell_matrix)}."""
    fft10_name = _find_full_file(pack_id, '1.0C', data_folder)
    fft03_name = _find_full_file(pack_id, '0.3C', data_folder)
    if not fft10_name or not fft03_name:
        return None

    path10 = os.path.join(data_folder, fft10_name)
    path03 = os.path.join(data_folder, fft03_name)
    cell_cols10 = get_cell_columns(pd.read_csv(path10, nrows=1))
    cell_cols03 = get_cell_columns(pd.read_csv(path03, nrows=1))

    real_1_0c, real_0_3c, mod_cols_by_module = {}, {}, {}
    for m in range(1, N_MODULES + 1):
        real_1_0c[m] = _load_module_raw(path10, m, cell_cols10)
        real_0_3c[m] = _load_module_raw(path03, m, cell_cols03)
        mod_cols_by_module[m] = module_cell_columns(cell_cols10, m)
    return mod_cols_by_module, real_1_0c, real_0_3c


def build_module_perturbation_rows(n_per_pack=N_PER_PACK, seed=None, data_folder=DATA_FOLDER,
                                    slice_pcts=SLICE_PCTS, packs=PACKS):
    rng = np.random.default_rng(seed)
    rows = []

    for pack_id in packs:
        loaded = _load_pack_modules(pack_id, data_folder)
        if loaded is None:
            continue
        mod_cols_by_module, real_1_0c, real_0_3c = loaded

        for i in range(n_per_pack):
            pack_scale = rng.normal(1.0, PACK_SCALE_JITTER_STD)

            # Perturb all 9 modules together (one instance = one synthetic
            # virtual pack, correlated via pack_scale) before slicing/feature
            # extraction, so sibling features compare a module against its
            # own instance's packmates -- not against unrelated draws.
            perturbed_1_0c, true_soh_0_3c = {}, {}
            for m in range(1, N_MODULES + 1):
                module_scale = pack_scale * rng.normal(1.0, MODULE_JITTER_STD)
                scale10 = module_scale * rng.normal(1.0, GAP_JITTER_STD)
                scale03 = module_scale * rng.normal(1.0, GAP_JITTER_STD)

                ah10, cells10 = real_1_0c[m]
                new_ah10 = ah10 * scale10
                new_cells10 = cells10 + rng.normal(0, VOLTAGE_NOISE_STD, size=cells10.shape)
                perturbed_1_0c[m] = (new_ah10, new_cells10)

                ah03, _cells03 = real_0_3c[m]
                new_cap03 = ah03[-1] * scale03
                # SOH can't physically exceed 100% -- upward jitter compounding
                # across pack_scale/module_scale/gap_jitter can otherwise push
                # an already-healthy real module (e.g. pk1's ~95-97%) just
                # over 100.
                true_soh_0_3c[m] = float(np.clip(compute_soh(new_cap03), 0.0, 100.0))

            for pct in slice_pcts:
                feats_by_module = {}
                for m in range(1, N_MODULES + 1):
                    new_ah10, new_cells10 = perturbed_1_0c[m]
                    start_row = int(len(new_ah10) * pct)
                    slice_df = pd.DataFrame(new_cells10[start_row:], columns=mod_cols_by_module[m])
                    slice_df['AHDischarge'] = new_ah10[start_row:]
                    f = extract_module_features_from_slice(slice_df, mod_cols_by_module[m])
                    if f is not None:
                        feats_by_module[m] = f
                if len(feats_by_module) < N_MODULES:
                    continue

                add_sibling_features(feats_by_module)
                for m, f in feats_by_module.items():
                    row = dict(f)
                    row.update({'c_rate': 1.0, 'is_sfct': 0.0, 'slice_start_pct': pct,
                                'module_idx': m, 'true_soh': true_soh_0_3c[m],
                                'pack_id': f'{pack_id}_pert_{i:04d}'})
                    rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_module_perturbation_rows(n_per_pack=20, seed=42)
    print(f"Generated {len(df)} rows from {df['pack_id'].nunique()} synthetic pack instances")
    print(f"true_soh range: {df['true_soh'].min():.2f} - {df['true_soh'].max():.2f}")
    print(df[['pack_id', 'module_idx', 'true_soh', 'sibling_rank']].head(9).to_string(index=False))
