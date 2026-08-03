"""Cross-rate synthetic per-module training data: extends
synthetic_module_generator.py's physics-informed virtual-pack generator with
a sampled 0.3C-vs-1.0C rate gap, so each synthetic module gets BOTH a 1.0C
source curve (for feature extraction, reusing generate_virtual_pack_dataframe
completely unchanged -- no need to re-derive curve shape/IR-sag/imbalance
physics, all of that is already validated) and a 0.3C target label (for
module_soh_cross_rate_train.py).

The rate gap itself is NOT a physical constant -- rate_gap_analysis.py (see
the scratchpad from this same investigation) measured it directly from the 6
real packs and found it's PACK-DEPENDENT, not a smooth function of SOH: pk5's
gap is ~0 SOH points while pk2/pk4/pk6 sit around 2.9-4.5 points, with
pk1/pk3 in between. There's no single physical formula to derive it from, so
it's sampled by BOOTSTRAPPING from the empirical per-pack mean gaps (computed
fresh from build_module_dataset.py every call -- data-driven, not hardcoded,
so it stays current if more real packs are added) plus small Gaussian
smoothing jitter, rather than assuming a distribution shape the data doesn't
actually support.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))

from build_module_dataset import build_module_dataset, DATA_FOLDER
from curve_utils import (
    N_MODULES, get_cell_columns, module_cell_columns,
    extract_module_features_from_slice, add_sibling_features,
)
from synthetic_module_generator import (
    generate_virtual_pack_dataframe, SLICE_PCTS, SYNTH_SOH_FLOOR, SYNTH_SOH_CEILING,
)

# Smooths the bootstrap over the 6 discrete real pack-mean gaps -- without
# this, every synthetic pack's gap would land on exactly one of 6 values.
BOOTSTRAP_JITTER_STD = 0.8


def _c_rate_bucket(c_rate):
    return '1.0C' if c_rate >= 0.9 else '0.3C'


def compute_real_rate_gaps(data_folder=DATA_FOLDER):
    """Per-pack mean (0.3C_soh - 1.0C_soh) across that pack's 9 modules, and
    the average within-pack std of that same gap -- both computed fresh from
    real data every call. Returns (pack_gap_values, within_pack_std)."""
    df = build_module_dataset(data_folder, verbose=False)
    df['c_bucket'] = df['c_rate'].apply(_c_rate_bucket)
    wide = df.pivot_table(index=['pack_id', 'module_idx'], columns='c_bucket', values='module_soh', aggfunc='first')
    wide = wide.dropna(subset=[c for c in ('0.3C', '1.0C') if c in wide.columns])
    wide['gap'] = wide['0.3C'] - wide['1.0C']

    pack_gaps = wide.groupby('pack_id')['gap'].mean()
    within_pack_std = wide.groupby('pack_id')['gap'].std().mean()
    if not np.isfinite(within_pack_std):
        within_pack_std = 0.5
    return pack_gaps.values, float(within_pack_std)


def sample_cross_rate_gaps(rng, real_pack_gaps, within_pack_std, n_modules=N_MODULES):
    """One pack-level gap, bootstrapped from the real per-pack means (plus
    smoothing jitter -- only ~6 discrete values are known) then per-module
    jitter matching the observed real within-pack spread. Modules within one
    synthetic pack share a common gap plus small noise, mirroring how real
    packs' modules move together (see compute_real_rate_gaps)."""
    pack_gap = float(rng.choice(real_pack_gaps)) + float(rng.normal(0, BOOTSTRAP_JITTER_STD))
    return {m: pack_gap + float(rng.normal(0, within_pack_std)) for m in range(1, n_modules + 1)}


def build_synthetic_cross_rate_module_rows(n_instances, seed=None, data_folder=DATA_FOLDER, slice_pcts=SLICE_PCTS):
    """Each instance: one synthetic 1.0C virtual pack (real curve-generation
    physics, unchanged) + one sampled rate-gap applied per module to get the
    0.3C target label. Returns rows shaped like
    synthetic_module_generator.build_synthetic_module_rows, but 'true_soh' is
    the CROSS-RATE (0.3C) label and 'source_soh_1_0c' carries the underlying
    1.0C truth the curve was actually generated from (diagnostic only -- not
    a model feature, dropped by the caller before training)."""
    rng = np.random.default_rng(seed)
    real_pack_gaps, within_pack_std = compute_real_rate_gaps(data_folder)
    rows = []

    for i in range(n_instances):
        df, module_soh_1_0c, module_capacity_ah_1_0c = generate_virtual_pack_dataframe('1.0C', rng, data_folder)
        gaps = sample_cross_rate_gaps(rng, real_pack_gaps, within_pack_std)
        module_soh_0_3c = {
            m: float(np.clip(module_soh_1_0c[m] + gaps[m], SYNTH_SOH_FLOOR, SYNTH_SOH_CEILING))
            for m in module_soh_1_0c
        }

        cell_cols = get_cell_columns(df)
        total_rows = len(df)
        pack_id = f"synth_cross_{i:05d}"
        is_sfct_value = float(rng.integers(0, 2))

        for pct in slice_pcts:
            start_row = int(total_rows * pct)
            df_slice = df.iloc[start_row:]
            feats_by_module = {}
            for m in range(1, N_MODULES + 1):
                f = extract_module_features_from_slice(df_slice, module_cell_columns(cell_cols, m))
                if f is not None:
                    feats_by_module[m] = f
            if len(feats_by_module) < N_MODULES:
                continue

            add_sibling_features(feats_by_module)
            for m, f in feats_by_module.items():
                row = dict(f)
                row.update({'c_rate': 1.0, 'is_sfct': is_sfct_value, 'slice_start_pct': pct,
                            'module_idx': m, 'true_soh': module_soh_0_3c[m],
                            'source_soh_1_0c': module_soh_1_0c[m],
                            'source_capacity_1_0c': module_capacity_ah_1_0c[m],
                            'pack_id': pack_id})
                rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    pack_gaps, within_std = compute_real_rate_gaps()
    print(f"Real per-pack mean gaps (0.3C - 1.0C SOH points): {sorted(pack_gaps.round(2))}")
    print(f"Average within-pack std: {within_std:.3f}")

    df = build_synthetic_cross_rate_module_rows(50, seed=42)
    print(f"\nGenerated {len(df)} synthetic cross-rate rows")
    print(df[['pack_id', 'module_idx', 'source_soh_1_0c', 'true_soh']].head(10).to_string(index=False))
