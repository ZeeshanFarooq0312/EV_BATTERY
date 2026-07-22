"""Physics-informed synthetic per-module training data.

Unlike the old template-copy augmentation (module_soh_train.py's retired
_legacy_build_synthetic_rows), this generates a genuinely new 108-cell
discharge DataFrame per synthetic instance from independently-SAMPLED
physical parameters (calibrated ranges in physics_calibration.py) rather than
perturbing one real pack's own feature row. Every feature the model sees is
recomputed from a real, distinct synthesized curve every time -- there is no
feature the model could memorize as "this is what template #3 always looks
like", which was the actual generalization problem the client flagged.

Design (see the approved plan for full rationale):
  1. One averaged, normalized "canonical shape" per C-rate bucket (deliberately
     not per-pack -- ~10 real curves can't support richer shape variation
     without fitting noise).
  2. Per virtual-pack instance: sample 9 modules' SOH -- either a tight
     cluster (matches pk1/pk2/pk3/pk6's real ~0.8-2.5pt spreads) or one
     standout-weak module (matches pk4/pk5's ~2.8-5.1pt spreads), both
     patterns observed in real data.
  3. Build ONE shared Ah axis from 0 to the MINIMUM of the 9 sampled module
     capacities -- the same series-circuit stopping mechanism
     module_capacity_extrapolation.py relies on for real data -- then
     synthesize each module's own voltage trace by warping the canonical
     shape to its own capacity/knee-timing/IR-sag/imbalance, all sampled
     independently per module from physics_calibration.py.
  4. True labels are known exactly by construction (no extrapolation needed).
"""

import os
import numpy as np
import pandas as pd

from curve_utils import (
    extract_and_resample_curve, NOMINAL_CAPACITY, N_MODULES, MODULE_SIZE,
    list_full_curve_files, get_cell_columns, module_cell_columns,
    extract_module_features_from_slice, add_sibling_features,
)
from module_capacity_extrapolation import detect_gated_knee
import physics_calibration as pc

DEFAULT_DATA_FOLDER = pc.DEFAULT_DATA_FOLDER

# Widened vs. module_soh_train.py's real-data SLICE_PCTS (0.40-0.70) --
# synthetic data is cheap, and wider slice-depth coverage directly targets
# generalization to SFT files observed at depths our 6 real packs happen not
# to have exercised.
SLICE_PCTS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

SYNTH_SOH_FLOOR = 82.0
SYNTH_SOH_CEILING = 99.0
# Real packs' own module SOH spans ~86.6-98.2% (pk5's weakest module to
# pk6's strongest); was (80,97)/floor=70, wide enough that synthetic curves
# routinely simulated end-voltages no real pack has ever reached (validated
# range [2.04V,2.80V] vs real [2.71V,3.18V]) -- the generalization gate's
# root failure. Narrowed to track real data with a little headroom rather
# than extrapolating degradation levels nothing in the dataset supports.
BASELINE_SOH_RANGE = (85.0, 98.0)

# Tunable knobs calibrated from n=6 packs (see plan's "explicit scope cuts") --
# not precise physical constants, adjust if the generalization gate wants it.
STANDOUT_PROBABILITY = 0.3
CLUSTERED_JITTER_STD = 0.6           # SOH points; matches pk1/pk2/pk3/pk6's ~0.8-2.5pt spreads
STANDOUT_EXTRA_DEFICIT_RANGE = (2.0, 5.0)  # SOH points; matches pk4/pk5's standout module

_shape_cache = {}


def _c_rate_bucket(c_rate):
    return '1.0C' if c_rate >= 0.9 else '0.3C'


def build_canonical_shape(c_rate_name, data_folder=DEFAULT_DATA_FOLDER, n_points=500, force=False):
    """One averaged, normalized discharge shape per C-rate bucket: fraction
    in [0,1] of that curve's own total capacity -> normalized voltage
    (~0 at the plateau, -1 at cutoff). Cached per (c_rate_name, data_folder)."""
    key = (c_rate_name, data_folder)
    if not force and key in _shape_cache:
        return _shape_cache[key]

    grid = np.linspace(0.0, 1.0, n_points)
    curves = []
    for entry in list_full_curve_files(data_folder):
        if _c_rate_bucket(entry['c_rate']) != c_rate_name:
            continue
        ah, v, cell_std = extract_and_resample_curve(entry['path'])
        if ah is None:
            continue
        total = ah[-1]
        frac = ah / total
        plateau_mask = (frac > 0.2) & (frac < 0.4)
        plateau = float(np.mean(v[plateau_mask])) if np.any(plateau_mask) else float(v[0])
        end_v = float(v[-1])
        span = plateau - end_v
        if span <= 0:
            continue
        normalized = (v - plateau) / span
        curves.append(np.interp(grid, frac, normalized))

    if not curves:
        raise ValueError(f"No real curves found for C-rate bucket {c_rate_name} to build a canonical shape")

    canonical = np.mean(curves, axis=0)
    knee_idx = detect_gated_knee(grid, canonical)
    canonical_knee_frac = float(grid[knee_idx]) if knee_idx is not None else 0.9

    result = {'grid': grid, 'shape': canonical, 'knee_frac': canonical_knee_frac, 'n_curves': len(curves)}
    _shape_cache[key] = result
    return result


def _warp_fraction(frac, canonical_knee_frac, target_knee_frac):
    """Monotonic piecewise-linear remap moving the canonical shape's own knee
    to a newly sampled location, stretching/compressing the pre-/post-knee
    segments independently while keeping both endpoints fixed at 0 and 1."""
    frac = np.asarray(frac)
    warped = np.empty_like(frac)
    pre = frac <= canonical_knee_frac
    warped[pre] = frac[pre] * (target_knee_frac / canonical_knee_frac)
    post = ~pre
    warped[post] = target_knee_frac + (frac[post] - canonical_knee_frac) * (
        (1.0 - target_knee_frac) / (1.0 - canonical_knee_frac)
    )
    return np.clip(warped, 0.0, 1.0)


def sample_module_capacities(rng):
    """Returns {module_idx: soh}: either one standout-weak module (pk4/pk5's
    real pattern) or a tight cluster (pk1/pk2/pk3/pk6's real pattern)."""
    base = float(rng.uniform(*BASELINE_SOH_RANGE))
    sohs = {}
    if rng.random() < STANDOUT_PROBABILITY:
        weak_module = int(rng.integers(1, N_MODULES + 1))
        extra_deficit = float(rng.uniform(*STANDOUT_EXTRA_DEFICIT_RANGE))
        for m in range(1, N_MODULES + 1):
            jitter = rng.normal(0, CLUSTERED_JITTER_STD)
            sohs[m] = base - (extra_deficit if m == weak_module else 0.0) + jitter
    else:
        for m in range(1, N_MODULES + 1):
            sohs[m] = base + rng.normal(0, CLUSTERED_JITTER_STD)
    return {m: float(np.clip(s, SYNTH_SOH_FLOOR, SYNTH_SOH_CEILING)) for m, s in sohs.items()}


# Real files sample at a roughly constant RATE, not a fixed row count, so a
# lower-capacity file naturally has FEWER total rows (measured directly:
# ~76.6 rows/Ah on a real 0.3C full curve). This matters a lot: a fixed
# n_rows regardless of capacity means a percentage-of-rows slice always
# captures the exact same FRACTION of Ah no matter the true capacity,
# destroying the row-count-based signal extract_features_from_slice actually
# depends on -- this exact failure mode is called out in
# SOH_MODELS_TRAIN.PY's own comments from an earlier mistake ("the first
# attempt at this kept the same row count for every synthetic SOH... made
# LOPO accuracy worse, not better"), and reproducing it here caused the
# generalization gate to fail outright (synthetic-only model worse than a
# trivial linear baseline) until this was fixed.
ROWS_PER_AH = 76.6


def generate_virtual_pack_dataframe(c_rate_name, rng, data_folder=DEFAULT_DATA_FOLDER):
    """Builds a synthetic DataFrame with real-named Cell 001..108 + AHDischarge
    columns (so it can be fed through the UNMODIFIED extract_module_features_from_slice).
    Returns (df, module_soh, module_capacity_ah)."""
    shape_info = build_canonical_shape(c_rate_name, data_folder)
    grid, canonical = shape_info['grid'], shape_info['shape']
    canonical_knee_frac = shape_info['knee_frac']

    module_soh = sample_module_capacities(rng)
    module_capacity_ah = {m: (soh / 100.0) * NOMINAL_CAPACITY for m, soh in module_soh.items()}
    pack_capacity_ah = min(module_capacity_ah.values())  # series string: stops at the weakest module

    n_rows = max(500, int(pack_capacity_ah * ROWS_PER_AH))
    ah = np.linspace(0.0, pack_capacity_ah, n_rows)
    # Real-data-calibrated (see physics_calibration.py's plateau_v_range/
    # span_v_range) -- was a hardcoded (3.45,3.75)/(1.0,1.4) guess
    # uncorrelated with real data, which left synthetic end_voltage sitting
    # well below real end_voltage regardless of the SOH/capacity range
    # (caught by validate_synthetic_module_generator.py's distributional
    # sanity check).
    plateau_v = pc.sample_plateau_v(c_rate_name, rng, data_folder)
    span_v = pc.sample_span_v(c_rate_name, rng, data_folder)  # plateau-to-cutoff voltage span

    data = {'AHDischarge': ah}
    for m in range(1, N_MODULES + 1):
        cap_m = module_capacity_ah[m]
        soh_deficit = 100.0 - module_soh[m]
        target_knee_frac = pc.sample_knee_fraction(rng, data_folder)

        ir_sag = pc.sample_ir_sag_v(c_rate_name, soh_deficit, rng, data_folder)
        # Sag fades in over the first ~5% of capacity rather than existing at
        # ah=0, mirroring curve_train.py's IR_DROP_DECAY_FRACTION mechanism.
        decay_window = 0.05 * cap_m
        sag_profile = ir_sag * (1.0 - np.exp(-ah / (decay_window + 1e-6)))

        # Cell-to-cell imbalance within a module is modeled as each of the 12
        # cells having its OWN slightly different capacity (manufacturing/
        # aging mismatch), not uniform noise added on top of one shared
        # curve. This matters: real data shows imbalance GROWING sharply near
        # the knee (max_cell_spread_at_end is roughly an order of magnitude
        # larger than mid-discharge imbalance), which uniform noise can't
        # reproduce -- confirmed by the validator flagging a zero-overlap
        # distribution on that exact feature. With per-cell capacity jitter,
        # cells reach their own individual knees at slightly different Ah,
        # so small mid-discharge differences amplify near the end (where
        # dV/d(fraction) is steepest) exactly like real packs.
        intra_module_cv = float(rng.uniform(0.003, 0.015))
        cell_caps = cap_m * (1.0 + rng.normal(0, intra_module_cv, size=MODULE_SIZE))
        if rng.random() < 0.15:  # occasional single mismatched cell, matches real long-tailed spread
            outlier_idx = int(rng.integers(0, MODULE_SIZE))
            cell_caps[outlier_idx] = cap_m * (1.0 - rng.uniform(0.03, 0.08))
        cell_caps = np.clip(cell_caps, cap_m * 0.85, cap_m * 1.05)

        residual_noise_std = pc.sample_cell_imbalance_v(c_rate_name, rng, data_folder) * 0.5
        for cell_offset in range(MODULE_SIZE):
            cell_num = (m - 1) * MODULE_SIZE + cell_offset + 1
            cell_cap = cell_caps[cell_offset]
            cell_frac = np.clip(ah / cell_cap, 0.0, 1.0)
            cell_warped = _warp_fraction(cell_frac, canonical_knee_frac, target_knee_frac)
            cell_shape = np.interp(cell_warped, grid, canonical)
            cell_v = plateau_v + cell_shape * span_v - sag_profile
            noise = rng.normal(0, residual_noise_std, size=n_rows)
            data[f'Cell {cell_num:03d}'] = cell_v + noise

    return pd.DataFrame(data), module_soh, module_capacity_ah


def build_synthetic_module_rows(n_instances, c_rate_name, seed=None, data_folder=DEFAULT_DATA_FOLDER,
                                 slice_pcts=SLICE_PCTS):
    rng = np.random.default_rng(seed)
    c_rate_value = 1.0 if c_rate_name == '1.0C' else 0.3
    rows = []

    for i in range(n_instances):
        df, module_soh, module_capacity_ah = generate_virtual_pack_dataframe(c_rate_name, rng, data_folder)
        cell_cols = get_cell_columns(df)
        total_rows = len(df)
        pack_id = f"synth_{c_rate_name}_{i:05d}"
        # Real data has both full-curve (is_sfct=0) and short-test (is_sfct=1)
        # files sliced the same way -- sample per instance so this feature
        # doesn't collapse to a constant (caught by the distributional
        # sanity check: every synthetic row had is_sfct=1.0 before this fix).
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
                row.update({'c_rate': c_rate_value, 'is_sfct': is_sfct_value, 'slice_start_pct': pct,
                            'module_idx': m, 'true_soh': module_soh[m], 'pack_id': pack_id})
                rows.append(row)

    return pd.DataFrame(rows)


def build_synthetic_module_dataset(n_per_crate=300, seed=None, data_folder=DEFAULT_DATA_FOLDER):
    dfs = [build_synthetic_module_rows(n_per_crate, c_rate_name, seed=seed, data_folder=data_folder)
           for c_rate_name in ('0.3C', '1.0C')]
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    df = build_synthetic_module_dataset(n_per_crate=50, seed=42)
    print(f"Generated {len(df)} synthetic rows")
    print(df[['pack_id', 'c_rate', 'module_idx', 'true_soh', 'end_voltage', 'voltage_drop',
               'mean_cell_imbalance_std']].head(20).to_string(index=False))
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic_module_dataset.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
