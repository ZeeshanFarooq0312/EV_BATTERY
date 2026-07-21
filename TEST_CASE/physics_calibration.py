"""Single source of truth for every real-data-measured constant used by the
synthetic module generator (synthetic_module_generator.py) -- replaces the
inconsistent EXTRA_IR_DROP_PER_SOH_POINT that used to be hardcoded separately
in curve_train.py ({'0.3C': 0.003, '1.0C': 0.015}) and module_soh_train.py/
SOH_MODELS_TRAIN.PY ({'0.3C': 0.03, '1.0C': 0.015} -- 10x too large at 0.3C
and the wrong C-rate ratio direction; confirmed wrong by measuring real data
directly, see calibrate()'s ir_sag_fit).

Every quantity here is exposed as a SAMPLING function (drawing from a fitted
mean/std or an empirical percentile range measured across all real full-curve
tests), not a fixed constant -- the whole point is that the synthetic
generator should never see the same physical parameter value twice.
"""

import os
import numpy as np
import pandas as pd

from curve_utils import (
    list_full_curve_files, extract_and_resample_curve, compute_soh,
    get_cell_columns, module_cell_columns, N_MODULES,
    extract_module_features_from_slice, parse_pack_and_crate,
)
from module_capacity_extrapolation import detect_gated_knee, find_crossing_index, _interp_crossing_ah

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_FOLDER = os.path.join(_REPO_ROOT, 'clean_data_for_test', 'OneDrive_1_7-9-2026_CLEANED')

_CACHE = {}


def _c_rate_bucket(c_rate):
    return '1.0C' if c_rate >= 0.9 else '0.3C'


def _measure_initial_ir_sag_v(ah, v, window_frac=0.01):
    total = ah[-1]
    mask = ah <= window_frac * total
    if mask.sum() < 3:
        return None
    return float(v[0] - v[mask][-1])


def _measure_pack_heterogeneity_mv(modules):
    end_vs = [modules[m]['min_v'][-1] for m in range(1, N_MODULES + 1)]
    return float((max(end_vs) - min(end_vs)) * 1000.0)


def _measure_cell_imbalance_mv(df, cell_cols, module_idx, window=500):
    mid = len(df) // 2
    cols = module_cell_columns(cell_cols, module_idx)
    lo, hi = max(0, mid - window), min(len(df), mid + window)
    return float(df[cols].iloc[lo:hi].std(axis=1).mean() * 1000.0)


def calibrate(data_folder=DEFAULT_DATA_FOLDER, force=False):
    """Scans every real full-curve file once; result cached per data_folder.
    Call calibrate(force=True) to recompute (e.g. after new data arrives)."""
    if not force and data_folder in _CACHE:
        return _CACHE[data_folder]

    ir_sag_rows = {'0.3C': [], '1.0C': []}  # (soh_deficit_points, sag_v)
    heterogeneity_mv = []
    imbalance_mv = {'0.3C': [], '1.0C': []}
    knee_fractions = []  # ah_at_knee / (best-available capacity estimate)
    # (soh_deficit_points, max_cell_spread_at_end, mean_cell_imbalance_std) per module,
    # sampled at a late-discharge slice -- used to teach the legacy synthetic
    # generator that these two features should GROW with degradation instead
    # of being frozen at the template's real value (see
    # _legacy_build_synthetic_rows: freezing them means a wide synthetic SOH
    # sweep pairs the SAME real spread/imbalance value with wildly different
    # SOH targets, actively teaching the model those features are irrelevant).
    structural_rows = {'0.3C': [], '1.0C': []}

    module_soh_lookup = None

    for entry in list_full_curve_files(data_folder):
        ah, pack_v, cell_std, modules = extract_and_resample_curve(entry['path'], want_modules=True)
        if ah is None:
            continue
        bucket = _c_rate_bucket(entry['c_rate'])
        pack_soh = compute_soh(ah[-1])

        sag = _measure_initial_ir_sag_v(ah, pack_v)
        if sag is not None:
            ir_sag_rows[bucket].append((100.0 - pack_soh, sag))

        heterogeneity_mv.append(_measure_pack_heterogeneity_mv(modules))

        df = pd.read_csv(entry['path'])
        cell_cols = get_cell_columns(df)
        for m in range(1, N_MODULES + 1):
            imbalance_mv[bucket].append(_measure_cell_imbalance_mv(df, cell_cols, m))

        if module_soh_lookup is None:
            from build_module_dataset import build_module_dataset
            mdf = build_module_dataset(data_folder, verbose=False)
            mdf['c_rate_bucket'] = mdf['c_rate'].apply(_c_rate_bucket)
            module_soh_lookup = mdf.groupby(['pack_id', 'c_rate_bucket', 'module_idx'])['module_soh'].mean().to_dict()

        pack_id, _ = parse_pack_and_crate(os.path.basename(entry['path']))
        if pack_id is not None:
            total_rows = len(df)
            late_slice = df.iloc[int(total_rows * 0.5):]
            for m in range(1, N_MODULES + 1):
                soh = module_soh_lookup.get((pack_id, bucket, m))
                if soh is None:
                    continue
                f = extract_module_features_from_slice(late_slice, module_cell_columns(cell_cols, m))
                if f is None:
                    continue
                structural_rows[bucket].append((100.0 - soh, f['max_cell_spread_at_end'], f['mean_cell_imbalance_std']))

        cutoff_v = float(df[cell_cols].min(axis=1).iloc[-1])
        for m in range(1, N_MODULES + 1):
            mv = modules[m]['min_v']
            idx = find_crossing_index(ah, mv, cutoff_v)
            if idx is not None:
                # Real Tier-0 module: exact capacity known, so an exact knee fraction.
                capacity_ah = _interp_crossing_ah(ah, mv, idx, cutoff_v)
                knee_idx = detect_gated_knee(ah, mv)
                if knee_idx is not None:
                    knee_fractions.append(ah[knee_idx] / capacity_ah)
            else:
                # Real Tier-1 module: true capacity is unknown, but the knee
                # ONSET itself is a directly-observed point regardless -- use
                # ah[-1] (a slight underestimate of true capacity, since the
                # module hadn't crossed yet) as the capacity proxy. This is
                # what lets the ~74 Tier-1 module-tests contribute here too,
                # not just the ~10 Tier-0 ones.
                knee_idx = detect_gated_knee(ah, mv)
                if knee_idx is not None:
                    knee_fractions.append(ah[knee_idx] / ah[-1])

    ir_sag_fit = {}
    for bucket, rows in ir_sag_rows.items():
        if len(rows) < 2:
            continue
        x = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        # Through-origin fit (slope = sum(xy)/sum(x^2)), not a free-intercept
        # regression: EXTRA_IR_DROP_PER_SOH_POINT is consumed as
        # `sag = slope * extra_degradation`, i.e. zero extra degradation must
        # imply zero extra sag by construction. A free-intercept fit on only
        # 5 points per bucket let the intercept (extrapolated to a
        # hypothetical 0-deficit pack) absorb most of the real C-rate
        # difference, leaving the slope itself nearly IDENTICAL between
        # 0.3C and 1.0C (and even inverted, 0.3C slightly above 1.0C) --
        # wrong, and not what this quantity is supposed to represent.
        slope = float(np.sum(x * y) / np.sum(x * x))
        resid_std = float(np.std(y - slope * x))
        ir_sag_fit[bucket] = {'slope': slope, 'resid_std': max(resid_std, 1e-4), 'n': len(rows)}

    imbalance_range = {b: (float(np.percentile(vals, 5)), float(np.percentile(vals, 95)))
                        for b, vals in imbalance_mv.items() if vals}

    structural_growth_fit = {}
    for bucket, rows in structural_rows.items():
        if len(rows) < 5:
            continue
        x = np.array([r[0] for r in rows])
        A = np.vstack([x, np.ones_like(x)]).T
        fit = {}
        for name, idx in (('spread', 1), ('imbalance', 2)):
            y = np.array([r[idx] for r in rows])
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            fit[name] = {'slope': float(slope), 'intercept': float(intercept)}
        structural_growth_fit[bucket] = fit

    result = {
        'ir_sag_fit': ir_sag_fit,
        'structural_growth_fit': structural_growth_fit,
        'heterogeneity_mv_range': (
            (float(np.percentile(heterogeneity_mv, 5)), float(np.percentile(heterogeneity_mv, 95)))
            if heterogeneity_mv else (400.0, 1000.0)
        ),
        'imbalance_mv_range': imbalance_range,
        'knee_fraction_range': (
            (float(np.percentile(knee_fractions, 5)), float(np.percentile(knee_fractions, 95)))
            if knee_fractions else (0.75, 0.95)
        ),
        'knee_fraction_median': float(np.median(knee_fractions)) if knee_fractions else 0.85,
        'n_knee_samples': len(knee_fractions),
    }
    _CACHE[data_folder] = result
    return result


def sample_ir_sag_v(c_rate_name, soh_deficit_points, rng, data_folder=DEFAULT_DATA_FOLDER):
    """Initial-of-discharge IR sag (volts) for a module at `soh_deficit_points`
    (100 - soh) below full, at C-rate bucket ('0.3C' or '1.0C'), sampled
    around the real-data-regressed (through-origin) slope plus residual noise."""
    cal = calibrate(data_folder)
    fit = cal['ir_sag_fit'].get(c_rate_name, {'slope': 0.02, 'resid_std': 0.005})
    mean_sag = fit['slope'] * soh_deficit_points
    return float(max(0.0, mean_sag + rng.normal(0, fit['resid_std'])))


def structural_growth_delta(c_rate_name, extra_degradation, data_folder=DEFAULT_DATA_FOLDER):
    """How much max_cell_spread_at_end / mean_cell_imbalance_std should shift
    for a module synthesized `extra_degradation` SOH-points weaker (or, if
    negative, healthier) than its real template -- real-data-regressed slope
    (see calibrate()'s structural_growth_fit), not zero (frozen), which is
    what _legacy_build_synthetic_rows used to do and which taught the model
    these features carry no information."""
    cal = calibrate(data_folder)
    fit = cal['structural_growth_fit'].get(c_rate_name)
    if not fit:
        return 0.0, 0.0
    return fit['spread']['slope'] * extra_degradation, fit['imbalance']['slope'] * extra_degradation


def sample_cell_imbalance_v(c_rate_name, rng, data_folder=DEFAULT_DATA_FOLDER):
    cal = calibrate(data_folder)
    lo, hi = cal['imbalance_mv_range'].get(c_rate_name, (1.0, 3.0))
    return float(rng.uniform(lo, hi)) / 1000.0


def sample_pack_heterogeneity_v(rng, data_folder=DEFAULT_DATA_FOLDER):
    cal = calibrate(data_folder)
    lo, hi = cal['heterogeneity_mv_range']
    return float(rng.uniform(lo, hi)) / 1000.0


def sample_knee_fraction(rng, data_folder=DEFAULT_DATA_FOLDER):
    cal = calibrate(data_folder)
    lo, hi = cal['knee_fraction_range']
    spread = max((hi - lo) / 4.0, 1e-3)
    return float(np.clip(rng.normal(cal['knee_fraction_median'], spread), lo - 0.05, 0.99))


if __name__ == "__main__":
    cal = calibrate()
    print("IR sag fit (through-origin: sag_mV = slope * soh_deficit_points):")
    for bucket, fit in cal['ir_sag_fit'].items():
        print(f"  {bucket}: slope={fit['slope']*1000:.2f} mV/pt  "
              f"resid_std={fit['resid_std']*1000:.2f} mV  (n={fit['n']})")
    print(f"\nHeterogeneity range (mV): {cal['heterogeneity_mv_range']}")
    print(f"Cell imbalance range (mV): {cal['imbalance_mv_range']}")
    print(f"Knee fraction range: {cal['knee_fraction_range']}  median={cal['knee_fraction_median']:.3f}  "
          f"(n={cal['n_knee_samples']})")
