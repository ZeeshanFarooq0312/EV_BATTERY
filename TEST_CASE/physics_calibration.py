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
    # Same problem, different features: ah_per_voltage_drop/mean_dV_dAh/
    # std_dV_dAh/min_dV_dAh describe the CURVE'S SHAPE (how fast voltage moves
    # per Ah, and how that varies), not just its level -- _legacy_build_synthetic_rows
    # used to leave these frozen at the template's real value too, so a
    # synthetic row's degraded voltage LEVEL was never accompanied by the
    # shape change a genuinely more-degraded module actually has (steeper,
    # more variable dV/dAh as the knee approaches). Confirmed this matters:
    # pk5 (the one real pack with genuinely low SOH, ~86-92%) was the worst-
    # or near-worst-generalizing pack under leave-one-pack-out even AFTER
    # its own real data was added to training -- the model had almost no
    # signal that these shape features move at all with degradation.
    shape_rows = {'0.3C': [], '1.0C': []}
    # (soh_deficit_points, voltage at ~50% capacity) per module -- fits how
    # much the PERSISTING mid-discharge voltage level actually drops per
    # degradation point, as distinct from ir_sag_fit above (which measures
    # only the fast INITIAL transient over the first 1% of capacity). See
    # persisting_sag_fit below for why these two must not be conflated.
    voltage_level_rows = {'0.3C': [], '1.0C': []}
    # (plateau_v, span_v) per real full-curve file, per C-rate bucket --
    # synthetic_module_generator.py previously sampled these from hardcoded
    # guesses (plateau ~3.45-3.75V, span ~1.0-1.4V) completely independent of
    # real data, which is why its synthetic end_voltage [2.03V,2.81V] sat well
    # below real end_voltage [2.71V,3.18V] even after the SOH-range fix (that
    # only fixed CAPACITY, not voltage LEVEL) -- caught by
    # validate_synthetic_module_generator.py's distributional sanity check.
    plateau_span_rows = {'0.3C': [], '1.0C': []}

    module_soh_lookup = None

    for entry in list_full_curve_files(data_folder):
        ah, pack_v, cell_std, modules = extract_and_resample_curve(entry['path'], want_modules=True)
        if ah is None:
            continue
        bucket = _c_rate_bucket(entry['c_rate'])
        pack_soh = compute_soh(ah[-1])

        sag = _measure_initial_ir_sag_v(ah, pack_v)
        # Physically, an initial-of-discharge IR sag can't be negative (a
        # rested pack's voltage only drops as current ramps up, never rises
        # by a meaningful amount) and shouldn't plausibly exceed ~0.5V given
        # every other real pack measures 0.03-0.17V. Caught via a visual
        # synthetic-vs-real curve overlay: pk6's 0.3C file measured -1.09V --
        # a single resampling-edge-case artifact (extract_and_resample_curve's
        # mask+sort lets one anomalous low-voltage row from the raw file sort
        # to ah=0 ahead of the real plateau readings), not a real sag -- which
        # alone flipped the through-origin fit's slope negative and inflated
        # its residual std to 432mV (vs 1.0C's clean 27mV), causing the
        # synthetic generator to occasionally sample a huge, physically
        # nonsensical sag that crashed a curve's voltage far too early.
        # Filtering here fixes the calibration without touching
        # extract_and_resample_curve, which is shared by the whole pipeline.
        if sag is not None and 0.0 <= sag <= 0.5:
            ir_sag_rows[bucket].append((100.0 - pack_soh, sag))

        heterogeneity_mv.append(_measure_pack_heterogeneity_mv(modules))

        # Same plateau window build_canonical_shape() itself uses to
        # normalize a curve (frac in (0.2, 0.4)) -- kept consistent so the
        # sampled plateau_v/span_v describe the same reference point the
        # canonical shape is warped around.
        frac = ah / ah[-1]
        plateau_mask = (frac > 0.2) & (frac < 0.4)
        plateau_v = float(np.mean(pack_v[plateau_mask])) if np.any(plateau_mask) else float(pack_v[0])
        end_v = float(pack_v[-1])
        span_v = plateau_v - end_v
        if span_v > 0:
            plateau_span_rows[bucket].append((plateau_v, span_v))

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
                shape_rows[bucket].append((100.0 - soh, f['ah_per_voltage_drop'], f['mean_dV_dAh'],
                                            f['std_dV_dAh'], f['min_dV_dAh']))
                # late_slice starts at 50% of the file, so f['start_voltage']
                # (mean of its own first 10 rows) is this module's voltage at
                # ~50% capacity delivered -- squarely inside the 40%-70%
                # window SLICE_PCTS actually samples for training features.
                voltage_level_rows[bucket].append((100.0 - soh, f['start_voltage']))

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

    # generate_virtual_pack_dataframe applies its sag_profile as a PERSISTING
    # depression (rises over the first ~5-20% of capacity, then holds for
    # the rest of the discharge) -- it was reusing ir_sag_fit's slope for
    # this, which measures something physically different (the fast initial
    # transient over just the first 1% of capacity). Measured directly: real
    # mid-discharge (~50%) voltage drops only 0.0025-0.0043 V per SOH-deficit
    # point, vs ir_sag_fit's 0.0053-0.0155 V/pt -- i.e. the persisting sag was
    # 2-3.6x too large, sitting synthetic curves systematically ~0.1-0.2V
    # below real ones across most of the discharge (confirmed on a
    # synthetic-vs-real per-module overlay). Free-intercept fit (like
    # structural/shape_growth_fit above), since only the SLOPE is consumed --
    # the intercept (a hypothetical 0-deficit module's absolute voltage
    # level) is irrelevant, sample_plateau_v already supplies that baseline.
    persisting_sag_fit = {}
    for bucket, rows in voltage_level_rows.items():
        if len(rows) < 5:
            continue
        x = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        resid_std = float(np.std(y - (slope * x + intercept)))
        # Real degradation depresses voltage (slope <= 0 expected); clip a
        # noise-driven sign flip to 0 rather than let sag go negative (which
        # would mean synthetic voltage RISING with degradation).
        persisting_sag_fit[bucket] = {'slope': float(max(0.0, -slope)), 'resid_std': max(resid_std, 1e-4), 'n': len(rows)}

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

    shape_growth_fit = {}
    for bucket, rows in shape_rows.items():
        if len(rows) < 5:
            continue
        x = np.array([r[0] for r in rows])
        A = np.vstack([x, np.ones_like(x)]).T
        fit = {}
        for name, idx in (('ah_per_voltage_drop', 1), ('mean_dV_dAh', 2), ('std_dV_dAh', 3), ('min_dV_dAh', 4)):
            y = np.array([r[idx] for r in rows])
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            fit[name] = {'slope': float(slope), 'intercept': float(intercept)}
        shape_growth_fit[bucket] = fit

    plateau_v_range = {b: (float(np.percentile([r[0] for r in rows], 5)), float(np.percentile([r[0] for r in rows], 95)))
                        for b, rows in plateau_span_rows.items() if rows}
    span_v_range = {b: (float(np.percentile([r[1] for r in rows], 5)), float(np.percentile([r[1] for r in rows], 95)))
                     for b, rows in plateau_span_rows.items() if rows}

    result = {
        'ir_sag_fit': ir_sag_fit,
        'persisting_sag_fit': persisting_sag_fit,
        'structural_growth_fit': structural_growth_fit,
        'shape_growth_fit': shape_growth_fit,
        'plateau_v_range': plateau_v_range,
        'span_v_range': span_v_range,
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


def sample_persisting_sag_v(c_rate_name, soh_deficit_points, rng, data_folder=DEFAULT_DATA_FOLDER):
    """Voltage depression (volts) that PERSISTS through the rest of the
    discharge for a module `soh_deficit_points` below full -- NOT the same
    quantity as sample_ir_sag_v (that's the fast initial transient over the
    first ~1% of capacity only). generate_virtual_pack_dataframe previously
    reused sample_ir_sag_v's slope for its persisting sag_profile, which is
    2-3.6x too large relative to what real mid-discharge voltage actually
    does per degradation point (see persisting_sag_fit in calibrate())."""
    cal = calibrate(data_folder)
    fit = cal['persisting_sag_fit'].get(c_rate_name, {'slope': 0.004, 'resid_std': 0.02})
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


def shape_growth_delta(c_rate_name, extra_degradation, data_folder=DEFAULT_DATA_FOLDER):
    """How much ah_per_voltage_drop / mean_dV_dAh / std_dV_dAh / min_dV_dAh --
    the features describing the discharge curve's SHAPE at this slice, not
    just its voltage level -- should shift for a module synthesized
    `extra_degradation` SOH-points weaker than its real template, per
    calibrate()'s shape_growth_fit. Same rationale as structural_growth_delta:
    freezing these at the template's value pairs one real shape with an
    entire synthetic SOH sweep, teaching the model they carry no signal --
    the specific gap that left pk5 (the one real pack with genuinely low
    SOH) poorly generalized even after its own real data was in training.
    Returns 0.0 for any feature without enough real data to fit."""
    cal = calibrate(data_folder)
    fit = cal['shape_growth_fit'].get(c_rate_name)
    if not fit:
        return 0.0, 0.0, 0.0, 0.0
    return (fit['ah_per_voltage_drop']['slope'] * extra_degradation,
            fit['mean_dV_dAh']['slope'] * extra_degradation,
            fit['std_dV_dAh']['slope'] * extra_degradation,
            fit['min_dV_dAh']['slope'] * extra_degradation)


def sample_plateau_v(c_rate_name, rng, data_folder=DEFAULT_DATA_FOLDER):
    """Plateau (near-start) voltage a synthetic virtual pack's curve is built
    around, drawn from real full-curve packs' own measured plateau (mean
    voltage in the 20-40% capacity fraction window) at this C-rate -- was a
    hardcoded (3.45, 3.75) guess uncorrelated with real data."""
    cal = calibrate(data_folder)
    lo, hi = cal['plateau_v_range'].get(c_rate_name, (3.45, 3.75))
    return float(rng.uniform(lo, hi))


def sample_span_v(c_rate_name, rng, data_folder=DEFAULT_DATA_FOLDER):
    """Plateau-to-cutoff voltage span, drawn from real full-curve packs' own
    measured (plateau_v - final_v) at this C-rate -- was a hardcoded
    (1.0, 1.4) guess."""
    cal = calibrate(data_folder)
    lo, hi = cal['span_v_range'].get(c_rate_name, (1.0, 1.4))
    return float(rng.uniform(lo, hi))


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
    print("\nPersisting sag fit (mid-discharge depression = slope * soh_deficit_points):")
    for bucket, fit in cal['persisting_sag_fit'].items():
        print(f"  {bucket}: slope={fit['slope']*1000:.2f} mV/pt  "
              f"resid_std={fit['resid_std']*1000:.2f} mV  (n={fit['n']})")
    print(f"\nHeterogeneity range (mV): {cal['heterogeneity_mv_range']}")
    print(f"Cell imbalance range (mV): {cal['imbalance_mv_range']}")
    print(f"Knee fraction range: {cal['knee_fraction_range']}  median={cal['knee_fraction_median']:.3f}  "
          f"(n={cal['n_knee_samples']})")
