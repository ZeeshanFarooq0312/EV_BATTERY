"""Honest validation of module_capacity_extrapolation.py using only real,
directly-measured ground truth -- no assumed/hardcoded capacity numbers.

Method: across all full-curve (FFCT/FFT) files, find every module that
actually crossed cutoff within observed data (a Tier-0 "measured" capacity --
see module_capacity_extrapolation.py's docstring for why this is rare, ~11
total across this dataset). Re-truncate each of those real curves at several
points BEFORE its true crossing Ah (simulating it being censored earlier than
it really was), run the extrapolation method on the truncated version, and
compare the prediction to the real, known capacity.

This is a genuinely thin validation (n ~= 11 underlying curves, so results
below are reported per-pack/per-depth, not just pooled, and should be read as
"is this method directionally trustworthy", not "proven to high confidence").

Every held-out curve is real -- but the TEMPLATE used for shape-transfer
during validation is deliberately taken from a DIFFERENT test (since using
the curve's own real tail as its template would be circular). This is a
harder scenario than production actually faces: build_module_dataset.py
always has that SAME test's own crossing module available as template. So
Tier-2 error reported here is a conservative (pessimistic) estimate of
production accuracy, not an exact match to it.
"""

import os
import sys
import numpy as np
import pandas as pd

_ML_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ML_PIPELINE_DIR, 'core'))
sys.path.insert(0, os.path.join(_ML_PIPELINE_DIR, 'training'))

from curve_utils import compute_soh, list_full_curve_files, extract_and_resample_curve
from module_capacity_extrapolation import (
    find_crossing_index, estimate_module_capacity, naive_linear_capacity, _interp_crossing_ah,
)

DATA_FOLDER = os.path.join(os.path.dirname(_ML_PIPELINE_DIR),
                           'clean_data_for_test', 'OneDrive_1_7-9-2026_CLEANED')

DEPTH_FRACS = [0.5, 0.65, 0.75, 0.85, 0.90, 0.95, 0.98, 0.99]
MIN_VALID_MIN_V = 2.1
MAX_VALID_MIN_V = 2.4


def _c_rate_bucket(c_rate):
    return '1.0C' if c_rate >= 0.9 else '0.3C'


def build_ground_truth_pool(data_folder=DATA_FOLDER):
    """One entry per real Tier-0 (measured) crossing module found across all
    unique full-curve files: {pack_id, c_rate, module_idx, ah, v, true_capacity_ah}."""
    pool = []
    for entry in list_full_curve_files(data_folder):
        ah, pack_v, cell_std, modules = extract_and_resample_curve(entry['path'], want_modules=True)
        if ah is None:
            continue
        # cutoff_v is the pack's own final min-cell voltage -- read it directly
        # off the raw file's min_v column (verified byte-for-byte identical to
        # the literal running min of all 108 cells).
        cutoff_v = float(pd.read_csv(entry['path'], usecols=['min_v'])['min_v'].iloc[-1])
        if not (MIN_VALID_MIN_V <= cutoff_v <= MAX_VALID_MIN_V):
            print(f"  [skip] {os.path.basename(entry['path'])}: final min_v={cutoff_v:.3f} outside sanity band")
            continue

        for m, traces in modules.items():
            idx = find_crossing_index(ah, traces['min_v'], cutoff_v)
            if idx is None:
                continue
            true_cap = _interp_crossing_ah(ah, traces['min_v'], idx, cutoff_v)
            pool.append({
                'pack_id': entry['pack_id'], 'c_rate': entry['c_rate'], 'module_idx': m,
                'ah': ah, 'v': traces['min_v'], 'cutoff_v': cutoff_v, 'true_capacity_ah': true_cap,
            })
    return pool


def _pick_template(template_pool, exclude_pack_id, c_rate):
    """A different pack's crossing module, preferring the same C-rate bucket,
    deterministically (first match by list order). template_pool must never
    include the pack currently being evaluated."""
    candidates = [t for t in template_pool if t['pack_id'] != exclude_pack_id]
    same_bucket = [t for t in candidates if _c_rate_bucket(t['c_rate']) == _c_rate_bucket(c_rate)]
    if same_bucket:
        return same_bucket[0]
    if candidates:
        return candidates[0]
    return None


def run_validation(eval_items, template_pool, tail_fit_frac=0.1, min_pre_knee_slope=-0.005):
    """Evaluate extrapolation accuracy on `eval_items` (curves to re-truncate
    and predict), drawing shape-transfer templates from `template_pool` --
    kept as a separate argument so LOPO tuning can restrict template_pool to
    exclude the pack currently held out, without also starving the held-out
    pack's own evaluation of any valid template (see module docstring: using
    a curve's own tail as its template would be circular, so templates always
    come from a different pack). Returns a list of result rows."""
    rows = []
    for item in eval_items:
        template = _pick_template(template_pool, item['pack_id'], item['c_rate'])
        if template is None:
            continue
        for depth in DEPTH_FRACS:
            truncate_ah = depth * item['true_capacity_ah']
            mask = item['ah'] <= truncate_ah
            if mask.sum() < 20:
                continue
            trunc_ah, trunc_v = item['ah'][mask], item['v'][mask]

            try:
                tiered_cap, source, diag = estimate_module_capacity(
                    trunc_ah, trunc_v, item['cutoff_v'],
                    template_ah=template['ah'], template_v=template['v'],
                    tail_fit_frac=tail_fit_frac, min_pre_knee_slope=min_pre_knee_slope,
                )
            except Exception as e:
                tiered_cap, source = None, f'error:{e}'

            naive_cap = naive_linear_capacity(trunc_ah, trunc_v, item['cutoff_v'])

            rows.append({
                'pack_id': item['pack_id'], 'c_rate': item['c_rate'], 'module_idx': item['module_idx'],
                'depth_frac': depth, 'source': source,
                'true_ah': item['true_capacity_ah'],
                'tiered_pred_ah': tiered_cap,
                'naive_pred_ah': naive_cap,
                'tiered_err_ah': (abs(tiered_cap - item['true_capacity_ah']) if tiered_cap is not None else None),
                'naive_err_ah': (abs(naive_cap - item['true_capacity_ah']) if naive_cap is not None else None),
            })
    return rows


def _mae(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(vals)) if vals else float('nan')


def report(rows):
    print(f"\n{'='*100}")
    print(f"Validation pool: {len(set((r['pack_id'], r['module_idx'], r['c_rate']) for r in rows))} real crossing-module curves, "
          f"{len(rows)} (curve x depth) truncation trials")
    print(f"{'='*100}")

    print(f"\n{'Pack':<6}{'C-rate':<8}{'Depth':<8}{'Tiered MAE(Ah)':<16}{'Naive MAE(Ah)':<16}"
          f"{'Tiered MAE(SOH%)':<18}{'Naive MAE(SOH%)':<16}{'n':<4}")
    print('-' * 100)
    packs = sorted(set(r['pack_id'] for r in rows))
    for pack in packs:
        for depth in DEPTH_FRACS:
            sub = [r for r in rows if r['pack_id'] == pack and r['depth_frac'] == depth]
            if not sub:
                continue
            tiered_soh_err = [abs(compute_soh(r['tiered_pred_ah']) - compute_soh(r['true_ah']))
                               for r in sub if r['tiered_pred_ah'] is not None]
            naive_soh_err = [abs(compute_soh(r['naive_pred_ah']) - compute_soh(r['true_ah']))
                              for r in sub if r['naive_pred_ah'] is not None]
            print(f"{pack:<6}{sub[0]['c_rate']:<8}{depth:<8.2f}"
                  f"{_mae(sub,'tiered_err_ah'):<16.2f}{_mae(sub,'naive_err_ah'):<16.2f}"
                  f"{(np.mean(tiered_soh_err) if tiered_soh_err else float('nan')):<18.2f}"
                  f"{(np.mean(naive_soh_err) if naive_soh_err else float('nan')):<16.2f}{len(sub):<4}")

    print('-' * 100)
    overall_tiered_soh = [abs(compute_soh(r['tiered_pred_ah']) - compute_soh(r['true_ah']))
                           for r in rows if r['tiered_pred_ah'] is not None]
    overall_naive_soh = [abs(compute_soh(r['naive_pred_ah']) - compute_soh(r['true_ah']))
                          for r in rows if r['naive_pred_ah'] is not None]
    print(f"OVERALL (pooled across all packs/depths -- correlated draws from ~{len(packs)} packs, read with caution):")
    print(f"  Tiered method: MAE = {_mae(rows,'tiered_err_ah'):.2f} Ah / {np.mean(overall_tiered_soh):.2f} SOH points")
    print(f"  Naive linear:  MAE = {_mae(rows,'naive_err_ah'):.2f} Ah / {np.mean(overall_naive_soh):.2f} SOH points")
    improvement = np.mean(overall_naive_soh) / max(np.mean(overall_tiered_soh), 1e-9)
    print(f"  Tiered method beats naive linear by {improvement:.1f}x on SOH-point MAE")

    tier_counts = {}
    for r in rows:
        tier_counts[r['source']] = tier_counts.get(r['source'], 0) + 1
    print(f"  Tier usage across trials: {tier_counts}")


def tune_params_lopo(pool):
    """Small grid search over the extrapolation's free parameters, tuned via
    leave-one-pack-out: for each held-out pack, pick the (tail_fit_frac,
    min_pre_knee_slope) that minimizes SOH-point MAE on all OTHER packs, then
    report that combo's error on the held-out pack. Avoids overfitting the
    method to this small pool."""
    grid = [(tf, sl) for tf in (0.10, 0.15, 0.20, 0.25) for sl in (-0.005, -0.01, -0.02, -0.04)]
    packs = sorted(set(item['pack_id'] for item in pool))
    holdout_errors = []

    for held_out_pack in packs:
        train_items = [it for it in pool if it['pack_id'] != held_out_pack]
        test_items = [it for it in pool if it['pack_id'] == held_out_pack]
        if not train_items or not test_items:
            continue
        best_params, best_score = None, float('inf')
        for tf, sl in grid:
            rows = run_validation(train_items, train_items, tail_fit_frac=tf, min_pre_knee_slope=sl)
            soh_errs = [abs(compute_soh(r['tiered_pred_ah']) - compute_soh(r['true_ah']))
                        for r in rows if r['tiered_pred_ah'] is not None]
            score = np.mean(soh_errs) if soh_errs else float('inf')
            if score < best_score:
                best_score, best_params = score, (tf, sl)

        # Templates for the held-out pack's evaluation still come only from
        # train_items -- never from the pack being held out.
        test_rows = run_validation(test_items, train_items, tail_fit_frac=best_params[0], min_pre_knee_slope=best_params[1])
        test_soh_errs = [abs(compute_soh(r['tiered_pred_ah']) - compute_soh(r['true_ah']))
                         for r in test_rows if r['tiered_pred_ah'] is not None]
        held_err = np.mean(test_soh_errs) if test_soh_errs else float('nan')
        holdout_errors.append(held_err)
        print(f"  Held out {held_out_pack}: best params on other packs = {best_params} "
              f"(train MAE {best_score:.2f} SOH pts) -> held-out MAE = {held_err:.2f} SOH pts")

    print(f"\nLOPO-tuned mean held-out SOH-point MAE: {np.nanmean(holdout_errors):.2f}")


if __name__ == "__main__":
    print(f"Building ground-truth pool from {DATA_FOLDER} ...")
    pool = build_ground_truth_pool()
    print(f"Found {len(pool)} real Tier-0 (measured) crossing-module curves:")
    for item in pool:
        print(f"  {item['pack_id']} @ {item['c_rate']}C, module {item['module_idx']}: "
              f"true capacity = {item['true_capacity_ah']:.2f} Ah ({compute_soh(item['true_capacity_ah']):.2f}% SOH)")

    print("\n--- Validation with default (LOPO-tuned) parameters (tail_fit_frac=0.1, min_pre_knee_slope=-0.005) ---")
    rows = run_validation(pool, pool)
    report(rows)

    print("\n--- Leave-one-pack-out parameter tuning ---")
    tune_params_lopo(pool)
