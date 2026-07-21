"""Builds the per-module capacity/SOH labeled dataset from real full-curve
(FFCT/FFT) test files.

For each unique (pack_id, c_rate) test: splits the pack's 108 cells into 9
modules (sequential 12-cell blocks), reads that test's own realized cutoff
voltage directly off the min_v column, identifies which module(s) actually
crossed it within observed data (Tier 0 -- real, measured capacity), and
estimates every other module's capacity via the validated tiered
extrapolation method (see module_capacity_extrapolation.py), using that same
test's own weakest measured module as the shape-transfer template.

Sanity-checks the computed pack-level (= weakest module's) SOH against the
already-trusted actual_soh_by_crate.csv before accepting anything else from
that file, so a bug here can't silently drift from numbers already validated
by the existing pack-level pipeline.
"""

import os
import pandas as pd

from curve_utils import compute_soh, list_full_curve_files, extract_and_resample_curve
from module_capacity_extrapolation import find_crossing_index, _interp_crossing_ah, estimate_module_capacity

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(_REPO_ROOT, 'clean_data_for_test', 'OneDrive_1_7-9-2026_CLEANED')
ACTUAL_SOH_CSV = os.path.join(_REPO_ROOT, 'actual_soh_by_crate.csv')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'module_capacity_dataset.csv')

MIN_VALID_MIN_V = 2.1
MAX_VALID_MIN_V = 2.4
SOH_SANITY_TOL_PCT = 0.5  # percentage points


def _load_actual_soh_lookup(path=ACTUAL_SOH_CSV):
    df = pd.read_csv(path)
    return dict(zip(df['file'], df['actual_soh']))


def build_module_dataset(data_folder=DATA_FOLDER, verbose=True):
    actual_soh_lookup = _load_actual_soh_lookup()
    rows = []

    for entry in list_full_curve_files(data_folder):
        fname = os.path.basename(entry['path'])
        ah, pack_v, cell_std, modules = extract_and_resample_curve(entry['path'], want_modules=True)
        if ah is None:
            if verbose:
                print(f"  [skip] {fname}: missing AHDischarge")
            continue

        cutoff_v = float(pd.read_csv(entry['path'], usecols=['min_v'])['min_v'].iloc[-1])
        if not (MIN_VALID_MIN_V <= cutoff_v <= MAX_VALID_MIN_V):
            if verbose:
                print(f"  [skip] {fname}: final min_v={cutoff_v:.3f} outside sanity band "
                      f"[{MIN_VALID_MIN_V}, {MAX_VALID_MIN_V}]")
            continue

        tier0 = {}
        for m, traces in modules.items():
            idx = find_crossing_index(ah, traces['min_v'], cutoff_v)
            if idx is not None:
                tier0[m] = _interp_crossing_ah(ah, traces['min_v'], idx, cutoff_v)

        if not tier0:
            if verbose:
                print(f"  [skip] {fname}: no module crossed cutoff within observed data (unexpected)")
            continue

        pack_soh_here = compute_soh(ah[-1])
        if fname in actual_soh_lookup:
            expected = actual_soh_lookup[fname]
            if abs(pack_soh_here - expected) > SOH_SANITY_TOL_PCT:
                raise AssertionError(
                    f"{fname}: computed pack SOH {pack_soh_here:.2f}% doesn't match the already-"
                    f"trusted actual_soh_by_crate.csv value {expected:.2f}% (tolerance {SOH_SANITY_TOL_PCT} pts). "
                    f"Something in the new pipeline disagrees with the existing, validated pack-level numbers."
                )
        elif verbose:
            print(f"  [note] {fname}: not in actual_soh_by_crate.csv (new file/pack) -- no cross-check available")

        weakest_module = min(tier0, key=tier0.get)
        template_v = modules[weakest_module]['min_v']

        for m, traces in modules.items():
            if m in tier0:
                capacity_ah, source, diag = tier0[m], 'measured', {'tier': 0}
            else:
                capacity_ah, source, diag = estimate_module_capacity(
                    ah, traces['min_v'], cutoff_v, template_ah=ah, template_v=template_v,
                )
            rows.append({
                'file': fname, 'pack_id': entry['pack_id'], 'c_rate': entry['c_rate'],
                'module_idx': m, 'capacity_ah': capacity_ah, 'module_soh': compute_soh(capacity_ah),
                'label_source': source, 'cutoff_v': cutoff_v,
                'template_module': weakest_module if m not in tier0 else None,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(f"Building module dataset from {DATA_FOLDER} ...")
    df = build_module_dataset()
    print(f"\nBuilt module dataset: {len(df)} rows across {df['file'].nunique()} files "
          f"({df['pack_id'].nunique()} packs)")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")

    print("\nPer-pack/module summary (weakest module first, averaged across C-rates):")
    summary = df.groupby(['pack_id', 'module_idx'])['module_soh'].agg(['mean', 'std', 'count']).reset_index()
    for pack in sorted(summary['pack_id'].unique()):
        sub = summary[summary['pack_id'] == pack].sort_values('mean')
        print(f"\n  {pack}:")
        for _, r in sub.iterrows():
            print(f"    module {int(r['module_idx'])}: mean SOH={r['mean']:.2f}%  "
                  f"std={r['std']:.2f}  n={int(r['count'])}")

    print("\nLabel source breakdown:")
    print(df['label_source'].value_counts().to_string())
