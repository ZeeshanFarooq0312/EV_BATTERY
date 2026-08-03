"""Cleans raw pack characterisation-test CSV exports down to just the real
discharge window, matching the convention every file under DATA_FOLDER
already uses (see BS_Data/ for what a raw export looks like vs. DATA_FOLDER
for the cleaned result).

Raw exports bundle several phases in one file: a pre-test CC-CV charge
(LoadUnitCurrent positive, tapering as cells approach full), a short rest
(current ~0), the real constant-current DISCHARGE test (current negative --
this is the only part any downstream code, curve_utils.py /
build_module_dataset.py / app.py, actually uses), then a post-test rest and
the start of the next recharge (current flips positive again, and the
AHDischarge counter resets back near 0 -- so a naive "AHDischarge < threshold"
filter can't tell that apart from the pre-test phase and leaves the real junk
in place).

This is the single, importable source of truth for that cleaning step --
used by both the manual project-root new_raw_file.py CLI and (once wired up)
the UI's "add new training data" upload route, so both go through the exact
same discharge-block detection rather than two copies drifting apart.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_module_dataset import DATA_FOLDER

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Drop zone for raw CSVs a user has uploaded (via the UI, once that route
# exists, or manually) that still need cleaning before they belong in
# DATA_FOLDER.
RAW_UPLOADS_FOLDER = os.path.join(_REPO_ROOT, 'raw_uploads')

CURRENT_COL = 'LoadUnitCurrent'
# Comfortably above rest-phase sensor noise (observed ~+-0.1A) and comfortably
# below the smallest real discharge current observed (a several-second ramp
# starting around -20A before settling at the target C-rate current).
DISCHARGE_CURRENT_THRESHOLD = -1.0
MIN_BLOCK_LEN = 30  # guards against a spurious short run winning on a bad/near-empty file


def find_discharge_block(df, current_col=CURRENT_COL, threshold=DISCHARGE_CURRENT_THRESHOLD):
    """(start, end) inclusive row indices of the longest contiguous run where
    current_col < threshold -- the real discharge test, isolated from the
    pre-test charge/rest and the post-test rest/recharge on either side."""
    mask = (df[current_col].values < threshold).astype(int)
    padded = np.concatenate(([0], mask, [0]))
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    if len(starts) == 0:
        raise ValueError(f"No rows with {current_col} < {threshold} found -- is this really a discharge test file?")

    lengths = ends - starts + 1
    best = int(np.argmax(lengths))
    if lengths[best] < MIN_BLOCK_LEN:
        raise ValueError(f"Longest discharging run is only {lengths[best]} rows -- too short to trust; check the file.")
    return int(starts[best]), int(ends[best])


def clean_raw_file(input_path, output_path):
    """Reads input_path, trims it to just the real discharge block, and
    writes the result to output_path. Returns the cleaned DataFrame."""
    df = pd.read_csv(input_path)
    start, end = find_discharge_block(df)
    cleaned = df.iloc[start:end + 1].reset_index(drop=True)

    print(f"{os.path.basename(input_path)}: {len(df)} rows -> kept rows {start}-{end} "
          f"({len(cleaned)} rows) -- dropped {start} pre-test rows "
          f"(charge/rest) and {len(df) - end - 1} post-test rows (rest/recharge)")

    # Every file under DATA_FOLDER carries a precomputed 'min_v' column
    # (row-wise min across all 108 Cell NNN columns) -- build_module_dataset.py
    # reads it directly via usecols=['min_v'] rather than recomputing from the
    # cell columns each time. Raw exports don't have it; add it here so this
    # output matches that convention exactly.
    if 'min_v' not in cleaned.columns:
        cell_cols = [c for c in cleaned.columns if c.startswith('Cell ')]
        cleaned['min_v'] = cleaned[cell_cols].min(axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return cleaned


def clean_uploaded_file(input_path, output_dir=DATA_FOLDER):
    """Cleans one uploaded raw CSV and writes it into output_dir under its
    own filename. This is what a per-file "add new training data" upload
    route should call on each file right after it's saved to disk."""
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    clean_raw_file(input_path, output_path)
    return output_path


def clean_all_uploads(input_dir=RAW_UPLOADS_FOLDER, output_dir=DATA_FOLDER):
    """Batch-cleans every CSV in input_dir (e.g. everything a user has
    uploaded since the last run) straight into output_dir, ready for the next
    retrain. Skips files that fail (e.g. not actually a discharge test, or
    already cleaned/no charge phase to trim) rather than aborting the whole
    batch, and prints a summary at the end."""
    if not os.path.isdir(input_dir):
        print(f"No raw uploads folder at {input_dir} -- nothing to clean.")
        return []

    csv_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.csv'))
    if not csv_files:
        print(f"No CSV files found in {input_dir}.")
        return []

    results = []
    for fname in csv_files:
        input_path = os.path.join(input_dir, fname)
        try:
            clean_uploaded_file(input_path, output_dir)
            results.append((fname, 'ok'))
        except Exception as e:
            print(f"  [skip] {fname}: {e}")
            results.append((fname, f'skipped: {e}'))

    ok = sum(1 for _, status in results if status == 'ok')
    print(f"\nCleaned {ok}/{len(csv_files)} file(s) into {output_dir}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean raw pack CSV export(s) down to just the real discharge window.")
    parser.add_argument(
        'input', nargs='?', default=RAW_UPLOADS_FOLDER,
        help=f"A raw CSV file, or a folder of raw CSVs to clean (default: {RAW_UPLOADS_FOLDER})")
    parser.add_argument(
        '--output-dir', default=DATA_FOLDER,
        help=f"Folder to write the cleaned CSV(s) into (default: {DATA_FOLDER})")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        clean_all_uploads(args.input, args.output_dir)
    else:
        clean_uploaded_file(args.input, args.output_dir)
