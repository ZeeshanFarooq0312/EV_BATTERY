"""Standalone script: given a folder of raw test CSVs, compute each file's
ACTUAL C-rate from its own measured LoadUnitCurrent column -- not the
C-rate label in the filename. See app.py's investigation into pk4's two
1.0C-labeled sessions actually running at different currents (~0.93C vs
~0.99C) for why this can differ from the label.

Usage: edit FOLDER_PATH (and OUTPUT_CSV_PATH if desired) below, then run:
    python3 get_actual_c_rate.py
"""

import os
import pandas as pd

# Paste the folder to scan here.
FOLDER_PATH = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/clean_data_for_test/OneDrive_1_7-9-2026_CLEANED"
OUTPUT_CSV_PATH = "actual_c_rates.csv"

NOMINAL_CAPACITY = 156.0  # Ah, matches curve_utils.NOMINAL_CAPACITY


def get_actual_c_rate(csv_path):
    df = pd.read_csv(csv_path)
    if 'LoadUnitCurrent' not in df.columns:
        raise ValueError("'LoadUnitCurrent' column not found")
    mean_current = df['LoadUnitCurrent'].abs().mean()
    return mean_current, mean_current / NOMINAL_CAPACITY


def scan_folder(folder_path):
    rows = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith('.csv'):
            continue
        path = os.path.join(folder_path, fname)
        try:
            mean_current, actual_c_rate = get_actual_c_rate(path)
            rows.append({'filename': fname, 'mean_current_A': round(mean_current, 2),
                         'actual_c_rate': round(actual_c_rate, 3), 'error': ''})
        except Exception as e:
            rows.append({'filename': fname, 'mean_current_A': None,
                         'actual_c_rate': None, 'error': str(e)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = scan_folder(FOLDER_PATH)
    results.to_csv(OUTPUT_CSV_PATH, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved {len(results)} rows to {OUTPUT_CSV_PATH}")
