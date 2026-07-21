import pandas as pd
import numpy as np
import joblib
import os
import re
import warnings
warnings.filterwarnings('ignore')

from curve_utils import (
    NOMINAL_CAPACITY, HEAD_CHECKPOINTS_PCT, SFT_SAMPLE_FRACTIONS,
    compute_soh, extract_and_resample_curve, extract_enhanced_features,
)

# --- CONFIGURATION ---
DATA_FOLDER = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/fft_raw_data"
CUTOFF_PCT = 0.50  # simulate a short field test covering the last 50% of the discharge


def run_diagnostic(target_c_rate):
    model_path = f"reconstruction_model_{'0_3C' if target_c_rate == 0.3 else '1_0C'}_v10_knee.pkl"
    print(f"Loading model: {model_path}")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Run curve_train.py first!")
        return

    checkpoint_errors = {pct: [] for pct in HEAD_CHECKPOINTS_PCT}
    valid_files_count = 0

    print(f"\nStarting diagnostic evaluation for {target_c_rate}C files (cutoff at {int(CUTOFF_PCT*100)}%)...")

    for file in os.listdir(DATA_FOLDER):
        if not file.endswith('.csv'):
            continue
        if 'FFCT' not in file.upper() and 'FFT' not in file.upper():
            continue

        c_rate_match = re.search(r'(\d+(?:\.\d+)?)C', file)
        if not c_rate_match:
            continue
        c_rate = float(c_rate_match.group(1))
        if target_c_rate == 0.3 and abs(c_rate - 0.3) > 0.1:
            continue
        if target_c_rate == 1.0 and c_rate < 0.9:
            continue

        ah, v, cell_std = extract_and_resample_curve(os.path.join(DATA_FOLDER, file))
        if ah is None or len(ah) < 100:
            continue

        total_cap = ah[-1]
        true_soh = compute_soh(total_cap)

        cutoff_ah = float(total_cap * CUTOFF_PCT)
        cutoff_idx = np.searchsorted(ah, cutoff_ah)
        if cutoff_idx < 20:
            continue

        sft_ah = ah[cutoff_idx:] - ah[cutoff_idx]
        sft_v = v[cutoff_idx:]
        sft_cell_imbalance = cell_std[cutoff_idx:] if cell_std is not None else None

        # True target voltages AT THE PERCENTAGE CHECKPOINTS OF THE COMPLETE CURVE
        true_target_v = [float(np.interp(pct * total_cap, ah, v)) for pct in HEAD_CHECKPOINTS_PCT]

        sft_sampled_v = [float(np.interp(frac * sft_ah[-1], sft_ah, sft_v)) for frac in SFT_SAMPLE_FRACTIONS]
        feats = extract_enhanced_features(sft_v, sft_ah, sft_cell_imbalance)

        features = sft_sampled_v + [
            true_soh,
            feats['initial_slope'], feats['final_slope'], feats['overall_slope'],
            feats['mean_curvature'], feats['max_curvature'], feats['voltage_std'],
            feats['voltage_range'], feats['plateau_length'], feats['end_voltage_drop'],
            feats['end_slope'], feats['cell_imbalance_mean'], feats['cell_imbalance_std'],
            feats['tail_knee_pct'], feats['tail_knee_slope'],
            float(cutoff_ah)
        ]

        pred_v = model.predict(np.array([features]))[0]

        for i, pct in enumerate(HEAD_CHECKPOINTS_PCT):
            error = abs(pred_v[i] - true_target_v[i]) * 1000  # mV
            checkpoint_errors[pct].append(error)

        valid_files_count += 1
        print(f"  Processed: {file} (True SOH: {true_soh:.2f}%)")

    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC RESULTS for {target_c_rate}C (Evaluated on {valid_files_count} real files)")
    print(f"{'='*80}")
    print(f"{'Curve %':<10} | {'Avg Error (mV)':<15} | {'Max Error (mV)':<15} | {'Sample Count'}")
    print("-" * 80)

    knee_region_errors = []
    for pct in HEAD_CHECKPOINTS_PCT:
        errors = checkpoint_errors[pct]
        if pct >= 0.72:
            knee_region_errors.extend(errors)
        if len(errors) > 0:
            print(f"{pct*100:<9.2f}% | {np.mean(errors):<15.2f} | {np.max(errors):<15.2f} | {len(errors)}")
        else:
            print(f"{pct*100:<9.2f}% | {'N/A':<15} | {'N/A':<15} | 0")

    if knee_region_errors:
        print("-" * 80)
        print(f"Knee region (72-100%) avg error: {np.mean(knee_region_errors):.2f} mV | max: {np.max(knee_region_errors):.2f} mV")


if __name__ == "__main__":
    for target_c_rate in [0.3, 1.0]:
        run_diagnostic(target_c_rate)
        print()
