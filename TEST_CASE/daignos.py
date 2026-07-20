import pandas as pd
import numpy as np
import joblib
import os
import re
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_FOLDER = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/fft_raw_data"
NOMINAL_CAPACITY = 156.0

HEAD_CHECKPOINTS_PCT = [
    0.0, 0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075, 0.0875, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325,
    0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.7125, 0.725, 0.7375,
    0.75, 0.7625, 0.775, 0.7875, 0.8, 0.8125, 0.825, 0.8375, 0.85, 0.8625, 0.875, 0.8875, 0.9, 0.9125, 0.925, 0.9375,
    0.95, 0.9625, 0.975, 0.9875, 1.0
]
SFT_SAMPLE_FRACTIONS = [
    0.0, 0.026, 0.052, 0.078, 0.104, 0.13, 0.156, 0.182, 0.208, 0.234, 0.26, 0.286, 0.312, 0.338, 0.364, 0.39, 0.416,
    0.442, 0.468, 0.494, 0.52, 0.546, 0.572, 0.598, 0.624, 0.65, 0.6635, 0.6764, 0.6894, 0.7023, 0.7153, 0.7282, 0.7412,
    0.7541, 0.767, 0.78, 0.7929, 0.8059, 0.8188, 0.8318, 0.8447, 0.8576, 0.8706, 0.8835, 0.8965, 0.9094, 0.9223, 0.9353,
    0.9482, 0.9612, 0.9741, 0.9871, 1.0
]
CUTOFF_PCT = 0.50  # simulate a short field test covering the last 50% of the discharge


def compute_soh(total_capacity_ah):
    return float((total_capacity_ah / NOMINAL_CAPACITY) * 100.0)


def extract_and_resample_curve(file_path):
    df = pd.read_csv(file_path)
    cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
    df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)
    mask = (df['Mean_Cell_Voltage'] < 4.15) & (df['Mean_Cell_Voltage'] > 2.0)
    discharge_df = df[mask].copy()
    if 'AHDischarge' not in discharge_df.columns:
        return None, None, None
    discharge_df = discharge_df.sort_values('AHDischarge')
    discharge_df['Ah_Relative'] = discharge_df['AHDischarge'] - discharge_df['AHDischarge'].iloc[0]

    ah = discharge_df['Ah_Relative'].values
    v = discharge_df['Mean_Cell_Voltage'].values
    cell_std = df[cell_cols].std(axis=1)[mask].values if len(cell_cols) > 0 else None
    return ah, v, cell_std


def extract_enhanced_features(sft_v, sft_ah, cell_data=None):
    """Identical to curve_train.py -- must match exactly, feature order is positional."""
    features = {}
    dv_dah = np.gradient(sft_v, sft_ah)
    d2v_dah2 = np.gradient(dv_dah, sft_ah)
    mid_idx = len(sft_v) // 2
    features['initial_slope'] = float((sft_v[mid_idx] - sft_v[0]) / (sft_ah[mid_idx] - sft_ah[0] + 1e-5))
    features['final_slope'] = float((sft_v[-1] - sft_v[mid_idx]) / (sft_ah[-1] - sft_ah[mid_idx] + 1e-5))
    features['overall_slope'] = float((sft_v[-1] - sft_v[0]) / (sft_ah[-1] - sft_ah[0] + 1e-5))
    features['mean_curvature'] = float(np.mean(np.abs(d2v_dah2)))
    features['max_curvature'] = float(np.max(np.abs(d2v_dah2)))
    features['curvature_std'] = float(np.std(d2v_dah2))
    features['voltage_std'] = float(np.std(sft_v))
    features['voltage_range'] = float(sft_v[0] - sft_v[-1])
    features['voltage_mean'] = float(np.mean(sft_v))
    plateau_mask = (sft_v > 3.4) & (sft_v < 3.7)
    features['plateau_length'] = float(np.sum(plateau_mask) / len(sft_v))
    end_idx = int(len(sft_v) * 0.8)
    features['end_slope'] = float((sft_v[-1] - sft_v[end_idx]) / (sft_ah[-1] - sft_ah[end_idx] + 1e-5))
    features['end_voltage_drop'] = float(sft_v[end_idx] - sft_v[-1])
    if cell_data is not None and len(cell_data) > 0:
        features['cell_imbalance_mean'] = float(np.mean(cell_data))
        features['cell_imbalance_std'] = float(np.std(cell_data))
    else:
        features['cell_imbalance_mean'] = 0.0
        features['cell_imbalance_std'] = 0.0
    knee_idx = int(np.argmax(np.abs(dv_dah)))
    features['tail_knee_pct'] = float(sft_ah[knee_idx] / (sft_ah[-1] + 1e-5))
    features['tail_knee_slope'] = float(dv_dah[knee_idx])
    return features


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
