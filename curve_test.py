import pandas as pd
import numpy as np
import joblib
import os
import re
import warnings
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

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

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TEST_CASE')


# ==========================================
# 1. PART 1: Capacity / SOH prediction from a short (SFT) test
# ==========================================
def extract_soh_features(file_path):
    """Matches the feature set soh_model_{0_3c,1_0c}.pkl were trained on (see SOH_MODELS_TRAIN.PY)."""
    df = pd.read_csv(file_path)
    filename = os.path.basename(file_path)
    features = {}

    c_rate_match = re.search(r'(\d+(?:\.\d+)?)C', filename)
    c_rate = float(c_rate_match.group(1)) if c_rate_match else 0.3
    features['c_rate'] = c_rate
    features['is_sfct'] = 1.0
    features['slice_start_pct'] = 0.5

    cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
    df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)
    df['Min_Cell_Voltage'] = df[cell_cols].min(axis=1)
    df['Max_Cell_Voltage'] = df[cell_cols].max(axis=1)
    df['Cell_Voltage_Std'] = df[cell_cols].std(axis=1)

    features['start_voltage'] = df['Mean_Cell_Voltage'].iloc[:10].mean()
    features['end_voltage'] = df['Mean_Cell_Voltage'].iloc[-10:].mean()
    features['voltage_drop'] = features['start_voltage'] - features['end_voltage']
    features['mean_cell_imbalance_std'] = df['Cell_Voltage_Std'].mean()
    features['max_cell_spread'] = (df['Max_Cell_Voltage'] - df['Min_Cell_Voltage']).mean()
    features['max_cell_spread_at_end'] = (df['Max_Cell_Voltage'].iloc[-10:] - df['Min_Cell_Voltage'].iloc[-10:]).mean()
    features['min_cell_voltage_at_end'] = df['Min_Cell_Voltage'].iloc[-10:].mean()

    temp_cols = [c for c in df.columns if 'Temperature' in c]
    if temp_cols:
        df['Mean_Temp'] = df[temp_cols].mean(axis=1)
        features['mean_temp'] = df['Mean_Temp'].mean()
        features['temp_rise'] = df['Mean_Temp'].iloc[-1] - df['Mean_Temp'].iloc[0]
    else:
        features['mean_temp'] = 0.0
        features['temp_rise'] = 0.0

    if 'AHDischarge' in df.columns:
        features['delta_Ah'] = df['AHDischarge'].iloc[-1] - df['AHDischarge'].iloc[0]
        features['ah_per_voltage_drop'] = features['delta_Ah'] / (features['voltage_drop'] + 1e-5)
        dV = df['Mean_Cell_Voltage'].diff().dropna()
        dAh = df['AHDischarge'].diff().dropna()
        dV_dAh = dV / dAh.replace(0, 1e-5)
        features['mean_dV_dAh'] = dV_dAh.mean()
        features['std_dV_dAh'] = dV_dAh.std()
        features['min_dV_dAh'] = dV_dAh.min()
    else:
        features['delta_Ah'] = 0.0
        features['ah_per_voltage_drop'] = 0.0
        features['mean_dV_dAh'] = 0.0
        features['std_dV_dAh'] = 0.0
        features['min_dV_dAh'] = 0.0

    features['voltage_drop_norm'] = features['voltage_drop'] / (c_rate + 1e-5)
    features['delta_Ah_norm'] = features['delta_Ah'] / (c_rate + 1e-5)
    return features, c_rate


def predict_capacity_from_sft(sft_file_path):
    features, c_rate = extract_soh_features(sft_file_path)

    if abs(c_rate - 0.3) < 0.05:
        model = joblib.load(os.path.join(MODEL_DIR, 'soh_model_0_3c.pkl'))
        feat_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names_0_3c.pkl'))
    else:
        model = joblib.load(os.path.join(MODEL_DIR, 'soh_model_1_0c.pkl'))
        feat_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names_1_0c.pkl'))

    X = np.array([[features.get(f, 0.0) for f in feat_names]])
    pred_soh = float(model.predict(X)[0])
    pred_capacity = (pred_soh / 100.0) * NOMINAL_CAPACITY
    return pred_soh, pred_capacity, c_rate


# ==========================================
# 2. PART 2: Complete global curve reconstruction (v8 model)
# ==========================================
def extract_and_resample_curve(file_path):
    df = pd.read_csv(file_path)
    cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
    df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)

    discharge_df = df[(df['Mean_Cell_Voltage'] < 4.15) & (df['Mean_Cell_Voltage'] > 2.0)].copy()
    discharge_df = discharge_df.sort_values('AHDischarge')
    discharge_df['Ah_Relative'] = discharge_df['AHDischarge'] - discharge_df['AHDischarge'].iloc[0]

    ah = discharge_df['Ah_Relative'].values
    v = discharge_df['Mean_Cell_Voltage'].values

    if len(cell_cols) > 0:
        mask = (df['Mean_Cell_Voltage'] < 4.15) & (df['Mean_Cell_Voltage'] > 2.0)
        cell_std = df[cell_cols].std(axis=1)[mask].values
    else:
        cell_std = None

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


def reconstruct_full_curve(sft_file_path, predicted_total_capacity, c_rate):
    """SFT data is treated as the tail segment of the full discharge (it ends where
    the full discharge ends), exactly like the tail-after-cutoff segments used in
    training. The model then predicts all 41 checkpoints across the COMPLETE curve
    (0-100% of predicted_total_capacity), not just the missing head."""
    if abs(c_rate - 0.3) < 0.05:
        model = joblib.load(os.path.join(MODEL_DIR, 'reconstruction_model_0_3C_v10_knee.pkl'))
    else:
        model = joblib.load(os.path.join(MODEL_DIR, 'reconstruction_model_1_0C_v10_knee.pkl'))

    sft_ah, sft_v, sft_cell_std = extract_and_resample_curve(sft_file_path)
    sft_delta_ah = sft_ah[-1]
    cutoff_ah = predicted_total_capacity - sft_delta_ah

    estimated_soh = (predicted_total_capacity / NOMINAL_CAPACITY) * 100.0
    sft_sampled_v = [np.interp(frac * sft_ah[-1], sft_ah, sft_v) for frac in SFT_SAMPLE_FRACTIONS]
    feats = extract_enhanced_features(sft_v, sft_ah, sft_cell_std)

    features = list(sft_sampled_v) + [
        estimated_soh,
        feats['initial_slope'], feats['final_slope'], feats['overall_slope'],
        feats['mean_curvature'], feats['max_curvature'], feats['voltage_std'],
        feats['voltage_range'], feats['plateau_length'], feats['end_voltage_drop'],
        feats['end_slope'], feats['cell_imbalance_mean'], feats['cell_imbalance_std'],
        feats['tail_knee_pct'], feats['tail_knee_slope'],
        float(cutoff_ah)
    ]

    pred_v = model.predict(np.array([features]))[0]
    recon_ah = np.array([pct * predicted_total_capacity for pct in HEAD_CHECKPOINTS_PCT])
    return recon_ah, pred_v, cutoff_ah


# ==========================================
# 3. FFCT ground truth extraction (for validation only)
# ==========================================
def extract_true_fft(fft_file_path):
    ah, v, _ = extract_and_resample_curve(fft_file_path)
    return ah, v


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
    SFT_FILE_PATH = os.path.join(DATA_ROOT, "stf_raw_data", "pk1-62-08062021-SFT-0.3C 202604281055 Characterisation Test.csv")
    FFT_FILE_PATH = os.path.join(DATA_ROOT, "fft_raw_data", "pk1-62-08062021-FFCT-0.3C 202605151215 Characterisation Test.csv")

    if not os.path.exists(SFT_FILE_PATH) or not os.path.exists(FFT_FILE_PATH):
        print("Error: One or both files not found.")
    else:
        print(f"Analyzing SFT: {os.path.basename(SFT_FILE_PATH)}")
        pred_soh, pred_capacity, c_rate = predict_capacity_from_sft(SFT_FILE_PATH)
        print(f"    Part 1 Predicted SOH: {pred_soh:.2f}%")
        print(f"   ➔ Part 1 Predicted Capacity: {pred_capacity:.2f} Ah\n")

        print("Reconstructing complete global curve using C-Rate Specific ML Model...")
        recon_ah, recon_v, cutoff_ah = reconstruct_full_curve(SFT_FILE_PATH, pred_capacity, c_rate)
        print(f"   ➔ Reconstruction complete. SFT tail starts at {cutoff_ah:.2f} Ah of the reconstructed curve.\n")

        print("Extracting ground truth from FFCT...")
        fft_ah, fft_v = extract_true_fft(FFT_FILE_PATH)
        print(f"   ➔ Found {len(fft_ah)} points in real FFCT discharge phase.\n")

        interp_func = interp1d(recon_ah, recon_v, kind='cubic', fill_value='extrapolate')
        dense_ah = np.linspace(recon_ah.min(), recon_ah.max(), 300)
        dense_v = interp_func(dense_ah)

        min_ah = max(fft_ah.min(), dense_ah.min())
        max_ah = min(fft_ah.max(), dense_ah.max())
        mask_fft = (fft_ah >= min_ah) & (fft_ah <= max_ah)
        mask_recon = (dense_ah >= min_ah) & (dense_ah <= max_ah)

        fft_interp_func = interp1d(fft_ah[mask_fft], fft_v[mask_fft], kind='linear', fill_value='extrapolate')
        v_fft_interp = fft_interp_func(dense_ah[mask_recon])

        mae = np.mean(np.abs(dense_v[mask_recon] - v_fft_interp))
        rmse = np.sqrt(np.mean((dense_v[mask_recon] - v_fft_interp) ** 2))

        print("=" * 60)
        print(" VALIDATION METRICS (Reconstructed vs Real FFCT)")
        print("=" * 60)
        print(f"   Mean Absolute Error (MAE):  {mae*1000:.1f} mV")
        print(f"   Root Mean Square Error (RMSE): {rmse*1000:.1f} mV")
        print("=" * 60)

        plt.figure(figsize=(12, 7))
        plt.plot(fft_ah, fft_v, label='Real FFCT (Ground Truth)', color='blue', linewidth=2.5, alpha=0.8)
        plt.plot(dense_ah, dense_v, label='ML Reconstructed (Complete Global Curve)', color='red', linewidth=2, linestyle='--')
        plt.axvline(x=cutoff_ah, color='green', linestyle=':', linewidth=2, label=f'SFT Tail Start ({cutoff_ah:.2f} Ah)')

        plt.title(f'End-to-End Validation: ML Reconstruction vs Real FFCT\nMAE: {mae*1000:.1f} mV', fontsize=14, fontweight='bold')
        plt.xlabel('Capacity Delivered (Ah)', fontsize=12)
        plt.ylabel('Mean Cell Voltage (V)', fontsize=12)
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.ylim(1.8, 4.3)
        plt.tight_layout()

        print("\nDisplaying overlay plot... Close window to exit.")
        plt.show()
