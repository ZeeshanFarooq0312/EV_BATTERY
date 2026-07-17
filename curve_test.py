import pandas as pd
import numpy as np
import joblib
import os
import re
import warnings
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# Constants
NOMINAL_CAPACITY = 156.0
SFT_CHECKPOINTS_COUNT = 15

# ==========================================
# 1. PART 1: SOH & Capacity Prediction Logic
# ==========================================
def extract_soh_features(file_path):
    """Extracts features exactly as the Part 1 V5.1 model expects."""
    df = pd.read_csv(file_path)
    features = {}
    filename = os.path.basename(file_path)
    
    c_rate_match = re.search(r'(\d+\.\d+)C', filename)
    features['c_rate'] = float(c_rate_match.group(1)) if c_rate_match else 0.0
    features['is_sfct'] = 1.0 if 'SFCT' in filename.upper() or 'SFT' in filename.upper() else 0.0
    
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
        df['Max_Temp'] = df[temp_cols].max(axis=1)
        features['mean_temp'] = df['Mean_Temp'].mean()
        features['max_temp'] = df['Max_Temp'].max()
        features['temp_rise'] = df['Mean_Temp'].iloc[-1] - df['Mean_Temp'].iloc[0]
        
    if 'AHDischarge' in df.columns:
        features['delta_Ah'] = df['AHDischarge'].iloc[-1] - df['AHDischarge'].iloc[0]
        features['ah_per_voltage_drop'] = features['delta_Ah'] / (features['voltage_drop'] + 1e-5)
        dV = df['Mean_Cell_Voltage'].diff().dropna()
        dAh = df['AHDischarge'].diff().dropna()
        dV_dAh = dV / (dAh.replace(0, 1e-5)) 
        features['mean_dV_dAh'] = dV_dAh.mean()
        features['std_dV_dAh'] = dV_dAh.std()
    else:
        features['delta_Ah'] = 0.0
        features['ah_per_voltage_drop'] = 0.0
        
    return features

# ==========================================
# 2. PART 2: C-Rate Specific ML Reconstruction
# ==========================================
def reconstruct_with_ml_crate_specific(sft_file_path, predicted_total_capacity):
    # Extract C-rate from filename
    filename = os.path.basename(sft_file_path)
    c_rate_match = re.search(r'(\d+\.\d+)C', filename)
    c_rate = float(c_rate_match.group(1)) if c_rate_match else 0.3
    
    # Load the appropriate C-rate specific model
    if abs(c_rate - 0.3) < 0.01:
        model = joblib.load('reconstruction_model_0_3C.pkl')
        crate_name = '0.3C'
    elif abs(c_rate - 1.0) < 0.01:
        model = joblib.load('reconstruction_model_1_0C.pkl')
        crate_name = '1.0C'
    else:
        # Fallback to 0.3C model
        model = joblib.load('reconstruction_model_0_3C.pkl')
        crate_name = '0.3C (fallback)'
    
    print(f"   ➔ Using {crate_name} specific reconstruction model")

    # Extract SFT data
    df = pd.read_csv(sft_file_path)
    cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
    df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)
    
    sft_v = df['Mean_Cell_Voltage'].values
    sft_ah = df['AHDischarge'].values
    sft_start_v = sft_v[0]
    
    sft_delta_ah = sft_ah[-1] - sft_ah[0]
    target_sft_start_ah = predicted_total_capacity - sft_delta_ah
    
    # Prepare input features EXACTLY matching the new training script
    sft_sampled_v = np.linspace(sft_v[0], sft_v[-1], SFT_CHECKPOINTS_COUNT)
    sft_slope = (sft_v[-1] - sft_v[0]) / (len(sft_v) + 1e-5)
    sft_mean_v = np.mean(sft_v)
    
    estimated_soh = (predicted_total_capacity / NOMINAL_CAPACITY) * 100.0
    
    # Features: [15 SFT points] + [SOH] + [sft_slope] + [sft_mean_v] + [cutoff_ah]
    features = list(sft_sampled_v) + [estimated_soh, sft_slope, sft_mean_v, target_sft_start_ah]
    
    X_input = np.array([features])
    predicted_head_v = model.predict(X_input)[0]
    
    # Build the head curve
    HEAD_CHECKPOINTS_AH = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60]
    valid_mask = (predicted_head_v > 2.0) & ~np.isnan(predicted_head_v)
    head_ah_valid = np.array(HEAD_CHECKPOINTS_AH)[valid_mask]
    head_v_valid = predicted_head_v[valid_mask]
    
    head_ah_dense = np.linspace(0, target_sft_start_ah, 100)
    
    all_head_ah = np.append(head_ah_valid, target_sft_start_ah)
    all_head_v = np.append(head_v_valid, sft_start_v)
    
    sort_idx = np.argsort(all_head_ah)
    all_head_ah = all_head_ah[sort_idx]
    all_head_v = all_head_v[sort_idx]
    
    interp_func = interp1d(all_head_ah, all_head_v, kind='cubic', fill_value='extrapolate')
    head_v_dense = interp_func(head_ah_dense)
    
    rebased_sft_ah = np.linspace(target_sft_start_ah, predicted_total_capacity, len(sft_ah))
    
    full_ah = np.concatenate([head_ah_dense, rebased_sft_ah])
    full_v = np.concatenate([head_v_dense, sft_v])
    
    return full_ah, full_v, target_sft_start_ah

# ==========================================
# 3. FFT Ground Truth Extraction
# ==========================================
def extract_true_fft(fft_file_path):
    df = pd.read_csv(fft_file_path)
    cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
    df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)
    
    discharge_df = df[(df['Mean_Cell_Voltage'] < 4.15) & (df['Mean_Cell_Voltage'] > 2.0)].copy()
    
    if 'AHDischarge' in discharge_df.columns and discharge_df['AHDischarge'].max() > 0:
        discharge_df = discharge_df.sort_values('AHDischarge')
        discharge_df['Ah_Relative'] = discharge_df['AHDischarge'] - discharge_df['AHDischarge'].iloc[0]
    else:
        discharge_df['Ah_Relative'] = np.arange(len(discharge_df))

    return discharge_df['Ah_Relative'].values, discharge_df['Mean_Cell_Voltage'].values

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- FILE PATHS (Update these to test different files) ---
    SFT_FILE_PATH = r"D:\ev_battery_version_2\new_tech\stf_raw_data\pk4-60pc-29052021-SFT-0.3C 202605121634 Characterisation Test (1).csv"
    FFT_FILE_PATH = r"D:\ev_battery_version_2\new_tech\fft_raw_data\pk4-60pc-29052021-FFCT-0.3C 202605130802 Characterisation Test (1).csv"
    
    print("Loading SOH model...")
    soh_model = joblib.load('soh_model_v5_1.pkl')
    soh_feature_names = joblib.load('feature_names_v5_1.pkl')
    print("✅ SOH Model loaded.\n")

    if not os.path.exists(SFT_FILE_PATH) or not os.path.exists(FFT_FILE_PATH):
        print("Error: One or both files not found.")
    else:
        # 1. PREDICT CAPACITY (Part 1)
        print(f"Analyzing SFT: {os.path.basename(SFT_FILE_PATH)}")
        soh_feats = extract_soh_features(SFT_FILE_PATH)
        X_soh = pd.DataFrame([soh_feats]).reindex(columns=soh_feature_names, fill_value=0)
        
        pred_soh = soh_model.predict(X_soh)[0]
        pred_capacity = (pred_soh / 100.0) * NOMINAL_CAPACITY
        print(f"    Part 1 Predicted SOH: {pred_soh:.2f}%")
        print(f"   ➔ Part 1 Predicted Capacity: {pred_capacity:.2f} Ah\n")

        # 2. RECONSTRUCT CURVE (Part 2 - C-Rate Specific)
        print("Reconstructing missing head using C-Rate Specific ML Model...")
        recon_ah, recon_v, splice_ah = reconstruct_with_ml_crate_specific(SFT_FILE_PATH, pred_capacity)
        print(f"   ➔ Reconstruction complete. Splice point at {splice_ah:.2f} Ah.\n")

        # 3. EXTRACT REAL FFT FOR VALIDATION
        print("Extracting ground truth from FFT...")
        fft_ah, fft_v = extract_true_fft(FFT_FILE_PATH)
        print(f"   ➔ Found {len(fft_ah)} points in real FFT discharge phase.\n")

        # 4. CALCULATE ERROR & PLOT
        min_ah = max(fft_ah.min(), recon_ah.min())
        max_ah = min(fft_ah.max(), recon_ah.max())
        
        mask_fft = (fft_ah >= min_ah) & (fft_ah <= max_ah)
        mask_recon = (recon_ah >= min_ah) & (recon_ah <= max_ah)
        
        interp_func = interp1d(fft_ah[mask_fft], fft_v[mask_fft], kind='linear', fill_value='extrapolate')
        v_fft_interp = interp_func(recon_ah[mask_recon])
        
        mae = np.mean(np.abs(recon_v[mask_recon] - v_fft_interp))
        rmse = np.sqrt(np.mean((recon_v[mask_recon] - v_fft_interp) ** 2))

        print("="*60)
        print(" VALIDATION METRICS (Reconstructed vs Real FFT)")
        print("="*60)
        print(f"   Mean Absolute Error (MAE):  {mae*1000:.1f} mV")
        print(f"   Root Mean Square Error (RMSE): {rmse*1000:.1f} mV")
        print("="*60)

        # Plotting
        plt.figure(figsize=(12, 7))
        plt.plot(fft_ah, fft_v, label='Real FFT (Ground Truth)', color='blue', linewidth=2.5, alpha=0.8)
        plt.plot(recon_ah, recon_v, label='ML Reconstructed from SFT', color='red', linewidth=2, linestyle='--')
        plt.axvline(x=splice_ah, color='green', linestyle=':', linewidth=2, label=f'ML Splice Point ({splice_ah:.2f} Ah)')
        
        plt.title(f'End-to-End Validation: ML Reconstruction vs Real FFT\nMAE: {mae*1000:.1f} mV', fontsize=14, fontweight='bold')
        plt.xlabel('Capacity Delivered (Ah)', fontsize=12)
        plt.ylabel('Mean Cell Voltage (V)', fontsize=12)
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.ylim(1.8, 4.3)
        plt.tight_layout()
        
        print("\n📊 Displaying overlay plot... Close window to exit.")
        plt.show()