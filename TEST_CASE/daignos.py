# import pandas as pd
# import numpy as np
# import joblib
# import os
# import re
# from sklearn.metrics import mean_absolute_error

# # --- CONFIGURATION ---
# DATA_FOLDER = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/PACK_1_0.1C_FILE" # Update if needed
# MODEL_PATH = 'reconstruction_model_1_0C_v5.pkl' # Change to 0_3C to test the other
# TARGET_C_RATE = 1.0

# # MUST MATCH TRAINING SCRIPT EXACTLY
# SOH_GROUND_TRUTH = {
#     'pk1': {0.3: 95.88, 1.0: 94.40},
#     'pk2': {0.3: 95.14, 0.95: 91.96},
#     'pk3': {0.3: 95.32, 1.0: 91.67},
#     'pk4': {0.3: 93.63, 1.0: 89.08},
#     'pk5': {0.3: 86.78}
# }

# HEAD_CHECKPOINTS_AH = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
# SFT_CHECKPOINTS_COUNT = 20

# print(f"Loading model: {MODEL_PATH}")
# model = joblib.load(MODEL_PATH)

# # --- FEATURE EXTRACTION (Copied from training script) ---
# def extract_and_resample_curve(file_path):
#     df = pd.read_csv(file_path)
#     cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
#     df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)
#     mask = (df['Mean_Cell_Voltage'] < 4.15) & (df['Mean_Cell_Voltage'] > 2.0)
#     discharge_df = df[mask].copy()
#     if 'AHDischarge' not in discharge_df.columns:
#         return None, None, None
#     discharge_df = discharge_df.sort_values('AHDischarge')
#     discharge_df['Ah_Relative'] = discharge_df['AHDischarge'] - discharge_df['AHDischarge'].iloc[0]
    
#     ah = discharge_df['Ah_Relative'].values
#     v = discharge_df['Mean_Cell_Voltage'].values
#     cell_std = df[cell_cols].std(axis=1)[mask].values if len(cell_cols) > 0 else None
#     return ah, v, cell_std

# def extract_enhanced_features(sft_v, sft_ah, cell_data=None):
#     features = {}
#     dv_dah = np.gradient(sft_v, sft_ah)
#     d2v_dah2 = np.gradient(dv_dah, sft_ah)
#     mid_idx = len(sft_v) // 2
#     features['initial_slope'] = float((sft_v[mid_idx] - sft_v[0]) / (sft_ah[mid_idx] - sft_ah[0] + 1e-5))
#     features['final_slope'] = float((sft_v[-1] - sft_v[mid_idx]) / (sft_ah[-1] - sft_ah[mid_idx] + 1e-5))
#     features['overall_slope'] = float((sft_v[-1] - sft_v[0]) / (sft_ah[-1] - sft_ah[0] + 1e-5))
#     features['mean_curvature'] = float(np.mean(np.abs(d2v_dah2)))
#     features['max_curvature'] = float(np.max(np.abs(d2v_dah2)))
#     features['curvature_std'] = float(np.std(d2v_dah2))
#     features['voltage_std'] = float(np.std(sft_v))
#     features['voltage_range'] = float(sft_v[0] - sft_v[-1])
#     features['plateau_length'] = float(np.sum((sft_v > 3.4) & (sft_v < 3.7)) / len(sft_v))
#     features['end_slope'] = float((sft_v[-1] - sft_v[int(len(sft_v) * 0.8)]) / (sft_ah[-1] - sft_ah[int(len(sft_ah) * 0.8)] + 1e-5))
#     features['cell_imbalance_mean'] = float(np.mean(cell_data)) if cell_data is not None and len(cell_data) > 0 else 0.0
#     features['cell_imbalance_std'] = float(np.std(cell_data)) if cell_data is not None and len(cell_data) > 0 else 0.0
#     return features

# # --- DIAGNOSTIC EVALUATION ---
# checkpoint_errors = {cp: [] for cp in HEAD_CHECKPOINTS_AH}
# valid_files_count = 0

# print(f"\nStarting diagnostic evaluation for {TARGET_C_RATE}C files...")

# for file in os.listdir(DATA_FOLDER):
#     if not file.endswith('.csv'): continue
    
#     # 1. Extract Pack ID and C-Rate from filename
#     pack_match = re.search(r'(pk\d+)', file)
#     pack_id = pack_match.group(1) if pack_match else 'unknown'
    
#     c_rate_match = re.search(r'(\d+(?:\.\d+)?)C', file)
#     if not c_rate_match: continue
#     c_rate = float(c_rate_match.group(1))
    
#     # Filter by target C-rate
#     if abs(c_rate - TARGET_C_RATE) > 0.1: continue 
    
#     # 2. Get the CORRECT, pack-specific SOH (NO MORE HARDCODING 95.0!)
#     true_soh = 95.0 # fallback
#     if pack_id in SOH_GROUND_TRUTH:
#         soh_dict = SOH_GROUND_TRUTH[pack_id]
#         c_rate_key = 1.0 if abs(c_rate - 0.95) < 0.05 else c_rate
#         true_soh = soh_dict.get(c_rate_key, soh_dict.get(0.3, 95.0))
    
#     ah, v, cell_std = extract_and_resample_curve(os.path.join(DATA_FOLDER, file))
#     if ah is None or len(ah) < 100: continue
    
#     # Simulate a 50% cutoff (head is first half)
#     cutoff_pct = 0.50
#     cutoff_ah = float(ah[-1] * cutoff_pct)
#     cutoff_idx = np.searchsorted(ah, cutoff_ah)
#     if cutoff_idx < 20: continue
    
#     head_ah = ah[:cutoff_idx]
#     head_v = v[:cutoff_idx]
#     sft_ah = ah[cutoff_idx:] - ah[cutoff_idx]
#     sft_v = v[cutoff_idx:]
#     sft_cell_imbalance = cell_std[cutoff_idx:] if cell_std is not None else None
    
#     # Get True Target Voltages
#     true_target_v = []
#     for cp_ah in HEAD_CHECKPOINTS_AH:
#         if cp_ah <= head_ah[-1]:
#             true_target_v.append(float(np.interp(cp_ah, head_ah, head_v)))
#         else:
#             true_target_v.append(np.nan) 
            
#     # Extract Features
#     sft_sampled_v = [float(x) for x in np.linspace(sft_v[0], sft_v[-1], SFT_CHECKPOINTS_COUNT)]
#     enhanced_feats = extract_enhanced_features(sft_v, sft_ah, sft_cell_imbalance)
    
#     # EXACT 34-FEATURE ORDER MATCHING TRAINING SCRIPT
#     features = sft_sampled_v + [
#         true_soh,
#         enhanced_feats['initial_slope'],
#         enhanced_feats['final_slope'],
#         enhanced_feats['overall_slope'],
#         enhanced_feats['mean_curvature'],
#         enhanced_feats['max_curvature'],
#         enhanced_feats['curvature_std'],
#         enhanced_feats['voltage_std'],
#         enhanced_feats['voltage_range'],
#         enhanced_feats['plateau_length'],
#         enhanced_feats['end_slope'],
#         enhanced_feats['cell_imbalance_mean'],
#         enhanced_feats['cell_imbalance_std'],
#         cutoff_ah
#     ]
    
#     # Predict
#     X_input = np.array([features])
#     pred_v = model.predict(X_input)[0]
    
#     # Calculate error per checkpoint (only where we have ground truth)
#     for i, cp_ah in enumerate(HEAD_CHECKPOINTS_AH):
#         if not np.isnan(true_target_v[i]):
#             error = abs(pred_v[i] - true_target_v[i]) * 1000 # in mV
#             checkpoint_errors[cp_ah].append(error)
            
#     valid_files_count += 1
#     print(f"  Processed: {file} (Pack: {pack_id}, True SOH: {true_soh}%)")

# print(f"\n{'='*75}")
# print(f"DIAGNOSTIC RESULTS for {TARGET_C_RATE}C (Evaluated on {valid_files_count} real files)")
# print(f"{'='*75}")
# print(f"{'Checkpoint (Ah)':<18} | {'Avg Error (mV)':<15} | {'Max Error (mV)':<15} | {'Sample Count'}")
# print("-" * 75)

# for cp_ah in HEAD_CHECKPOINTS_AH:
#     errors = checkpoint_errors[cp_ah]
#     if len(errors) > 0:
#         avg_err = np.mean(errors)
#         max_err = np.max(errors)
#         print(f"{cp_ah:<18} | {avg_err:<15.2f} | {max_err:<15.2f} | {len(errors)}")
#     else:
#         print(f"{cp_ah:<18} | {'N/A':<15} | {'N/A':<15} | 0")




import pandas as pd
import numpy as np
import joblib
import os
import re
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
# Update this to the folder containing your real test files
DATA_FOLDER = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/fft_raw_data" 

# Point this to the newly trained percentage-based model
MODEL_PATH = 'reconstruction_model_0_3C_v6_pct.pkl' 
TARGET_C_RATE = 1.0

# MUST MATCH TRAINING SCRIPT EXACTLY
SOH_GROUND_TRUTH = {
    'pk1': {0.3: 95.88, 1.0: 94.40},
    'pk2': {0.3: 95.14, 0.95: 91.96},
    'pk3': {0.3: 95.32, 1.0: 91.67},
    'pk4': {0.3: 93.63, 1.0: 89.08},
    'pk5': {0.3: 86.78}
}

# Percentage-based checkpoints (21 points from 0% to 100% of the head)
HEAD_CHECKPOINTS_PCT = [
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0
]
SFT_CHECKPOINTS_COUNT = 40

print(f"Loading model: {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"❌ Model not found at {MODEL_PATH}. Please run the training script first!")
    exit()

# --- FEATURE EXTRACTION (Copied EXACTLY from training script) ---
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
    
    return features

# --- DIAGNOSTIC EVALUATION ---
checkpoint_errors = {pct: [] for pct in HEAD_CHECKPOINTS_PCT}
valid_files_count = 0

print(f"\nStarting diagnostic evaluation for {TARGET_C_RATE}C files...")

for file in os.listdir(DATA_FOLDER):
    if not file.endswith('.csv'): continue
    
    # 1. Extract Pack ID and C-Rate from filename
    pack_match = re.search(r'(pk\d+)', file)
    pack_id = pack_match.group(1) if pack_match else 'unknown'
    
    c_rate_match = re.search(r'(\d+(?:\.\d+)?)C', file)
    if not c_rate_match: continue
    c_rate = float(c_rate_match.group(1))
    
    # Filter by target C-rate
    if abs(c_rate - TARGET_C_RATE) > 0.1: continue 
    
    # 2. Get the CORRECT, pack-specific SOH
    true_soh = 95.0 # fallback
    if pack_id in SOH_GROUND_TRUTH:
        soh_dict = SOH_GROUND_TRUTH[pack_id]
        c_rate_key = 1.0 if abs(c_rate - 0.95) < 0.05 else c_rate
        true_soh = soh_dict.get(c_rate_key, soh_dict.get(0.3, 95.0))
    
    ah, v, cell_std = extract_and_resample_curve(os.path.join(DATA_FOLDER, file))
    if ah is None or len(ah) < 100: continue
    
    # Simulate a 50% cutoff (head is first half)
    cutoff_pct = 0.50
    cutoff_ah = float(ah[-1] * cutoff_pct)
    cutoff_idx = np.searchsorted(ah, cutoff_ah)
    if cutoff_idx < 20: continue
    
    head_ah = ah[:cutoff_idx]
    head_v = v[:cutoff_idx]
    sft_ah = ah[cutoff_idx:] - ah[cutoff_idx]
    sft_v = v[cutoff_idx:]
    sft_cell_imbalance = cell_std[cutoff_idx:] if cell_std is not None else None
    
    # Get True Target Voltages AT THE PERCENTAGE CHECKPOINTS
    true_target_v = []
    for pct in HEAD_CHECKPOINTS_PCT:
        target_ah = pct * head_ah[-1]
        v_at_pct = float(np.interp(target_ah, head_ah, head_v))
        true_target_v.append(v_at_pct)
        
    # Extract Features (EXACTLY matching training script)
    sft_sampled_v = [float(x) for x in np.linspace(sft_v[0], sft_v[-1], SFT_CHECKPOINTS_COUNT)]
    enhanced_feats = extract_enhanced_features(sft_v, sft_ah, sft_cell_imbalance)
    
    features = sft_sampled_v + [
        true_soh,
        enhanced_feats['initial_slope'],
        enhanced_feats['final_slope'],
        enhanced_feats['overall_slope'],
        enhanced_feats['mean_curvature'],
        enhanced_feats['max_curvature'],
        enhanced_feats['voltage_std'],
        enhanced_feats['voltage_range'],
        enhanced_feats['plateau_length'],
        enhanced_feats['end_voltage_drop'],
        enhanced_feats['end_slope'],
        enhanced_feats['cell_imbalance_mean'],
        enhanced_feats['cell_imbalance_std'],
        float(cutoff_ah) # Keep absolute head length as a feature so the model knows the scale
    ]
    
    # Predict
    X_input = np.array([features])
    pred_v = model.predict(X_input)[0]
    
    # Calculate error per checkpoint
    for i, pct in enumerate(HEAD_CHECKPOINTS_PCT):
        error = abs(pred_v[i] - true_target_v[i]) * 1000 # in mV
        checkpoint_errors[pct].append(error)
        
    valid_files_count += 1
    print(f"  Processed: {file} (Pack: {pack_id}, True SOH: {true_soh}%)")

print(f"\n{'='*80}")
print(f"DIAGNOSTIC RESULTS for {TARGET_C_RATE}C (Evaluated on {valid_files_count} real files)")
print(f"{'='*80}")
print(f"{'Head %':<10} | {'Avg Error (mV)':<15} | {'Max Error (mV)':<15} | {'Sample Count'}")
print("-" * 80)

for pct in HEAD_CHECKPOINTS_PCT:
    errors = checkpoint_errors[pct]
    if len(errors) > 0:
        avg_err = np.mean(errors)
        max_err = np.max(errors)
        print(f"{int(pct*100):<10}% | {avg_err:<15.2f} | {max_err:<15.2f} | {len(errors)}")
    else:
        print(f"{int(pct*100):<10}% | {'N/A':<15} | {'N/A':<15} | 0")