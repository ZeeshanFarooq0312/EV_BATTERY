
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import re
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')

# FIX 1: C-rate specific SOH
SOH_GROUND_TRUTH = {
    'pk1': {0.3: 95.88, 1.0: 94.40},
    'pk2': {0.3: 95.14, 0.95: 91.96},
    'pk3': {0.3: 95.32, 1.0: 91.67},
    'pk4': {0.3: 93.63, 1.0: 89.08},
    'pk5': {0.3: 86.78}
}

# INCREASED CHECKPOINTS: Every 2.5% instead of 5% for better resolution
HEAD_CHECKPOINTS_PCT = [
    0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475,
    0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0
]
SFT_CHECKPOINTS_COUNT = 40

def extract_and_resample_curve(file_path):
    df = pd.read_csv(file_path)
    cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
    df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)
    
    discharge_df = df[(df['Mean_Cell_Voltage'] < 4.15) & (df['Mean_Cell_Voltage'] > 2.0)].copy()
    if 'AHDischarge' not in discharge_df.columns:
        return None, None, None
        
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

def generate_synthetic_samples(X_base, y_base, n_samples=8):
    X_synthetic, y_synthetic = [], []
    for i in range(len(X_base)):
        for j in range(n_samples):
            noise_level = np.random.uniform(0.005, 0.03)
            X_new = X_base[i] + np.random.normal(0, noise_level, X_base[i].shape)
            
            voltage_noise = np.random.normal(0, 0.003, y_base[i].shape)
            drift = np.linspace(0, np.random.uniform(-0.003, 0.003), len(y_base[i]))
            y_new = np.clip(y_base[i] + voltage_noise + drift, 2.0, 4.2)
            
            X_synthetic.append(X_new)
            y_synthetic.append(y_new)
    return np.array(X_synthetic), np.array(y_synthetic)

def interpolate_between_packs(X1, y1, X2, y2, n_interpolations=3):
    X_interp, y_interp = [], []
    min_len = min(len(X1), len(X2))
    for i in range(min_len):
        for alpha in np.linspace(0.2, 0.8, n_interpolations):
            X_interp.append(alpha * X1[i] + (1 - alpha) * X2[i])
            y_interp.append(np.clip(alpha * y1[i] + (1 - alpha) * y2[i], 2.0, 4.2))
    return np.array(X_interp), np.array(y_interp)

def prepare_training_data_by_crate(data_folder):
    data_0_3c = {'X': [], 'y': []}
    data_1_0c = {'X': [], 'y': []}
    all_0_3c_X, all_0_3c_y, all_1_0c_X, all_1_0c_y = [], [], [], []
    
    for file in os.listdir(data_folder):
        if not file.endswith('.csv'): continue
        
        pack_match = re.search(r'(pk\d+)', file)
        pack_id = pack_match.group(1) if pack_match else 'unknown'
        if pack_id not in SOH_GROUND_TRUTH: continue
        
        c_rate_match = re.search(r'(\d+(?:\.\d+)?)C', file)
        if not c_rate_match: continue
        c_rate = float(c_rate_match.group(1))
        
        soh_dict = SOH_GROUND_TRUTH[pack_id]
        c_rate_key = 1.0 if abs(c_rate - 0.95) < 0.05 else c_rate
        true_soh = float(soh_dict.get(c_rate_key, soh_dict.get(0.3, 95.0)))
        
        ah, v, cell_std_data = extract_and_resample_curve(os.path.join(data_folder, file))
        if ah is None or len(ah) < 50: continue
        
        total_cap = ah[-1]
        
        for cutoff_pct in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            cutoff_ah = float(total_cap * cutoff_pct)
            cutoff_idx = np.searchsorted(ah, cutoff_ah)
            if cutoff_idx < 15: continue
            
            head_ah = ah[:cutoff_idx]
            head_v = v[:cutoff_idx]
            sft_ah = ah[cutoff_idx:] - ah[cutoff_idx]
            sft_v = v[cutoff_idx:]
            sft_cell_imbalance = cell_std_data[cutoff_idx:] if cell_std_data is not None else None
            
            # Predict voltage at percentage checkpoints
            target_v = []
            for pct in HEAD_CHECKPOINTS_PCT:
                target_ah = pct * head_ah[-1]
                v_at_pct = float(np.interp(target_ah, head_ah, head_v))
                target_v.append(v_at_pct)
            
            sft_sampled_v = [float(x) for x in np.linspace(sft_v[0], sft_v[-1], SFT_CHECKPOINTS_COUNT)]
            enhanced_feats = extract_enhanced_features(sft_v, sft_ah, sft_cell_imbalance)
            
            features = sft_sampled_v + [
                true_soh,
                enhanced_feats['initial_slope'], enhanced_feats['final_slope'], enhanced_feats['overall_slope'],
                enhanced_feats['mean_curvature'], enhanced_feats['max_curvature'], enhanced_feats['voltage_std'],
                enhanced_feats['voltage_range'], enhanced_feats['plateau_length'], enhanced_feats['end_voltage_drop'],
                enhanced_feats['end_slope'], enhanced_feats['cell_imbalance_mean'], enhanced_feats['cell_imbalance_std'],
                float(cutoff_ah)
            ]
            
            if abs(c_rate - 0.3) < 0.05:
                data_0_3c['X'].append(features); data_0_3c['y'].append(target_v)
                all_0_3c_X.append(features); all_0_3c_y.append(target_v)
            elif c_rate >= 0.9:
                data_1_0c['X'].append(features); data_1_0c['y'].append(target_v)
                all_1_0c_X.append(features); all_1_0c_y.append(target_v)
    
    print("Generating synthetic samples...")
    X_03c_base, y_03c_base = np.array(all_0_3c_X), np.array(all_0_3c_y)
    X_10c_base, y_10c_base = np.array(all_1_0c_X), np.array(all_1_0c_y)
    
    X_synth_03c, y_synth_03c = generate_synthetic_samples(X_03c_base, y_03c_base, n_samples=8)
    X_synth_10c, y_synth_10c = generate_synthetic_samples(X_10c_base, y_10c_base, n_samples=8)
    
    if len(X_03c_base) > 1:
        X_i, y_i = interpolate_between_packs(X_03c_base[:-1], y_03c_base[:-1], X_03c_base[1:], y_03c_base[1:], 3)
        X_synth_03c, y_synth_03c = np.vstack([X_synth_03c, X_i]), np.vstack([y_synth_03c, y_i])
    
    if len(X_10c_base) > 1:
        X_i, y_i = interpolate_between_packs(X_10c_base[:-1], y_10c_base[:-1], X_10c_base[1:], y_10c_base[1:], 3)
        X_synth_10c, y_synth_10c = np.vstack([X_synth_10c, X_i]), np.vstack([y_synth_10c, y_i])
    
    data_0_3c['X'].extend(X_synth_03c.tolist()); data_0_3c['y'].extend(y_synth_03c.tolist())
    data_1_0c['X'].extend(X_synth_10c.tolist()); data_1_0c['y'].extend(y_synth_10c.tolist())
    
    print(f"0.3C samples: {len(data_0_3c['X'])}")
    print(f"1.0C samples: {len(data_1_0c['X'])}")
    return data_0_3c, data_1_0c

def train_crate_specific_models(data_folder):
    print("Preparing C-rate specific training data...")
    data_0_3c, data_1_0c = prepare_training_data_by_crate(data_folder)
    models = {}
    
    for c_rate_name, data in [('0.3C', data_0_3c), ('1.0C', data_1_0c)]:
        if len(data['X']) == 0: continue
            
        lengths = [len(x) for x in data['X']]
        if len(set(lengths)) > 1:
            most_common_len = Counter(lengths).most_common(1)[0][0]
            X_list = [x for x, l in zip(data['X'], lengths) if l == most_common_len]
            y_list = [y for y, l in zip(data['y'], lengths) if l == most_common_len]
        else:
            X_list, y_list = data['X'], data['y']
            
        X, y = np.array(X_list), np.array(y_list)
        print(f"\nTraining {c_rate_name} model with {len(X)} samples...")
        print(f"  Input features: {X.shape[1]}")
        print(f"  Output checkpoints: {y.shape[1]}")
        
        model = xgb.XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.015, random_state=42,
                                 reg_alpha=0.15, reg_lambda=2.5, subsample=0.85, colsample_bytree=0.85, min_child_weight=3)
        multi_model = MultiOutputRegressor(model)
        
        if len(X) > 50:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            multi_model.fit(X_train, y_train)
            mae = mean_absolute_error(y_test, multi_model.predict(X_test))
            print(f"✅ {c_rate_name} Model MAE: {mae*1000:.1f} mV")
        else:
            multi_model.fit(X, y)
        models[c_rate_name] = multi_model
    return models

if __name__ == "__main__":
    DATA_FOLDER = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/fft_raw_data"
    if os.path.exists(DATA_FOLDER):
        models = train_crate_specific_models(DATA_FOLDER)
        for c_rate_name, model in models.items():
            safe_name = c_rate_name.replace('.', '_')
            joblib.dump(model, f'reconstruction_model_{safe_name}_v7_41chk.pkl')
            print(f"✅ Saved reconstruction_model_{safe_name}_v7_41chk.pkl")