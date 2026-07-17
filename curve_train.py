import pandas as pd
import numpy as np
import xgboost as xgb
import os
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')

SOH_GROUND_TRUTH = {
    'pk1': 95.9, 'pk2': 95.1, 'pk3': 95.3, 'pk4': 93.6, 'pk5': 86.8
}

HEAD_CHECKPOINTS_AH = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
SFT_CHECKPOINTS_COUNT = 20

def extract_and_resample_curve(file_path):
    df = pd.read_csv(file_path)
    cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
    df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)
    
    discharge_df = df[(df['Mean_Cell_Voltage'] < 4.15) & (df['Mean_Cell_Voltage'] > 2.0)].copy()
    if 'AHDischarge' in discharge_df.columns:
        discharge_df = discharge_df.sort_values('AHDischarge')
        discharge_df['Ah_Relative'] = discharge_df['AHDischarge'] - discharge_df['AHDischarge'].iloc[0]
    else:
        return None, None
        
    ah = discharge_df['Ah_Relative'].values
    v = discharge_df['Mean_Cell_Voltage'].values
    return ah, v

def extract_enhanced_features(sft_v, sft_ah, cell_data=None):
    features = {}
    
    mid_idx = len(sft_v) // 2
    features['initial_slope'] = (sft_v[mid_idx] - sft_v[0]) / (mid_idx + 1e-5)
    features['final_slope'] = (sft_v[-1] - sft_v[mid_idx]) / (len(sft_v) - mid_idx + 1e-5)
    features['overall_slope'] = (sft_v[-1] - sft_v[0]) / (len(sft_v) + 1e-5)
    
    if len(sft_v) > 10:
        first_diff = np.diff(sft_v)
        second_diff = np.diff(first_diff)
        features['mean_curvature'] = np.mean(second_diff)
        features['max_curvature'] = np.max(np.abs(second_diff))
        features['curvature_std'] = np.std(second_diff)
    else:
        features['mean_curvature'] = 0
        features['max_curvature'] = 0
        features['curvature_std'] = 0
    
    features['voltage_std'] = np.std(sft_v)
    features['voltage_range'] = sft_v[0] - sft_v[-1]
    features['voltage_mean'] = np.mean(sft_v)
    
    plateau_mask = (sft_v > 3.4) & (sft_v < 3.7)
    features['plateau_length'] = np.sum(plateau_mask) / len(sft_v)
    
    end_idx = int(len(sft_v) * 0.8)
    features['end_slope'] = (sft_v[-1] - sft_v[end_idx]) / (len(sft_v) - end_idx + 1e-5)
    features['end_voltage_drop'] = sft_v[end_idx] - sft_v[-1]
    
    if cell_data is not None and len(cell_data) > 0:
        features['cell_imbalance_mean'] = np.mean(cell_data)
        features['cell_imbalance_std'] = np.std(cell_data)
    else:
        features['cell_imbalance_mean'] = 0
        features['cell_imbalance_std'] = 0
    
    return features

def generate_synthetic_samples(X_base, y_base, true_soh, n_samples=8):
    """Enhanced synthetic generation with multiple noise patterns"""
    X_synthetic = []
    y_synthetic = []
    
    for i in range(len(X_base)):
        for j in range(n_samples):
            # Vary noise level for diversity
            noise_level = np.random.uniform(0.005, 0.04)
            noise = np.random.normal(0, noise_level, X_base[i].shape)
            X_new = X_base[i] + noise
            
            # Add voltage noise with slight drift
            voltage_noise = np.random.normal(0, 0.004, y_base[i].shape)
            drift = np.linspace(0, np.random.uniform(-0.005, 0.005), len(y_base[i]))
            y_new = y_base[i] + voltage_noise + drift
            
            y_new = np.clip(y_new, 2.0, 4.2)
            
            X_synthetic.append(X_new)
            y_synthetic.append(y_new)
    
    return np.array(X_synthetic), np.array(y_synthetic)

def interpolate_between_packs(X1, y1, X2, y2, n_interpolations=3):
    """Create synthetic samples by interpolating between two different packs"""
    X_interp = []
    y_interp = []
    
    for i in range(min(len(X1), len(X2))):
        for alpha in np.linspace(0.2, 0.8, n_interpolations):
            X_new = alpha * X1[i] + (1 - alpha) * X2[i]
            y_new = alpha * y1[i] + (1 - alpha) * y2[i]
            y_new = np.clip(y_new, 2.0, 4.2)
            
            X_interp.append(X_new)
            y_interp.append(y_new)
    
    return np.array(X_interp), np.array(y_interp)

def prepare_training_data_by_crate(data_folder):
    """Enhanced preparation with aggressive augmentation"""
    data_0_3c = {'X': [], 'y': []}
    data_1_0c = {'X': [], 'y': []}
    
    # Store all base samples for cross-pack interpolation
    all_0_3c_X = []
    all_0_3c_y = []
    all_1_0c_X = []
    all_1_0c_y = []
    
    for file in os.listdir(data_folder):
        if not file.endswith('.csv'): continue
        
        pack_match = re.search(r'(pk\d+)', file)
        pack_id = pack_match.group(1) if pack_match else 'unknown'
        if pack_id not in SOH_GROUND_TRUTH: continue
        
        c_rate_match = re.search(r'(\d+\.\d+)C', file)
        c_rate = float(c_rate_match.group(1)) if c_rate_match else 0.3
        
        true_soh = SOH_GROUND_TRUTH[pack_id]
        ah, v = extract_and_resample_curve(os.path.join(data_folder, file))
        if ah is None or len(ah) < 100: continue
        
        total_cap = ah[-1]
        
        df = pd.read_csv(os.path.join(data_folder, file))
        cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
        if len(cell_cols) > 0:
            df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)
            df['Cell_Std'] = df[cell_cols].std(axis=1)
            cell_std_data = df['Cell_Std'].values
        else:
            cell_std_data = None
        
        # MORE CUTOFFS: Added 0.20, 0.25, 0.35, 0.45, 0.55, 0.65, 0.70, 0.75
        for cutoff_pct in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            cutoff_ah = total_cap * cutoff_pct
            cutoff_idx = np.searchsorted(ah, cutoff_ah)
            
            if cutoff_idx < 15: continue  # Lowered minimum
            
            head_ah = ah[:cutoff_idx]
            head_v = v[:cutoff_idx]
            sft_ah = ah[cutoff_idx:] - ah[cutoff_idx]
            sft_v = v[cutoff_idx:]
            
            if cell_std_data is not None and cutoff_idx < len(cell_std_data):
                sft_cell_imbalance = cell_std_data[cutoff_idx:]
            else:
                sft_cell_imbalance = None
            
            # Extract Target
            target_v = []
            for cp_ah in HEAD_CHECKPOINTS_AH:
                if cp_ah <= head_ah[-1]:
                    target_v.append(np.interp(cp_ah, head_ah, head_v))
                else:
                    target_v.append(head_v[-1] - 0.01)
            
            # Extract Enhanced Features
            sft_sampled_v = np.linspace(sft_v[0], sft_v[-1], SFT_CHECKPOINTS_COUNT)
            enhanced_feats = extract_enhanced_features(sft_v, sft_ah, sft_cell_imbalance)
            
            features = list(sft_sampled_v) + [
                true_soh, 
                enhanced_feats['initial_slope'],
                enhanced_feats['final_slope'],
                enhanced_feats['overall_slope'],
                enhanced_feats['mean_curvature'],
                enhanced_feats['max_curvature'],
                enhanced_feats['voltage_std'],
                enhanced_feats['voltage_range'],
                enhanced_feats['plateau_length'],
                enhanced_feats['end_slope'],
                enhanced_feats['cell_imbalance_mean'],
                enhanced_feats['cell_imbalance_std'],
                cutoff_ah
            ]
            
            if abs(c_rate - 0.3) < 0.01:
                data_0_3c['X'].append(features)
                data_0_3c['y'].append(target_v)
                all_0_3c_X.append(features)
                all_0_3c_y.append(target_v)
            elif abs(c_rate - 1.0) < 0.01:
                data_1_0c['X'].append(features)
                data_1_0c['y'].append(target_v)
                all_1_0c_X.append(features)
                all_1_0c_y.append(target_v)
    
    # AGGRESSIVE SYNTHETIC AUGMENTATION
    print("Generating synthetic samples...")
    X_03c_base = np.array(all_0_3c_X)
    y_03c_base = np.array(all_0_3c_y)
    X_10c_base = np.array(all_1_0c_X)
    y_10c_base = np.array(all_1_0c_y)
    
    # Generate 8 synthetic samples per real sample
    X_synth_03c, y_synth_03c = generate_synthetic_samples(X_03c_base, y_03c_base, 95.0, n_samples=8)
    X_synth_10c, y_synth_10c = generate_synthetic_samples(X_10c_base, y_10c_base, 95.0, n_samples=8)
    
    # Cross-pack interpolation (create hybrid samples)
    if len(X_03c_base) > 1:
        # FIXED: Added the y arrays
        X_interp_03c, y_interp_03c = interpolate_between_packs(
            X_03c_base[:-1], y_03c_base[:-1], X_03c_base[1:], y_03c_base[1:], n_interpolations=3
        )
        X_synth_03c = np.vstack([X_synth_03c, X_interp_03c])
        y_synth_03c = np.vstack([y_synth_03c, y_interp_03c])
    
    if len(X_10c_base) > 1:
        # FIXED: Added the y arrays
        X_interp_10c, y_interp_10c = interpolate_between_packs(
            X_10c_base[:-1], y_10c_base[:-1], X_10c_base[1:], y_10c_base[1:], n_interpolations=3
        )
        X_synth_10c = np.vstack([X_synth_10c, X_interp_10c])
        y_synth_10c = np.vstack([y_synth_10c, y_interp_10c])
    
    # Combine real and synthetic
    data_0_3c['X'].extend(X_synth_03c.tolist())
    data_0_3c['y'].extend(y_synth_03c.tolist())
    data_1_0c['X'].extend(X_synth_10c.tolist())
    data_1_0c['y'].extend(y_synth_10c.tolist())
    
    print(f"0.3C samples (with aggressive augmentation): {len(data_0_3c['X'])}")
    print(f"1.0C samples (with aggressive augmentation): {len(data_1_0c['X'])}")
    
    return data_0_3c, data_1_0c

def train_crate_specific_models(data_folder):
    print("Preparing C-rate specific training data with aggressive augmentation...")
    data_0_3c, data_1_0c = prepare_training_data_by_crate(data_folder)
    
    models = {}
    
    for c_rate_name, data in [('0.3C', data_0_3c), ('1.0C', data_1_0c)]:
        if len(data['X']) == 0:
            print(f"⚠️  No training data for {c_rate_name}")
            continue
            
        X = np.array(data['X'])
        y = np.array(data['y'])
        
        print(f"\nTraining {c_rate_name} model with {len(X)} samples...")
        print(f"  Input features: {X.shape[1]}")
        print(f"  Output checkpoints: {y.shape[1]}")
        
        model = xgb.XGBRegressor(
            n_estimators=600,  # Increased
            max_depth=8,       # Increased
            learning_rate=0.015,  # Lower for stability
            random_state=42,
            reg_alpha=0.15,
            reg_lambda=2.5,    # Stronger regularization
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3
        )
        
        multi_model = MultiOutputRegressor(model)
        
        if len(X) > 50:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            multi_model.fit(X_train, y_train)
            
            y_pred = multi_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            print(f"✅ {c_rate_name} Model MAE: {mae*1000:.1f} mV")
        else:
            print(f"️  Training on all data...")
            multi_model.fit(X, y)
        
        models[c_rate_name] = multi_model
    
    return models

if __name__ == "__main__":
    # Update this path to your actual data folder
    DATA_FOLDER = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/fft_raw_data"
    if os.path.exists(DATA_FOLDER):
        models = train_crate_specific_models(DATA_FOLDER)
        
        for c_rate_name, model in models.items():
            safe_name = c_rate_name.replace('.', '_')
            joblib.dump(model, f'reconstruction_model_{safe_name}_v4.pkl')
            print(f"✅ Saved reconstruction_model_{safe_name}_v4.pkl")
    else:
        print("Error: Data folder not found.")