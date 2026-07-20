
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import re
import joblib
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# CORRECTED: Ground truth SOH for each pack at each C-rate
SOH_GROUND_TRUTH = {
    'pk1': {0.3: 95.88, 1.0: 94.40},
    'pk2': {0.3: 95.14, 0.95: 91.96},  # 0.95C treated as 1.0C
    'pk3': {0.3: 95.32, 1.0: 91.67},
    'pk4': {0.3: 93.63, 1.0: 89.08},
    'pk5': {0.3: 86.78}  # No 1.0C data available yet
}

def get_soh_for_pack_and_crate(pack_id, c_rate):
    """Get the correct SOH value for a specific pack and C-rate."""
    if pack_id not in SOH_GROUND_TRUTH:
        return None
    
    # Handle 0.95C as 1.0C
    if abs(c_rate - 0.95) < 0.05:
        c_rate = 1.0
    
    soh_dict = SOH_GROUND_TRUTH[pack_id]
    
    # Try exact match first
    if c_rate in soh_dict:
        return soh_dict[c_rate]
    
    # Try to find closest C-rate
    available_rates = list(soh_dict.keys())
    if available_rates:
        closest_rate = min(available_rates, key=lambda x: abs(x - c_rate))
        return soh_dict[closest_rate]
    
    return None

def extract_features_from_slice(df, start_row):
    """Extracts physics-based features from a partial or full discharge slice."""
    df_slice = df.iloc[start_row:].copy()
    if len(df_slice) < 50: 
        return None 
        
    features = {}
    cell_cols = [c for c in df_slice.columns if 'Cell' in c and 'Temperature' not in c]
    
    # Cell voltage stats
    df_slice['Mean_Cell_Voltage'] = df_slice[cell_cols].mean(axis=1)
    df_slice['Min_Cell_Voltage'] = df_slice[cell_cols].min(axis=1)
    df_slice['Max_Cell_Voltage'] = df_slice[cell_cols].max(axis=1)
    df_slice['Cell_Voltage_Std'] = df_slice[cell_cols].std(axis=1)
    
    features['start_voltage'] = df_slice['Mean_Cell_Voltage'].iloc[:10].mean()
    features['end_voltage'] = df_slice['Mean_Cell_Voltage'].iloc[-10:].mean()
    features['voltage_drop'] = features['start_voltage'] - features['end_voltage']
    
    # Imbalance features
    features['mean_cell_imbalance_std'] = df_slice['Cell_Voltage_Std'].mean()
    features['max_cell_spread'] = (df_slice['Max_Cell_Voltage'] - df_slice['Min_Cell_Voltage']).mean()
    features['max_cell_spread_at_end'] = (df_slice['Max_Cell_Voltage'].iloc[-10:] - df_slice['Min_Cell_Voltage'].iloc[-10:]).mean()
    features['min_cell_voltage_at_end'] = df_slice['Min_Cell_Voltage'].iloc[-10:].mean()
    
    # Temperature features
    temp_cols = [c for c in df_slice.columns if 'Temperature' in c]
    if temp_cols:
        df_slice['Mean_Temp'] = df_slice[temp_cols].mean(axis=1)
        features['mean_temp'] = df_slice['Mean_Temp'].mean()
        features['temp_rise'] = df_slice['Mean_Temp'].iloc[-1] - df_slice['Mean_Temp'].iloc[0]
        
    # Capacity and Resistance proxy features
    if 'AHDischarge' in df_slice.columns:
        features['delta_Ah'] = df_slice['AHDischarge'].iloc[-1] - df_slice['AHDischarge'].iloc[0]
        features['ah_per_voltage_drop'] = features['delta_Ah'] / (features['voltage_drop'] + 1e-5)
        
        # dV/dAh (Slope of the curve - highly indicative of SOH)
        dV = df_slice['Mean_Cell_Voltage'].diff().dropna()
        dAh = df_slice['AHDischarge'].diff().dropna()
        dV_dAh = dV / (dAh.replace(0, 1e-5)) 
        features['mean_dV_dAh'] = dV_dAh.mean()
        features['std_dV_dAh'] = dV_dAh.std()
        features['min_dV_dAh'] = dV_dAh.min()
        
    return features

def build_dataset(data_folder, target_c_rate=None):
    """
    Builds dataset with correct C-rate specific SOH labels.
    If target_c_rate is provided, filters for it (e.g., 0.3 or 1.0).
    """
    real_data = []
    
    for file in os.listdir(data_folder):
        if not file.endswith('.csv'): continue
        
        pack_match = re.search(r'(pk\d+)', file)
        pack_id = pack_match.group(1) if pack_match else 'unknown'
        if pack_id not in SOH_GROUND_TRUTH: continue
        
        c_rate_match = re.search(r'(\d+\.\d+)C', file)
        if not c_rate_match: continue
        c_rate = float(c_rate_match.group(1))
        
        # Filter by C-rate if requested
        if target_c_rate is not None:
            if target_c_rate == 0.3:
                if abs(c_rate - 0.3) > 0.05: continue
            elif target_c_rate == 1.0:
                if c_rate < 0.9: continue  # Include 0.95C and 1.0C
        
        # Get the CORRECT SOH for this specific pack and C-rate combination
        true_soh = get_soh_for_pack_and_crate(pack_id, c_rate)
        if true_soh is None:
            print(f"Warning: No SOH found for {pack_id} at {c_rate}C")
            continue
            
        is_sfct = 1.0 if 'SFCT' in file.upper() or 'SFT' in file.upper() else 0.0
        
        df = pd.read_csv(os.path.join(data_folder, file))
        total_rows = len(df)
        
        # Extract features from different partial discharge starting points
        for pct in [0.40, 0.50, 0.60, 0.70]:
            start_row = int(total_rows * pct)
            feats = extract_features_from_slice(df, start_row)
            if feats:
                feats['c_rate'] = c_rate
                feats['is_sfct'] = is_sfct
                feats['true_soh'] = true_soh
                feats['pack_id'] = pack_id
                feats['slice_start_pct'] = pct
                real_data.append(feats)
                
    return pd.DataFrame(real_data)

def train_separate_models(data_folder):
    print("Building Physics-Augmented Dataset with C-rate specific SOH labels...")
    full_df = build_dataset(data_folder)
    
    if full_df.empty:
        print("No data found! Check file paths and naming conventions.")
        return None

    print(f"Total samples loaded: {len(full_df)}")
    print(f"Pack distribution:\n{full_df['pack_id'].value_counts()}")
    print(f"\nC-rate distribution:\n{full_df['c_rate'].value_counts()}")
    
    feature_cols = [c for c in full_df.columns if c not in ['true_soh', 'pack_id']]
    
    # --- TRAIN SEPARATE MODELS FOR 0.3C AND 1.0C ---
    models = {}
    
    for target_c in [0.3, 1.0]:
        print(f"\n{'='*60}")
        print(f"Training Model for C-Rate: {target_c}C")
        print(f"{'='*60}")
        
        # Filter data for this C-rate
        if target_c == 0.3:
            df_c = full_df[abs(full_df['c_rate'] - 0.3) < 0.05].copy()
        else:  # 1.0C
            df_c = full_df[full_df['c_rate'] >= 0.9].copy()
        
        if len(df_c) == 0:
            print(f"No data available for {target_c}C training!")
            continue
            
        # Add C-rate normalized features
        if 'voltage_drop' in df_c.columns:
            df_c['voltage_drop_norm'] = df_c['voltage_drop'] / (target_c + 1e-5)
        if 'delta_Ah' in df_c.columns:
            df_c['delta_Ah_norm'] = df_c['delta_Ah'] / (target_c + 1e-5)
            
        X = df_c[feature_cols + ['voltage_drop_norm', 'delta_Ah_norm']]
        y = df_c['true_soh'].values
        groups = df_c['pack_id'].values
        
        print(f"Samples for {target_c}C: {len(X)}")
        print(f"SOH range: {y.min():.2f}% - {y.max():.2f}%")
        print(f"Features ({len(X.columns)}): {list(X.columns)}\n")
        
        # Use Leave-One-Pack-Out CV
        logo = LeaveOneGroupOut()
        logo_scores = []
        
        print("Running Leave-One-Pack-Out Cross-Validation...")
        for train_idx, test_idx in logo.split(X, y, groups):
            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05, 
                objective='reg:squarederror', random_state=42, 
                reg_alpha=0.1, reg_lambda=1.0
            )
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[test_idx])
            mae = mean_absolute_error(y[test_idx], pred)
            logo_scores.append(mae)
            print(f"  Left out {groups[test_idx][0]} -> MAE: {mae:.2f}%")
            
        if logo_scores:
            avg_mae = np.mean(logo_scores)
            print(f"\n>>> AVERAGE LOPO SOH ERROR for {target_c}C: {avg_mae:.2f}% <<<\n")
            
            # Train final model on all data for this C-rate
            final_model = xgb.XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05, 
                objective='reg:squarederror', random_state=42, 
                reg_alpha=0.1, reg_lambda=1.0
            )
            final_model.fit(X, y)
            models[target_c] = (final_model, list(X.columns))
        else:
            print(f"No validation scores for {target_c}C\n")
        
    return models

def predict_soh(model_path, feature_names_path, input_features):
    """
    Load a trained model and make predictions.
    input_features: dict with feature names and values
    """
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)
    
    # Create feature vector in correct order
    X = np.array([[input_features.get(f, 0) for f in feature_names]])
    
    prediction = model.predict(X)[0]
    return prediction

if __name__ == "__main__":
    DATA_FOLDER = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/raw_dataset" 
    
    if os.path.exists(DATA_FOLDER):
        trained_models = train_separate_models(DATA_FOLDER)
        
        if trained_models:
            # Save models
            for c_rate, (model, feats) in trained_models.items():
                c_str = str(c_rate).replace('.', '_')
                joblib.dump(model, f'soh_model_{c_str}c.pkl')
                joblib.dump(feats, f'feature_names_{c_str}c.pkl')
                print(f":white_check_mark: Model for {c_rate}C saved successfully!")
        else:
            print(" No models were trained!")
    else:
        print(":x: Data folder not found.")