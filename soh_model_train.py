# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import os
# import re
# import joblib
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_absolute_error
# import warnings
# warnings.filterwarnings('ignore')

# SOH_GROUND_TRUTH = {
#     'pk1': 95.9, 'pk2': 95.1, 'pk3': 95.3, 'pk4': 93.6, 'pk5': 86.8
# }

# def extract_features_from_slice(df, start_row):
#     df_slice = df.iloc[start_row:].copy()
#     if len(df_slice) < 50: return None 
#     features = {}
#     cell_cols = [c for c in df_slice.columns if 'Cell' in c and 'Temperature' not in c]
#     df_slice['Mean_Cell_Voltage'] = df_slice[cell_cols].mean(axis=1)
#     df_slice['Min_Cell_Voltage'] = df_slice[cell_cols].min(axis=1)
#     df_slice['Max_Cell_Voltage'] = df_slice[cell_cols].max(axis=1)
#     df_slice['Cell_Voltage_Std'] = df_slice[cell_cols].std(axis=1)
#     features['start_voltage'] = df_slice['Mean_Cell_Voltage'].iloc[:10].mean()
#     features['end_voltage'] = df_slice['Mean_Cell_Voltage'].iloc[-10:].mean()
#     features['voltage_drop'] = features['start_voltage'] - features['end_voltage']
#     features['mean_cell_imbalance_std'] = df_slice['Cell_Voltage_Std'].mean()
#     features['max_cell_spread'] = (df_slice['Max_Cell_Voltage'] - df_slice['Min_Cell_Voltage']).mean()
#     features['max_cell_spread_at_end'] = (df_slice['Max_Cell_Voltage'].iloc[-10:] - df_slice['Min_Cell_Voltage'].iloc[-10:]).mean()
#     features['min_cell_voltage_at_end'] = df_slice['Min_Cell_Voltage'].iloc[-10:].mean()
#     temp_cols = [c for c in df_slice.columns if 'Temperature' in c]
#     if temp_cols:
#         df_slice['Mean_Temp'] = df_slice[temp_cols].mean(axis=1)
#         df_slice['Max_Temp'] = df_slice[temp_cols].max(axis=1)
#         features['mean_temp'] = df_slice['Mean_Temp'].mean()
#         features['max_temp'] = df_slice['Max_Temp'].max()
#         features['temp_rise'] = df_slice['Mean_Temp'].iloc[-1] - df_slice['Mean_Temp'].iloc[0]
#     if 'AHDischarge' in df_slice.columns:
#         features['delta_Ah'] = df_slice['AHDischarge'].iloc[-1] - df_slice['AHDischarge'].iloc[0]
#         features['ah_per_voltage_drop'] = features['delta_Ah'] / (features['voltage_drop'] + 1e-5)
#         dV = df_slice['Mean_Cell_Voltage'].diff().dropna()
#         dAh = df_slice['AHDischarge'].diff().dropna()
#         dV_dAh = dV / (dAh.replace(0, 1e-5)) 
#         features['mean_dV_dAh'] = dV_dAh.mean()
#         features['std_dV_dAh'] = dV_dAh.std()
#     else:
#         features['delta_Ah'] = 0.0
#         features['ah_per_voltage_drop'] = 0.0
#     return features

# def generate_soh_synthetic_samples(X_base, y_base, n_samples=6):
#     """Generate synthetic SOH samples with realistic noise"""
#     X_synthetic = []
#     y_synthetic = []
    
#     for i in range(len(X_base)):
#         for j in range(n_samples):
#             # Add small noise (0.5-2%) to features
#             noise_level = np.random.uniform(0.005, 0.02)
#             noise = np.random.normal(0, noise_level, X_base[i].shape)
#             X_new = X_base[i] + noise
            
#             # Ensure features stay physically valid (non-negative where needed)
#             X_new = np.clip(X_new, 0, None)
            
#             # Add tiny noise to SOH target (0.1-0.5%)
#             soh_noise = np.random.uniform(-0.005, 0.005)
#             y_new = np.clip(y_base[i] + soh_noise, 80.0, 100.0)
            
#             X_synthetic.append(X_new)
#             y_synthetic.append(y_new)
    
#     return np.array(X_synthetic), np.array(y_synthetic)

# def interpolate_soh_samples(X1, y1, X2, y2, n_interpolations=4):
#     """Create synthetic samples by interpolating between packs"""
#     X_interp = []
#     y_interp = []
    
#     for i in range(min(len(X1), len(X2))):
#         for alpha in np.linspace(0.2, 0.8, n_interpolations):
#             X_new = alpha * X1[i] + (1 - alpha) * X2[i]
#             y_new = alpha * y1[i] + (1 - alpha) * y2[i]
#             y_new = np.clip(y_new, 80.0, 100.0)
            
#             X_interp.append(X_new)
#             y_interp.append(y_new)
    
#     return np.array(X_interp), np.array(y_interp)

# def build_dataset_for_crate(data_folder, target_crate):
#     real_data = []
#     for file in os.listdir(data_folder):
#         if not file.endswith('.csv'): continue
        
#         c_rate_match = re.search(r'(\d+\.\d+)C', file)
#         c_rate = float(c_rate_match.group(1)) if c_rate_match else 0.0
        
#         # STRICT FILTER: Only process files matching the target C-rate
#         if abs(c_rate - target_crate) > 0.05: continue
            
#         pack_match = re.search(r'(pk\d+)', file)
#         pack_id = pack_match.group(1) if pack_match else 'unknown'
#         if pack_id not in SOH_GROUND_TRUTH: continue
        
#         is_sfct = 1.0 if 'SFCT' in file.upper() or 'SFT' in file.upper() else 0.0
#         true_soh = SOH_GROUND_TRUTH[pack_id]
        
#         file_path = os.path.join(data_folder, file)
#         df = pd.read_csv(file_path)
#         total_rows = len(df)
        
#         for pct in [0.30, 0.40, 0.50, 0.60, 0.70]:
#             start_row = int(total_rows * pct)
#             feats = extract_features_from_slice(df, start_row)
#             if feats:
#                 feats['c_rate'] = c_rate
#                 feats['is_sfct'] = is_sfct
#                 feats['true_soh'] = true_soh
#                 real_data.append(feats)
                
#     real_df = pd.DataFrame(real_data)
#     print(f"Extracted {len(real_df)} real base feature sets for {target_crate}C.")
    
#     # Bounded Interpolation
#     target_soh_levels = [95.0, 94.0, 93.0, 92.0, 91.0, 90.0, 89.0, 88.0, 87.0]
#     augmented_data = []
#     feature_cols = [c for c in real_df.columns if c not in ['true_soh', 'pack_id']]
    
#     if len(real_df) > 0:
#         real_sohs = real_df['true_soh'].values
#         real_features = real_df[feature_cols].values
#         sorted_indices = np.argsort(real_sohs)
#         real_sohs = real_sohs[sorted_indices]
#         real_features = real_features[sorted_indices]
        
#         for target_soh in target_soh_levels:
#             if target_soh < real_sohs.min() or target_soh > real_sohs.max(): continue
#             idx = np.searchsorted(real_sohs, target_soh)
#             soh_low, soh_high = real_sohs[idx-1], real_sohs[idx]
#             weight_high = (target_soh - soh_low) / (soh_high - soh_low)
#             weight_low = 1.0 - weight_high
#             synth_features = (weight_low * real_features[idx-1]) + (weight_high * real_features[idx])
#             noise = np.random.normal(0, 0.005, synth_features.shape)
#             final_features = np.clip(synth_features + noise, 0, None) 
#             row_dict = dict(zip(feature_cols, final_features))
#             row_dict['true_soh'] = target_soh
#             augmented_data.append(row_dict)
    
#     # Combine real + bounded interpolation
#     combined_df = pd.DataFrame(augmented_data + real_df.to_dict('records'))
    
#     # AGGRESSIVE AUGMENTATION FOR 1.0C
#     if target_crate == 1.0 and len(combined_df) > 0:
#         print(f"Applying aggressive augmentation for 1.0C model...")
#         X_base = combined_df[feature_cols].values
#         y_base = combined_df['true_soh'].values
        
#         # Generate 6 synthetic samples per real sample
#         X_synth, y_synth = generate_soh_synthetic_samples(X_base, y_base, n_samples=6)
        
#         # Cross-pack interpolation
#         if len(X_base) > 1:
#             X_interp, y_interp = interpolate_soh_samples(
#                 X_base[:-1], y_base[:-1], X_base[1:], y_base[1:], n_interpolations=4
#             )
#             X_synth = np.vstack([X_synth, X_interp])
#             y_synth = np.hstack([y_synth, y_interp])
        
#         # Combine everything
#         X_final = np.vstack([X_base, X_synth])
#         y_final = np.hstack([y_base, y_synth])
        
#         print(f"  1.0C samples: {len(X_base)} real → {len(X_final)} total (with augmentation)")
        
#         final_df = pd.DataFrame(X_final, columns=feature_cols)
#         final_df['true_soh'] = y_final
#         return final_df
    
#     return combined_df

# def train_specific_soh_model(data_folder, target_crate):
#     print(f"\n{'='*50}")
#     print(f"Training SOH Model specifically for {target_crate}C")
#     print(f"{'='*50}")
#     X_df = build_dataset_for_crate(data_folder, target_crate)
    
#     feature_cols = [c for c in X_df.columns if c not in ['true_soh', 'pack_id']]
#     X = X_df[feature_cols]
#     y = X_df['true_soh'].values
    
#     print(f"Total samples for training: {len(X_df)}")
    
#     model_soh = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, 
#                                  objective='reg:squarederror', random_state=42, reg_alpha=0.1)
#     model_soh.fit(X, y)
#     return model_soh, feature_cols

# if __name__ == "__main__":
#     DATA_FOLDER = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/raw_dataset" 
#     if os.path.exists(DATA_FOLDER):
#         # Train 0.3C Model
#         model_03c, features_03c = train_specific_soh_model(DATA_FOLDER, 0.3)
#         joblib.dump(model_03c, 'soh_model_0_3C_v6.pkl')
#         joblib.dump(features_03c, 'feature_names_0_3C_v6.pkl')
#         print("✅ Saved soh_model_0_3C_v6.pkl")
        
#         # Train 1.0C Model
#         model_10c, features_10c = train_specific_soh_model(DATA_FOLDER, 1.0)
#         joblib.dump(model_10c, 'soh_model_1_0C_v6.pkl')
#         joblib.dump(features_10c, 'feature_names_1_0C_v6.pkl')
#         print("✅ Saved soh_model_1_0C_v6.pkl")
#     else:
#         print("Error: Data folder not found.")