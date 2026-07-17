# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import os
# import re
# import joblib
# from sklearn.model_selection import train_test_split, LeaveOneGroupOut
# from sklearn.metrics import mean_absolute_error
# from sklearn.multioutput import MultiOutputRegressor
# import warnings
# warnings.filterwarnings('ignore')

# # C-rate specific SOH ground truth
# SOH_GROUND_TRUTH = {
#     'pk1': {0.3: 95.88, 1.0: 94.40},
#     'pk2': {0.3: 95.14, 0.95: 91.96},
#     'pk3': {0.3: 95.32, 1.0: 91.67},
#     'pk4': {0.3: 93.63, 1.0: 89.08},
#     'pk5': {0.3: 86.78}
# }

# # Checkpoints for prediction (Ah values where we predict voltage)
# HEAD_CHECKPOINTS_AH = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
# SFT_CHECKPOINTS_COUNT = 20  # Number of sampled voltages from SFT

# def extract_and_resample_curve(file_path):
#     """Extract voltage and capacity curves from CSV, handling cell imbalance."""
#     df = pd.read_csv(file_path)
#     cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
    
#     if not cell_cols:
#         return None, None, None
    
#     # Calculate mean cell voltage and cell imbalance (std)
#     df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)
#     df['Cell_Imbalance_Std'] = df[cell_cols].std(axis=1)
    
#     # Filter to discharge region (4.15V to 2.0V)
#     mask = (df['Mean_Cell_Voltage'] < 4.15) & (df['Mean_Cell_Voltage'] > 2.0)
#     discharge_df = df[mask].copy()
    
#     if 'AHDischarge' not in discharge_df.columns:
#         return None, None, None
    
#     discharge_df = discharge_df.sort_values('AHDischarge')
#     discharge_df['Ah_Relative'] = discharge_df['AHDischarge'] - discharge_df['AHDischarge'].iloc[0]
    
#     ah = discharge_df['Ah_Relative'].values
#     v = discharge_df['Mean_Cell_Voltage'].values
#     cell_imbalance = discharge_df['Cell_Imbalance_Std'].values
    
#     return ah, v, cell_imbalance

# def generate_physics_curve(total_capacity, c_rate, soh, noise_level=0.003):
#     """
#     Generate a physics-based discharge curve using real battery equations.
    
#     V(q) = V_ocv - I*R_int - V_sag_1C - k*(q/Q)^3
    
#     Where:
#     - V_ocv: Open circuit voltage (~4.15-4.18V)
#     - I*R_int: IR drop (larger for 1.0C)
#     - V_sag_1C: Initial voltage sag for 1.0C (exponential decay)
#     - k*(q/Q)^3: Cubic discharge curve (creates the knee at end)
#     """
#     NOMINAL_CAPACITY = 156.0
#     BASE_RESISTANCE_mOhm = 8.0
    
#     # Scale capacity by SOH
#     actual_capacity = (soh / 100.0) * NOMINAL_CAPACITY
#     current_A = c_rate * NOMINAL_CAPACITY
    
#     # Internal resistance increases as SOH drops
#     degradation_factor = (100.0 - soh) / 20.0
#     internal_resistance = BASE_RESISTANCE_mOhm * (1 + 1.5 * degradation_factor)
    
#     # Generate capacity points
#     ah = np.linspace(0, actual_capacity, 500)
    
#     # Base OCV (slight variation)
#     v_ocv = np.random.uniform(4.15, 4.18)
    
#     # IR drop (constant throughout discharge)
#     ir_drop = (current_A * internal_resistance) / 1000.0
    
#     # Total voltage drop from OCV to end (~1.15-1.25V)
#     total_ocv_drop = np.random.uniform(1.15, 1.25)
    
#     # Cubic discharge curve (flat start, steep end)
#     v_curve = v_ocv - ir_drop - total_ocv_drop * (ah / actual_capacity)**3
    
#     # 1.0C Initial Voltage Sag (exponential decay in first 5% of discharge)
#     if c_rate >= 0.9:
#         sag_magnitude = ir_drop * 0.3  # 30% of IR drop as initial sag
#         sag_tau = actual_capacity * 0.02  # Time constant (2% of capacity)
#         v_sag = sag_magnitude * np.exp(-ah / sag_tau)
#         v_curve = v_curve - v_sag
    
#     # Add realistic noise
#     noise = np.random.normal(0, noise_level, size=len(ah))
#     v_curve = v_curve + noise
    
#     # Enforce monotonic decreasing (physics constraint)
#     for i in range(1, len(v_curve)):
#         if v_curve[i] >= v_curve[i-1]:
#             v_curve[i] = v_curve[i-1] - 0.001
    
#     # Generate cell imbalance (increases with discharge)
#     cell_imbalance = np.linspace(0.005, 0.025, len(ah)) * (1 + degradation_factor)
#     cell_imbalance = cell_imbalance + np.random.normal(0, 0.002, size=len(ah))
    
#     return ah, v_curve, cell_imbalance

# def extract_enhanced_features(sft_v, sft_ah, cell_imbalance=None):
#     """Extract physics-based features from SFT slice."""
#     features = {}
    
#     # Calculate derivatives (curvature)
#     if len(sft_ah) > 10:
#         dv_dah = np.gradient(sft_v, sft_ah)
#         d2v_dah2 = np.gradient(dv_dah, sft_ah)
#         features['mean_curvature'] = np.mean(np.abs(d2v_dah2))
#         features['max_curvature'] = np.max(np.abs(d2v_dah2))
#         features['curvature_std'] = np.std(d2v_dah2)
#     else:
#         features['mean_curvature'] = 0
#         features['max_curvature'] = 0
#         features['curvature_std'] = 0
    
#     # Slope features
#     mid_idx = len(sft_v) // 2
#     features['initial_slope'] = (sft_v[mid_idx] - sft_v[0]) / (sft_ah[mid_idx] - sft_ah[0] + 1e-5)
#     features['final_slope'] = (sft_v[-1] - sft_v[mid_idx]) / (sft_ah[-1] - sft_ah[mid_idx] + 1e-5)
#     features['overall_slope'] = (sft_v[-1] - sft_v[0]) / (sft_ah[-1] - sft_ah[0] + 1e-5)
    
#     # Voltage statistics
#     features['voltage_std'] = np.std(sft_v)
#     features['voltage_range'] = sft_v[0] - sft_v[-1]
#     features['voltage_mean'] = np.mean(sft_v)
    
#     # Plateau detection (flat region around 3.4-3.7V)
#     plateau_mask = (sft_v > 3.4) & (sft_v < 3.7)
#     features['plateau_length'] = np.sum(plateau_mask) / len(sft_v)
    
#     # End-of-discharge features
#     end_idx = int(len(sft_v) * 0.8)
#     features['end_slope'] = (sft_v[-1] - sft_v[end_idx]) / (sft_ah[-1] - sft_ah[end_idx] + 1e-5)
#     features['end_voltage_drop'] = sft_v[end_idx] - sft_v[-1]
    
#     # Cell imbalance features
#     if cell_imbalance is not None and len(cell_imbalance) > 0:
#         features['cell_imbalance_mean'] = np.mean(cell_imbalance)
#         features['cell_imbalance_std'] = np.std(cell_imbalance)
#         features['cell_imbalance_max'] = np.max(cell_imbalance)
#     else:
#         features['cell_imbalance_mean'] = 0
#         features['cell_imbalance_std'] = 0
#         features['cell_imbalance_max'] = 0
    
#     return features

# def prepare_training_data_by_crate(data_folder):
#     """Prepare training data from real FFT files with slicing."""
#     data_0_3c = {'X': [], 'y': []}
#     data_1_0c = {'X': [], 'y': []}
    
#     all_0_3c_X, all_0_3c_y = [], []
#     all_1_0c_X, all_1_0c_y = [], []
    
#     for file in os.listdir(data_folder):
#         if not file.endswith('.csv'):
#             continue
        
#         pack_match = re.search(r'(pk\d+)', file)
#         pack_id = pack_match.group(1) if pack_match else 'unknown'
#         if pack_id not in SOH_GROUND_TRUTH:
#             continue
        
#         c_rate_match = re.search(r'(\d+(?:\.\d+)?)C', file)
#         if not c_rate_match:
#             continue
#         c_rate = float(c_rate_match.group(1))
        
#         # Get C-rate specific SOH
#         soh_dict = SOH_GROUND_TRUTH[pack_id]
#         c_rate_key = 1.0 if abs(c_rate - 0.95) < 0.05 else c_rate
#         true_soh = soh_dict.get(c_rate_key, soh_dict.get(0.3, 95.0))
        
#         ah, v, cell_imbalance = extract_and_resample_curve(os.path.join(data_folder, file))
#         if ah is None or len(ah) < 100:
#             continue
        
#         total_cap = ah[-1]
        
#         # Slice at different cutoff points (40%, 50%, 60%, 70%)
#         for cutoff_pct in [0.40, 0.50, 0.60, 0.70]:
#             cutoff_ah = total_cap * cutoff_pct
#             cutoff_idx = np.searchsorted(ah, cutoff_ah)
            
#             if cutoff_idx < 20:
#                 continue
            
#             head_ah = ah[:cutoff_idx]
#             head_v = v[:cutoff_idx]
#             sft_ah = ah[cutoff_idx:] - ah[cutoff_idx]
#             sft_v = v[cutoff_idx:]
#             sft_cell_imbalance = cell_imbalance[cutoff_idx:] if cell_imbalance is not None else None
            
#             # Extract target: voltages at checkpoints
#             target_v = []
#             for cp_ah in HEAD_CHECKPOINTS_AH:
#                 if cp_ah <= head_ah[-1]:
#                     target_v.append(np.interp(cp_ah, head_ah, head_v))
#                 else:
#                     # Extrapolate using end slope
#                     end_slope = (sft_v[0] - head_v[-1]) / (sft_ah[0] + 1e-5)
#                     ext_v = head_v[-1] + end_slope * (cp_ah - head_ah[-1])
#                     target_v.append(max(2.5, ext_v))
            
#             # Extract enhanced features from SFT
#             sft_sampled_v = np.linspace(sft_v[0], sft_v[-1], SFT_CHECKPOINTS_COUNT)
#             enhanced_feats = extract_enhanced_features(sft_v, sft_ah, sft_cell_imbalance)
            
#             features = list(sft_sampled_v) + [
#                 true_soh,
#                 enhanced_feats['initial_slope'],
#                 enhanced_feats['final_slope'],
#                 enhanced_feats['overall_slope'],
#                 enhanced_feats['mean_curvature'],
#                 enhanced_feats['max_curvature'],
#                 enhanced_feats['voltage_std'],
#                 enhanced_feats['voltage_range'],
#                 enhanced_feats['plateau_length'],
#                 enhanced_feats['end_slope'],
#                 enhanced_feats['cell_imbalance_mean'],
#                 enhanced_feats['cell_imbalance_std'],
#                 enhanced_feats['cell_imbalance_max'],
#                 cutoff_ah
#             ]
            
#             if abs(c_rate - 0.3) < 0.05:
#                 data_0_3c['X'].append(features)
#                 data_0_3c['y'].append(target_v)
#                 all_0_3c_X.append(features)
#                 all_0_3c_y.append(target_v)
#             elif c_rate >= 0.9:
#                 data_1_0c['X'].append(features)
#                 data_1_0c['y'].append(target_v)
#                 all_1_0c_X.append(features)
#                 all_1_0c_y.append(target_v)
    
#     print(f"Real 0.3C samples: {len(data_0_3c['X'])}")
#     print(f"Real 1.0C samples: {len(data_1_0c['X'])}")
    
#     # Generate physics-based synthetic data
#     print("\nGenerating physics-based synthetic curves...")
    
#     # Generate 400 synthetic curves per C-rate
#     for _ in range(400):
#         soh = np.random.uniform(85.0, 100.0)
        
#         # 0.3C synthetic
#         ah_03c, v_03c, imb_03c = generate_physics_curve(156.0, 0.3, soh, noise_level=0.003)
#         total_cap_03c = ah_03c[-1]
        
#         for cutoff_pct in [0.40, 0.50, 0.60, 0.70]:
#             cutoff_ah = total_cap_03c * cutoff_pct
#             cutoff_idx = np.searchsorted(ah_03c, cutoff_ah)
#             if cutoff_idx < 20:
#                 continue
            
#             head_ah = ah_03c[:cutoff_idx]
#             head_v = v_03c[:cutoff_idx]
#             sft_ah = ah_03c[cutoff_idx:] - ah_03c[cutoff_idx]
#             sft_v = v_03c[cutoff_idx:]
#             sft_imb = imb_03c[cutoff_idx:]
            
#             target_v = []
#             for cp_ah in HEAD_CHECKPOINTS_AH:
#                 if cp_ah <= head_ah[-1]:
#                     target_v.append(np.interp(cp_ah, head_ah, head_v))
#                 else:
#                     end_slope = (sft_v[0] - head_v[-1]) / (sft_ah[0] + 1e-5)
#                     ext_v = head_v[-1] + end_slope * (cp_ah - head_ah[-1])
#                     target_v.append(max(2.5, ext_v))
            
#             sft_sampled_v = np.linspace(sft_v[0], sft_v[-1], SFT_CHECKPOINTS_COUNT)
#             enhanced_feats = extract_enhanced_features(sft_v, sft_ah, sft_imb)
            
#             features = list(sft_sampled_v) + [
#                 soh,
#                 enhanced_feats['initial_slope'],
#                 enhanced_feats['final_slope'],
#                 enhanced_feats['overall_slope'],
#                 enhanced_feats['mean_curvature'],
#                 enhanced_feats['max_curvature'],
#                 enhanced_feats['voltage_std'],
#                 enhanced_feats['voltage_range'],
#                 enhanced_feats['plateau_length'],
#                 enhanced_feats['end_slope'],
#                 enhanced_feats['cell_imbalance_mean'],
#                 enhanced_feats['cell_imbalance_std'],
#                 enhanced_feats['cell_imbalance_max'],
#                 cutoff_ah
#             ]
            
#             data_0_3c['X'].append(features)
#             data_0_3c['y'].append(target_v)
        
#         # 1.0C synthetic (with initial sag)
#         ah_10c, v_10c, imb_10c = generate_physics_curve(156.0, 1.0, soh, noise_level=0.005)
#         total_cap_10c = ah_10c[-1]
        
#         for cutoff_pct in [0.40, 0.50, 0.60, 0.70]:
#             cutoff_ah = total_cap_10c * cutoff_pct
#             cutoff_idx = np.searchsorted(ah_10c, cutoff_ah)
#             if cutoff_idx < 20:
#                 continue
            
#             head_ah = ah_10c[:cutoff_idx]
#             head_v = v_10c[:cutoff_idx]
#             sft_ah = ah_10c[cutoff_idx:] - ah_10c[cutoff_idx]
#             sft_v = v_10c[cutoff_idx:]
#             sft_imb = imb_10c[cutoff_idx:]
            
#             target_v = []
#             for cp_ah in HEAD_CHECKPOINTS_AH:
#                 if cp_ah <= head_ah[-1]:
#                     target_v.append(np.interp(cp_ah, head_ah, head_v))
#                 else:
#                     end_slope = (sft_v[0] - head_v[-1]) / (sft_ah[0] + 1e-5)
#                     ext_v = head_v[-1] + end_slope * (cp_ah - head_ah[-1])
#                     target_v.append(max(2.5, ext_v))
            
#             sft_sampled_v = np.linspace(sft_v[0], sft_v[-1], SFT_CHECKPOINTS_COUNT)
#             enhanced_feats = extract_enhanced_features(sft_v, sft_ah, sft_imb)
            
#             features = list(sft_sampled_v) + [
#                 soh,
#                 enhanced_feats['initial_slope'],
#                 enhanced_feats['final_slope'],
#                 enhanced_feats['overall_slope'],
#                 enhanced_feats['mean_curvature'],
#                 enhanced_feats['max_curvature'],
#                 enhanced_feats['voltage_std'],
#                 enhanced_feats['voltage_range'],
#                 enhanced_feats['plateau_length'],
#                 enhanced_feats['end_slope'],
#                 enhanced_feats['end_voltage_drop'],
#                 enhanced_feats['cell_imbalance_mean'],
#                 enhanced_feats['cell_imbalance_std'],
#                 enhanced_feats['cell_imbalance_max'],
#                 cutoff_ah
#             ]
            
#             data_1_0c['X'].append(features)
#             data_1_0c['y'].append(target_v)
    
#     print(f"Total 0.3C samples (real + synth): {len(data_0_3c['X'])}")
#     print(f"Total 1.0C samples (real + synth): {len(data_1_0c['X'])}")
    
#     return data_0_3c, data_1_0c

# def train_crate_specific_models(data_folder):
#     """Train separate XGBoost models for 0.3C and 1.0C."""
#     print("Preparing C-rate specific training data...")
#     data_0_3c, data_1_0c = prepare_training_data_by_crate(data_folder)
    
#     models = {}
    
#     for c_rate_name, data in [('0.3C', data_0_3c), ('1.0C', data_1_0c)]:
#         if len(data['X']) == 0:
#             print(f"⚠️  No training data for {c_rate_name}")
#             continue
        
#         X = np.array(data['X'])
#         y = np.array(data['y'])
        
#         print(f"\n{'='*60}")
#         print(f"Training {c_rate_name} model with {len(X)} samples...")
#         print(f"{'='*60}")
#         print(f"  Input features: {X.shape[1]}")
#         print(f"  Output checkpoints: {y.shape[1]}")
        
#         # XGBoost with regularization for smoothness
#         model = xgb.XGBRegressor(
#             n_estimators=500,
#             max_depth=5,  # Prevent overfitting
#             learning_rate=0.02,
#             random_state=42,
#             reg_alpha=0.2,  # L1 regularization
#             reg_lambda=2.0,  # L2 regularization
#             subsample=0.85,
#             colsample_bytree=0.85,
#             min_child_weight=3
#         )
        
#         multi_model = MultiOutputRegressor(model)
        
#         # Train-test split for validation
#         if len(X) > 100:
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#             multi_model.fit(X_train, y_train)
            
#             y_pred = multi_model.predict(X_test)
#             mae = mean_absolute_error(y_test, y_pred)
#             print(f"\n✅ {c_rate_name} Model MAE: {mae*1000:.1f} mV")
#         else:
#             print(f"⚠️  Not enough samples for validation, training on all data...")
#             multi_model.fit(X, y)
        
#         models[c_rate_name] = multi_model
    
#     return models

# if __name__ == "__main__":
#     DATA_FOLDER = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/fft_raw_data"
    
#     if os.path.exists(DATA_FOLDER):
#         models = train_crate_specific_models(DATA_FOLDER)
        
#         for c_rate_name, model in models.items():
#             safe_name = c_rate_name.replace('.', '_')
#             joblib.dump(model, f'reconstruction_model_{safe_name}_v6.pkl')
#             print(f"✅ Saved reconstruction_model_{safe_name}_v6.pkl")
#     else:
#         print(f"❌ Error: Data folder not found at {DATA_FOLDER}")

# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import os
# import re
# import joblib
# from sklearn.model_selection import LeaveOneGroupOut
# from sklearn.metrics import mean_absolute_error
# import warnings
# warnings.filterwarnings('ignore')

# # CORRECTED: Ground truth SOH for each pack at each C-rate
# SOH_GROUND_TRUTH = {
#     'pk1': {0.3: 95.88, 1.0: 94.40},
#     'pk2': {0.3: 95.14, 0.95: 91.96},  # 0.95C treated as 1.0C
#     'pk3': {0.3: 95.32, 1.0: 91.67},
#     'pk4': {0.3: 93.63, 1.0: 89.08},
#     'pk5': {0.3: 86.78}  # No 1.0C data available yet
# }

# def get_soh_for_pack_and_crate(pack_id, c_rate):
#     """Get the correct SOH value for a specific pack and C-rate."""
#     if pack_id not in SOH_GROUND_TRUTH:
#         return None
    
#     # Handle 0.95C as 1.0C
#     if abs(c_rate - 0.95) < 0.05:
#         c_rate = 1.0
    
#     soh_dict = SOH_GROUND_TRUTH[pack_id]
    
#     # Try exact match first
#     if c_rate in soh_dict:
#         return soh_dict[c_rate]
    
#     # Try to find closest C-rate
#     available_rates = list(soh_dict.keys())
#     if available_rates:
#         closest_rate = min(available_rates, key=lambda x: abs(x - c_rate))
#         return soh_dict[closest_rate]
    
#     return None

# def extract_features_from_slice(df, start_row):
#     """Extracts physics-based features from a partial or full discharge slice."""
#     df_slice = df.iloc[start_row:].copy()
#     if len(df_slice) < 50: 
#         return None 
        
#     features = {}
#     cell_cols = [c for c in df_slice.columns if 'Cell' in c and 'Temperature' not in c]
    
#     # Cell voltage stats
#     df_slice['Mean_Cell_Voltage'] = df_slice[cell_cols].mean(axis=1)
#     df_slice['Min_Cell_Voltage'] = df_slice[cell_cols].min(axis=1)
#     df_slice['Max_Cell_Voltage'] = df_slice[cell_cols].max(axis=1)
#     df_slice['Cell_Voltage_Std'] = df_slice[cell_cols].std(axis=1)
    
#     features['start_voltage'] = df_slice['Mean_Cell_Voltage'].iloc[:10].mean()
#     features['end_voltage'] = df_slice['Mean_Cell_Voltage'].iloc[-10:].mean()
#     features['voltage_drop'] = features['start_voltage'] - features['end_voltage']
    
#     # Imbalance features
#     features['mean_cell_imbalance_std'] = df_slice['Cell_Voltage_Std'].mean()
#     features['max_cell_spread'] = (df_slice['Max_Cell_Voltage'] - df_slice['Min_Cell_Voltage']).mean()
#     features['max_cell_spread_at_end'] = (df_slice['Max_Cell_Voltage'].iloc[-10:] - df_slice['Min_Cell_Voltage'].iloc[-10:]).mean()
#     features['min_cell_voltage_at_end'] = df_slice['Min_Cell_Voltage'].iloc[-10:].mean()
    
#     # Temperature features
#     temp_cols = [c for c in df_slice.columns if 'Temperature' in c]
#     if temp_cols:
#         df_slice['Mean_Temp'] = df_slice[temp_cols].mean(axis=1)
#         features['mean_temp'] = df_slice['Mean_Temp'].mean()
#         features['temp_rise'] = df_slice['Mean_Temp'].iloc[-1] - df_slice['Mean_Temp'].iloc[0]
        
#     # Capacity and Resistance proxy features
#     if 'AHDischarge' in df_slice.columns:
#         features['delta_Ah'] = df_slice['AHDischarge'].iloc[-1] - df_slice['AHDischarge'].iloc[0]
#         features['ah_per_voltage_drop'] = features['delta_Ah'] / (features['voltage_drop'] + 1e-5)
        
#         # dV/dAh (Slope of the curve - highly indicative of SOH)
#         dV = df_slice['Mean_Cell_Voltage'].diff().dropna()
#         dAh = df_slice['AHDischarge'].diff().dropna()
#         dV_dAh = dV / (dAh.replace(0, 1e-5)) 
#         features['mean_dV_dAh'] = dV_dAh.mean()
#         features['std_dV_dAh'] = dV_dAh.std()
#         features['min_dV_dAh'] = dV_dAh.min()
        
#     return features

# def build_dataset(data_folder, target_c_rate=None):
#     """
#     Builds dataset with correct C-rate specific SOH labels.
#     If target_c_rate is provided, filters for it (e.g., 0.3 or 1.0).
#     """
#     real_data = []
    
#     for file in os.listdir(data_folder):
#         if not file.endswith('.csv'): continue
        
#         pack_match = re.search(r'(pk\d+)', file)
#         pack_id = pack_match.group(1) if pack_match else 'unknown'
#         if pack_id not in SOH_GROUND_TRUTH: continue
        
#         c_rate_match = re.search(r'(\d+\.\d+)C', file)
#         if not c_rate_match: continue
#         c_rate = float(c_rate_match.group(1))
        
#         # Filter by C-rate if requested
#         if target_c_rate is not None:
#             if target_c_rate == 0.3:
#                 if abs(c_rate - 0.3) > 0.05: continue
#             elif target_c_rate == 1.0:
#                 if c_rate < 0.9: continue  # Include 0.95C and 1.0C
        
#         # Get the CORRECT SOH for this specific pack and C-rate combination
#         true_soh = get_soh_for_pack_and_crate(pack_id, c_rate)
#         if true_soh is None:
#             print(f"Warning: No SOH found for {pack_id} at {c_rate}C")
#             continue
            
#         is_sfct = 1.0 if 'SFCT' in file.upper() or 'SFT' in file.upper() else 0.0
        
#         df = pd.read_csv(os.path.join(data_folder, file))
#         total_rows = len(df)
        
#         # Extract features from different partial discharge starting points
#         for pct in [0.40, 0.50, 0.60, 0.70]:
#             start_row = int(total_rows * pct)
#             feats = extract_features_from_slice(df, start_row)
#             if feats:
#                 feats['c_rate'] = c_rate
#                 feats['is_sfct'] = is_sfct
#                 feats['true_soh'] = true_soh
#                 feats['pack_id'] = pack_id
#                 feats['slice_start_pct'] = pct
#                 real_data.append(feats)
                
#     return pd.DataFrame(real_data)

# def train_separate_models(data_folder):
#     print("Building Physics-Augmented Dataset with C-rate specific SOH labels...")
#     full_df = build_dataset(data_folder)
    
#     if full_df.empty:
#         print("No data found! Check file paths and naming conventions.")
#         return None

#     print(f"Total samples loaded: {len(full_df)}")
#     print(f"Pack distribution:\n{full_df['pack_id'].value_counts()}")
#     print(f"\nC-rate distribution:\n{full_df['c_rate'].value_counts()}")
    
#     feature_cols = [c for c in full_df.columns if c not in ['true_soh', 'pack_id']]
    
#     # --- TRAIN SEPARATE MODELS FOR 0.3C AND 1.0C ---
#     models = {}
    
#     for target_c in [0.3, 1.0]:
#         print(f"\n{'='*60}")
#         print(f"Training Model for C-Rate: {target_c}C")
#         print(f"{'='*60}")
        
#         # Filter data for this C-rate
#         if target_c == 0.3:
#             df_c = full_df[abs(full_df['c_rate'] - 0.3) < 0.05].copy()
#         else:  # 1.0C
#             df_c = full_df[full_df['c_rate'] >= 0.9].copy()
        
#         if len(df_c) == 0:
#             print(f"No data available for {target_c}C training!")
#             continue
            
#         # Add C-rate normalized features
#         if 'voltage_drop' in df_c.columns:
#             df_c['voltage_drop_norm'] = df_c['voltage_drop'] / (target_c + 1e-5)
#         if 'delta_Ah' in df_c.columns:
#             df_c['delta_Ah_norm'] = df_c['delta_Ah'] / (target_c + 1e-5)
            
#         X = df_c[feature_cols + ['voltage_drop_norm', 'delta_Ah_norm']]
#         y = df_c['true_soh'].values
#         groups = df_c['pack_id'].values
        
#         print(f"Samples for {target_c}C: {len(X)}")
#         print(f"SOH range: {y.min():.2f}% - {y.max():.2f}%")
#         print(f"Features ({len(X.columns)}): {list(X.columns)}\n")
        
#         # Use Leave-One-Pack-Out CV
#         logo = LeaveOneGroupOut()
#         logo_scores = []
        
#         print("Running Leave-One-Pack-Out Cross-Validation...")
#         for train_idx, test_idx in logo.split(X, y, groups):
#             model = xgb.XGBRegressor(
#                 n_estimators=300, max_depth=3, learning_rate=0.05, 
#                 objective='reg:squarederror', random_state=42, 
#                 reg_alpha=0.1, reg_lambda=1.0
#             )
#             model.fit(X.iloc[train_idx], y[train_idx])
#             pred = model.predict(X.iloc[test_idx])
#             mae = mean_absolute_error(y[test_idx], pred)
#             logo_scores.append(mae)
#             print(f"  Left out {groups[test_idx][0]} -> MAE: {mae:.2f}%")
            
#         if logo_scores:
#             avg_mae = np.mean(logo_scores)
#             print(f"\n>>> AVERAGE LOPO SOH ERROR for {target_c}C: {avg_mae:.2f}% <<<\n")
            
#             # Train final model on all data for this C-rate
#             final_model = xgb.XGBRegressor(
#                 n_estimators=300, max_depth=3, learning_rate=0.05, 
#                 objective='reg:squarederror', random_state=42, 
#                 reg_alpha=0.1, reg_lambda=1.0
#             )
#             final_model.fit(X, y)
#             models[target_c] = (final_model, list(X.columns))
#         else:
#             print(f"No validation scores for {target_c}C\n")
        
#     return models

# def predict_soh(model_path, feature_names_path, input_features):
#     """
#     Load a trained model and make predictions.
#     input_features: dict with feature names and values
#     """
#     model = joblib.load(model_path)
#     feature_names = joblib.load(feature_names_path)
    
#     # Create feature vector in correct order
#     X = np.array([[input_features.get(f, 0) for f in feature_names]])
    
#     prediction = model.predict(X)[0]
#     return prediction

# if __name__ == "__main__":
#     DATA_FOLDER = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/raw_dataset" 
    
#     if os.path.exists(DATA_FOLDER):
#         trained_models = train_separate_models(DATA_FOLDER)
        
#         if trained_models:
#             # Save models
#             for c_rate, (model, feats) in trained_models.items():
#                 c_str = str(c_rate).replace('.', '_')
#                 joblib.dump(model, f'soh_model_{c_str}c.pkl')
#                 joblib.dump(feats, f'feature_names_{c_str}c.pkl')
#                 print(f":white_check_mark: Model for {c_rate}C saved successfully!")
#         else:
#             print(" No models were trained!")
#     else:
#         print(":x: Data folder not found.")


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