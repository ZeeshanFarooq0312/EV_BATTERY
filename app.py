
import os
import io
import base64
import joblib
import numpy as np
import pandas as pd
import re
import warnings
import traceback

# CRITICAL FIX: Set matplotlib backend to 'Agg' BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==========================================
# LOAD MODELS AT STARTUP
# ==========================================
print("Loading models...")

# 1. Load SOH Models
soh_models = {}
soh_feature_names = {}
for c_rate in [0.3, 1.0]:
    c_str = str(c_rate).replace('.', '_')
    try:
        soh_models[c_rate] = joblib.load(f'soh_model_{c_str}c.pkl')
        soh_feature_names[c_rate] = joblib.load(f'feature_names_{c_str}c.pkl')
        print(f"✅ SOH model for {c_rate}C loaded")
    except Exception as e:
        print(f"❌ Error loading SOH model for {c_rate}C: {e}")

# 2. Load Reconstruction Models
try:
    recon_model_0_3c = joblib.load('/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/TEST_CASE/reconstruction_model_0_3C_v7_41chk.pkl')
    print("✅ 0.3C reconstruction model loaded")
except Exception as e:
    print(f"⚠️ 0.3C reconstruction model not found: {e}")
    recon_model_0_3c = None

try:
    recon_model_1_0c = joblib.load('/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/TEST_CASE/reconstruction_model_1_0C_v7_41chk.pkl')
    print("✅ 1.0C reconstruction model loaded")
except Exception as e:
    print(f"⚠️ 1.0C reconstruction model not found: {e}")
    recon_model_1_0c = None

print("✅ Startup complete!\n")

# ==========================================
# CONSTANTS
# ==========================================
NOMINAL_CAPACITY = 156.0
SFT_CHECKPOINTS_COUNT = 40
HEAD_CHECKPOINTS_PCT = [round(i * 0.025, 3) for i in range(41)]

# ==========================================
# CORE LOGIC FUNCTIONS
# ==========================================
def get_actual_capacity_from_fft(df):
    if 'AHDischarge' in df.columns:
        return float(df['AHDischarge'].max())
    return 0.0

def extract_soh_features(df, c_rate):
    features = {}
    cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
    df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)
    df['Min_Cell_Voltage'] = df[cell_cols].min(axis=1)
    df['Max_Cell_Voltage'] = df[cell_cols].max(axis=1)
    df['Cell_Voltage_Std'] = df[cell_cols].std(axis=1)
    
    features['start_voltage'] = float(df['Mean_Cell_Voltage'].iloc[:10].mean())
    features['end_voltage'] = float(df['Mean_Cell_Voltage'].iloc[-10:].mean())
    features['voltage_drop'] = float(features['start_voltage'] - features['end_voltage'])
    features['mean_cell_imbalance_std'] = float(df['Cell_Voltage_Std'].mean())
    features['max_cell_spread'] = float((df['Max_Cell_Voltage'] - df['Min_Cell_Voltage']).mean())
    features['max_cell_spread_at_end'] = float((df['Max_Cell_Voltage'].iloc[-10:] - df['Min_Cell_Voltage'].iloc[-10:]).mean())
    features['min_cell_voltage_at_end'] = float(df['Min_Cell_Voltage'].iloc[-10:].mean())
    
    temp_cols = [c for c in df.columns if 'Temperature' in c]
    if temp_cols:
        df['Mean_Temp'] = df[temp_cols].mean(axis=1)
        features['mean_temp'] = float(df['Mean_Temp'].mean())
        features['temp_rise'] = float(df['Mean_Temp'].iloc[-1] - df['Mean_Temp'].iloc[0])
    else:
        features['mean_temp'] = 25.0
        features['temp_rise'] = 0.0
        
    if 'AHDischarge' in df.columns:
        features['delta_Ah'] = float(df['AHDischarge'].iloc[-1] - df['AHDischarge'].iloc[0])
        features['ah_per_voltage_drop'] = float(features['delta_Ah'] / (features['voltage_drop'] + 1e-5))
        
        dV = df['Mean_Cell_Voltage'].diff().dropna()
        dAh = df['AHDischarge'].diff().dropna()
        dV_dAh = dV / (dAh.replace(0, 1e-5)) 
        features['mean_dV_dAh'] = float(dV_dAh.mean())
        features['std_dV_dAh'] = float(dV_dAh.std())
        features['min_dV_dAh'] = float(dV_dAh.min())
    else:
        features['delta_Ah'] = 0.0
        features['ah_per_voltage_drop'] = 0.0
        features['mean_dV_dAh'] = 0.0
        features['std_dV_dAh'] = 0.0
        features['min_dV_dAh'] = 0.0

    features['voltage_drop_norm'] = float(features['voltage_drop'] / (c_rate + 1e-5))
    features['delta_Ah_norm'] = float(features['delta_Ah'] / (c_rate + 1e-5))
    
    features['c_rate'] = float(c_rate)
    features['is_sfct'] = 1.0
    features['slice_start_pct'] = 0.0 
    
    return features

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
    features['voltage_std'] = float(np.std(sft_v))
    features['voltage_range'] = float(sft_v[0] - sft_v[-1])
    
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

def extract_true_fft(df):
    cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
    df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)
    discharge_df = df[(df['Mean_Cell_Voltage'] < 4.15) & (df['Mean_Cell_Voltage'] > 2.0)].copy()
    if 'AHDischarge' in discharge_df.columns and discharge_df['AHDischarge'].max() > 0:
        discharge_df = discharge_df.sort_values('AHDischarge')
        discharge_df['Ah_Relative'] = discharge_df['AHDischarge'] - discharge_df['AHDischarge'].iloc[0]
    else:
        discharge_df['Ah_Relative'] = np.arange(len(discharge_df))
    return discharge_df['Ah_Relative'].values, discharge_df['Mean_Cell_Voltage'].values

def reconstruct_curve(sft_df, pred_capacity, c_rate=0.3, target_end_voltage=2.5):
    """Reconstruct full discharge curve with smooth splicing using low-pass filtering."""
    cell_cols = [c for c in sft_df.columns if 'Cell' in c and 'Temperature' not in c]
    sft_df['Mean_Cell_Voltage'] = sft_df[cell_cols].mean(axis=1)
    
    sft_v = sft_df['Mean_Cell_Voltage'].values
    sft_ah = sft_df['AHDischarge'].values
    sft_start_v = sft_v[0]
    
    sft_delta_ah = sft_ah[-1] - sft_ah[0]
    target_sft_start_ah = pred_capacity - sft_delta_ah
    
    sft_sampled_v = [float(x) for x in np.linspace(sft_v[0], sft_v[-1], SFT_CHECKPOINTS_COUNT)]
    
    dv_dah = np.gradient(sft_v, sft_ah)
    d2v_dah2 = np.gradient(dv_dah, sft_ah)
    mid_idx = len(sft_v) // 2
    
    cell_std_data = sft_df[cell_cols].std(axis=1).values if len(cell_cols) > 0 else None
    imb_mean = float(np.mean(cell_std_data)) if cell_std_data is not None else 0.0
    imb_std = float(np.std(cell_std_data)) if cell_std_data is not None else 0.0
    
    estimated_soh = (pred_capacity / NOMINAL_CAPACITY) * 100.0
    
    features = sft_sampled_v + [
        estimated_soh,
        float((sft_v[mid_idx] - sft_v[0]) / (sft_ah[mid_idx] - sft_ah[0] + 1e-5)),
        float((sft_v[-1] - sft_v[mid_idx]) / (sft_ah[-1] - sft_ah[mid_idx] + 1e-5)),
        float((sft_v[-1] - sft_v[0]) / (sft_ah[-1] - sft_ah[0] + 1e-5)),
        float(np.mean(np.abs(d2v_dah2))),
        float(np.max(np.abs(d2v_dah2))),
        float(np.std(sft_v)),
        float(sft_v[0] - sft_v[-1]),
        float(np.sum((sft_v > 3.4) & (sft_v < 3.7)) / len(sft_v)),
        float(sft_v[int(len(sft_v) * 0.8)] - sft_v[-1]),
        float((sft_v[-1] - sft_v[int(len(sft_v) * 0.8)]) / (sft_ah[-1] - sft_ah[int(len(sft_ah) * 0.8)] + 1e-5)),
        imb_mean,
        imb_std,
        float(target_sft_start_ah)
    ]
    
    if abs(c_rate - 1.0) < 0.1 and recon_model_1_0c is not None:
        model = recon_model_1_0c
    elif recon_model_0_3c is not None:
        model = recon_model_0_3c
    else:
        raise ValueError("No reconstruction model available")
    
    X_input = np.array([features])
    predicted_head_v = model.predict(X_input)[0]
    
    head_ah_valid = [pct * target_sft_start_ah for pct in HEAD_CHECKPOINTS_PCT]
    head_v_valid = list(predicted_head_v)
    
    valid_mask = [(v > 2.0) and not np.isnan(v) for v in head_v_valid]
    head_ah_valid = [ah for ah, valid in zip(head_ah_valid, valid_mask) if valid]
    head_v_valid = [v for v, valid in zip(head_v_valid, valid_mask) if valid]
    
    head_ah_dense = np.linspace(0, target_sft_start_ah, 200)
    interp_func = interp1d(head_ah_valid, head_v_valid, kind='cubic', fill_value='extrapolate')
    head_v_dense = interp_func(head_ah_dense)
    
    for i in range(1, len(head_v_dense)):
        if head_v_dense[i] > head_v_dense[i-1]:
            head_v_dense[i] = head_v_dense[i-1] - 0.0005
    
    rebased_sft_ah = np.linspace(target_sft_start_ah, pred_capacity, len(sft_ah))
    
    full_ah = np.concatenate([head_ah_dense, rebased_sft_ah])
    full_v = np.concatenate([head_v_dense, sft_v])
    
    # ==========================================
    # SMOOTH BLENDING AT SPLICE POINT
    # ==========================================
    n_head = len(head_v_dense)
    n_sft = len(sft_v)
    
    transition_size = min(20, n_head - 1, n_sft - 1)
    splice_idx = n_head - 1
    transition_start = max(0, splice_idx - transition_size)
    transition_end = min(len(full_v), splice_idx + transition_size + 1)
    
    head_segment = full_v[transition_start:splice_idx+1].copy()
    sft_segment = full_v[splice_idx:transition_end].copy()
    
    blend_length = min(len(head_segment), len(sft_segment))
    head_segment = head_segment[:blend_length]
    sft_segment = sft_segment[:blend_length]
    
    x = np.linspace(-6, 6, blend_length)
    weights = 1 / (1 + np.exp(-x))
    
    blended_segment = (1 - weights) * head_segment + weights * sft_segment
    full_v[transition_start:transition_start+blend_length] = blended_segment
    
    # Apply Savitzky-Golay filter for smoothness
    window_length = min(101, len(full_v) // 3)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length >= 5:
        full_v = savgol_filter(full_v, window_length, polyorder=3)
    
    # Apply Gaussian smoothing at splice region only
    splice_region_start = max(0, splice_idx - 60)
    splice_region_end = min(len(full_v), splice_idx + 60)
    
    splice_region = full_v[splice_region_start:splice_region_end].copy()
    smoothed_region = gaussian_filter1d(splice_region, sigma=2.0)
    
    full_v[splice_region_start:splice_region_end] = 0.5 * splice_region + 0.5 * smoothed_region
    
    # ==========================================
    # EXTENSION LOGIC
    # ==========================================
    extended_mask = np.zeros(len(full_v), dtype=bool)
    
    if full_v[-1] > target_end_voltage:
        print(f"⚠️  Curve ends at {full_v[-1]:.2f}V, extending to {target_end_voltage}V...")
        
        n_tail_samples = min(20, len(full_v) - 1)
        tail_ah_sample = full_ah[-n_tail_samples:]
        tail_v_sample = full_v[-n_tail_samples:]
        
        if len(tail_ah_sample) >= 5:
            coeffs = np.polyfit(tail_ah_sample, tail_v_sample, 2)
            poly_func = np.poly1d(coeffs)
            
            a, b, c = coeffs[0], coeffs[1], coeffs[2] - target_end_voltage
            discriminant = b**2 - 4*a*c
            
            if discriminant >= 0 and a != 0:
                ah_at_cutoff_1 = (-b + np.sqrt(discriminant)) / (2*a)
                ah_at_cutoff_2 = (-b - np.sqrt(discriminant)) / (2*a)
                ah_at_cutoff = max(ah_at_cutoff_1, ah_at_cutoff_2)
                
                if ah_at_cutoff > full_ah[-1]:
                    n_extend = 50
                    extend_ah = np.linspace(full_ah[-1], ah_at_cutoff, n_extend)[1:]
                    extend_v = poly_func(extend_ah)
                    extend_v = np.maximum(extend_v, target_end_voltage)
                    
                    extended_mask = np.zeros(len(full_ah) + len(extend_ah), dtype=bool)
                    extended_mask[len(full_ah):] = True
                    
                    full_ah = np.concatenate([full_ah, extend_ah])
                    full_v = np.concatenate([full_v, extend_v])
                    
                    print(f"✅ Extended curve to {target_end_voltage}V (at {ah_at_cutoff:.2f} Ah)")
    
    return full_ah, full_v, target_sft_start_ah, extended_mask

# ==========================================
# FLASK ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'sft_file' not in request.files or 'fft_file' not in request.files:
            return jsonify({'error': 'Please upload both SFT and FFT files.'}), 400

        sft_file = request.files['sft_file']
        fft_file = request.files['fft_file']
        voltage_cutoff = float(request.form.get('voltage_cutoff', 3.2))

        if sft_file.filename == '' or fft_file.filename == '':
            return jsonify({'error': 'No selected file.'}), 400

        print(f"Reading SFT file: {sft_file.filename}")
        sft_df = pd.read_csv(sft_file)
        print(f"Reading FFT file: {fft_file.filename}")
        fft_df = pd.read_csv(fft_file)

        c_rate_match = re.search(r'(\d+(?:\.\d+)?)C', sft_file.filename)
        c_rate = float(c_rate_match.group(1)) if c_rate_match else 0.3
        if abs(c_rate - 0.95) < 0.05:
            c_rate = 1.0
            
        print(f"Detected C-rate: {c_rate}")

        print("Extracting SOH features...")
        soh_feats = extract_soh_features(sft_df, c_rate)
        
        if c_rate not in soh_models:
            return jsonify({'error': f'No SOH model available for {c_rate}C'}), 400
            
        model = soh_models[c_rate]
        feats = soh_feature_names[c_rate]
        
        X_soh = pd.DataFrame([soh_feats]).reindex(columns=feats, fill_value=0)
        
        print("Predicting SOH...")
        pred_soh = float(model.predict(X_soh)[0])
        
        # CRITICAL FIX: Convert SOH to whole number for clean capacity calculation
        soh_whole_number = int(pred_soh)
        pred_capacity = (soh_whole_number / 100.0) * NOMINAL_CAPACITY
        print(f"Predicted SOH: {pred_soh:.2f}%, Capacity: {pred_capacity:.2f} Ah")

        print("Extracting actual capacity from FFT...")
        actual_capacity = get_actual_capacity_from_fft(fft_df)
        print(f"Actual Capacity (FFT): {actual_capacity:.2f} Ah")

        print(f"Reconstructing curve (extending to {voltage_cutoff}V if needed)...")
        recon_ah, recon_v, splice_ah, extended_mask = reconstruct_curve(sft_df, pred_capacity, c_rate, voltage_cutoff)
        print(f"Reconstruction complete. Splice at: {splice_ah:.2f} Ah")

        print("Extracting FFT ground truth...")
        fft_ah, fft_v = extract_true_fft(fft_df)
        print(f"Found {len(fft_ah)} FFT points")

        cutoff_idx = np.where(recon_v <= voltage_cutoff)[0]
        if len(cutoff_idx) > 0:
            cutoff_capacity = float(recon_ah[cutoff_idx[0]])
            cutoff_point_idx = cutoff_idx[0]
            print(f"✅ Found {voltage_cutoff}V cutoff at {cutoff_capacity:.2f} Ah (index {cutoff_point_idx})")
        else:
            cutoff_capacity = float(recon_ah[-1])
            cutoff_point_idx = -1
            print(f"⚠️ Curve doesn't reach {voltage_cutoff}V, using final capacity: {cutoff_capacity:.2f} Ah")

        min_ah = max(fft_ah.min(), recon_ah.min())
        max_ah = min(fft_ah.max(), recon_ah.max())
        mask_fft = (fft_ah >= min_ah) & (fft_ah <= max_ah)
        mask_recon = (recon_ah >= min_ah) & (recon_ah <= max_ah)
        
        if np.sum(mask_fft) > 0 and np.sum(mask_recon) > 0:
            interp_func = interp1d(fft_ah[mask_fft], fft_v[mask_fft], kind='linear', fill_value='extrapolate')
            v_fft_interp = interp_func(recon_ah[mask_recon])
            mae = float(np.mean(np.abs(recon_v[mask_recon] - v_fft_interp))) * 1000
        else:
            mae = 0.0
        print(f"Reconstruction MAE: {mae:.2f} mV")

        print("Generating plot...")
        plt.figure(figsize=(10, 6))
        
        plt.plot(fft_ah, fft_v, label='Real FFT (Ground Truth)', color='#2563eb', linewidth=2.5, alpha=0.9)
        
        if np.any(extended_mask):
            plt.plot(recon_ah[~extended_mask], recon_v[~extended_mask], 
                    label='ML Reconstructed from SFT', color='#dc2626', linewidth=2, linestyle='--')
            plt.plot(recon_ah[extended_mask], recon_v[extended_mask], 
                    label=f'Extended to {voltage_cutoff}V', color='#f59e0b', linewidth=2, linestyle=':')
        else:
            plt.plot(recon_ah, recon_v, label='ML Reconstructed from SFT', color='#dc2626', linewidth=2, linestyle='--')
        
        plt.axvline(x=splice_ah, color='#16a34a', linestyle=':', linewidth=2, label=f'Splice Point ({splice_ah:.2f} Ah)')
        
        if cutoff_point_idx >= 0:
            plt.scatter([recon_ah[cutoff_point_idx]], [recon_v[cutoff_point_idx]], 
                       color='#f59e0b', s=150, zorder=5, label=f'{voltage_cutoff}V Cutoff',
                       marker='o', edgecolors='white', linewidths=2)
        
        plt.title(f'Reconstruction Validation | Pred SOH: {pred_soh:.1f}% | Actual: {actual_capacity:.1f} Ah', 
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Capacity Delivered (Ah)', fontsize=12)
        plt.ylabel('Mean Cell Voltage (V)', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.ylim(1.8, 4.3)
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        print("Plot generated successfully")

        return jsonify({
            'soh': round(pred_soh, 2),
            'capacity': int(round(pred_capacity)),  # <-- FIXED: Returns clean whole number
            'actual_capacity': round(actual_capacity, 2),
            'cutoff_capacity': round(cutoff_capacity, 2),
            'mae': round(mae, 2),
            'plot': plot_url
        })
        
    except Exception as e:
        print(f"❌ Error in analyze endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)


