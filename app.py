
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

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CASE_DIR = os.path.join(APP_DIR, 'TEST_CASE')

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
        soh_models[c_rate] = joblib.load(os.path.join(APP_DIR, f'soh_model_{c_str}c.pkl'))
        soh_feature_names[c_rate] = joblib.load(os.path.join(APP_DIR, f'feature_names_{c_str}c.pkl'))
        print(f"✅ SOH model for {c_rate}C loaded")
    except Exception as e:
        print(f"❌ Error loading SOH model for {c_rate}C: {e}")

# 2. Load Reconstruction Models (v10 - complete global curve, knee-focused resolution)
try:
    recon_model_0_3c = joblib.load(os.path.join(TEST_CASE_DIR, 'reconstruction_model_0_3C_v10_knee.pkl'))
    print("✅ 0.3C reconstruction model (v10 knee) loaded")
except Exception as e:
    print(f"⚠️ 0.3C reconstruction model not found: {e}")
    recon_model_0_3c = None

try:
    recon_model_1_0c = joblib.load(os.path.join(TEST_CASE_DIR, 'reconstruction_model_1_0C_v10_knee.pkl'))
    print("✅ 1.0C reconstruction model (v10 knee) loaded")
except Exception as e:
    print(f"⚠️ 1.0C reconstruction model not found: {e}")
    recon_model_1_0c = None

print("✅ Startup complete!\n")

# ==========================================
# CONSTANTS
# ==========================================
NOMINAL_CAPACITY = 156.0
# Non-uniform, knee-dense checkpoint grid -- must match curve_train.py exactly (positional).
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

# ==========================================
# CORE LOGIC FUNCTIONS
# ==========================================
def extract_and_resample_curve(df):
    """Filters to the valid discharge window and re-zeros Ah. Shared by both the
    uploaded SFT and FFT curves so features/targets line up with training."""
    cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
    df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)
    mask = (df['Mean_Cell_Voltage'] < 4.15) & (df['Mean_Cell_Voltage'] > 2.0)
    discharge_df = df[mask].copy()
    discharge_df = discharge_df.sort_values('AHDischarge')
    discharge_df['Ah_Relative'] = discharge_df['AHDischarge'] - discharge_df['AHDischarge'].iloc[0]

    ah = discharge_df['Ah_Relative'].values
    v = discharge_df['Mean_Cell_Voltage'].values
    cell_std = df[cell_cols].std(axis=1)[mask].values if len(cell_cols) > 0 else None
    return ah, v, cell_std


def compute_soh(total_capacity_ah):
    return float((total_capacity_ah / NOMINAL_CAPACITY) * 100.0)

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

    knee_idx = int(np.argmax(np.abs(dv_dah)))
    features['tail_knee_pct'] = float(sft_ah[knee_idx] / (sft_ah[-1] + 1e-5))
    features['tail_knee_slope'] = float(dv_dah[knee_idx])

    return features

def detect_knee(ah, v, search_start_pct=0.5, search_end_pct=0.99):
    """Finds the sharpest bend (max perpendicular distance from the chord
    connecting the search window's endpoints) in the tail of a discharge
    curve -- the discharge knee. Same method used to measure real knee
    locations (0.3C/1.0C packs show it at ~83-92% of total capacity)."""
    total = ah[-1]
    mask = (ah >= search_start_pct * total) & (ah <= search_end_pct * total)
    idx = np.where(mask)[0]
    if len(idx) < 3:
        return None, None

    ah_w, v_w = ah[idx], v[idx]
    ah_n = (ah_w - ah_w[0]) / (ah_w[-1] - ah_w[0] + 1e-9)
    v_n = (v_w - v_w[0]) / (v_w[-1] - v_w[0] + 1e-9)

    x1, y1 = ah_n[0], v_n[0]
    x2, y2 = ah_n[-1], v_n[-1]
    num = np.abs((y2 - y1) * ah_n - (x2 - x1) * v_n + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-9
    dist = num / den

    knee_local_idx = int(np.argmax(dist))
    knee_idx = idx[knee_local_idx]
    return float(ah[knee_idx]), float(v[knee_idx])


def reconstruct_full_curve(sft_df, pred_capacity, c_rate=0.3):
    """The v10 model predicts all checkpoints across the COMPLETE global curve
    (0-100% of pred_capacity) directly from the observed SFT tail segment -- no
    splicing/blending needed. Checkpoints and SFT sampling are both non-uniform
    (dense near 72-100%) to better resolve the discharge knee."""
    sft_ah, sft_v, sft_cell_std = extract_and_resample_curve(sft_df)
    sft_delta_ah = sft_ah[-1]
    cutoff_ah = pred_capacity - sft_delta_ah

    estimated_soh = compute_soh(pred_capacity)
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

    if abs(c_rate - 1.0) < 0.1 and recon_model_1_0c is not None:
        model = recon_model_1_0c
    elif recon_model_0_3c is not None:
        model = recon_model_0_3c
    else:
        raise ValueError("No reconstruction model available")

    pred_v = model.predict(np.array([features]))[0]
    recon_ah = np.array([pct * pred_capacity for pct in HEAD_CHECKPOINTS_PCT])

    dense_ah = np.linspace(recon_ah.min(), recon_ah.max(), 300)
    interp_func = interp1d(recon_ah, pred_v, kind='cubic', fill_value='extrapolate')
    dense_v = np.clip(interp_func(dense_ah), 1.8, 4.3)

    return dense_ah, dense_v, cutoff_ah

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

        print("Extracting FFT ground truth...")
        fft_ah, fft_v, _ = extract_and_resample_curve(fft_df)
        actual_capacity = float(fft_ah[-1])
        actual_soh = compute_soh(actual_capacity)
        print(f"Found {len(fft_ah)} FFT points. Actual Capacity: {actual_capacity:.2f} Ah, Actual SOH: {actual_soh:.2f}%")

        print("Reconstructing complete global curve from SFT...")
        recon_ah, recon_v, cutoff_ah = reconstruct_full_curve(sft_df, pred_capacity, c_rate)
        print(f"Reconstruction complete. SFT tail starts at {cutoff_ah:.2f} Ah of the reconstructed curve.")

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

        print("Detecting discharge knee point...")
        pred_knee_ah, pred_knee_v = detect_knee(recon_ah, recon_v)
        actual_knee_ah, actual_knee_v = detect_knee(fft_ah, fft_v)
        knee_error_ah = abs(pred_knee_ah - actual_knee_ah) if (pred_knee_ah is not None and actual_knee_ah is not None) else None
        if pred_knee_ah is not None:
            print(f"   Predicted knee: {pred_knee_ah:.2f} Ah @ {pred_knee_v:.3f}V")
        if actual_knee_ah is not None:
            print(f"   Actual knee:    {actual_knee_ah:.2f} Ah @ {actual_knee_v:.3f}V")

        print("Generating plot...")
        plt.figure(figsize=(10, 6))

        plt.plot(fft_ah, fft_v, label='Real FFT (Ground Truth)', color='#2563eb', linewidth=2.5, alpha=0.9)
        plt.plot(recon_ah, recon_v, label='ML Reconstructed (Complete Global Curve)', color='#dc2626', linewidth=2, linestyle='--')
        plt.axvline(x=cutoff_ah, color='#16a34a', linestyle=':', linewidth=2, label=f'SFT Tail Start ({cutoff_ah:.2f} Ah)')

        if cutoff_point_idx >= 0:
            plt.scatter([recon_ah[cutoff_point_idx]], [recon_v[cutoff_point_idx]],
                       color='#f59e0b', s=150, zorder=5, label=f'{voltage_cutoff}V Cutoff',
                       marker='o', edgecolors='white', linewidths=2)

        if actual_knee_ah is not None:
            plt.scatter([actual_knee_ah], [actual_knee_v], color='#2563eb', s=220, zorder=6,
                       label=f'Actual Knee ({actual_knee_ah:.2f} Ah)', marker='*', edgecolors='white', linewidths=1.5)
        if pred_knee_ah is not None:
            plt.scatter([pred_knee_ah], [pred_knee_v], color='#dc2626', s=220, zorder=6,
                       label=f'Predicted Knee ({pred_knee_ah:.2f} Ah)', marker='*', edgecolors='white', linewidths=1.5)

        plt.title(f'Complete Global Curve Reconstruction | Pred SOH: {pred_soh:.1f}% | Actual SOH: {actual_soh:.1f}%',
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
            'capacity': int(round(pred_capacity)),
            'actual_capacity': round(actual_capacity, 2),
            'actual_soh': round(actual_soh, 2),
            'cutoff_capacity': round(cutoff_capacity, 2),
            'mae': round(mae, 2),
            'pred_knee_ah': round(pred_knee_ah, 2) if pred_knee_ah is not None else None,
            'pred_knee_v': round(pred_knee_v, 3) if pred_knee_v is not None else None,
            'actual_knee_ah': round(actual_knee_ah, 2) if actual_knee_ah is not None else None,
            'actual_knee_v': round(actual_knee_v, 3) if actual_knee_v is not None else None,
            'knee_error_ah': round(knee_error_ah, 2) if knee_error_ah is not None else None,
            'plot': plot_url
        })
        
    except Exception as e:
        print(f"❌ Error in analyze endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)


