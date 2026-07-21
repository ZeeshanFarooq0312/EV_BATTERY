
import os
import sys
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
sys.path.insert(0, TEST_CASE_DIR)

from curve_utils import (
    NOMINAL_CAPACITY, HEAD_CHECKPOINTS_PCT, SFT_SAMPLE_FRACTIONS,
    compute_soh, extract_and_resample_curve, extract_enhanced_features,
    detect_knee, get_cell_columns, module_cell_columns, N_MODULES,
    extract_module_features_from_slice, add_sibling_features,
)
from module_capacity_extrapolation import find_crossing_index, estimate_module_capacity, _interp_crossing_ah

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
        soh_models[c_rate] = joblib.load(os.path.join(TEST_CASE_DIR, f'soh_model_{c_str}c.pkl'))
        soh_feature_names[c_rate] = joblib.load(os.path.join(TEST_CASE_DIR, f'feature_names_{c_str}c.pkl'))
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

# 3. Load per-module SOH models
module_soh_models = {}
module_soh_feature_names = {}
for c_rate in [0.3, 1.0]:
    c_str = str(c_rate).replace('.', '_')
    try:
        module_soh_models[c_rate] = joblib.load(os.path.join(TEST_CASE_DIR, f'module_soh_model_{c_str}c.pkl'))
        module_soh_feature_names[c_rate] = joblib.load(os.path.join(TEST_CASE_DIR, f'module_feature_names_{c_str}c.pkl'))
        print(f"✅ Module SOH model for {c_rate}C loaded")
    except Exception as e:
        print(f"⚠️ Module SOH model for {c_rate}C not found: {e}")

print("✅ Startup complete!\n")

# ==========================================
# CORE LOGIC FUNCTIONS
# ==========================================
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

def reconstruct_full_curve(sft_df, pred_capacity, c_rate=0.3, actual_capacity=None):
    """The v10 model predicts all checkpoints across the COMPLETE global curve
    (0-100% of the curve's total capacity) directly from the observed SFT tail
    segment -- no splicing/blending needed. Checkpoints and SFT sampling are
    both non-uniform (dense near 72-100%) to better resolve the discharge knee.

    `pred_capacity` is Part 1's own SOH-model output and is always reported as
    its own metric. `actual_capacity` (the real FFT ground truth, when a
    comparison file is uploaded) is used instead to anchor the checkpoint Ah
    positions and the estimated-SOH feature, matching exactly what the model
    saw during training (real curves, real capacity) and keeping a Part-1 SOH
    miss from stretching/shifting the whole reconstructed curve on the plot.
    Falls back to pred_capacity when no ground truth is available."""
    anchor_capacity = actual_capacity if actual_capacity is not None else pred_capacity

    sft_ah, sft_v, sft_cell_std = extract_and_resample_curve(sft_df)
    sft_delta_ah = sft_ah[-1]
    cutoff_ah = anchor_capacity - sft_delta_ah

    estimated_soh = compute_soh(anchor_capacity)
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
    recon_ah = np.array([pct * anchor_capacity for pct in HEAD_CHECKPOINTS_PCT])

    dense_ah = np.linspace(recon_ah.min(), recon_ah.max(), 300)
    interp_func = interp1d(recon_ah, pred_v, kind='cubic', fill_value='extrapolate')
    dense_v = np.clip(interp_func(dense_ah), 1.8, 4.3)

    return dense_ah, dense_v, cutoff_ah


def analyze_modules(sft_df, fft_df, c_rate):
    """Per-module SOH: predicted from the uploaded (partial) SFT file via the
    module SOH model, and actual/ground-truth from the uploaded (full) FFT
    file via the same tiered capacity extrapolation used to build the
    training labels (see build_module_dataset.py) -- since the FFT file is a
    full curve, most modules resolve to a real measured or self-extrapolated
    value, only rarely needing the cross-module shape-transfer fallback.
    Returns (result_dict, error_message) -- exactly one is None."""
    c_bucket = 1.0 if c_rate >= 0.9 else 0.3
    if c_bucket not in module_soh_models:
        return None, f"No module SOH model available for {c_bucket}C"

    sft_cell_cols = get_cell_columns(sft_df)
    pred_feats_by_module = {}
    for m in range(1, N_MODULES + 1):
        f = extract_module_features_from_slice(sft_df, module_cell_columns(sft_cell_cols, m))
        if f is not None:
            pred_feats_by_module[m] = f
    if len(pred_feats_by_module) < N_MODULES:
        return None, "SFT file too short to extract all 9 modules' features"

    add_sibling_features(pred_feats_by_module)
    model = module_soh_models[c_bucket]
    feature_names = module_soh_feature_names[c_bucket]

    predicted = {}
    for m, f in pred_feats_by_module.items():
        row = dict(f)
        row['c_rate'] = c_rate
        row['is_sfct'] = 1.0
        row['slice_start_pct'] = 0.0
        row['voltage_drop_norm'] = row['voltage_drop'] / (c_rate + 1e-5)
        row['delta_Ah_norm'] = row['delta_Ah'] / (c_rate + 1e-5)
        X = pd.DataFrame([row]).reindex(columns=feature_names, fill_value=0)
        predicted[m] = float(model.predict(X)[0])

    fft_cell_cols = get_cell_columns(fft_df)
    ah, pack_v, cell_std, modules = extract_and_resample_curve(fft_df, want_modules=True)
    cutoff_v = float(fft_df[fft_cell_cols].min(axis=1).iloc[-1])

    tier0 = {}
    for m, traces in modules.items():
        idx = find_crossing_index(ah, traces['min_v'], cutoff_v)
        if idx is not None:
            tier0[m] = _interp_crossing_ah(ah, traces['min_v'], idx, cutoff_v)

    actual, label_source = {}, {}
    if tier0:
        weakest_module = min(tier0, key=tier0.get)
        template_v = modules[weakest_module]['min_v']
        for m, traces in modules.items():
            if m in tier0:
                capacity_ah, source = tier0[m], 'measured'
            else:
                capacity_ah, source, _diag = estimate_module_capacity(
                    ah, traces['min_v'], cutoff_v, template_ah=ah, template_v=template_v,
                )
            actual[m] = compute_soh(capacity_ah)
            label_source[m] = source

    modules_result = [{
        'module_idx': m,
        'predicted_soh': round(predicted[m], 2) if m in predicted else None,
        'actual_soh': round(actual[m], 2) if m in actual else None,
        'label_source': label_source.get(m),
    } for m in range(1, N_MODULES + 1)]

    return {
        'modules': modules_result,
        'weakest_module_predicted': min(predicted, key=predicted.get) if predicted else None,
        'weakest_module_actual': min(actual, key=actual.get) if actual else None,
    }, None


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

        print("Analyzing per-module SOH...")
        module_result, module_error = None, None
        try:
            module_result, module_error = analyze_modules(sft_df, fft_df, c_rate)
            if module_error:
                print(f"⚠️ Module analysis skipped: {module_error}")
            else:
                print(f"   Weakest module (predicted): {module_result['weakest_module_predicted']}  "
                      f"(actual): {module_result['weakest_module_actual']}")
        except Exception as e:
            module_error = str(e)
            print(f"⚠️ Module analysis failed: {module_error}")
            print(traceback.format_exc())

        # Headline "Predicted SOH" is derived from the per-module model: in a
        # series string, pack capacity is set by its weakest module, so the
        # weakest module's own predicted SOH IS the pack's predicted SOH --
        # not an aggregate/average. This replaces the old pack-level
        # soh_model_*.pkl (trained by SOH_MODELS_TRAIN.PY against a hardcoded
        # SOH_GROUND_TRUTH table covering only pk1-pk5) as the primary
        # source, since that table's blind spot for any other pack (e.g.
        # pk6) made it extrapolate badly -- a real ~10-point miss observed in
        # production. The legacy pack-level model is kept ONLY as a fallback
        # for when module analysis itself isn't available (e.g. the uploaded
        # SFT file is too short to extract all 9 modules' features).
        if module_result is not None:
            module_predicted_sohs = {m['module_idx']: m['predicted_soh'] for m in module_result['modules']}
            pred_soh = min(module_predicted_sohs.values())
            pred_soh_source = 'module_derived'
        else:
            print("   Module analysis unavailable -- falling back to legacy pack-level model")
            soh_feats = extract_soh_features(sft_df, c_rate)
            if c_rate not in soh_models:
                return jsonify({'error': f'No SOH model available for {c_rate}C'}), 400
            legacy_model = soh_models[c_rate]
            legacy_feats = soh_feature_names[c_rate]
            X_soh = pd.DataFrame([soh_feats]).reindex(columns=legacy_feats, fill_value=0)
            pred_soh = float(legacy_model.predict(X_soh)[0])
            pred_soh_source = 'legacy_pack_model'

        # CRITICAL FIX: Convert SOH to whole number for clean capacity calculation
        soh_whole_number = int(pred_soh)
        pred_capacity = (soh_whole_number / 100.0) * NOMINAL_CAPACITY
        print(f"Predicted SOH: {pred_soh:.2f}%, Capacity: {pred_capacity:.2f} Ah  (source={pred_soh_source})")

        print("Extracting FFT ground truth...")
        fft_ah, fft_v, _ = extract_and_resample_curve(fft_df)
        actual_capacity = float(fft_ah[-1])
        actual_soh = compute_soh(actual_capacity)
        print(f"Found {len(fft_ah)} FFT points. Actual Capacity: {actual_capacity:.2f} Ah, Actual SOH: {actual_soh:.2f}%")

        print("Reconstructing complete global curve from SFT...")
        recon_ah, recon_v, cutoff_ah = reconstruct_full_curve(sft_df, pred_capacity, c_rate, actual_capacity=actual_capacity)
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

        # if actual_knee_ah is not None:
        #     plt.scatter([actual_knee_ah], [actual_knee_v], color='#2563eb', s=220, zorder=6,
        #                label=f'Actual Knee ({actual_knee_ah:.2f} Ah)', marker='*', edgecolors='white', linewidths=1.5)
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
            'soh_source': pred_soh_source,
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
            'modules': module_result['modules'] if module_result else None,
            'weakest_module_predicted': module_result['weakest_module_predicted'] if module_result else None,
            'weakest_module_actual': module_result['weakest_module_actual'] if module_result else None,
            'module_error': module_error,
            'plot': plot_url
        })
        
    except Exception as e:
        print(f"❌ Error in analyze endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)


