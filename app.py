
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
import subprocess
import threading
import time
import uuid

# CRITICAL FIX: Set matplotlib backend to 'Agg' BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, median_filter
from sklearn.isotonic import IsotonicRegression


def _enforce_monotonic_nonincreasing(x, y):
    """Least-squares monotonic non-increasing fit to (x, y) via isotonic
    regression (pool-adjacent-violators) -- used to guarantee a reconstructed
    discharge curve never rises, since voltage cannot increase as cumulative
    Ah delivered increases. Preferred over a hard np.minimum.accumulate
    clamp: a clamp FREEZES the sequence flat at the last good value the
    instant a violation occurs (visible as an unnatural flat "shelf" right
    before a steep final cliff -- found on pk1 1.0C module 1, where one
    checkpoint's raw prediction jumped +69mV). Isotonic regression instead
    pools the violating point(s) with their neighbors and redistributes the
    correction smoothly across them, which is both a better fit to the raw
    (mostly-good) predictions and visually smoother.

    NOTE: isotonic regression on its own still leaves a visible flat
    "shelf" wherever the raw checkpoints have a sizeable local violation --
    found on pk3 1.0C module 7, where a +46mV bump around ah=40-55 forced a
    ~15-checkpoint-wide flat patch to reconcile it (each of the 57
    checkpoints is an independent regressor with no smoothness constraint
    between neighbors, so this kind of local noise is expected). Callers
    should smooth the raw predictions (see _smooth_checkpoints) BEFORE
    calling this, so isotonic has less amplitude to pool away in the first
    place."""
    return IsotonicRegression(increasing=False, out_of_bounds='clip').fit_transform(x, y)


def _smooth_checkpoints(y, sigma=1.2, tail_protect=5):
    """Damps local checkpoint-to-checkpoint noise before the isotonic step
    (see _enforce_monotonic_nonincreasing's docstring).

    Median filter FIRST, then Gaussian: found on pk6 0.3C module 8, where
    the checkpoint at ah=0 (full-charge voltage, should be the single
    HIGHEST point on the whole curve) was mispredicted 170mV LOW of the
    very next checkpoint -- a single-point outlier at the sequence's own
    boundary. Gaussian smoothing alone barely dents a single sharp outlier
    (it just spreads a fraction of it onto neighbors), and isotonic
    regression's pool-adjacent-violators then has to cascade the fix
    forward across ~30 checkpoints before the (slowly-descending) pooled
    average finally drops back below the raw curve -- visible as a long,
    unnaturally flat/near-linear stretch at the start of the curve, exactly
    the "looks like the old mean-voltage reconstruction" complaint. A
    median filter (mode='reflect', so it compares an edge point against its
    mirrored neighbors instead of padding with copies of itself) removes a
    single-point outlier outright instead of just softening it, so isotonic
    never sees a violation large enough to need a long cascade.

    tail_protect excludes the last few checkpoints from the median step:
    found on pk4 0.3C module 7, where the genuine, sharp discharge-knee
    collapse in the last ~2 checkpoints looks exactly like a single-point
    outlier to a size-5 median filter (reflect padding compares it against
    its own flatter neighbors) and gets flattened back toward the mid-curve
    trend. Combined with MODULE_END_ANCHOR_V forcing only the very last
    checkpoint back down, this produced a visible "flattens, then suddenly
    plunges" step instead of one continuous accelerating decline. The head
    (checkpoint 0) doesn't need this protection: its true signal IS
    near-constant across packs (see MODULE_START_VOLTAGE), so median
    filtering it is correct; the tail's true signal is a real acceleration,
    so median filtering it is not."""
    if tail_protect > 0 and len(y) > tail_protect:
        head = median_filter(y[:-tail_protect], size=5, mode='reflect')
        y = np.concatenate([head, y[-tail_protect:]])
    else:
        y = median_filter(y, size=5, mode='reflect')
    return gaussian_filter1d(y, sigma=sigma, mode='nearest')

warnings.filterwarnings('ignore')

app = FastAPI()
os.makedirs('uploads', exist_ok=True)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Every route that generates a plot saves it here too (as a real .png,
# alongside the base64 'plot' field already in the JSON response) --
# lets anyone testing via Postman/curl just open the file directly instead
# of decoding base64 by hand.
PLOTS_DIR = os.path.join(APP_DIR, 'generated_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_plot_image(img_bytes, prefix):
    """Writes a plot's raw PNG bytes to PLOTS_DIR under a unique filename,
    returns the absolute path. Filename includes a timestamp (for sorting/
    cleanup) and a short uuid (so two requests in the same second never
    collide)."""
    fname = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(PLOTS_DIR, fname)
    with open(path, 'wb') as f:
        f.write(img_bytes)
    return path


ML_PIPELINE_DIR = os.path.join(APP_DIR, 'ml_pipeline')
MODELS_LEGACY_DIR = os.path.join(ML_PIPELINE_DIR, 'models_legacy')  # disabled/legacy models -- see README
sys.path.insert(0, os.path.join(ML_PIPELINE_DIR, 'core'))
sys.path.insert(0, os.path.join(ML_PIPELINE_DIR, 'training'))

from model_versioning import resolve_active_models_dir

# Picks the highest-numbered ml_pipeline/models/vN/ folder automatically,
# unless ml_pipeline/models/ACTIVE_VERSION pins it to a specific one -- see
# model_versioning.py.
MODELS_DIR, MODELS_VERSION = resolve_active_models_dir()
print(f"Using models version: {MODELS_VERSION or '(unversioned models/ folder)'} ({MODELS_DIR})")

from curve_utils import (
    NOMINAL_CAPACITY, HEAD_CHECKPOINTS_PCT, SFT_SAMPLE_FRACTIONS,
    compute_soh, extract_and_resample_curve, extract_enhanced_features,
    detect_knee, get_cell_columns, module_cell_columns, N_MODULES,
    extract_module_features_from_slice, add_sibling_features, parse_pack_and_crate,
    detect_settle_index, is_full_curve_file,
)
from module_capacity_extrapolation import (
    find_crossing_index, estimate_module_capacity, estimate_module_capacity_at_targets, _interp_crossing_ah,
)
from build_module_dataset import load_module_templates, DATA_FOLDER
from raw_file_cleaner import clean_raw_file, RAW_UPLOADS_FOLDER

os.makedirs(RAW_UPLOADS_FOLDER, exist_ok=True)

# ==========================================
# LOAD MODELS AT STARTUP
# ==========================================
print("Loading models...")

# 1. Load SOH Models
# TEMPORARILY DISABLED for a test run focused on the module-level pipeline.
# This is the legacy pack-level model -- only used as a fallback in analyze()
# when module analysis itself is unavailable (see the `if module_result is
# not None` branch there); with soh_models left empty, that branch's own
# `if c_rate not in soh_models: return JSONResponse({'error': ...}, 400)` guard
# already handles it as a clean error instead of a crash. Re-enable by
# uncommenting below.
soh_models = {}
soh_feature_names = {}
# for c_rate in [0.3, 1.0]:
#     c_str = str(c_rate).replace('.', '_')
#     try:
#         soh_models[c_rate] = joblib.load(os.path.join(MODELS_LEGACY_DIR, f'soh_model_{c_str}c.pkl'))
#         soh_feature_names[c_rate] = joblib.load(os.path.join(MODELS_LEGACY_DIR, f'feature_names_{c_str}c.pkl'))
#         print(f"✅ SOH model for {c_rate}C loaded")
#     except Exception as e:
#         print(f"❌ Error loading SOH model for {c_rate}C: {e}")

# 2. Load Reconstruction Models (v10 - complete global curve, knee-focused resolution)
# TEMPORARILY DISABLED for a test run focused on the module-level pipeline
# (module_soh_model_* + module_curve_model_*). This pack-level model is used
# unconditionally in analyze() for the whole-pack reconstructed curve (feeds
# the 'mae'/knee fields) AND as the fallback head-curve source if the
# module-level reconstruction model fails -- both call sites are guarded
# (see reconstruct_full_curve() callers) so leaving these as None degrades
# gracefully instead of crashing /analyze. Re-enable by uncommenting below.
# try:
#     recon_model_0_3c = joblib.load(os.path.join(MODELS_LEGACY_DIR, 'reconstruction_model_0_3C_v10_knee.pkl'))
#     print("✅ 0.3C reconstruction model (v10 knee) loaded")
# except Exception as e:
#     print(f"⚠️ 0.3C reconstruction model not found: {e}")
#     recon_model_0_3c = None
recon_model_0_3c = None

# try:
#     recon_model_1_0c = joblib.load(os.path.join(MODELS_LEGACY_DIR, 'reconstruction_model_1_0C_v10_knee.pkl'))
#     print("✅ 1.0C reconstruction model (v10 knee) loaded")
# except Exception as e:
#     print(f"⚠️ 1.0C reconstruction model not found: {e}")
#     recon_model_1_0c = None
recon_model_1_0c = None

# 3. Load per-module SOH models
module_soh_models = {}
module_soh_feature_names = {}
for c_rate in [0.3, 1.0]:
    c_str = str(c_rate).replace('.', '_')
    try:
        module_soh_models[c_rate] = joblib.load(os.path.join(MODELS_DIR, f'module_soh_model_{c_str}c.pkl'))
        module_soh_feature_names[c_rate] = joblib.load(os.path.join(MODELS_DIR, f'module_feature_names_{c_str}c.pkl'))
        print(f"✅ Module SOH model for {c_rate}C loaded")
    except Exception as e:
        print(f"⚠️ Module SOH model for {c_rate}C not found: {e}")

# 3a2. Cross-rate module SOH model (module_soh_cross_rate_train.py) --
# predicts a module's 0.3C SOH from FEATURES EXTRACTED AT 1.0C (the
# module_soh_models[1.0]/[0.3] pair above are same-rate: features and label
# share one C-rate). Used by /analyze_cross_rate for the "upload a 1.0C SFT,
# see predicted 0.3C SOH/capacity vs an uploaded 0.3C FFT's actual value" flow.
try:
    module_soh_model_cross_rate = joblib.load(os.path.join(MODELS_DIR, 'module_soh_model_1_0c_to_0_3c.pkl'))
    module_soh_feature_names_cross_rate = joblib.load(os.path.join(MODELS_DIR, 'module_feature_names_1_0c_to_0_3c.pkl'))
    print("✅ Cross-rate module SOH model (1.0C -> 0.3C) loaded")
except Exception as e:
    module_soh_model_cross_rate = None
    print(f"⚠️ Cross-rate module SOH model not found: {e}")

# 3b. Load per-module curve-reconstruction models (module_curve_train.py) --
# predicts one module's complete voltage curve directly, replacing the old
# head+observed+tail three-piece stitch (visible seam at the joins) with one
# smooth model output for the weakest-module plot.
try:
    module_curve_model_0_3c = joblib.load(os.path.join(MODELS_DIR, 'module_curve_reconstruction_model_0_3C.pkl'))
    print("✅ 0.3C module curve-reconstruction model loaded")
except Exception as e:
    print(f"⚠️ 0.3C module curve-reconstruction model not found: {e}")
    module_curve_model_0_3c = None

try:
    module_curve_model_1_0c = joblib.load(os.path.join(MODELS_DIR, 'module_curve_reconstruction_model_1_0C.pkl'))
    print("✅ 1.0C module curve-reconstruction model loaded")
except Exception as e:
    print(f"⚠️ 1.0C module curve-reconstruction model not found: {e}")
    module_curve_model_1_0c = None

# 3c. Real weakest-module full-charge voltage is a tight, near-constant
# cluster across packs (see module_curve_train.typical_start_voltage) -- the
# curve model's own checkpoint-0 regressor sometimes undershoots this by
# 0.1-0.2V, since that single point carries almost no real signal from the
# SFT input features. Computed once at startup and used to anchor
# reconstruct_module_curve's first checkpoint directly.
try:
    from module_curve_train import typical_start_voltage
    MODULE_START_VOLTAGE = {0.3: typical_start_voltage('0.3C'), 1.0: typical_start_voltage('1.0C')}
    print(f"✅ Calibrated module start voltage: {MODULE_START_VOLTAGE}")
except Exception as e:
    MODULE_START_VOLTAGE = {}
    print(f"⚠️ Could not calibrate module start voltage: {e}")

# 3d. reconstruct_module_curve's last checkpoint (pct=1.0) is placed at
# anchor_capacity, which its caller always computes as THIS module's
# predicted capacity at the point its voltage crosses this value -- but the
# curve model's own last-checkpoint regressor was trained to predict each
# curve's absolute terminal voltage (real weakest-module curves keep going
# well below this, e.g. ~2.2-2.4V, since the pack-level test only stops once
# EVERY module including the strong ones has crossed cutoff). Left
# uncorrected, the reconstructed curve visibly flattens out short of this
# value instead of continuing its decline through the anchor point.
MODULE_END_ANCHOR_V = 2.5

# 4. Load per-(pack, C-rate bucket) weakest-module template curves, used as the
# Tier-2 shape-transfer template for the SFT-only weakest-module endpoint (no
# ground-truth FFT file to draw a same-test template from at inference time).
# Was temporarily disabled for a test run to isolate Tier 0/Tier 1 behavior
# from this Tier-2 fallback -- re-enabled so /analyze_weakest_module (and its
# plot) works again instead of unconditionally 500ing.
try:
    MODULE_TEMPLATES = load_module_templates()
    print(f"✅ Loaded {len(MODULE_TEMPLATES)} module template curves: {sorted(MODULE_TEMPLATES.keys())}")
except Exception as e:
    MODULE_TEMPLATES = {}
    print(f"⚠️ Could not load module template curves: {e}")


def pick_template_curve(pack_id, c_bucket):
    """The uploaded pack's own template if we have one; else any real template
    at the same C-rate bucket (cross-pack shape borrowing -- already the
    established Tier-2 fallback, see module_capacity_extrapolation.py); else
    whatever's loaded at all. (None, None) only if nothing loaded."""
    key = (pack_id, c_bucket)
    if key in MODULE_TEMPLATES:
        return MODULE_TEMPLATES[key]
    same_bucket = [v for (p, b), v in MODULE_TEMPLATES.items() if b == c_bucket]
    if same_bucket:
        return same_bucket[0]
    return next(iter(MODULE_TEMPLATES.values()), (None, None))


# 5. Real 0.3C weakest-module template curves for cross-rate CURVE
# reconstruction (/analyze_cross_rate's plot). Unlike the same-rate
# reconstruct_module_curve, there is no 0.3C voltage shape signal available
# at inference time at all -- only a 1.0C SFT is uploaded -- so a checkpoint
# regressor has nothing 0.3C-shaped to condition on. This borrows the same
# Tier-2 shape-transfer idea as pick_template_curve above (kept independent
# of MODULE_TEMPLATES/its test-isolation toggle, since this is a separate
# feature): each pack's own real historical 0.3C weakest-module curve SHAPE,
# rescaled to the predicted capacity. See reconstruct_cross_rate_module_curve.
try:
    _all_module_templates = load_module_templates()
    CROSS_RATE_TEMPLATES_03C = {pid: curve for (pid, bucket), curve in _all_module_templates.items() if bucket == 0.3}
    print(f"✅ Loaded {len(CROSS_RATE_TEMPLATES_03C)} real 0.3C templates for cross-rate curve reconstruction: "
          f"{sorted(CROSS_RATE_TEMPLATES_03C.keys())}")
except Exception as e:
    CROSS_RATE_TEMPLATES_03C = {}
    print(f"⚠️ Could not load cross-rate 0.3C curve templates: {e}")


def pick_cross_rate_template(pack_id):
    """The uploaded pack's own real 0.3C weakest-module curve if we have one
    on disk; else any other pack's (cross-pack shape borrowing, same
    fallback idea as pick_template_curve). (None, None) if nothing loaded."""
    if pack_id in CROSS_RATE_TEMPLATES_03C:
        return CROSS_RATE_TEMPLATES_03C[pack_id]
    return next(iter(CROSS_RATE_TEMPLATES_03C.values()), (None, None))


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


def reconstruct_module_curve(sft_mod_ah, module_v_sft, module_mean_v_sft, anchor_capacity, c_rate):
    """Module-level analog of reconstruct_full_curve -- one smooth model
    output for a single module's complete voltage curve, replacing the old
    head+observed+tail three-piece stitch. See module_curve_train.py for how
    the underlying model was trained (same architecture and feature/target
    definitions as the pack-level model above, just module-scoped; same
    build_rows_from_curve function reused for both).

    `anchor_capacity` is this module's own predicted capacity at
    module_curve_train.TARGET_CUTOFF_V (2.5V) -- the same value already
    computed for the "Pred @2.5V" card, so no separate estimate is needed."""
    sft_delta_ah = sft_mod_ah[-1]
    cutoff_ah = anchor_capacity - sft_delta_ah

    estimated_soh = compute_soh(anchor_capacity)
    sft_sampled_v = [np.interp(frac * sft_mod_ah[-1], sft_mod_ah, module_v_sft) for frac in SFT_SAMPLE_FRACTIONS]
    imbalance_proxy = module_mean_v_sft - module_v_sft
    feats = extract_enhanced_features(module_v_sft, sft_mod_ah, imbalance_proxy)

    features = list(sft_sampled_v) + [
        estimated_soh,
        feats['initial_slope'], feats['final_slope'], feats['overall_slope'],
        feats['mean_curvature'], feats['max_curvature'], feats['voltage_std'],
        feats['voltage_range'], feats['plateau_length'], feats['end_voltage_drop'],
        feats['end_slope'], feats['cell_imbalance_mean'], feats['cell_imbalance_std'],
        feats['tail_knee_pct'], feats['tail_knee_slope'],
        float(cutoff_ah)
    ]

    if abs(c_rate - 1.0) < 0.1 and module_curve_model_1_0c is not None:
        model = module_curve_model_1_0c
    elif module_curve_model_0_3c is not None:
        model = module_curve_model_0_3c
    else:
        raise ValueError("No module curve-reconstruction model available")

    pred_v = model.predict(np.array([features]))[0]
    recon_ah = np.array([pct * anchor_capacity for pct in HEAD_CHECKPOINTS_PCT])

    # Each of the 57 checkpoints is predicted by an independent per-output
    # regressor (MultiOutputRegressor), with no ordering constraint between
    # them -- in the noisy deep-knee-to-cutoff region (real training curves'
    # tail-fill occasionally kinks right at the real/extrapolated boundary,
    # see module_curve_train.py's docstring), this can produce a checkpoint
    # that's HIGHER than the one before it, which cubic interpolation then
    # renders as a visible upward "hike". A discharge voltage curve is
    # physically non-increasing in cumulative Ah by definition, so enforcing
    # it is a hard physical constraint, not a heuristic smoothing choice --
    # see _enforce_monotonic_nonincreasing for why isotonic regression is
    # used here instead of a simpler np.minimum.accumulate clamp. Smoothed
    # first (see _smooth_checkpoints) so isotonic has less local noise
    # amplitude to pool away into a flat patch in the first place.
    pred_v = _smooth_checkpoints(pred_v)
    pred_v = _enforce_monotonic_nonincreasing(recon_ah, pred_v)

    # Checkpoint 0 (Ah=0, full charge) carries almost no real signal from
    # the SFT input features -- real modules cluster tightly at 4.10-4.14V
    # there regardless of degradation (see module_curve_train's
    # typical_start_voltage), so the regressor's own prediction for this ONE
    # point is pure noise relative to a direct data-calibrated anchor. Found
    # on pk5 1.0C module 6: raw prediction was 3.91V vs the calibrated
    # ~4.13V. Applied AFTER smoothing/isotonic, not before: setting it
    # earlier gets undone by the median filter, which treats the corrected
    # point as an outlier relative to its (still-low) raw neighbors and
    # pulls it back down. Safe to set directly here -- raising the first
    # point of an already non-increasing sequence can't violate monotonicity.
    calibrated_start = MODULE_START_VOLTAGE.get(1.0 if abs(c_rate - 1.0) < 0.1 else 0.3)
    if calibrated_start is not None and calibrated_start > pred_v[0]:
        pred_v[0] = calibrated_start

    # anchor_capacity IS this module's predicted capacity at
    # MODULE_END_ANCHOR_V by definition (see caller) -- force the curve to
    # actually reach that voltage there instead of trusting the raw
    # last-checkpoint regressor (see MODULE_END_ANCHOR_V's docstring above).
    # Isotonic re-applied so the preceding checkpoints pool smoothly toward
    # the anchored endpoint instead of leaving a last-step jump/kink.
    pred_v[-1] = MODULE_END_ANCHOR_V
    pred_v = _enforce_monotonic_nonincreasing(recon_ah, pred_v)

    dense_ah = np.linspace(recon_ah.min(), recon_ah.max(), 300)
    interp_func = interp1d(recon_ah, pred_v, kind='cubic', fill_value='extrapolate')
    dense_v = np.clip(interp_func(dense_ah), 1.8, 4.3)
    dense_v = _enforce_monotonic_nonincreasing(dense_ah, dense_v)  # cubic spline can still overshoot between monotonic checkpoints

    return dense_ah, dense_v


def reconstruct_cross_rate_module_curve(pack_id, pred_capacity_03c):
    """Cross-rate analog of reconstruct_module_curve: since /analyze_cross_
    rate only ever has a 1.0C SFT at inference time (no 0.3C voltage shape
    signal at all -- that's the whole point of the feature), there's nothing
    to feed a checkpoint regressor the way the same-rate model needs. Instead
    this borrows the SAME PACK's own real historical 0.3C weakest-module
    curve SHAPE (see CROSS_RATE_TEMPLATES_03C / pick_cross_rate_template --
    the established Tier-2 shape-transfer idea already used elsewhere in this
    codebase) and rescales its Ah-axis so the endpoint lands exactly on this
    module's predicted 0.3C capacity from the cross-rate SOH model -- the
    same rescale-a-real-template technique curve_train.generate_synthetic_
    curve uses to turn one real curve into a family of different-SOH curves.
    Voltage values are left untouched (shape-preserving rescale); only the
    Ah axis stretches/compresses to hit the new endpoint."""
    template_ah, template_v = pick_cross_rate_template(pack_id)
    if template_ah is None:
        raise ValueError("No real 0.3C template curve available for cross-rate reconstruction")

    scale = pred_capacity_03c / float(template_ah[-1])
    recon_ah = template_ah * scale
    return recon_ah, template_v


def _interp_target_from_reconstructed_curve(recon_ah, recon_v, target_v):
    """Where a shape-transferred (monotonic-by-construction, real-template)
    curve crosses target_v -- plain interpolation, no tiered fallback needed
    since this curve already spans the full 0-100% range by construction
    (reconstruct_cross_rate_module_curve rescales a COMPLETE real template).
    Returns None if target_v is outside the curve's observed voltage range."""
    v, ah = np.asarray(recon_v), np.asarray(recon_ah)
    if target_v > v[0] or target_v < v[-1]:
        return None
    # v is decreasing as ah increases (real discharge curve) -- np.interp
    # needs its x-array increasing, so feed it both arrays reversed.
    return float(np.interp(target_v, v[::-1], ah[::-1]))


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

    actual, actual_capacity, label_source = {}, {}, {}
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
            actual_capacity[m] = capacity_ah
            label_source[m] = source

    modules_result = [{
        'module_idx': m,
        'predicted_soh': round(predicted[m], 2) if m in predicted else None,
        'actual_soh': round(actual[m], 2) if m in actual else None,
        'predicted_capacity': round(predicted[m] / 100.0 * NOMINAL_CAPACITY, 2) if m in predicted else None,
        'actual_capacity': round(actual_capacity[m], 2) if m in actual_capacity else None,
        'label_source': label_source.get(m),
    } for m in range(1, N_MODULES + 1)]

    return {
        'modules': modules_result,
        'weakest_module_predicted': min(predicted, key=predicted.get) if predicted else None,
        'weakest_module_actual': min(actual, key=actual.get) if actual else None,
    }, None


def analyze_modules_cross_rate(sft_df, sft_c_rate, fft_df=None, pack_id=None):
    """Cross-rate per-module SOH: predicted 0.3C SOH from features extracted
    from an uploaded 1.0C-bucket SFT file (module_soh_cross_rate_train.py's
    model), and actual/ground-truth 0.3C SOH from an uploaded 0.3C FFT file
    via the SAME tiered capacity extrapolation analyze_modules uses -- the
    only difference from analyze_modules is that the SFT and FFT here are
    deliberately different C-rates (1.0C source, 0.3C target), not the same
    rate compared against itself. Returns (result_dict, error_message) --
    exactly one is None.

    fft_df is optional -- pass None (see /analyze_cross_rate_weakest_module)
    to get predictions only, with every actual/ground-truth field coming
    back None, for the case where no 0.3C FFT was uploaded to validate
    against.

    Also computes capacity/SOH at the two FIXED voltage cutoffs (3.25V, 2.5V)
    for every module, same idea as analyze_modules's target_cutoffs -- but
    the "predicted" side can't use analyze_modules's tiered-extrapolation
    method directly: there's no 0.3C voltage trace at all at inference time
    (only a 1.0C SFT), which is the whole reason cross-rate needs shape
    transfer in the first place (see reconstruct_cross_rate_module_curve).
    So "predicted @ cutoff" here means: build that same shape-transferred
    0.3C curve for THIS module (real template shape, rescaled to this
    module's own predicted capacity), then read off where it crosses each
    target voltage. "actual @ cutoff" still uses the real tiered-
    extrapolation method against the uploaded 0.3C FFT, exactly like
    analyze_modules, since that ground truth IS a real 0.3C curve."""
    if module_soh_model_cross_rate is None:
        return None, "Cross-rate module SOH model not available"

    sft_cell_cols = get_cell_columns(sft_df)
    pred_feats_by_module = {}
    for m in range(1, N_MODULES + 1):
        f = extract_module_features_from_slice(sft_df, module_cell_columns(sft_cell_cols, m))
        if f is not None:
            pred_feats_by_module[m] = f
    if len(pred_feats_by_module) < N_MODULES:
        return None, "SFT file too short to extract all 9 modules' features"

    add_sibling_features(pred_feats_by_module)

    predicted = {}
    for m, f in pred_feats_by_module.items():
        row = dict(f)
        row['c_rate'] = sft_c_rate
        row['is_sfct'] = 1.0
        row['slice_start_pct'] = 0.0
        row['voltage_drop_norm'] = row['voltage_drop'] / (sft_c_rate + 1e-5)
        row['delta_Ah_norm'] = row['delta_Ah'] / (sft_c_rate + 1e-5)
        X = pd.DataFrame([row]).reindex(columns=module_soh_feature_names_cross_rate, fill_value=0)
        predicted[m] = float(module_soh_model_cross_rate.predict(X)[0])

    ah, modules = None, {}
    actual, actual_capacity, label_source = {}, {}, {}
    if fft_df is not None:
        fft_cell_cols = get_cell_columns(fft_df)
        ah, pack_v, cell_std, modules = extract_and_resample_curve(fft_df, want_modules=True)
        cutoff_v = float(fft_df[fft_cell_cols].min(axis=1).iloc[-1])

        tier0 = {}
        for m, traces in modules.items():
            idx = find_crossing_index(ah, traces['min_v'], cutoff_v)
            if idx is not None:
                tier0[m] = _interp_crossing_ah(ah, traces['min_v'], idx, cutoff_v)

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
                actual_capacity[m] = capacity_ah
                label_source[m] = source

    # Capacity/SOH at the two fixed voltage cutoffs (3.25V, 2.5V), per module.
    # Tier-2 fallback template for the ACTUAL side: pick_template_curve() reads
    # from MODULE_TEMPLATES, which is intentionally left empty right now (see
    # "TEMPORARILY DISABLED for a test run" above) -- that made this always
    # resolve to (None, None), so any module without a Tier-0/Tier-1 result
    # raised an uncaught error and crashed the whole request. Use the
    # already-loaded, always-populated cross-rate 0.3C templates instead.
    target_cutoffs = [3.25, 2.5]
    cr_template_ah, cr_template_v = pick_cross_rate_template(pack_id)
    targets_by_module = {}
    for m in range(1, N_MODULES + 1):
        if m not in predicted:
            continue
        module_targets = {}

        # Predicted: shape-transfer this module's own real-template curve
        # (same technique as reconstruct_cross_rate_module_curve, applied
        # per-module here) to its own predicted capacity, then read off
        # where it crosses each target voltage.
        pred_capacity_m = predicted[m] / 100.0 * NOMINAL_CAPACITY
        pred_recon_ah, pred_recon_v = None, None
        try:
            pred_recon_ah, pred_recon_v = reconstruct_cross_rate_module_curve(pack_id, pred_capacity_m)
        except Exception as e:
            print(f"⚠️ [cross-rate] Per-target curve reconstruction unavailable for module {m}: {e}")

        # Actual: real tiered extrapolation against the uploaded 0.3C FFT --
        # exactly analyze_modules's method, since this ground truth IS a
        # real 0.3C curve.
        actual_targets = {}
        if m in modules:
            try:
                actual_targets, _tail_ah, _tail_v = estimate_module_capacity_at_targets(
                    ah, modules[m]['min_v'], target_cutoffs, template_ah=cr_template_ah, template_v=cr_template_v,
                )
            except Exception as e:
                print(f"⚠️ [cross-rate] Actual-side target capacity unavailable for module {m}: {e}")

        for cv in target_cutoffs:
            pred_entry = None
            if pred_recon_v is not None:
                cap_ah = _interp_target_from_reconstructed_curve(pred_recon_ah, pred_recon_v, cv)
                if cap_ah is not None:
                    pred_entry = {'capacity_ah': round(cap_ah, 2), 'soh': round(compute_soh(cap_ah), 2)}

            actual_entry = None
            if cv in actual_targets:
                cap_ah, source, _diag = actual_targets[cv]
                actual_entry = {'capacity_ah': round(cap_ah, 2), 'soh': round(compute_soh(cap_ah), 2), 'source': source}

            module_targets[str(cv)] = {'predicted': pred_entry, 'actual': actual_entry}
        targets_by_module[m] = module_targets

    modules_result = [{
        'module_idx': m,
        'predicted_soh': round(predicted[m], 2) if m in predicted else None,
        'actual_soh': round(actual[m], 2) if m in actual else None,
        'predicted_capacity': round(predicted[m] / 100.0 * NOMINAL_CAPACITY, 2) if m in predicted else None,
        'actual_capacity': round(actual_capacity[m], 2) if m in actual_capacity else None,
        'label_source': label_source.get(m),
        'targets': targets_by_module.get(m),
    } for m in range(1, N_MODULES + 1)]

    return {
        'modules': modules_result,
        'weakest_module_predicted': min(predicted, key=predicted.get) if predicted else None,
        'weakest_module_actual': min(actual, key=actual.get) if actual else None,
    }, None



# ==========================================
# FLASK ROUTES
# ==========================================
@app.get('/')
def index():
    return FileResponse(os.path.join(APP_DIR, 'templates', 'index.html'))

@app.post('/analyze')
def analyze(sft_file: UploadFile | None = File(None), fft_file: UploadFile | None = File(None)):
    try:
        if sft_file is None or fft_file is None:
            return JSONResponse({'error': 'Please upload both SFT and FFT files.'}, status_code=400)

        if sft_file.filename == '' or fft_file.filename == '':
            return JSONResponse({'error': 'No selected file.'}, status_code=400)

        print(f"Reading SFT file: {sft_file.filename}")
        sft_df = pd.read_csv(sft_file.file)
        print(f"Reading FFT file: {fft_file.filename}")
        fft_df = pd.read_csv(fft_file.file)

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
                return JSONResponse({'error': f'No SOH model available for {c_rate}C'}, status_code=400)
            legacy_model = soh_models[c_rate]
            legacy_feats = soh_feature_names[c_rate]
            X_soh = pd.DataFrame([soh_feats]).reindex(columns=legacy_feats, fill_value=0)
            pred_soh = float(legacy_model.predict(X_soh)[0])
            pred_soh_source = 'legacy_pack_model'

        pred_capacity = (pred_soh / 100.0) * NOMINAL_CAPACITY
        print(f"Predicted SOH: {pred_soh:.2f}%, Capacity: {pred_capacity:.2f} Ah  (source={pred_soh_source})")

        print("Extracting FFT ground truth...")
        fft_ah, fft_v, _ = extract_and_resample_curve(fft_df)
        actual_capacity = float(fft_ah[-1])
        actual_soh = compute_soh(actual_capacity)
        print(f"Found {len(fft_ah)} FFT points. Actual Capacity: {actual_capacity:.2f} Ah, Actual SOH: {actual_soh:.2f}%")

        print("Reconstructing complete global curve from SFT...")
        try:
            recon_ah, recon_v, cutoff_ah = reconstruct_full_curve(sft_df, pred_capacity, c_rate, actual_capacity=actual_capacity)
            print(f"Reconstruction complete. SFT tail starts at {cutoff_ah:.2f} Ah of the reconstructed curve.")
        except Exception as e:
            print(f"⚠️ Pack-level reconstruction unavailable (disabled for this test run): {e}")
            recon_ah, recon_v, cutoff_ah = None, None, None

        # Weakest-module capacity/SOH at BOTH fixed cutoffs (3.25V, 2.5V) --
        # predicted from the SFT (same tiered extrapolation as
        # /analyze_weakest_module) and, since this route has the FFT ground
        # truth too, the actual value measured the SAME way directly from the
        # FFT's own weakest-module curve. Replaces the old single-cutoff
        # dropdown: the weakest module's own tiered estimate already computes
        # both cutoffs in one pass, so forcing the user to pick one first was
        # pure friction, not a real constraint of the method.
        weakest_module_results = None
        weakest_module_plot_url = None
        weakest_module_plot_path = None
        target_module = module_result['weakest_module_predicted'] if module_result else None
        if target_module is not None:
            pack_id, _ = parse_pack_and_crate(sft_file.filename)
            c_bucket = 1.0 if c_rate >= 0.9 else 0.3
            template_ah, template_v = pick_template_curve(pack_id, c_bucket)
            target_cutoffs = [3.25, 2.5]

            sft_mod_ah, _sft_pack_v, _sft_cell_std, sft_modules = extract_and_resample_curve(sft_df, want_modules=True)
            module_v_sft = sft_modules[target_module]['min_v']
            module_anchor_capacity = module_predicted_sohs[target_module] / 100.0 * NOMINAL_CAPACITY
            ah_offset = max(0.0, module_anchor_capacity - float(sft_mod_ah[-1]))

            pred_targets, pred_tail_ah, pred_tail_v = estimate_module_capacity_at_targets(
                sft_mod_ah, module_v_sft, target_cutoffs, template_ah=template_ah, template_v=template_v,
            )
            pred_results = {}
            for cv, (local_cap_ah, source, _diag) in pred_targets.items():
                global_cap_ah = ah_offset + local_cap_ah
                pred_results[cv] = {'capacity_ah': round(global_cap_ah, 2), 'soh': round(compute_soh(global_cap_ah), 2), 'source': source}

            fft_mod_ah, _fft_pack_v, _fft_cell_std, fft_modules = extract_and_resample_curve(fft_df, want_modules=True)
            module_v_fft = fft_modules[target_module]['min_v']
            actual_targets, _actual_tail_ah, _actual_tail_v = estimate_module_capacity_at_targets(
                fft_mod_ah, module_v_fft, target_cutoffs, template_ah=template_ah, template_v=template_v,
            )
            actual_results = {}
            for cv, (local_cap_ah, source, _diag) in actual_targets.items():
                actual_results[cv] = {'capacity_ah': round(local_cap_ah, 2), 'soh': round(compute_soh(local_cap_ah), 2), 'source': source}

            weakest_module_results = {
                str(cv): {'predicted': pred_results[cv], 'actual': actual_results[cv]}
                for cv in target_cutoffs
            }
            print(f"Weakest module {target_module}: "
                  + ", ".join(f"@{cv}V pred={pred_results[cv]['capacity_ah']}Ah/{pred_results[cv]['soh']}% "
                              f"actual={actual_results[cv]['capacity_ah']}Ah/{actual_results[cv]['soh']}%" for cv in target_cutoffs))

            # Same startup-transient trim as /analyze_weakest_module (see its
            # comments for the full rationale) -- the real observed SFT
            # segment is always overlaid on top of the reconstruction below,
            # whichever path produces it.
            settle_idx = detect_settle_index(sft_mod_ah, module_v_sft)
            ah_settled, v_settled = sft_mod_ah[settle_idx:], module_v_sft[settle_idx:]
            ah_global_settled = ah_offset + (ah_settled - sft_mod_ah[0])

            print("Generating weakest-module plot...")
            plt.figure(figsize=(10, 6))
            plt.plot(fft_mod_ah, module_v_fft, label=f'Module {target_module} (Actual, FFT)',
                     color='#64748b', linewidth=2.5, alpha=0.85)

            # Preferred: one smooth model-predicted curve for this module,
            # end-to-end (module_curve_train.py) -- replaces the old
            # head+observed+tail three-piece stitch, which showed a visible
            # seam at the segment joins. Falls back to the old stitch if the
            # new model isn't available/loaded.
            recon_drawn = False
            try:
                module_mean_v_sft = sft_modules[target_module]['mean_v']
                recon_ah, recon_v = reconstruct_module_curve(
                    sft_mod_ah, module_v_sft, module_mean_v_sft, pred_results[2.5]['capacity_ah'], c_rate,
                )
                plt.plot(recon_ah, recon_v, label=f'Module {target_module} (Reconstructed)',
                         color='#dc2626', linewidth=2, linestyle='--')
                recon_drawn = True
            except Exception as e:
                print(f"⚠️ Smooth module curve reconstruction unavailable, falling back to stitched curve: {e}")

            if not recon_drawn:
                head_ah, head_v = np.array([]), np.array([])
                try:
                    mod_recon_ah, mod_recon_v, _ = reconstruct_full_curve(sft_df, module_anchor_capacity, c_rate)
                    ah_offset_settled = ah_offset + (ah_settled[0] - sft_mod_ah[0])
                    recon_v_at_module_ah = np.interp(ah_global_settled, mod_recon_ah, mod_recon_v)
                    boundary_n = min(10, len(ah_global_settled))
                    pack_mean_offset = float(np.mean(v_settled[:boundary_n] - recon_v_at_module_ah[:boundary_n]))
                    head_mask = mod_recon_ah < ah_offset_settled
                    if head_mask.sum() >= 2:
                        head_ah = mod_recon_ah[head_mask]
                        head_v = np.clip(mod_recon_v[head_mask] + pack_mean_offset, 1.8, 4.3)
                except Exception as e:
                    print(f"⚠️ Head reconstruction unavailable: {e}")

                pred_tail_ah_global = ah_offset + pred_tail_ah
                if len(head_ah) > 0:
                    plt.plot(head_ah, head_v, label='Predicted Head (0 → SFT start, approximate)',
                             color='#16a34a', linewidth=2, linestyle=':')
                if len(pred_tail_ah_global) > 0:
                    order = np.argsort(pred_tail_ah_global)
                    plt.plot(pred_tail_ah_global[order], pred_tail_v[order], label='Extrapolated Tail (physics-based)',
                             color='#dc2626', linewidth=2, linestyle='--')

            # plt.plot(ah_global_settled, v_settled, label=f'Module {target_module} (Observed, SFT)', color='#2563eb', linewidth=2.5)

            marker_colors = {3.25: '#f59e0b', 2.5: '#a855f7'}
            for cv in target_cutoffs:
                color = marker_colors.get(cv, '#f59e0b')
                plt.scatter([pred_results[cv]['capacity_ah']], [cv], color=color, s=150, zorder=5, marker='o',
                            edgecolors='white', linewidths=2,
                            label=f"Pred @{cv}V: {pred_results[cv]['capacity_ah']}Ah, {pred_results[cv]['soh']}%")
                plt.scatter([actual_results[cv]['capacity_ah']], [cv], color=color, s=150, zorder=5, marker='^',
                            edgecolors='white', linewidths=2,
                            label=f"Actual @{cv}V: {actual_results[cv]['capacity_ah']}Ah, {actual_results[cv]['soh']}%")

            plt.title(f'Weakest Module (Module {target_module}) — Predicted vs Actual', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel('Capacity Delivered (Ah)', fontsize=12)
            plt.ylabel('Min Cell Voltage in Module (V)', fontsize=12)
            plt.legend(loc='upper right', fontsize=8)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.ylim(1.8, 4.3)
            plt.tight_layout()

            wm_img = io.BytesIO()
            plt.savefig(wm_img, format='png', dpi=100, bbox_inches='tight')
            wm_img.seek(0)
            weakest_module_plot_url = base64.b64encode(wm_img.getvalue()).decode()
            weakest_module_plot_path = save_plot_image(wm_img.getvalue(), 'analyze_weakest_module')
            plt.close()
            print(f"Weakest-module plot generated successfully -> {weakest_module_plot_path}")

        if recon_ah is None:
            mae = None
            pred_knee_ah = pred_knee_v = None
        else:
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

        return {
            'soh': round(pred_soh, 2),
            'soh_source': pred_soh_source,
            'capacity': round(pred_capacity, 2),
            'actual_capacity': round(actual_capacity, 2),
            'actual_soh': round(actual_soh, 2),
            'weakest_module_results': weakest_module_results,
            'mae': round(mae, 2) if mae is not None else None,
            'pred_knee_ah': round(pred_knee_ah, 2) if pred_knee_ah is not None else None,
            'pred_knee_v': round(pred_knee_v, 3) if pred_knee_v is not None else None,
            'actual_knee_ah': round(actual_knee_ah, 2) if actual_knee_ah is not None else None,
            'actual_knee_v': round(actual_knee_v, 3) if actual_knee_v is not None else None,
            'knee_error_ah': round(knee_error_ah, 2) if knee_error_ah is not None else None,
            'modules': module_result['modules'] if module_result else None,
            'weakest_module_predicted': module_result['weakest_module_predicted'] if module_result else None,
            'weakest_module_actual': module_result['weakest_module_actual'] if module_result else None,
            'module_error': module_error,
            'plot': weakest_module_plot_url,
            'plot_path': weakest_module_plot_path,
        }

    except Exception as e:
        print(f"❌ Error in analyze endpoint: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse({'error': f'Server error: {str(e)}'}, status_code=500)


@app.post('/analyze_cross_rate')
def analyze_cross_rate(sft_file: UploadFile | None = File(None), fft_file: UploadFile | None = File(None)):
    """Cross-rate: upload a 1.0C SFT (source) + a 0.3C FFT (ground truth) --
    predicts every module's 0.3C SOH/capacity from the 1.0C SFT alone
    (module_soh_cross_rate_train.py's model), and reports the ACTUAL 0.3C
    SOH/capacity measured directly from the uploaded FFT-0.3C file, so
    predicted-vs-actual can be compared for a pack whose true 0.3C rating
    genuinely wasn't available at prediction time -- see module_soh_cross_
    rate_train.py's docstring for this model's validated accuracy (4/5
    weakest-module ranking match, pk5 the known unresolved exception)."""
    try:
        if sft_file is None or fft_file is None:
            return JSONResponse({'error': 'Please upload both a 1.0C SFT file and a 0.3C FFT file.'}, status_code=400)
        if sft_file.filename == '' or fft_file.filename == '':
            return JSONResponse({'error': 'No selected file.'}, status_code=400)

        print(f"[cross-rate] Reading SFT (1.0C) file: {sft_file.filename}")
        sft_df = pd.read_csv(sft_file.file)
        print(f"[cross-rate] Reading FFT (0.3C) file: {fft_file.filename}")
        fft_df = pd.read_csv(fft_file.file)

        c_rate_match = re.search(r'(\d+(?:\.\d+)?)C', sft_file.filename)
        sft_c_rate = float(c_rate_match.group(1)) if c_rate_match else 1.0
        if abs(sft_c_rate - 0.95) < 0.05:
            sft_c_rate = 1.0
        if abs(sft_c_rate - 1.0) > 0.15:
            print(f"⚠️ [cross-rate] SFT file's own C-rate ({sft_c_rate}) doesn't look like 1.0C -- "
                  f"this model was only trained/validated for a 1.0C-bucket source file.")

        pack_id, _ = parse_pack_and_crate(sft_file.filename)
        module_result, module_error = analyze_modules_cross_rate(sft_df, sft_c_rate, fft_df, pack_id=pack_id)
        if module_error:
            print(f"⚠️ [cross-rate] Module analysis failed: {module_error}")
            return JSONResponse({'error': module_error}, status_code=400)

        print("[cross-rate] Extracting FFT-0.3C ground truth (pack-level)...")
        fft_ah, fft_v, _ = extract_and_resample_curve(fft_df)
        actual_capacity = float(fft_ah[-1])
        actual_soh = compute_soh(actual_capacity)

        module_predicted_sohs = {m['module_idx']: m['predicted_soh'] for m in module_result['modules']}
        pred_soh = min(module_predicted_sohs.values())
        pred_capacity = (pred_soh / 100.0) * NOMINAL_CAPACITY

        print(f"[cross-rate] Predicted (weakest module, from 1.0C SFT): {pred_soh:.2f}% / {pred_capacity:.2f} Ah   "
              f"Actual (from 0.3C FFT, pack-level): {actual_soh:.2f}% / {actual_capacity:.2f} Ah")
        print(f"[cross-rate] Weakest module -- predicted: {module_result['weakest_module_predicted']}  "
              f"actual: {module_result['weakest_module_actual']}")

        # Curve reconstruction: the predicted-weakest module's full 0.3C
        # voltage curve (shape-transferred from a real template, rescaled to
        # this module's predicted capacity -- see reconstruct_cross_rate_
        # module_curve) plotted against that SAME module's real 0.3C curve
        # straight from the uploaded FFT (exact ground truth, no
        # extrapolation needed since the FFT is a complete discharge).
        plot_url = None
        plot_path = None
        target_module = module_result['weakest_module_predicted']
        if target_module is not None:
            try:
                module_pred_capacity = next(
                    m['predicted_capacity'] for m in module_result['modules'] if m['module_idx'] == target_module
                )
                recon_ah, recon_v = reconstruct_cross_rate_module_curve(pack_id, module_pred_capacity)

                fft_mod_ah, _fft_pack_v, _fft_cell_std, fft_modules = extract_and_resample_curve(fft_df, want_modules=True)
                module_v_fft = fft_modules[target_module]['min_v']

                print("[cross-rate] Generating curve reconstruction plot...")
                plt.figure(figsize=(10, 6))
                plt.plot(fft_mod_ah, module_v_fft, label=f'Module {target_module} (Actual, FFT-0.3C)',
                         color='#64748b', linewidth=2.5, alpha=0.85)
                plt.plot(recon_ah, recon_v, label=f'Module {target_module} (Predicted, from 1.0C SFT)',
                         color='#dc2626', linewidth=2, linestyle='--')
                plt.title(f'Module {target_module} (Predicted Weakest) — 1.0C SFT → 0.3C Curve Reconstruction',
                          fontsize=13, fontweight='bold', pad=15)
                plt.xlabel('Capacity Delivered (Ah)', fontsize=12)
                plt.ylabel('Min Cell Voltage in Module (V)', fontsize=12)
                plt.legend(loc='upper right', fontsize=9)
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.ylim(1.8, 4.3)
                plt.tight_layout()

                cr_img = io.BytesIO()
                plt.savefig(cr_img, format='png', dpi=100, bbox_inches='tight')
                cr_img.seek(0)
                plot_url = base64.b64encode(cr_img.getvalue()).decode()
                plot_path = save_plot_image(cr_img.getvalue(), 'analyze_cross_rate')
                plt.close()
                print(f"[cross-rate] Curve reconstruction plot generated successfully -> {plot_path}")
            except Exception as e:
                print(f"⚠️ [cross-rate] Curve reconstruction plot unavailable: {e}")
                print(traceback.format_exc())

        return {
            'soh': round(pred_soh, 2),
            'capacity': round(pred_capacity, 2),
            'actual_soh': round(actual_soh, 2),
            'actual_capacity': round(actual_capacity, 2),
            'modules': module_result['modules'],
            'weakest_module_predicted': module_result['weakest_module_predicted'],
            'weakest_module_actual': module_result['weakest_module_actual'],
            'plot': plot_url,
            'plot_path': plot_path,
        }

    except Exception as e:
        print(f"❌ Error in analyze_cross_rate endpoint: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse({'error': f'Server error: {str(e)}'}, status_code=500)


@app.post('/analyze_cross_rate_weakest_module')
def analyze_cross_rate_weakest_module(sft_file: UploadFile | None = File(None)):
    """SFT-only cross-rate: upload just a 1.0C SFT file to predict every
    module's 0.3C SOH/capacity via the cross-rate model (module_soh_cross_
    rate_train.py) -- no 0.3C FFT ground-truth file required. Mirrors
    /analyze_weakest_module's SFT-only same-rate flow, but for cross-rate:
    reuses analyze_modules_cross_rate with fft_df=None (which comes back with
    every actual/ground-truth field null, since there's no FFT to validate
    against), then strips those null fields out entirely before responding --
    this route's response only ever contains predicted values."""
    try:
        if sft_file is None:
            return JSONResponse({'error': 'Please upload a 1.0C SFT file.'}, status_code=400)
        if sft_file.filename == '':
            return JSONResponse({'error': 'No selected file.'}, status_code=400)

        print(f"[cross-rate, SFT-only] Reading SFT (1.0C) file: {sft_file.filename}")
        sft_df = pd.read_csv(sft_file.file)

        c_rate_match = re.search(r'(\d+(?:\.\d+)?)C', sft_file.filename)
        sft_c_rate = float(c_rate_match.group(1)) if c_rate_match else 1.0
        if abs(sft_c_rate - 0.95) < 0.05:
            sft_c_rate = 1.0
        if abs(sft_c_rate - 1.0) > 0.15:
            print(f"⚠️ [cross-rate, SFT-only] SFT file's own C-rate ({sft_c_rate}) doesn't look like 1.0C -- "
                  f"this model was only trained/validated for a 1.0C-bucket source file.")

        pack_id, _ = parse_pack_and_crate(sft_file.filename)
        module_result, module_error = analyze_modules_cross_rate(sft_df, sft_c_rate, fft_df=None, pack_id=pack_id)
        if module_error:
            print(f"⚠️ [cross-rate, SFT-only] Module analysis failed: {module_error}")
            return JSONResponse({'error': module_error}, status_code=400)

        # No FFT was uploaded, so every actual_soh/actual_capacity/
        # label_source/targets['actual'] field analyze_modules_cross_rate
        # returns is unconditionally null -- strip them here rather than
        # shipping a response full of nulls the caller can never populate
        # through this route.
        modules_result = []
        for m in module_result['modules']:
            targets = m.get('targets') or {}
            clean_targets = {cv: entry['predicted'] for cv, entry in targets.items() if entry.get('predicted')}
            modules_result.append({
                'module_idx': m['module_idx'],
                'predicted_soh': m['predicted_soh'],
                'predicted_capacity': m['predicted_capacity'],
                'targets': clean_targets,
            })

        weakest_module = module_result['weakest_module_predicted']
        weakest_entry = next(m for m in modules_result if m['module_idx'] == weakest_module)
        pred_soh = weakest_entry['predicted_soh']
        pred_capacity = weakest_entry['predicted_capacity']
        print(f"[cross-rate, SFT-only] Predicted weakest module (from 1.0C SFT): "
              f"{weakest_module} -> {pred_soh:.2f}% / {pred_capacity:.2f} Ah")

        plot_url = None
        plot_path = None
        try:
            recon_ah, recon_v = reconstruct_cross_rate_module_curve(pack_id, pred_capacity)
            plt.figure(figsize=(10, 6))
            plt.plot(recon_ah, recon_v, label=f'Module {weakest_module} (Predicted 0.3C, shape-transferred)',
                     color='#dc2626', linewidth=2.5, linestyle='--')
            marker_colors = {'3.25': '#f59e0b', '2.5': '#a855f7'}
            for cv_str, pred_entry in (weakest_entry.get('targets') or {}).items():
                plt.scatter([pred_entry['capacity_ah']], [float(cv_str)], color=marker_colors.get(cv_str, '#f59e0b'),
                            s=150, zorder=5, marker='o', edgecolors='white', linewidths=2,
                            label=f"{cv_str}V -> {pred_entry['capacity_ah']} Ah, SOH {pred_entry['soh']}%")
            plt.title(f'Weakest Module (Module {weakest_module}) -- Predicted 0.3C Curve (from 1.0C SFT)',
                      fontsize=13, fontweight='bold', pad=15)
            plt.xlabel('Capacity Delivered (Ah)', fontsize=12)
            plt.ylabel('Min Cell Voltage in Module (V)', fontsize=12)
            plt.legend(loc='upper right', fontsize=9)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.ylim(1.8, 4.3)
            plt.tight_layout()
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plot_path = save_plot_image(img.getvalue(), 'analyze_cross_rate_weakest_module')
            plt.close()
            print(f"[cross-rate, SFT-only] Plot generated successfully -> {plot_path}")
        except Exception as e:
            print(f"⚠️ [cross-rate, SFT-only] Curve reconstruction plot unavailable: {e}")
            print(traceback.format_exc())

        return {
            'soh': round(pred_soh, 2),
            'capacity': round(pred_capacity, 2),
            'weakest_module_predicted': weakest_module,
            'modules': modules_result,
            'plot': plot_url,
            'plot_path': plot_path,
        }

    except Exception as e:
        print(f"❌ Error in analyze_cross_rate_weakest_module endpoint: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse({'error': f'Server error: {str(e)}'}, status_code=500)


@app.post('/analyze_weakest_module')
def analyze_weakest_module(sft_file: UploadFile | None = File(None)):
    """SFT-only: identifies the weakest module from the uploaded (partial) SFT
    file via the module SOH model, then estimates that module's own capacity
    and SOH at two fixed voltage cutoffs (3.25V, 2.5V) via the tiered physics
    extrapolation in module_capacity_extrapolation.py -- no FFT ground-truth
    file required, since that method only needs a template curve (borrowed
    from the pack's own or a cross-pack real full-curve test) rather than this
    test's own full curve."""
    try:
        if sft_file is None:
            return JSONResponse({'error': 'Please upload an SFT file.'}, status_code=400)

        if sft_file.filename == '':
            return JSONResponse({'error': 'No selected file.'}, status_code=400)

        print(f"Reading SFT file: {sft_file.filename}")
        sft_df = pd.read_csv(sft_file.file)

        pack_id, _ = parse_pack_and_crate(sft_file.filename)
        c_rate_match = re.search(r'(\d+(?:\.\d+)?)C', sft_file.filename)
        c_rate = float(c_rate_match.group(1)) if c_rate_match else 0.3
        if abs(c_rate - 0.95) < 0.05:
            c_rate = 1.0
        c_bucket = 1.0 if c_rate >= 0.9 else 0.3
        print(f"Detected pack_id={pack_id}, C-rate={c_rate} (bucket {c_bucket}C)")

        if c_bucket not in module_soh_models:
            return JSONResponse({'error': f'No module SOH model available for {c_bucket}C'}, status_code=400)

        sft_cell_cols = get_cell_columns(sft_df)
        feats_by_module = {}
        for m in range(1, N_MODULES + 1):
            f = extract_module_features_from_slice(sft_df, module_cell_columns(sft_cell_cols, m))
            if f is not None:
                feats_by_module[m] = f
        if len(feats_by_module) < N_MODULES:
            return JSONResponse({'error': "SFT file too short to extract all 9 modules' features."}, status_code=400)

        add_sibling_features(feats_by_module)
        model = module_soh_models[c_bucket]
        feature_names = module_soh_feature_names[c_bucket]

        predicted_soh = {}
        for m, f in feats_by_module.items():
            row = dict(f)
            row['c_rate'] = c_rate
            row['is_sfct'] = 1.0
            row['slice_start_pct'] = 0.0
            row['voltage_drop_norm'] = row['voltage_drop'] / (c_rate + 1e-5)
            row['delta_Ah_norm'] = row['delta_Ah'] / (c_rate + 1e-5)
            X = pd.DataFrame([row]).reindex(columns=feature_names, fill_value=0)
            predicted_soh[m] = float(model.predict(X)[0])

        weakest_module = min(predicted_soh, key=predicted_soh.get)
        print(f"Weakest module (predicted from SFT): {weakest_module}")

        ah, pack_v, cell_std, modules = extract_and_resample_curve(sft_df, want_modules=True)
        if ah is None:
            return JSONResponse({'error': 'SFT file missing AHDischarge column.'}, status_code=400)
        module_v = modules[weakest_module]['min_v']

        # SFT files are TAIL-ONLY captures: their AHDischarge is zeroed at
        # wherever that short test itself started, not at full charge (same
        # issue reconstruct_full_curve() already handles for the whole-pack
        # case via cutoff_ah = anchor_capacity - sft_delta_ah). Apply the same
        # fix here at module granularity: anchor on this module's OWN
        # predicted capacity (from the module SOH model above) so the local
        # SFT Ah axis lines up with true full-curve Ah. Verified against a
        # real FFT ground truth (pk4 module 7): without this offset, @3.25V/
        # @2.5V capacity came out at 76/82 Ah (SOH ~49%/53%) vs the true
        # 140/145 Ah (SOH ~90%/93%) -- a silent, badly wrong result.
        module_anchor_capacity = (predicted_soh[weakest_module] / 100.0) * NOMINAL_CAPACITY
        ah_offset = max(0.0, module_anchor_capacity - float(ah[-1]))
        print(f"   Module {weakest_module} anchor capacity: {module_anchor_capacity:.2f} Ah, "
              f"SFT local span: {ah[-1]:.2f} Ah, offset: {ah_offset:.2f} Ah")

        template_ah, template_v = pick_template_curve(pack_id, c_bucket)
        if template_ah is None:
            return JSONResponse({'error': 'No template curve available for extrapolation.'}, status_code=500)
        template_used = 'pack_own' if (pack_id, c_bucket) in MODULE_TEMPLATES else 'cross_pack_fallback'

        target_cutoffs = [3.25, 2.5]
        targets, tail_ah, tail_v = estimate_module_capacity_at_targets(
            ah, module_v, target_cutoffs, template_ah=template_ah, template_v=template_v,
        )

        results = {}
        for cv, (local_cap_ah, source, _diag) in targets.items():
            global_cap_ah = ah_offset + local_cap_ah
            results[cv] = {'capacity_ah': round(global_cap_ah, 2), 'soh': round(compute_soh(global_cap_ah), 2), 'source': source}
            print(f"  @{cv}V -> {global_cap_ah:.2f} Ah, SOH {compute_soh(global_cap_ah):.2f}% (source={source})")

        # The SFT test starts from a rested/open-circuit state, so its first
        # stretch shows a real but LOCAL IR-drop/relaxation transient (sharp
        # initial sag settling into the normal gentle plateau slope) that
        # doesn't exist at that same Ah position in a genuinely continuous
        # discharge. Trim it before using this data as the "observed" curve
        # or as the calibration anchor -- otherwise the predicted head (which
        # has no such transient, since it's built from continuous curves) and
        # the observed data visibly disagree right at the join.
        settle_idx = detect_settle_index(ah, module_v)
        ah_settled, v_settled = ah[settle_idx:], module_v[settle_idx:]
        if settle_idx > 0:
            print(f"   Trimmed {ah[settle_idx] - ah[0]:.2f} Ah of startup transient from module "
                  f"{weakest_module}'s observed SFT data ({settle_idx} samples)")

        # Head (0 -> settled SFT start): the SFT only observes the TAIL of the
        # curve, so there's no real data for the module before that point.
        # Approximate it by reusing the pack-level ML reconstruction model
        # (same one the /analyze endpoint uses for the whole-pack curve) to
        # get the full 0->anchor_capacity Mean_Cell_Voltage curve, then shift
        # it by this module's own calibrated offset from the pack mean --
        # measured right at the join -- since a weak module sits at a fairly
        # consistent voltage gap below the pack mean until it nears its own
        # knee. Approximate, not a validated extrapolation like the tail;
        # labelled as such on the plot.
        head_ah, head_v = np.array([]), np.array([])
        try:
            recon_ah, recon_v, _recon_cutoff_ah = reconstruct_full_curve(sft_df, module_anchor_capacity, c_rate)
            ah_offset_settled = ah_offset + (ah_settled[0] - ah[0])
            ah_global_settled = ah_offset_settled + (ah_settled - ah_settled[0])
            recon_v_at_module_ah = np.interp(ah_global_settled, recon_ah, recon_v)
            boundary_n = min(10, len(ah_global_settled))
            pack_mean_offset = float(np.mean(v_settled[:boundary_n] - recon_v_at_module_ah[:boundary_n]))
            head_mask = recon_ah < ah_offset_settled
            if head_mask.sum() >= 2:
                head_ah = recon_ah[head_mask]
                head_v = np.clip(recon_v[head_mask] + pack_mean_offset, 1.8, 4.3)
        except Exception as e:
            print(f"⚠️ Head reconstruction unavailable: {e}")

        print("Generating weakest-module plot...")
        ah_global_settled = ah_offset + (ah_settled - ah[0])
        tail_ah_global = ah_offset + tail_ah
        plt.figure(figsize=(10, 6))
        if len(head_ah) > 0:
            plt.plot(head_ah, head_v, label='Predicted Head (0 → SFT start, approximate)',
                     color='#16a34a', linewidth=2, linestyle=':')
        plt.plot(ah_global_settled, v_settled, label=f'Module {weakest_module} (Observed, SFT)', color='#2563eb', linewidth=2.5)
        if len(tail_ah_global) > 0:
            order = np.argsort(tail_ah_global)
            plt.plot(tail_ah_global[order], tail_v[order], label='Extrapolated Tail (physics-based)',
                     color='#dc2626', linewidth=2, linestyle='--')

        marker_colors = {3.25: '#f59e0b', 2.5: '#a855f7'}
        for cv, res in results.items():
            plt.scatter([res['capacity_ah']], [cv], color=marker_colors.get(cv, '#f59e0b'), s=150, zorder=5,
                        marker='o', edgecolors='white', linewidths=2,
                        label=f"{cv}V -> {res['capacity_ah']} Ah, SOH {res['soh']}%")

        plt.title(f'Weakest Module (Module {weakest_module}) Voltage Curve', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Capacity Delivered (Ah)', fontsize=12)
        plt.ylabel('Min Cell Voltage in Module (V)', fontsize=12)
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.ylim(1.8, 4.3)
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plot_path = save_plot_image(img.getvalue(), 'analyze_weakest_module')
        plt.close()
        print(f"Plot generated successfully -> {plot_path}")

        return {
            'weakest_module': weakest_module,
            'predicted_soh_by_module': {str(m): round(v, 2) for m, v in predicted_soh.items()},
            'results': {str(cv): res for cv, res in results.items()},
            'template_used': template_used,
            'head_reconstructed': len(head_ah) > 0,
            'plot': plot_url,
            'plot_path': plot_path,
        }

    except Exception as e:
        print(f"❌ Error in analyze_weakest_module endpoint: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse({'error': f'Server error: {str(e)}'}, status_code=500)


@app.post('/upload_training_data')
def upload_training_data(files: list[UploadFile] = File(...)):
    """Training Pipeline: accepts one or more raw pack CSVs -- a single file,
    several files, or an entire folder selected via the UI (webkitdirectory)
    -- and cleans each one down to just its real discharge window (see
    raw_file_cleaner.py for what "clean" means: dropping the pre-test
    charge/rest and post-test rest/recharge sections). Each raw upload is
    saved into RAW_UPLOADS_FOLDER for a record of what came in, and its
    cleaned result is written straight into DATA_FOLDER -- ready for the
    next retrain (see 'Retrain the models' in the README for what to run
    next; this endpoint does NOT retrain anything itself).
    """
    results = []
    for f in files:
        # basename(), not the raw filename -- a folder upload's filename can
        # carry the browser's full relative subfolder path (webkitRelativePath),
        # and untouched that's also a directory-traversal risk against
        # RAW_UPLOADS_FOLDER/DATA_FOLDER.
        fname = os.path.basename((f.filename or '').replace('\\', '/'))
        if not fname.lower().endswith('.csv'):
            results.append({'filename': fname or '(unnamed)', 'status': 'skipped', 'reason': 'Not a .csv file'})
            continue

        raw_path = os.path.join(RAW_UPLOADS_FOLDER, fname)
        try:
            with open(raw_path, 'wb') as out:
                out.write(f.file.read())

            with open(raw_path) as raw_fh:
                raw_row_count = sum(1 for _ in raw_fh) - 1  # minus header

            pack_id, c_rate = parse_pack_and_crate(fname)
            upper = fname.upper()
            if is_full_curve_file(fname):
                file_type = 'FFCT/FFT (full)'
            elif 'SFT' in upper or 'SFCT' in upper:
                file_type = 'SFT/SFCT (short)'
            else:
                file_type = 'unknown'

            output_path = os.path.join(DATA_FOLDER, fname)
            cleaned_df = clean_raw_file(raw_path, output_path)

            results.append({
                'filename': fname,
                'status': 'cleaned',
                'raw_rows': raw_row_count,
                'cleaned_rows': len(cleaned_df),
                'pack_id': pack_id,
                'c_rate': c_rate,
                'file_type': file_type,
                'naming_warning': None if pack_id else (
                    "Filename doesn't match the pkN + C-rate + FFCT/FFT/SFT/SFCT convention -- "
                    "training scripts won't pick this file up until it does (see README)."
                ),
            })
        except Exception as e:
            results.append({'filename': fname, 'status': 'skipped', 'reason': str(e)})

    cleaned_count = sum(1 for r in results if r['status'] == 'cleaned')
    return JSONResponse({
        'results': results,
        'cleaned_count': cleaned_count,
        'total_count': len(results),
        'data_folder': DATA_FOLDER,
    })


# ==========================================
# TRAINING API (async job) -- POST /train_models to start, GET /train_status/{job_id} to poll
# ==========================================
# A full retrain (cleaning + all 3 active training scripts) takes 5-30
# minutes, far too long for a normal blocking HTTP request/timeout -- so
# /train_models kicks it off as a background subprocess and returns
# immediately with a job_id; /train_status/{job_id} reports progress (tailed
# log output) and the final result. TRAIN_JOBS is in-memory only -- like
# everything else in this file, it resets if the app restarts, so a job's
# status becomes unknown (404) across a restart; the training run itself
# (a separate subprocess) is unaffected and keeps writing its own log file.
TRAIN_JOBS_DIR = os.path.join(APP_DIR, 'train_jobs')
os.makedirs(TRAIN_JOBS_DIR, exist_ok=True)
TRAIN_JOBS = {}


class TrainRequest(BaseModel):
    raw_folder: str


def _run_training_job(job_id, raw_folder):
    log_path = os.path.join(TRAIN_JOBS_DIR, f'{job_id}.log')
    TRAIN_JOBS[job_id]['log_file'] = log_path
    pipeline_script = os.path.join(APP_DIR, 'run_full_pipeline.py')
    try:
        with open(log_path, 'w') as log_fh:
            proc = subprocess.run(
                [sys.executable, pipeline_script],
                input=raw_folder + '\n', text=True,
                stdout=log_fh, stderr=subprocess.STDOUT, cwd=APP_DIR,
            )
        TRAIN_JOBS[job_id]['returncode'] = proc.returncode
        TRAIN_JOBS[job_id]['status'] = 'completed' if proc.returncode == 0 else 'failed'
    except Exception as e:
        TRAIN_JOBS[job_id]['status'] = 'failed'
        TRAIN_JOBS[job_id]['error'] = str(e)
    finally:
        TRAIN_JOBS[job_id]['finished_at'] = time.time()


@app.post('/train_models')
def train_models(req: TrainRequest):
    """Starts a full retrain -- cleans every CSV in req.raw_folder (a path on
    the SERVER's filesystem, same as what run_full_pipeline.py's interactive
    prompt asks for) and retrains all 3 active models -- as a background
    subprocess, and returns a job_id immediately rather than blocking for the
    5-30 minutes the run actually takes. Poll GET /train_status/{job_id} for
    progress. A successful run lands in a new ml_pipeline/models/vN/ folder;
    restart the app afterward to start serving it (see model_versioning.py --
    it's picked up automatically, no config needed)."""
    if not os.path.isdir(req.raw_folder):
        return JSONResponse(
            {'error': f"'{req.raw_folder}' is not a folder that exists on the server running this app."},
            status_code=400,
        )
    job_id = uuid.uuid4().hex[:8]
    TRAIN_JOBS[job_id] = {
        'status': 'running', 'started_at': time.time(), 'raw_folder': req.raw_folder, 'log_file': None,
    }
    thread = threading.Thread(target=_run_training_job, args=(job_id, req.raw_folder), daemon=True)
    thread.start()
    return {'job_id': job_id, 'status': 'running', 'poll_at': f'/train_status/{job_id}'}


@app.get('/train_status/{job_id}')
def train_status(job_id: str):
    job = TRAIN_JOBS.get(job_id)
    if job is None:
        return JSONResponse(
            {'error': "Unknown job_id (job list is in-memory and resets if the app restarts)."},
            status_code=404,
        )
    resp = {
        'job_id': job_id, 'status': job['status'],
        'started_at': job['started_at'], 'raw_folder': job['raw_folder'],
    }
    if job.get('finished_at'):
        resp['finished_at'] = job['finished_at']
        resp['elapsed_minutes'] = round((job['finished_at'] - job['started_at']) / 60, 1)
        resp['returncode'] = job.get('returncode')
    if job.get('error'):
        resp['error'] = job['error']
    if job.get('log_file') and os.path.exists(job['log_file']):
        with open(job['log_file']) as f:
            content = f.read()
        resp['log_tail'] = content[-4000:]  # last ~4000 chars -- enough to see current progress/errors
    return resp


if __name__ == '__main__':
    import uvicorn
    # No reload: the models are already restarted manually after every
    # code/model change (auto-reload also isn't compatible with this file's
    # module-level model loading -- the reloader would re-run it in a
    # separate watcher process, doubling load time for no benefit here).
    uvicorn.run(app, host='0.0.0.0', port=5000)


