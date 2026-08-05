"""Shared curve/feature extraction utilities.

Consolidates logic that used to be copy-pasted (with the comment "must match
exactly, feature order is positional") across curve_train.py, daignos.py, and
app.py. Behavior for existing callers is unchanged -- extract_and_resample_curve
still returns the same (ah, v, cell_std) 3-tuple by default, and
extract_enhanced_features/detect_knee_index/detect_knee compute identically to
their old per-file copies. The only additions are module-level constants/
helpers and the optional want_modules=True path on extract_and_resample_curve,
needed for the new per-module pipeline.
"""

import os
import re
import numpy as np
import pandas as pd

NOMINAL_CAPACITY = 156.0

MODULE_SIZE = 12
N_MODULES = 9

# Non-uniform, knee-dense checkpoint grid -- must match across training and
# inference call sites (positional).
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

# Some real captures show a single-row sensor glitch on the very first
# sample of the file, before per-cell readings settle -- confirmed on pk6's
# 0.3C FFCT file: pack MEAN voltage at row 0 reads a normal ~4.19V, but
# several modules' MIN cell voltage at that SAME row reads 2.3-3.0V (a >1V
# single-row dropout on a subset of cells), snapping back to the normal
# ~4.14V by the very next row. A discharge curve cannot legitimately start
# more than this far below its own first-sample pack mean and then
# instantly recover -- this is a DAQ/relay-closure artifact, not real
# physics, and left in place it corrupts anything that borrows this file's
# curve SHAPE (e.g. the cross-rate curve-reconstruction template, or any
# per-module synthetic-data generator anchored on it) with a spurious
# vertical drop-and-recover at Ah=0. See _trim_leading_glitch_row.
GLITCH_ROW_DROP_THRESHOLD_V = 0.5


def _trim_leading_glitch_row(discharge_df, cell_cols, max_rows=3):
    """Drops up to `max_rows` leading rows where a single row's cell-to-cell
    spread (mean - min) exceeds GLITCH_ROW_DROP_THRESHOLD_V -- see that
    constant's docstring. Checked generically at the start of every file
    (not hardcoded to any one pack) so any file with the same artifact is
    caught; bounded so a genuinely noisy file can't be trimmed away
    entirely."""
    for _ in range(max_rows):
        if len(discharge_df) < 10 or not cell_cols:
            break
        first_row = discharge_df[cell_cols].iloc[0]
        if (first_row.mean() - first_row.min()) > GLITCH_ROW_DROP_THRESHOLD_V:
            discharge_df = discharge_df.iloc[1:]
        else:
            break
    return discharge_df


def compute_soh(total_capacity_ah):
    return float((total_capacity_ah / NOMINAL_CAPACITY) * 100.0)


def get_cell_columns(df):
    """All 108 cell-voltage columns, in true numeric tap order (Cell 001..108).

    Requires a digit in the name (not just the substring 'Cell') -- callers
    like extract_soh_features mutate the passed-in dataframe in place, adding
    derived columns such as 'Mean_Cell_Voltage'/'Cell_Voltage_Std' that also
    contain 'Cell'. Without the digit requirement those get silently swept
    into cell_cols on any later call sharing the same dataframe object,
    corrupting downstream means with a bogus extra "cell"."""
    cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c and re.search(r'\d+', c)]
    return sorted(cols, key=lambda c: int(re.search(r'\d+', c).group()))


def module_cell_columns(cell_cols, module_idx):
    """cell_cols for module_idx (1-based, 1..N_MODULES). Sequential 12-cell block."""
    start = (module_idx - 1) * MODULE_SIZE
    return cell_cols[start:start + MODULE_SIZE]


def extract_and_resample_curve(source, want_modules=False):
    """Filters to the valid discharge window and re-zeros Ah.

    `source` may be a file path (str) or an already-loaded DataFrame -- shared
    by uploaded SFT/FFT curves (app.py) and on-disk training files
    (curve_train.py, daignos.py).

    Returns (ah, v, cell_std) by default -- unchanged from the original
    per-file copies. With want_modules=True, also returns a 4th value: a
    dict {module_idx (1..N_MODULES): {'min_v': array, 'mean_v': array}}, each
    aligned to the same `ah` axis (computed from the same AHDischarge-sorted
    row set), for the new per-module pipeline.
    """
    df = pd.read_csv(source) if isinstance(source, str) else source
    cell_cols = get_cell_columns(df)
    df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)

    if 'AHDischarge' not in df.columns:
        return (None, None, None, None) if want_modules else (None, None, None)

    mask = (df['Mean_Cell_Voltage'] < 4.15) & (df['Mean_Cell_Voltage'] > 2.0)
    discharge_df = df[mask].copy()
    discharge_df = discharge_df.sort_values('AHDischarge')
    discharge_df = _trim_leading_glitch_row(discharge_df, cell_cols)
    discharge_df['Ah_Relative'] = discharge_df['AHDischarge'] - discharge_df['AHDischarge'].iloc[0]

    ah = discharge_df['Ah_Relative'].values
    v = discharge_df['Mean_Cell_Voltage'].values
    cell_std = discharge_df[cell_cols].std(axis=1).values if len(cell_cols) > 0 else None

    if not want_modules:
        return ah, v, cell_std

    modules = {}
    for m in range(1, N_MODULES + 1):
        mod_cols = module_cell_columns(cell_cols, m)
        modules[m] = {
            'min_v': discharge_df[mod_cols].min(axis=1).values,
            'mean_v': discharge_df[mod_cols].mean(axis=1).values,
        }
    return ah, v, cell_std, modules


def extract_enhanced_features(sft_v, sft_ah, cell_data=None):
    """Feature order is positional wherever this is consumed -- do not reorder."""
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

    knee_idx = int(np.argmax(np.abs(dv_dah)))
    features['tail_knee_pct'] = float(sft_ah[knee_idx] / (sft_ah[-1] + 1e-5))
    features['tail_knee_slope'] = float(dv_dah[knee_idx])

    return features


def detect_settle_index(ah, v, window=5, k=3.0, persist=15, max_search_ah_frac=0.08, max_search_ah_abs=5.0):
    """Index where a trace's initial IR-drop/relaxation transient has settled
    into its normal (much gentler) decline. A short test that starts
    discharging from a rested/open-circuit state shows a real, physical sharp
    initial sag (ohmic drop + polarization settling) before flattening into
    the usual plateau slope -- distinct from, and much steeper than, the rest
    of the curve. Finds the first point after which the rolling |dV/dAh|
    stays at/below k times the trace's own median |dV/dAh| for `persist`
    consecutive samples, searched only within the first
    min(max_search_ah_abs, max_search_ah_frac * total_span) of the trace (a
    genuine startup transient resolves quickly; searching further risks
    latching onto an unrelated calm stretch elsewhere in the curve).
    Returns 0 (no trim) if nothing qualifies within that window."""
    dv = np.diff(v) / np.clip(np.diff(ah), 1e-6, None)
    if len(dv) < window + persist:
        return 0
    roll = np.convolve(np.abs(dv), np.ones(window) / window, mode='valid')
    ref = np.median(np.abs(dv))
    thresh = max(k * ref, 1e-4)
    below = roll <= thresh
    search_limit_ah = min(max_search_ah_abs, max_search_ah_frac * (ah[-1] - ah[0]))
    search_limit_idx = np.searchsorted(ah, ah[0] + search_limit_ah)
    limit = min(len(below) - persist, search_limit_idx)
    for i in range(max(limit, 0)):
        if below[i:i + persist].all():
            return i
    return 0


def detect_knee_index(ah, v, search_start_pct=0.5, search_end_pct=0.99, fallback='last'):
    """Chord-distance elbow detection: the point in the search window with
    maximum perpendicular distance from the straight line joining the
    window's endpoints.

    fallback controls behavior when the window has < 3 points:
    'last' -> len(ah) - 1 (curve_train.py's original behavior), None -> None
    (app.py's original detect_knee behavior, via the detect_knee wrapper below).
    """
    total = ah[-1]
    mask = (ah >= search_start_pct * total) & (ah <= search_end_pct * total)
    idx = np.where(mask)[0]
    if len(idx) < 3:
        return (len(ah) - 1) if fallback == 'last' else None

    ah_w, v_w = ah[idx], v[idx]
    ah_n = (ah_w - ah_w[0]) / (ah_w[-1] - ah_w[0] + 1e-9)
    v_n = (v_w - v_w[0]) / (v_w[-1] - v_w[0] + 1e-9)
    x1, y1 = ah_n[0], v_n[0]
    x2, y2 = ah_n[-1], v_n[-1]
    num = np.abs((y2 - y1) * ah_n - (x2 - x1) * v_n + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-9
    dist = num / den
    return int(idx[np.argmax(dist)])


def detect_knee(ah, v, search_start_pct=0.5, search_end_pct=0.99):
    """Value-returning wrapper (ah, v) at the knee, or (None, None) -- matches
    app.py's original detect_knee."""
    idx = detect_knee_index(ah, v, search_start_pct, search_end_pct, fallback=None)
    if idx is None:
        return None, None
    return float(ah[idx]), float(v[idx])


def extract_module_features_from_slice(df_slice, module_cols):
    """Module-scoped version of SOH_MODELS_TRAIN.PY's extract_features_from_slice.
    Used identically for real (module_soh_train.py) and synthetic
    (synthetic_module_generator.py) rows, so both go through the exact same
    feature computation."""
    if len(df_slice) < 50:
        return None
    mv = df_slice[module_cols].mean(axis=1)
    minv = df_slice[module_cols].min(axis=1)
    maxv = df_slice[module_cols].max(axis=1)
    stdv = df_slice[module_cols].std(axis=1)

    features = {}
    features['start_voltage'] = float(mv.iloc[:10].mean())
    features['end_voltage'] = float(mv.iloc[-10:].mean())
    features['voltage_drop'] = features['start_voltage'] - features['end_voltage']
    features['mean_cell_imbalance_std'] = float(stdv.mean())
    features['max_cell_spread'] = float((maxv - minv).mean())
    features['max_cell_spread_at_end'] = float((maxv.iloc[-10:] - minv.iloc[-10:]).mean())
    features['min_cell_voltage_at_end'] = float(minv.iloc[-10:].mean())

    if 'AHDischarge' in df_slice.columns:
        ah = df_slice['AHDischarge']
        features['delta_Ah'] = float(ah.iloc[-1] - ah.iloc[0])
        features['ah_per_voltage_drop'] = features['delta_Ah'] / (features['voltage_drop'] + 1e-5)
        dV = mv.diff().dropna()
        dAh = ah.diff().dropna()
        dV_dAh = dV / dAh.replace(0, 1e-5)
        features['mean_dV_dAh'] = float(dV_dAh.mean())
        features['std_dV_dAh'] = float(dV_dAh.std())
        features['min_dV_dAh'] = float(dV_dAh.min())

        # Curve-SHAPE features (slopes at different sections, curvature, where
        # along the Ah axis the knee sits) -- already implemented in
        # extract_enhanced_features and used by the curve-reconstruction
        # pipeline (see reconstruct_module_curve), but never wired into the
        # per-module SOH model before. Unlike end_voltage/voltage_drop, these
        # are largely invariant to a module's own series resistance (a
        # resistance offset shifts the whole curve's voltage LEVEL, not its
        # slope/curvature/knee-position) -- meant to help the model tell
        # "genuinely lower capacity" apart from "same capacity, more IR sag"
        # (see pk1's documented m1-vs-m5 case: m5 sags more per Ah than m1
        # despite m1 being the module that actually measured weaker).
        #
        # end_slope/end_voltage_drop (computed from int(len(slice) * 0.8) --
        # a PROPORTIONAL position, not a fixed Ah window) were tried removed
        # in an earlier iteration on the theory that they're noisy across
        # real files of different lengths (confirmed on pk4's two real 1.0C
        # SFT files: 1754 vs 1943 rows, end_voltage_drop swinging 0.489 vs
        # 0.261 for the same module). That WAS true and fixed a cross-rate
        # miss, but net-regressed the same-rate model badly (live test 9/11
        # -> 6/11, breaking 3 previously-correct packs) -- kept in despite
        # the known noise, since removing them cost more than it gained.
        try:
            shape_feats = extract_enhanced_features(mv.values, ah.values, stdv.values)
            for key in ('initial_slope', 'final_slope', 'overall_slope', 'mean_curvature',
                        'max_curvature', 'curvature_std', 'voltage_std', 'voltage_range',
                        'plateau_length', 'end_slope', 'end_voltage_drop',
                        'tail_knee_pct', 'tail_knee_slope'):
                features[key] = shape_feats[key]
        except Exception:
            for key in ('initial_slope', 'final_slope', 'overall_slope', 'mean_curvature',
                        'max_curvature', 'curvature_std', 'voltage_std', 'voltage_range',
                        'plateau_length', 'end_slope', 'end_voltage_drop',
                        'tail_knee_pct', 'tail_knee_slope'):
                features[key] = 0.0
    else:
        features['delta_Ah'] = 0.0
        features['ah_per_voltage_drop'] = 0.0
        features['mean_dV_dAh'] = 0.0
        features['std_dV_dAh'] = 0.0
        features['min_dV_dAh'] = 0.0
        for key in ('initial_slope', 'final_slope', 'overall_slope', 'mean_curvature',
                    'max_curvature', 'curvature_std', 'voltage_std', 'voltage_range',
                    'plateau_length', 'end_slope', 'end_voltage_drop',
                    'tail_knee_pct', 'tail_knee_slope'):
            features[key] = 0.0

    return features


def add_sibling_features(feats_by_module):
    """feats_by_module: {module_idx: feature dict}. Adds rel_end_voltage
    (this module's end_voltage minus the pack's mean-of-module end_voltage)
    and sibling_rank (1 = weakest end_voltage among the 9) IN PLACE."""
    end_vs = {m: f['end_voltage'] for m, f in feats_by_module.items()}
    pack_mean = np.mean(list(end_vs.values()))
    ranks = {m: r + 1 for r, m in enumerate(sorted(end_vs, key=lambda k: end_vs[k]))}
    for m, f in feats_by_module.items():
        f['rel_end_voltage'] = end_vs[m] - pack_mean
        f['sibling_rank'] = ranks[m]


def parse_pack_and_crate(filename):
    """Returns (pack_id, c_rate) or (None, None) if the filename doesn't match.
    Uses the looser C-rate pattern (decimal optional) -- the stricter
    r'(\\d+\\.\\d+)C' used by SOH_MODELS_TRAIN.PY misses whole-number C-rates
    like 'pk4-60-29052021-SFT-1C ...csv'."""
    pack_match = re.search(r'(pk\d+)', filename)
    c_rate_match = re.search(r'(\d+(?:\.\d+)?)C', filename)
    if not pack_match or not c_rate_match:
        return None, None
    return pack_match.group(1), float(c_rate_match.group(1))


def is_full_curve_file(filename):
    upper = filename.upper()
    return 'FFCT' in upper or 'FFT' in upper


def list_full_curve_files(data_folder):
    """Full-curve (FFCT/FFT) CSVs in data_folder, deduped by (pack_id, c_rate)
    -- the raw exports include byte-identical '(1)'/'(2)' duplicates of the
    same test, which must not be double-counted. Returns a list of dicts:
    {path, pack_id, c_rate}, one per unique (pack_id, c_rate)."""
    seen = set()
    out = []
    for fname in sorted(os.listdir(data_folder)):
        if not fname.endswith('.csv') or not is_full_curve_file(fname):
            continue
        pack_id, c_rate = parse_pack_and_crate(fname)
        if pack_id is None:
            continue
        key = (pack_id, c_rate)
        if key in seen:
            continue
        seen.add(key)
        out.append({'path': os.path.join(data_folder, fname), 'pack_id': pack_id, 'c_rate': c_rate})
    return out
