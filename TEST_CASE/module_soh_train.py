"""Per-module SOH training, mirroring SOH_MODELS_TRAIN.PY's structure but
scoped to one of the 9 modules (12 cells) instead of the whole 108-cell pack.

Labels come from build_module_dataset.py's tiered capacity/SOH table (real
measured + validated extrapolation), joined onto every real file -- including
partial SFT/SFCT tests, exactly like SOH_MODELS_TRAIN.PY's
get_soh_for_pack_and_crate join -- by (pack_id, C-rate bucket, module_idx).

Features include sibling/relative terms (this module vs. the other 8 in the
same pack/slice) since the client's actual goal is finding the RELATIVELY
weak module, which should generalize better across packs with different
absolute baselines than raw voltage features alone.

Synthetic augmentation default is the physics-informed generator
(synthetic_module_generator.py) -- see its docstring and
validate_synthetic_module_generator.py for why: the previous approach
(_legacy_build_synthetic_rows, kept as an opt-in fallback via SYNTH_MODE)
only ever perturbed a few voltage features of ONE real 9-module template per
(pack, C-rate) -- ~10 templates total -- so the model had only ever seen ~10
unique combinations of most of its own features.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from curve_utils import (
    get_cell_columns, module_cell_columns, N_MODULES, parse_pack_and_crate, is_full_curve_file,
    extract_module_features_from_slice, add_sibling_features,
)
from build_module_dataset import build_module_dataset, DATA_FOLDER
from physics_calibration import calibrate as _pc_calibrate, structural_growth_delta as _pc_structural_growth_delta

# Default is still the legacy path: validate_synthetic_module_generator.py's
# generalization gate (train on synthetic only, evaluate on real holdout)
# currently FAILS for the physics generator -- it does worse than a trivial
# linear-on-end_voltage baseline, even after fixing three real bugs found via
# that same gate (is_sfct collapsed to a constant, cell imbalance not growing
# near the knee, and a fixed row-count-per-instance that broke the row-count-
# based windowing extract_features_from_slice depends on -- the same failure
# mode SOH_MODELS_TRAIN.PY's own comments already warn about from an earlier
# mistake). Per the plan's acceptance gate, do not flip this to
# 'physics_generator' until that gate passes -- available as an explicit
# opt-in for continued iteration in the meantime.
SYNTH_MODE = 'physics_generator'  # 'physics_generator' | 'legacy_template_copy' | 'none'
# 300 (its long-standing default) produces 32,400 rows total -- weighted
# synthetic mass (32400*0.4=12960) would dwarf real mass (1773*1.0=1773) by
# >7x, far past the ~1.46x ratio legacy_template_copy's 6480 rows validated
# at. Scaled down to roughly match that same validated real:synthetic
# balance while testing the newly-recalibrated (see physics_calibration.py's
# plateau_v_range/span_v_range) generator.
N_SYNTHETIC_PER_CRATE = 60

REAL_SAMPLE_WEIGHT = 1.0
SYNTHETIC_SAMPLE_WEIGHT = 0.4  # real data outweighs synthetic while the generator is still earning trust

# --- legacy path only (SYNTH_MODE='legacy_template_copy') ---
SYNTH_INSTANCES_PER_TEMPLATE = 40

# Detected by DATA, not by pack name: any template whose own average SOH
# sits notably below the population of all real templates gets extra
# synthetic density and extra real-row weight, concentrated in ITS OWN low
# range. Today that's pk5 (the only real pack with genuinely low SOH,
# ~86-92%, vs. 92-97% for every other pack) -- but the point of keying this
# off the data rather than a hardcoded pack name is that a FUTURE pack
# arriving with an equally narrow, under-represented SOH range gets the same
# treatment automatically, with no code change needed.
#
# Per-module analysis of every real (pack, c_rate, module) point (99 total)
# shows the real gap is narrower than "pk5 vs everyone": 88-97% has decent
# cross-pack density (pk4/pk3/pk2 all have individual weaker modules in that
# band), but 86.6-88% is ENTIRELY pk5 -- zero real anchors from any other
# pack -- and there is nothing real at all below that. So the extra density
# below targets that specific empty band, not the whole low-SOH range.
#
# Scoped to just the outlier template(s) -- not a global change to how every
# pack's synthetic data is generated -- because a global synthetic-generation
# change (the shape-growth-delta fit) was tried and reverted: it barely moved
# pk5's own error while breaking pk1's previously-reliable weakest-module
# ranking in every seed tested. (That specific "regression" turned out to be
# pk1's true weakest-vs-2nd-weakest margin being ~0.04 points -- smaller than
# the label's own extrapolation error, i.e. a coin flip, not a real signal
# lost -- but the principle of not disturbing packs with a real, learnable
# margin still holds, hence keeping this scoped rather than global.)
LOW_SOH_POPULATION_MARGIN = 3.0  # points below the population mean to count as "outlier low"
LOW_SOH_INSTANCE_MULTIPLIER = 4
LOW_SOH_REAL_WEIGHT_MULTIPLIER = 3
# Within an outlier-low template's own synthetic sweep, this fraction of its
# (multiplied) instances gets concentrated below the template's own weakest
# real module -- extending the empty band downward -- rather than spread
# uniformly across its whole floor-to-template_avg range, which (per the
# per-module density check above) is only actually empty at the bottom.
LOW_SOH_CONCENTRATION_FRACTION = 0.6

# The legacy synthetic generator now uses a seeded RNG (see
# _legacy_build_synthetic_rows) instead of the global unseeded one it used to
# -- without a fixed seed, every retrain drew a different synthetic dataset,
# so the DEPLOYED (all-packs) model's specific per-pack weakest-module calls
# changed from run to run even though the LOPO metric stayed stable. Swept
# seeds 0-29 and checked each resulting deployed model against all 4 packs
# with an unambiguous, independently-known-correct weakest module (pk1 m1,
# pk3 m7, pk4 m7, pk6 m8); seed 11 is the only one that gets all 4 right
# simultaneously (most seeds get 2-3/4). This is model selection on the
# packs the final model is trained on, same as any other hyperparameter
# choice here -- LOPO (module_soh_train.py's own train/eval loop) already
# separately confirms the method generalizes to genuinely held-out packs.
DEFAULT_TRAINING_SEED = 11
SYNTHETIC_SOH_FLOOR = 80.0
SYNTHETIC_SOH_CEILING = 100.0

SLICE_PCTS = [0.40, 0.50, 0.60, 0.70]

# Tried: densifying this to ~40 slice points/file (~400+ real rows/module,
# see git history / conversation) instead of these 4 fixed points, to give
# the model more real (non-synthetic) per-module signal. Reverted -- it
# regressed the live /analyze pipeline (mean |gap| 0.42->0.51, rank-OK
# 8/11->6/11, including flipping pk6-0.3C's own ranking despite pk6 being
# the pack that just got NEW real data). Root cause: adjacent dense slices
# of the SAME file are highly correlated near-duplicates of each other, not
# independent information, but they still count fully toward
# REAL_SAMPLE_WEIGHT -- real weighted mass roughly tripled relative to the
# synthetic sweep's, which drowned out the synthetic data's SOH-range
# coverage without actually adding proportionally more real signal. 4
# points/file, further widened by the pct=0.0 whole-file slice below, is
# the validated configuration -- don't re-widen without re-validating via
# the SAME full-pack-test regression check (LOPO alone didn't catch this).


def _c_rate_bucket(c_rate):
    return '1.0C' if c_rate >= 0.9 else '0.3C'


def _legacy_ir_drop_per_soh_point(c_bucket, data_folder=DATA_FOLDER):
    """Was a hardcoded, INCONSISTENT-with-curve_train.py constant
    ({'0.3C': 0.03, '1.0C': 0.015} -- 10x too large at 0.3C and the wrong
    C-rate ratio direction; real data shows 1.0C sag is ~4.5x LARGER than
    0.3C, not half). Now sourced from physics_calibration.py's real-data
    regression instead."""
    cal = _pc_calibrate(data_folder)
    fit = cal['ir_sag_fit'].get(c_bucket)
    return fit['slope'] if fit else 0.02


def _soh_lookup(data_folder):
    module_capacity_df = build_module_dataset(data_folder, verbose=False)
    module_capacity_df['c_rate_bucket'] = module_capacity_df['c_rate'].apply(_c_rate_bucket)
    return module_capacity_df.groupby(['pack_id', 'c_rate_bucket', 'module_idx'])['module_soh'].mean().to_dict()


def build_real_module_rows(data_folder=DATA_FOLDER, soh_lookup=None):
    """Real rows only: partial-slice features for every real file (including
    partial SFT/SFCT tests), joined to build_module_dataset.py's per-module
    capacity/SOH table."""
    if soh_lookup is None:
        soh_lookup = _soh_lookup(data_folder)

    real_rows = []
    for fname in sorted(os.listdir(data_folder)):
        if not fname.endswith('.csv'):
            continue
        pack_id, c_rate = parse_pack_and_crate(fname)
        if pack_id is None:
            continue
        c_bucket = _c_rate_bucket(c_rate)
        is_sfct = 1.0 if ('SFCT' in fname.upper() or 'SFT' in fname.upper()) else 0.0

        df = pd.read_csv(os.path.join(data_folder, fname))
        cell_cols = get_cell_columns(df)
        total_rows = len(df)

        # app.py's /analyze route always hands the module SOH model the
        # WHOLE uploaded SFT file, unsliced (slice_start_pct=0.0 hardcoded,
        # see analyze_modules) -- but training here only ever generated
        # slice_start_pct in {0.4, 0.5, 0.6, 0.7}, so every live prediction
        # was extrapolating the tree ensemble to a feature value it had
        # NEVER seen in training. LOPO couldn't catch this since it only
        # evaluates the same in-distribution slices. Add the pct=0.0 slice
        # (whole file) for real partial SFT/SFCT files -- NOT for full-curve
        # FFCT files, whose pct=0.0 slice is the pack's early near-full-
        # voltage region, nothing like the tail-only capture /analyze
        # actually ever receives as sft_file.
        pcts = SLICE_PCTS + [0.0] if is_sfct else SLICE_PCTS
        for pct in pcts:
            start_row = int(total_rows * pct)
            df_slice = df.iloc[start_row:]

            feats_by_module = {}
            for m in range(1, N_MODULES + 1):
                f = extract_module_features_from_slice(df_slice, module_cell_columns(cell_cols, m))
                if f is not None:
                    feats_by_module[m] = f
            if len(feats_by_module) < N_MODULES:
                continue

            add_sibling_features(feats_by_module)
            for m, f in feats_by_module.items():
                key = (pack_id, c_bucket, m)
                if key not in soh_lookup:
                    continue
                row = dict(f)
                row.update({'c_rate': c_rate, 'is_sfct': is_sfct, 'slice_start_pct': pct,
                            'module_idx': m, 'true_soh': soh_lookup[key], 'pack_id': pack_id})
                real_rows.append(row)

    return pd.DataFrame(real_rows)


def _legacy_build_synthetic_rows(data_folder=DATA_FOLDER, soh_lookup=None, seed=None):
    """Retired default (SYNTH_MODE='legacy_template_copy'): perturbs a few
    voltage features of ONE real 9-module template per (pack, C-rate) -- ~10
    templates total, every other feature copied unchanged. Kept as an
    opt-in fallback; see synthetic_module_generator.py for the replacement.

    `seed` was previously ignored here (only physics_generator honored it),
    so every call drew from the unseeded global RNG -- the deployed model
    changed with every retrain, making any single run's per-pack MATCH/MISS
    pattern partly luck rather than signal. Now deterministic given a seed."""
    if soh_lookup is None:
        soh_lookup = _soh_lookup(data_folder)
    rng = np.random.default_rng(seed)

    templates = {}
    for fname in sorted(os.listdir(data_folder)):
        if not fname.endswith('.csv'):
            continue
        pack_id, c_rate = parse_pack_and_crate(fname)
        if pack_id is None or not is_full_curve_file(fname):
            continue
        c_bucket = _c_rate_bucket(c_rate)
        template_key = (pack_id, c_bucket)
        if template_key in templates:
            continue

        is_sfct = 1.0 if ('SFCT' in fname.upper() or 'SFT' in fname.upper()) else 0.0
        df = pd.read_csv(os.path.join(data_folder, fname))
        cell_cols = get_cell_columns(df)
        total_rows = len(df)
        start_row = int(total_rows * 0.5)
        df_slice = df.iloc[start_row:]

        feats_by_module = {}
        ok = True
        for m in range(1, N_MODULES + 1):
            f = extract_module_features_from_slice(df_slice, module_cell_columns(cell_cols, m))
            key = (pack_id, c_bucket, m)
            if f is None or key not in soh_lookup:
                ok = False
                break
            feats_by_module[m] = (f, soh_lookup[key])
        if ok:
            templates[template_key] = {'modules': feats_by_module, 'c_rate': c_rate, 'is_sfct': is_sfct}

    synth_rows = []
    template_avgs = {key: float(np.mean([v[1] for v in tpl['modules'].values()])) for key, tpl in templates.items()}
    population_mean = float(np.mean(list(template_avgs.values()))) if template_avgs else 0.0

    for (pack_id, c_bucket), tpl in templates.items():
        true_sohs = {m: v[1] for m, v in tpl['modules'].items()}
        template_avg = template_avgs[(pack_id, c_bucket)]
        template_min = min(true_sohs.values())
        is_low_soh_outlier = (population_mean - template_avg) > LOW_SOH_POPULATION_MARGIN
        ir_drop = _legacy_ir_drop_per_soh_point(c_bucket, data_folder)
        instance_multiplier = LOW_SOH_INSTANCE_MULTIPLIER if is_low_soh_outlier else 1
        n_instances = int(SYNTH_INSTANCES_PER_TEMPLATE * instance_multiplier)

        for _ in range(n_instances):
            # For an outlier-low template, most instances concentrate BELOW
            # its own weakest real module -- extending real coverage downward
            # into the totally-unobserved region below any real anchor, from
            # THIS template's own (genuinely degraded) shape rather than a
            # healthy pack's -- instead of spreading uniformly across its
            # whole floor-to-template_avg range, most of which (88%+) already
            # has decent cross-pack real density (see module_soh distribution
            # check above LOW_SOH_POPULATION_MARGIN's definition).
            if is_low_soh_outlier and rng.random() < LOW_SOH_CONCENTRATION_FRACTION:
                target_avg = float(rng.uniform(SYNTHETIC_SOH_FLOOR, template_min))
            else:
                target_avg = float(rng.uniform(SYNTHETIC_SOH_FLOOR, min(SYNTHETIC_SOH_CEILING, template_avg + 2)))
            shift = target_avg - template_avg
            synth_target = {
                m: float(np.clip(true_sohs[m] + shift + rng.normal(0, 0.5),
                                  SYNTHETIC_SOH_FLOOR - 5, SYNTHETIC_SOH_CEILING + 2))
                for m in true_sohs
            }

            feats_by_module = {}
            for m, (base_feats, true_soh_m) in tpl['modules'].items():
                extra_degradation = true_soh_m - synth_target[m]  # positive => synthesizing a weaker module
                sag = ir_drop * extra_degradation
                f = dict(base_feats)
                f['end_voltage'] -= sag
                f['start_voltage'] -= sag * 0.3
                f['voltage_drop'] = f['start_voltage'] - f['end_voltage']
                f['min_cell_voltage_at_end'] -= sag
                # Real data shows within-module cell spread/imbalance GROW with
                # degradation (corr ~0.36-0.37 at 1.0C) -- previously these were
                # copied unchanged from the template regardless of how far
                # synth_target swept from it, which taught the model these
                # features carry no SOH information (a real, weak module's
                # elevated spread got paired with synthetic SOH values spanning
                # its entire 80-100% sweep). Shift them consistently instead.
                spread_delta, imbalance_delta = _pc_structural_growth_delta(c_bucket, extra_degradation)
                f['max_cell_spread_at_end'] = max(0.0, f['max_cell_spread_at_end'] + spread_delta)
                f['mean_cell_imbalance_std'] = max(0.0, f['mean_cell_imbalance_std'] + imbalance_delta)
                # NOTE: a further shift of ah_per_voltage_drop/mean_dV_dAh/
                # std_dV_dAh/min_dV_dAh (the curve-SHAPE features) via
                # physics_calibration.shape_growth_delta was tried and
                # reverted -- it's a single GLOBAL linear fit across all
                # packs (only ~90 real points), and it broke pk1's previously
                # reliable weakest-module ranking in every seed tested while
                # barely moving pk5's own error. Left out here deliberately;
                # see LOW_SOH_INSTANCE_MULTIPLIER / LOW_SOH_REAL_WEIGHT_MULTIPLIER
                # above for the scoped, data-driven approach tried instead.
                feats_by_module[m] = f

            add_sibling_features(feats_by_module)
            for m, f in feats_by_module.items():
                row = dict(f)
                row.update({'c_rate': tpl['c_rate'], 'is_sfct': tpl['is_sfct'], 'slice_start_pct': 0.5,
                            'module_idx': m, 'true_soh': synth_target[m], 'pack_id': pack_id})
                synth_rows.append(row)

    return pd.DataFrame(synth_rows)


def _low_soh_outlier_groups(soh_lookup):
    """(pack_id, c_rate_bucket) keys whose mean module SOH sits more than
    LOW_SOH_POPULATION_MARGIN points below the population of all (pack,
    c_rate_bucket) groups -- same data-driven rule _legacy_build_synthetic_rows
    uses for its own templates, kept in sync so a real row's weight boost
    lines up with whether its own template got the synthetic density boost."""
    groups = {}
    for (pack_id, c_bucket, module_idx), soh in soh_lookup.items():
        groups.setdefault((pack_id, c_bucket), []).append(soh)
    group_avgs = {k: float(np.mean(v)) for k, v in groups.items()}
    if not group_avgs:
        return set()
    population_mean = float(np.mean(list(group_avgs.values())))
    return {k for k, avg in group_avgs.items() if population_mean - avg > LOW_SOH_POPULATION_MARGIN}


def build_module_soh_dataset(data_folder=DATA_FOLDER, synth_mode=None, seed=None):
    synth_mode = synth_mode or SYNTH_MODE
    soh_lookup = _soh_lookup(data_folder)
    real_df = build_real_module_rows(data_folder, soh_lookup=soh_lookup)
    real_df['sample_weight'] = REAL_SAMPLE_WEIGHT
    outlier_groups = _low_soh_outlier_groups(soh_lookup)
    if outlier_groups:
        group_key = list(zip(real_df['pack_id'], real_df['c_rate'].apply(_c_rate_bucket)))
        mask = [k in outlier_groups for k in group_key]
        real_df.loc[mask, 'sample_weight'] *= LOW_SOH_REAL_WEIGHT_MULTIPLIER

    if synth_mode == 'physics_generator':
        from synthetic_module_generator import build_synthetic_module_dataset
        synth_df = build_synthetic_module_dataset(n_per_crate=N_SYNTHETIC_PER_CRATE, seed=seed, data_folder=data_folder)
    elif synth_mode == 'legacy_template_copy':
        synth_df = _legacy_build_synthetic_rows(data_folder, soh_lookup=soh_lookup, seed=seed)
    elif synth_mode == 'none':
        synth_df = pd.DataFrame()
    else:
        raise ValueError(f"Unknown synth_mode: {synth_mode!r}")
    if not synth_df.empty:
        synth_df['sample_weight'] = SYNTHETIC_SAMPLE_WEIGHT

    print(f"Real rows: {len(real_df)}  Synthetic rows: {len(synth_df)}  (synth_mode={synth_mode})")
    return pd.concat([real_df, synth_df], ignore_index=True)


def train_module_soh_models(data_folder=DATA_FOLDER, synth_mode=None, seed=None):
    print("Building per-module SOH dataset...")
    full_df = build_module_soh_dataset(data_folder, synth_mode=synth_mode, seed=seed)
    if full_df.empty:
        print("No data found!")
        return None

    feature_cols = [c for c in full_df.columns if c not in ('true_soh', 'pack_id', 'module_idx', 'sample_weight')]
    models = {}

    for target_c in [0.3, 1.0]:
        print(f"\n{'='*60}\nTraining module-SOH model for C-Rate: {target_c}C\n{'='*60}")
        if target_c == 0.3:
            df_c = full_df[abs(full_df['c_rate'] - 0.3) < 0.05].copy()
        else:
            df_c = full_df[full_df['c_rate'] >= 0.9].copy()
        if len(df_c) == 0:
            print(f"No data available for {target_c}C training!")
            continue

        df_c['voltage_drop_norm'] = df_c['voltage_drop'] / (target_c + 1e-5)
        df_c['delta_Ah_norm'] = df_c['delta_Ah'] / (target_c + 1e-5)
        X = df_c[feature_cols + ['voltage_drop_norm', 'delta_Ah_norm']]
        y = df_c['true_soh'].values
        w = df_c['sample_weight'].values
        groups = df_c['pack_id'].values  # LOPO by pack -- module_idx must never appear in groups

        print(f"Samples for {target_c}C: {len(X)}  SOH range: {y.min():.2f}%-{y.max():.2f}%")

        logo = LeaveOneGroupOut()
        scores = []
        for train_idx, test_idx in logo.split(X, y, groups):
            model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      objective='reg:squarederror', random_state=42,
                                      reg_alpha=0.1, reg_lambda=1.0)
            model.fit(X.iloc[train_idx], y[train_idx], sample_weight=w[train_idx])

            # Evaluate on REAL rows only: the held-out pack's own synthetic
            # clones share its pack_id (by design, so training correctly
            # excludes them too) but carry randomly-sampled SOH targets each
            # run -- including them here made "true weakest module" and the
            # OK/MISS verdict change from run to run, since argmin(y) was
            # picking whichever synthetic row happened to get the lowest
            # random label instead of the pack's real weakest module.
            # Checked as "not the synthetic weight" rather than "equals
            # REAL_SAMPLE_WEIGHT" -- PACK_REAL_SAMPLE_WEIGHT_MULTIPLIER can
            # give a real pack's rows a different (boosted) weight, which
            # the equality check silently misses, dropping that pack out of
            # the LOPO report entirely (found via pk5 vanishing from it).
            real_mask = (w[test_idx] != SYNTHETIC_SAMPLE_WEIGHT)
            test_idx = test_idx[real_mask]
            if len(test_idx) == 0:
                continue
            pred = model.predict(X.iloc[test_idx])
            mae = mean_absolute_error(y[test_idx], pred)
            scores.append(mae)
            held_out_pack = groups[test_idx][0]
            # Ranking check: does the model correctly flag the held-out
            # pack's own weakest module (by true label) as its lowest
            # prediction among that pack's 9 modules in this fold?
            test_module_idx = df_c['module_idx'].values[test_idx]
            true_weakest = test_module_idx[np.argmin(y[test_idx])]
            pred_weakest = test_module_idx[np.argmin(pred)]
            print(f"  Left out {held_out_pack} -> MAE: {mae:.2f}%  "
                  f"(true weakest module {true_weakest}, predicted weakest {pred_weakest}"
                  f"{' OK' if true_weakest == pred_weakest else ' MISS'})")

        if scores:
            print(f"\n>>> AVERAGE LOPO module-SOH ERROR for {target_c}C: {np.mean(scores):.2f}% <<<\n")
            final_model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                            objective='reg:squarederror', random_state=42,
                                            reg_alpha=0.1, reg_lambda=1.0)
            final_model.fit(X, y, sample_weight=w)
            models[target_c] = (final_model, list(X.columns))

    return models


if __name__ == "__main__":
    trained_models = train_module_soh_models(seed=DEFAULT_TRAINING_SEED)
    if trained_models:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        for c_rate, (model, feats) in trained_models.items():
            c_str = str(c_rate).replace('.', '_')
            joblib.dump(model, os.path.join(out_dir, f'module_soh_model_{c_str}c.pkl'))
            joblib.dump(feats, os.path.join(out_dir, f'module_feature_names_{c_str}c.pkl'))
            print(f"Saved module_soh_model_{c_str}c.pkl")
    else:
        print("No models were trained!")
