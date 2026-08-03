"""Cross-rate per-module SOH training: predicts a module's SOH/capacity AT
0.3C from features extracted from a 1.0C-bucket file (SFT/SFCT partial
captures, plus FFCT full-curve files sliced at the same partial-view
percentages used elsewhere -- see build_real_module_rows in
module_soh_train.py, whose row-building logic this mirrors).

Real ground truth for both rates comes from build_module_dataset.py's tiered
capacity/SOH table -- same as the same-rate module_soh_train.py -- but here
the label is deliberately pulled from the OPPOSITE C-rate bucket than the
row's own source file: "what would this exact module measure at 0.3C, given
only how it behaved at 1.0C."

Why this is a genuinely different, harder problem than same-rate prediction
(see rate_gap_analysis.py / rate_gap_0.3C_vs_1.0C.png in the scratchpad for
the full analysis this is based on):
  - The 0.3C-vs-1.0C capacity gap is real (mean +3.69 Ah / +2.36 SOH points,
    0.3C higher) but PACK-DEPENDENT, not a fixed constant or a smooth
    function of SOH: pk5's gap is ~0 while pk2/pk4/pk6 sit around 5-5.6 Ah,
    with pk1/pk3 in between. A single scalar rate-correction does not fit
    all packs.
  - The weakest module's IDENTITY is not rate-invariant: pk2 and pk3 (2 of
    6 real packs) have a DIFFERENT weakest module at 0.3C than at 1.0C. The
    model must predict every module's own 0.3C label independently and take
    min() over that, exactly like the same-rate pipeline -- it cannot just
    carry over "the 1.0C-weakest module" as an assumption.

Evaluated via leave-one-pack-out (LOPO), with a "realistic" sub-metric
computed on only the is_sfct=1, slice_start_pct=0.0 rows -- a whole uploaded
SFT file is EXACTLY what the live app hands this model (analyze_modules
always sends slice_start_pct=0.0 unsliced), so that sub-metric is more
representative of production than the pooled-across-all-slices number.

BASELINE (real rows only, no synthetic augmentation):
  >>> pooled MAE 3.03%, realistic MAE 2.95%, weakest-module ranking match
      2/5 evaluable packs (pk2 has no native 1.0C SFT file to evaluate the
      realistic case with).

TRIED: synthetic augmentation via synthetic_cross_rate_generator.py -- a
synthetic 1.0C virtual pack (reusing synthetic_module_generator.py's already-
validated curve-generation physics unchanged) paired with a 0.3C label built
from a rate-gap BOOTSTRAPPED from the 6 real packs' own measured gaps (see
that module's docstring for why a bootstrap, not a fitted/assumed
distribution). The fidelity check (cross_rate_synth_vs_real.png, scratchpad)
looked reasonable: synthetic gap mean/std (3.29/2.63 Ah) close to real
(3.69/2.33 Ah), synthetic cloud enveloping the real per-module scatter.

Swept SYNTHETIC_SAMPLE_WEIGHT at 0.0/0.1/0.2/0.3/0.4/0.6/0.8/1.0 (0.0 = the
real-only baseline above):
  weight  pooled MAE  realistic MAE  ranking
  0.0     3.03%       2.94%          2/5   <- real-only baseline
  0.1     2.87%       3.27%          2/5
  0.2     2.83%       2.95%          2/5
  0.3     2.61%       2.94%          1/5
  0.4     2.58%       2.90%          1/5
  0.6     2.76%       3.22%          0/5
  0.8     2.67%       3.14%          1/5
  1.0     2.65%       2.84%          0/5
Pooled MAE improves monotonically-ish with more synthetic weight, but
realistic MAE never clearly beats the real-only baseline (best case ties at
0.2/0.4/1.0, worst case is noticeably worse) and ranking match only ever
TIES the baseline (at low weight) or gets WORSE (0/5 at 0.6 and 1.0) -- never
improves it. Same pattern as the anchor-feature experiment: a well-motivated
addition that looks fine or better in the pooled/aggregate view but doesn't
clear the bar on the realistic, ranking-sensitive metric that actually
matters. Likely cause: only 6 real packs' worth of rate-gap observations to
bootstrap from is not enough to teach the model a reliable correction without
also adding noise that occasionally flips a close ranking call.

CURRENT default: real rows only (synth_mode='none'), matching the validated
baseline above. The synthetic generator and synth_mode='physics_generator'
option are kept available (not deleted) for future revisiting -- e.g. if
more real packs become available to better calibrate the rate-gap bootstrap
-- but are not the deployed default given this result.

TRIED AND REVERTED: adding the already-trained same-rate 1.0C model's own
prediction as an extra "same_rate_anchor_soh" input feature, so the model
only has to learn the residual gap on top of a same-rate anchor instead of
reconstructing absolute 0.3C capacity from raw voltage-shape features alone.
Computed leak-free (a fresh anchor model retrained excluding the held-out
pack's real rows, per LOPO fold -- synthetic rows use synth_* pack ids so
are unaffected). Unweighted, this looked WORSE on the metric that matters:
realistic MAE 2.95% -> 4.45%, ranking match 2/5 -> 1/5 (pk1 flipped from a
correct, 0.40%-MAE prediction to wrong at 4.47% MAE) -- despite the POOLED
(all-slices) metric improving (0/6 -> 4/6 ranking), which would have looked
like a clear win if that were the only number checked. Then tried
up-weighting the realistic (is_sfct=1, pct=0.0) rows during training, at
2x/4x/6x/8x/12x/20x -- ranking recovered to 2/5 (tying the no-anchor
baseline) but realistic MAE plateaued at 4.4-4.5% across the ENTIRE weight
range, never approaching the no-anchor baseline's 2.95%. Conclusion: the
anchor feature doesn't help here (possibly because it's highly correlated
with several of the model's own base features, destabilizing the small-
sample tree splits, or because the 1.0C same-rate estimate it's anchored to
carries its own ~2% error that the cross-rate model then has to partially
undo rather than build on) -- reverted rather than kept as an unproven
complexity. If reused later, see git history / this file's earlier version
for the _fit_same_rate_anchor / _anchor_feature_matrix implementation.

CURRENT DEFAULT: synth_mode='module_perturbation' (module_perturbation_
generator.py) -- the first synthetic approach in this whole cross-rate
investigation that actually beats the real-only baseline instead of trading
one metric for the other. Unlike the from-scratch physics generator, every
synthetic instance is a real (pack, module) FFT curve pair, lightly
rescaled + noised (see that module's docstring) -- anchored to real data by
construction, not reconstructed from a canonical shape. Fidelity confirmed
visually first (module_perturbation_capacity_scatter.png / _curve_examples.png):
10800 synthetic points (200/module x 54 real modules) form a tight cloud
directly around each real anchor, not spread randomly.

Swept SYNTHETIC_SAMPLE_WEIGHT at 0.0/0.1/0.2/0.3/0.4/0.6/0.8/1.0:
  weight  pooled MAE  realistic MAE  ranking
  0.0     3.03%       2.94%          2/5   <- real-only baseline
  0.1     1.50%       2.53%          1/5
  0.2     1.51%       2.28%          3/5
  0.3     1.69%       2.95%          3/5
  0.4     1.71%       3.09%          4/5   <- CHOSEN
  0.6     1.82%       3.40%          4/5
  0.8     1.83%       3.79%          3/5
  1.0     1.77%       3.48%          3/5
w=0.4 chosen: best ranking match (4/5) this investigation has produced, at
the cost of worse realistic MAE than w=0.2's 2.28% (a real tradeoff, not a
free win -- w=0.2 has better average error but only 3/5 ranking). Per-pack
at w=0.4: pk1 OK (0.53%), pk3 OK (1.27%, fixed from baseline's MISS), pk4 OK
(2.95%, MAE worse than baseline's 1.42% but still correct), pk6 OK (6.59%,
fixed from baseline's MISS but with much worse MAE), pk5 MISS (4.09%,
improved from baseline's 6.89% but still wrong) -- pk5 misses under EVERY
approach tried in this whole investigation (real-only, anchor, bootstrap
physics-generator, and this), consistent with it being the isolated-outlier
pack (see rate_gap_analysis.py). Prioritized ranking match over average MAE
since correctly identifying the weakest module is this system's core
deliverable -- revisit if pk4/pk6's larger absolute error at this weight
turns out to matter more in practice than getting their ranking right.

TRIED AND REVERTED: adding mean_temp/temp_rise (curve_utils.extract_temp_
features) as extra per-row features, motivated by a real finding on pk4 --
its two real 1.0C test sessions (~6 degC apart) disagree on whether module 1
or module 7 is weaker, and temperature looked like a plausible explanation
(module 1 loses the near-tie when cold, module 7 when warm). Implemented for
both real rows (build_cross_rate_module_rows) and synthetic rows
(module_perturbation_generator, which jitters a shared per-instance session
temperature since temperature is pack-wide, not module-specific -- 16
sensors read within ~2 degC of each other at any instant, confirmed on pk4).
Result, at the same w=0.4: pooled MAE 1.71%->1.86%, realistic MAE
3.09%->3.26%, ranking match 4/5->2/5 (pk1 and pk3 both flipped from OK to
MISS). Worse on every metric, most importantly the ranking match that
matters most. Likely cause: only 6 real packs' worth of temperature
readings -- each essentially a single fixed value per pack in the real
data -- isn't enough to teach the model a genuine temperature-sensitivity
correction; more likely it just adds another axis for small-sample tree
splits to latch onto noise. Reverted; see git history for the
_load_pack_temp_trace / extract_temp_features implementation if more
temperature-diverse real data becomes available later.

TRIED AND REVERTED: adding mean_current/actual_c_rate (curve_utils.extract_
current_features, from the file's own LoadUnitCurrent column) as extra
per-row features. Better motivated than the temperature attempt above: the
existing c_rate feature is only the FILENAME LABEL ("1C"/"1.0C"), not the
actual measured current, and pk4's two real 1.0C-labeled sessions turned out
to differ by ~7% in true mean current (144.9A vs 154.6A) -- a real gap the
label can't see. Implemented the same way as the temperature attempt (real
rows + module_perturbation_generator synthetic rows, sharing one perturbed
session-level value per synthetic instance) and retrained at the same
w=0.4. Result: pooled MAE 1.71%->1.59% (a bit better), realistic MAE
3.21%->3.27% (a bit worse), ranking match unchanged at 4/5 -- essentially a
wash on the aggregate LOPO numbers, neither a clear win nor a clear
regression like the temperature attempt. But checked directly against the
motivating case (pk4's two real files) and it changed NOTHING: the May-26
file still predicts module 1 (MISS vs true module 7) and the June-11 file
still predicts module 7 (OK), identical to before the feature was added,
across all 5 evaluable packs' live single-file predictions. Reverted since
it didn't move the actual thing it was meant to fix, despite the reasonable
motivation. See git history for the _load_pack_current_trace /
extract_current_features implementation if revisited later -- e.g. paired
with an explicit IR-correction term (voltage_drop adjusted by measured
current x an estimated resistance) rather than just handing the model a raw
extra number and hoping it finds the relationship with only 6 packs' worth
of examples.
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))

from curve_utils import (
    get_cell_columns, module_cell_columns, N_MODULES, parse_pack_and_crate,
    extract_module_features_from_slice, add_sibling_features,
)
from build_module_dataset import DATA_FOLDER
from module_soh_train import _c_rate_bucket, _soh_lookup, SLICE_PCTS
from synthetic_cross_rate_generator import build_synthetic_cross_rate_module_rows
from compute_device import get_xgb_device_params

SOURCE_BUCKET = '1.0C'
TARGET_BUCKET = '0.3C'

# Same convention as module_soh_train.py: real data outweighs synthetic
# while the generator is still earning trust.
REAL_SAMPLE_WEIGHT = 1.0
SYNTHETIC_SAMPLE_WEIGHT = 0.4
N_SYNTHETIC_INSTANCES = 60  # matches module_soh_train's N_SYNTHETIC_PER_CRATE order of magnitude
DEFAULT_SYNTH_SEED = 11  # matches module_soh_train.DEFAULT_TRAINING_SEED


def build_cross_rate_module_rows(data_folder=DATA_FOLDER, soh_lookup=None,
                                  source_bucket=SOURCE_BUCKET, target_bucket=TARGET_BUCKET):
    """One row per (source file, slice pct, module) whose FEATURES come from
    a source_bucket file and whose LABEL is that same (pack_id, module_idx)'s
    real target_bucket SOH -- deliberately mismatched rates, unlike
    module_soh_train.build_real_module_rows where both match."""
    if soh_lookup is None:
        soh_lookup = _soh_lookup(data_folder)

    rows = []
    for fname in sorted(os.listdir(data_folder)):
        if not fname.endswith('.csv'):
            continue
        pack_id, c_rate = parse_pack_and_crate(fname)
        if pack_id is None:
            continue
        if _c_rate_bucket(c_rate) != source_bucket:
            continue

        is_sfct = 1.0 if ('SFCT' in fname.upper() or 'SFT' in fname.upper()) else 0.0
        df = pd.read_csv(os.path.join(data_folder, fname))
        cell_cols = get_cell_columns(df)
        total_rows = len(df)

        # Same reasoning as build_real_module_rows: the live app always hands
        # this model a WHOLE uploaded SFT file (slice_start_pct=0.0), so a
        # genuine partial-test file needs that slice in training too.
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
                target_key = (pack_id, target_bucket, m)
                if target_key not in soh_lookup:
                    continue
                row = dict(f)
                row.update({'c_rate': c_rate, 'is_sfct': is_sfct, 'slice_start_pct': pct,
                            'module_idx': m, 'true_soh': soh_lookup[target_key], 'pack_id': pack_id})
                rows.append(row)

    return pd.DataFrame(rows)


def train_cross_rate_module_soh_model(data_folder=DATA_FOLDER, synth_mode='module_perturbation', seed=DEFAULT_SYNTH_SEED):
    print(f"Building cross-rate dataset: features @ {SOURCE_BUCKET} -> label @ {TARGET_BUCKET} ...")
    soh_lookup = _soh_lookup(data_folder)
    real_df = build_cross_rate_module_rows(data_folder, soh_lookup=soh_lookup)
    if real_df.empty:
        print("No data found!")
        return None
    real_df['sample_weight'] = REAL_SAMPLE_WEIGHT

    if synth_mode == 'physics_generator':
        synth_df = build_synthetic_cross_rate_module_rows(N_SYNTHETIC_INSTANCES, seed=seed, data_folder=data_folder)
        # Diagnostic-only columns from the generator (used for the real-vs-
        # synthetic fidelity plot) -- not real features, and real_df doesn't
        # have them, so drop before concatenating.
        synth_df = synth_df.drop(columns=['source_soh_1_0c', 'source_capacity_1_0c'], errors='ignore')
        synth_df['sample_weight'] = SYNTHETIC_SAMPLE_WEIGHT
    elif synth_mode == 'module_perturbation':
        from module_perturbation_generator import build_module_perturbation_rows, N_PER_PACK
        synth_df = build_module_perturbation_rows(n_per_pack=N_PER_PACK, seed=seed, data_folder=data_folder)
        synth_df['sample_weight'] = SYNTHETIC_SAMPLE_WEIGHT
    elif synth_mode == 'none':
        synth_df = pd.DataFrame()
    else:
        raise ValueError(f"Unknown synth_mode: {synth_mode!r}")

    print(f"Real rows: {len(real_df)}  Synthetic rows: {len(synth_df)}  (synth_mode={synth_mode})")
    df = pd.concat([real_df, synth_df], ignore_index=True)

    # Normalize by each row's OWN (source) c_rate, not the target -- the
    # voltage_drop/delta_Ah magnitudes reflect the ACTUAL 1.0C-ish discharge
    # that produced them, not the 0.3C rate they're being used to predict.
    df['voltage_drop_norm'] = df['voltage_drop'] / (df['c_rate'] + 1e-5)
    df['delta_Ah_norm'] = df['delta_Ah'] / (df['c_rate'] + 1e-5)

    feature_cols = [c for c in df.columns if c not in ('true_soh', 'pack_id', 'module_idx', 'sample_weight')]
    X = df[feature_cols]
    y = df['true_soh'].values
    w = df['sample_weight'].values
    groups = df['pack_id'].values

    real_pack_ids = sorted(set(real_df['pack_id']))
    print(f"Samples: {len(X)}  SOH range: {y.min():.2f}%-{y.max():.2f}%  Real packs: {real_pack_ids}")
    print(real_df.groupby('pack_id').size().to_string())

    logo = LeaveOneGroupOut()
    scores = []
    realistic_scores = []
    realistic_ranking_hits = []
    for train_idx, test_idx in logo.split(X, y, groups):
        held_out_pack = groups[test_idx][0]
        # Synthetic instances each carry their own unique pack_id, so
        # LeaveOneGroupOut also produces one trivial fold per synthetic
        # instance -- skip those before the (expensive) fit rather than
        # after, since every one of that fold's own rows is synthetic by
        # construction (module_soh_train.py's LOPO has this same pattern,
        # just filtered after fitting instead of before).
        if held_out_pack not in real_pack_ids:
            continue

        model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                  objective='reg:squarederror', random_state=42,
                                  reg_alpha=0.1, reg_lambda=1.0, **get_xgb_device_params())
        model.fit(X.iloc[train_idx], y[train_idx], sample_weight=w[train_idx])

        pred = model.predict(X.iloc[test_idx])
        mae = mean_absolute_error(y[test_idx], pred)
        scores.append(mae)
        test_module_idx = df['module_idx'].values[test_idx]
        true_weakest = test_module_idx[np.argmin(y[test_idx])]
        pred_weakest = test_module_idx[np.argmin(pred)]

        # Realistic single-file view: is_sfct=1 AND slice_start_pct==0.0 --
        # what the live app actually sends (a whole uploaded SFT file), not
        # an average over every training slice.
        realistic_mask = (df['is_sfct'].values[test_idx] == 1.0) & (df['slice_start_pct'].values[test_idx] == 0.0)
        if realistic_mask.sum() > 0:
            real_true = y[test_idx][realistic_mask]
            real_pred = pred[realistic_mask]
            real_mods = test_module_idx[realistic_mask]
            real_mae = mean_absolute_error(real_true, real_pred)
            real_true_weak = real_mods[np.argmin(real_true)]
            real_pred_weak = real_mods[np.argmin(real_pred)]
            real_str = (f"realistic MAE {real_mae:.2f}%, weakest true={real_true_weak} "
                        f"pred={real_pred_weak} {'OK' if real_true_weak == real_pred_weak else 'MISS'}")
            realistic_scores.append(real_mae)
            realistic_ranking_hits.append(real_true_weak == real_pred_weak)
        else:
            real_str = "realistic: no native SFT file for this pack"

        print(f"  Left out {held_out_pack} -> pooled MAE: {mae:.2f}%  "
              f"(true weakest module {true_weakest}, predicted weakest {pred_weakest}"
              f"{' OK' if true_weakest == pred_weakest else ' MISS'})   |   {real_str}")

    print(f"\n>>> AVERAGE LOPO cross-rate module-SOH ERROR ({SOURCE_BUCKET} -> {TARGET_BUCKET}), "
          f"pooled rows: {np.mean(scores):.2f}% <<<")
    if realistic_scores:
        n_hits = sum(realistic_ranking_hits)
        print(f">>> AVERAGE LOPO, realistic (pct=0.0) rows only: {np.mean(realistic_scores):.2f}%  "
              f"-- weakest-module ranking match: {n_hits}/{len(realistic_scores)} <<<\n")

    final_model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    objective='reg:squarederror', random_state=42,
                                    reg_alpha=0.1, reg_lambda=1.0, **get_xgb_device_params())
    final_model.fit(X, y, sample_weight=w)
    # See module_soh_train.py's identical comment: train fast on whatever
    # device is available, but the saved/pickled model must not require a
    # GPU to be present wherever it's later loaded for inference.
    final_model.set_params(device='cpu')
    return final_model, list(X.columns)


if __name__ == "__main__":
    result = train_cross_rate_module_soh_model()
    if result:
        model, feats = result
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        joblib.dump(model, os.path.join(out_dir, 'module_soh_model_1_0c_to_0_3c.pkl'))
        joblib.dump(feats, os.path.join(out_dir, 'module_feature_names_1_0c_to_0_3c.pkl'))
        print("Saved module_soh_model_1_0c_to_0_3c.pkl")
    else:
        print("No model was trained!")
