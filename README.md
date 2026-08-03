# EV Battery SOH & Discharge Curve Reconstruction

Machine-learning pipeline that estimates the **State of Health (SOH)** of an EV battery pack at the **module level**, identifies its **weakest module** (the one that determines the whole pack's usable capacity), and **reconstructs its complete discharge voltage curve** (0–100% of capacity) — all from a short field test (SFT/SFCT), without needing to run a full multi-hour characterization test (FFCT/FFT). It also supports a harder **cross-rate** case: predicting a module's SOH/capacity **at 0.3C** using only a **1.0C short test** as input, for when a full same-rate baseline isn't available.

## Why this exists

A full characterization test (**FFCT** — Full Function/Characterisation Test) takes hours and fully discharges the pack to measure its true capacity. That's impractical to run often in the field. A **short test** (**SFT/SFCT** — Short Function/Characterisation Test) only captures the tail end of a discharge and takes minutes.

This project trains models to go from a short test to:
1. Which of the pack's 9 modules is **weakest** (in a series-connected pack, the whole pack's capacity is set by its weakest module — not an average across modules),
2. That module's **SOH / capacity at fixed voltage cutoffs** (3.2V and 2.5V), and
3. Its reconstructed **complete voltage-vs-capacity discharge curve**, including the **discharge knee** — the sharp voltage drop near end-of-discharge.

...and, via a separate dedicated model, the same weakest-module SOH/capacity/curve reconstruction **at 0.3C from a 1.0C-only short test** (see [Cross-rate prediction](#cross-rate-prediction-10c--03c) below).

## The core insight: module-level, not pack-level

A "pack" is just a container — 9 modules × 12 cells = 108 cells, wired in series. The real unit of prediction is the **module**: each of a pack's 9 modules degrades independently and has its own SOH. Six real packs therefore give ~54 real module-level training examples per C-rate, not 6. Every model in this pipeline treats rows at the module level, and `pack_id` is explicitly excluded from the features fed to any model — it's only used for grouping (Leave-One-Pack-Out validation) and for computing sibling features (a module's voltage relative to its own pack's other 8 modules), never as a model input. This is what lets the models generalize to a pack they've never seen.

Because a pack's discharge always stops when its single weakest cell crosses the low safety cutoff, a full-curve (FFCT) file directly gives the *real, measured* capacity of whichever module owns that weakest cell — the other 8 modules' true capacity has to be estimated via tiered extrapolation (see `module_capacity_extrapolation.py`).

## How it works (pipeline)

```
Uploaded SFT file (partial, tail-only) + FFT file (full, ground truth)
        │
        ▼
┌────────────────────────────┐
│ Per-module SOH model       │  9 module-level feature rows (voltage drop,
│ (module_soh_model_*.pkl)   │  cell imbalance, dV/dAh shape, sibling rank
│                             │  relative to the pack's other 8 modules) →
│                             │  predicted SOH% per module
└─────────────┬───────────────┘
              │ weakest module = min(predicted SOH across the 9)
              ▼
┌────────────────────────────┐
│ Tiered capacity            │  measured (if the module's curve reaches the
│ extrapolation               │  target cutoff) → self-extrapolated exponential
│ (module_capacity_           │  decay → cross-module shape-transfer fallback
│ extrapolation.py)           │  → capacity/SOH at 3.2V AND 2.5V, from the SFT
└─────────────┬───────────────┘  (predicted) and the FFT (actual ground truth)
              │
              ▼
┌────────────────────────────┐
│ Module curve reconstruction │  one smooth model-predicted curve, direct
│ (module_curve_train.py's    │  from the SFT's own observed shape + the
│  model, reconstruct_module_ │  module's predicted capacity anchor — 57
│  curve() in app.py)         │  checkpoints across the whole 0→100% curve
└─────────────┬───────────────┘
              ▼
   Weakest-module voltage curve plot: predicted vs actual, both cutoffs marked
```

Two **separate module SOH models are trained per C-rate** (0.3C and 1.0C), because discharge current strongly shapes the curve (IR-drop sag scales with current — see `physics_calibration.py`).

### Cross-rate prediction (1.0C → 0.3C)

A separate, dedicated pipeline for a harder case: predicting a module's **0.3C** SOH/capacity/curve using only a **1.0C** short test as input — no 0.3C data available at all at prediction time.

```
Uploaded SFT file (1.0C, partial) [+ optional FFT file (0.3C, ground truth for validation only)]
        │
        ▼
┌────────────────────────────┐
│ Cross-rate module SOH model │  same per-module feature extraction as above,
│ (module_soh_model_1_0c_to_  │  but the model was trained to map 1.0C-shaped
│  0_3c.pkl)                  │  features directly to a 0.3C label
└─────────────┬───────────────┘
              │ weakest module = min(predicted 0.3C SOH across the 9)
              ▼
┌────────────────────────────┐
│ Shape-transfer curve        │  no 0.3C voltage shape exists at inference
│ reconstruction               │  time to feed a checkpoint regressor -- so
│ (reconstruct_cross_rate_    │  this borrows the SAME PACK's own real
│  module_curve() in app.py)  │  historical 0.3C curve SHAPE and rescales its
│                              │  Ah-axis to the predicted 0.3C capacity
└─────────────┬───────────────┘
              ▼
   Weakest-module voltage curve plot: predicted (from 1.0C SFT) vs actual (from 0.3C FFT, if uploaded)
```

This is fundamentally harder than same-rate prediction, and the model/UI are honest about that rather than hiding it:
- The 0.3C-vs-1.0C capacity gap is real but **pack-dependent**, not a fixed constant or a smooth function of SOH.
- The **weakest module's identity is not rate-invariant** — some packs have a different weakest module at 0.3C than at 1.0C.
- When two modules are genuinely near-tied in real capacity (sub-1-point margins), **which one a single real 1.0C test session calls "weakest" can flip** between two otherwise-valid captures of the same physical pack (confirmed on pk4: two real 1.0C sessions three weeks apart, ~6°C apart in pack temperature, disagree on whether module 1 or module 7 is weaker). This is a property of the physical pack and test-to-test measurement noise, not a model defect — see `module_soh_cross_rate_train.py`'s docstring for the full investigation, including two tried-and-reverted attempts to fix it (adding temperature and actual measured current as features — see `get_actual_c_rate.py` — neither survived validation).
- **Trained by**: `ml_pipeline/training/module_soh_cross_rate_train.py`, using `ml_pipeline/training/module_perturbation_generator.py` for synthetic augmentation (each synthetic instance is a real (pack, module) curve pair, lightly rescaled + noised — anchored to real data by construction, not synthesized from scratch).

## Key concepts

- **SOH formula**: `SOH % = (capacity_Ah / NOMINAL_CAPACITY) * 100`, where `NOMINAL_CAPACITY = 156.0 Ah`.
- **Weakest module = pack bottleneck**: in a series string, pack-level `AHDischarge` numerically equals the weakest module's own capacity. This is why the module SOH model's headline "pack SOH" is `min()` across its 9 module predictions, not an average.
- **Two voltage cutoffs, one pass**: 3.2V (a partial/usable-range cutoff) and 2.5V (near-total depletion, essentially where the FFT test itself stops) are both computed from a single tiered-extrapolation call — no need to pick one upfront.
- **SFT Ah-axis offset**: SFT files are tail-only captures — their `AHDischarge` is zeroed at wherever that short test happened to start, not at full charge. Every module-level SFT computation anchors on that module's own predicted capacity (`ah_offset = predicted_capacity - sft_local_span`) to align the local axis to the true global one.
- **IR-drop/relaxation transient**: SFT tests start from a rested state, producing a brief, real (not noise) sharp voltage sag that settles into the normal gentle plateau slope. `curve_utils.detect_settle_index` trims this before using SFT data as a curve-reconstruction anchor, so the predicted/observed join doesn't show an artificial spike.
- **Tail-protected smoothing**: `app.py`'s `_smooth_checkpoints` median-filters the reconstructed curve to remove noise, but excludes the last `tail_protect` (default 5) checkpoints from that filter. The steep discharge knee near end-of-life is a genuine feature, not noise — median-filtering it flattened the knee into a "step then plunge" artifact, so the tail is now left untouched.
- **Synthetic augmentation is physics-informed, not template-copying**: `synthetic_module_generator.py` builds independently-sampled *new* 108-cell virtual-pack curves (own capacity, knee timing, IR-sag, cell imbalance — all drawn from ranges measured in `physics_calibration.py`) rather than perturbing one real template's feature row. Every physical constant it samples from (plateau voltage, voltage span, IR-sag magnitude, knee-fraction range, cell imbalance) is measured from the real data, not hardcoded.
- **Leading-glitch-row trim**: a handful of real files show a single anomalous first row — pack mean voltage reads normal but one or more modules' *minimum* cell voltage reads >1V low, recovering by the very next row (a DAQ/relay-closure sensor artifact, not real physics; confirmed on pk6's 0.3C FFCT file). `curve_utils.extract_and_resample_curve` drops this generically (checked on every file via `GLITCH_ROW_DROP_THRESHOLD_V`, not hardcoded to any one pack) before any downstream consumer — training data, templates, or live inference — ever sees it.
- **Filename C-rate label ≠ actual measured current**: a file named `...-1.0C...` isn't necessarily running at exactly 156A (1C × 156Ah nominal capacity) — real sessions can differ by several percent (`ml_pipeline/diagnostics/get_actual_c_rate.py` computes the true value from the file's own `LoadUnitCurrent` column). This was investigated as a possible fix for the cross-rate ranking-flip issue above; it didn't help enough to keep (see that section), but the discrepancy itself is real and worth knowing about when interpreting any file's nominal C-rate.

## Directory structure

```
new_tech/
├── app.py                              # FastAPI web app (the interactive demo) — the only entry point
│                                        # that matters for live serving
├── run_full_pipeline.py                # ★ One-command retrain: prompts for a raw dataset folder,
│                                        #   cleans it, retrains all 3 active models, prints a summary
├── templates/index.html                # Web UI
├── clean_data_for_test/OneDrive_.../   # THE active training data folder (all 6 packs, both C-rates)
├── raw_uploads/                        # Drop zone for new raw CSVs awaiting cleaning (not read by
│                                        #   any training script directly — see raw_file_cleaner.py)
├── uploads/                            # File-upload scratch folder (live SFT/FFT inference uploads)
│
├── ml_pipeline/
│   ├── core/                           # Shared utilities imported by BOTH training and app.py at
│   │   │                                # runtime -- self-contained (only import each other)
│   │   ├── curve_utils.py              # Curve extraction, feature extraction, knee/settle
│   │   │                                #   detection, glitch-row trim — imported everywhere
│   │   ├── module_capacity_extrapolation.py # Tiered (measured/self-extrapolated/cross-module)
│   │   │                                #   capacity estimation at arbitrary target voltages
│   │   ├── build_module_dataset.py     # Real per-module capacity/SOH ground-truth table +
│   │   │                                #   real weakest-module template curves (load_module_templates)
│   │   │                                #   + DATA_FOLDER (the single source of truth for where
│   │   │                                #   every script reads raw CSVs from)
│   │   ├── raw_file_cleaner.py         # Cleans a raw pack CSV (or a whole folder of them, e.g.
│   │   │                                #   raw_uploads/) down to just the real discharge window —
│   │   │                                #   the single source of truth for that cleaning step, used
│   │   │                                #   by both the project-root new_raw_file.py CLI and (once
│   │   │                                #   wired up) any future UI "add training data" upload route
│   │   └── physics_calibration.py      # Single source of truth for every real-data-measured
│   │                                    #   constant the synthetic generator uses
│   │
│   ├── training/                       # Training scripts + synthetic-data generators. Each script
│   │   │                                # adds core/ to sys.path itself; sibling imports within this
│   │   │                                # folder resolve automatically when run directly.
│   │   ├── module_soh_train.py         # ★ ACTIVE: trains module_soh_model_{0_3c,1_0c}.pkl —
│   │   │                                #   same-rate per-module SOH model
│   │   ├── synthetic_module_generator.py # Physics-informed synthetic module data (default
│   │   │                                #   augmentation for module_soh_train.py)
│   │   ├── module_curve_train.py       # ★ ACTIVE: trains module_curve_reconstruction_model_
│   │   │                                #   {0_3C,1_0C}.pkl — same-rate per-module curve reconstruction
│   │   ├── module_soh_cross_rate_train.py # ★ ACTIVE: trains module_soh_model_1_0c_to_0_3c.pkl —
│   │   │                                #   cross-rate (1.0C SFT → 0.3C) per-module SOH model
│   │   ├── module_perturbation_generator.py # Default synthetic augmentation for the cross-rate
│   │   │                                #   model (real curve pairs, lightly rescaled + noised)
│   │   ├── synthetic_cross_rate_generator.py # Older cross-rate synthetic generator (bootstrap-
│   │   │                                #   based) — kept as an opt-in fallback (synth_mode=
│   │   │                                #   'physics_generator' in module_soh_cross_rate_train.py)
│   │   │
│   │   ├── curve_train.py              # DISABLED/legacy: trains reconstruction_model_{0_3C,1_0C}_
│   │   │                                #   v10_knee.pkl (pack-level curve reconstruction) — not
│   │   │                                #   loaded by app.py; kept as a documented fallback. Still
│   │   │                                #   actively imported by module_curve_train.py above, though
│   │   │                                #   (reuses its synthetic-curve generation) — don't delete.
│   │   └── soh_models_train.py         # DISABLED/legacy: trains soh_model_{0_3c,1_0c}.pkl —
│   │                                    #   pack-level SOH fallback; not loaded by app.py by default
│   │                                    #   (see Models below for how to re-enable either)
│   │
│   ├── diagnostics/                    # Validation gates + standalone diagnostic tools. Each adds
│   │   │                                # both core/ and training/ to sys.path itself.
│   │   ├── get_actual_c_rate.py        # Scans a folder of raw CSVs and reports each file's ACTUAL
│   │   │                                #   C-rate from its own measured current, not the filename label
│   │   ├── validate_module_extrapolation.py     # Acceptance gate for the tiered capacity method
│   │   ├── validate_synthetic_module_generator.py # Acceptance gate for the synthetic generator
│   │   │                                #   (distributional sanity + generalization + LOPO)
│   │   └── visualize_synthetic_vs_real.py # Visual overlay: synthetic vs real discharge curves,
│   │                                    #   per C-rate — catches shape/level issues the statistical
│   │                                    #   gate alone can miss
│   │
│   ├── models/                         # Every trained *.pkl artifact (active + legacy) — see Models below
│   └── generated_outputs/              # *.png / diagnostic *.csv byproducts (gitignored — regenerate
│                                        #   by re-running the diagnostics above, nothing here is source)
│
└── new_raw_file.py (project root)      # Backward-compat CLI wrapper around
                                         # ml_pipeline/core/raw_file_cleaner.py -- prefer that module directly
```

## Data format

Each raw CSV is a time series from a pack cycler with columns:

```
Timestamp, Cell 001 ... Cell 108, Temperature 001 ... Temperature 016,
AhCharge, AHDischarge, WhCharge, WhDischarge,
LoadUnitCurrent, LoadUnitVoltage, LoadUnitPower, SoC, PackCurrent,
isolation_resistance, min_v
```

108 cells = 9 modules × 12 cells each. `min_v` is the row-wise min across all 108 `Cell NNN` columns.

Filenames encode pack ID, test type, and C-rate, e.g.:
`pk1-62-08062021-FFCT-0.3C 202605151215 Characterisation Test.csv`
`pk4-60pc-29052021-SFCT-1.0C 202606110707 Characterisation Test.csv`

- `pkN` — pack identifier
- `FFCT`/`FFT` — full characterization test (ground truth / full-curve)
- `SFT`/`SFCT` — short (function) test (simulates the field input)
- `0.3C` / `0.95C` / `1.0C` — discharge C-rate (0.95C is treated as 1.0C throughout)

All curve extraction filters to the valid discharge window (`2.0V < mean cell voltage < 4.15V`) and re-zeros `AHDischarge` to start at 0 for each curve (`curve_utils.extract_and_resample_curve`).

## Models

All five models below are **active** — loaded by `app.py` at startup and used live. Each is retrained independently (no ordering dependency between them); see [Retrain the models](#retrain-the-models).

### Module SOH model (`ml_pipeline/models/module_soh_model_{0_3c,1_0c}.pkl`) — same-rate, primary

- **Algorithm**: XGBRegressor (300 trees, depth 3, L1/L2 regularized), one independent model per C-rate.
- **Input**: per-module voltage/shape features (start/end voltage, voltage drop, cell imbalance, dV/dAh mean/std/min, `ah_per_voltage_drop`) plus sibling features (`rel_end_voltage`, `sibling_rank` — this module vs. its own pack's other 8) plus C-rate/slice metadata. `pack_id` and `module_idx` are never features.
- **Training data**: real module rows (every real file, sliced at multiple depths including the exact whole-file slice the live app sends) + synthetic module rows (`synthetic_module_generator.py`, physics-informed, real weight 1.0 vs synthetic weight 0.4).
- **Validation**: Leave-One-Pack-Out (all 9 of a pack's modules held out together) + a live end-to-end pipeline test (upload real SFT+FFT pairs through the actual route) — LOPO alone has repeatedly missed real regressions in this project (e.g. a train/serve feature mismatch that LOPO's in-distribution sampling couldn't see), so both are required before deploying a retrain. `DEFAULT_TRAINING_SEED=11` was itself chosen by sweeping seeds and checking which one gets 4 independently-known-correct packs right on the FINAL (all-data) model — a different, easier check than LOPO's held-out-pack generalization test, and both should be checked after any retrain.
- **Trained by**: `ml_pipeline/training/module_soh_train.py`.

### Module curve reconstruction (`ml_pipeline/models/module_curve_reconstruction_model_{0_3C,1_0C}.pkl`) — same-rate

- One smooth model predicts all 57 checkpoints of a module's complete 0→100%-capacity voltage curve directly from its observed SFT tail, replacing an older head/observed/tail three-piece stitch that showed a visible seam at the joins.
- Trained only on each real file's own **weakest** module (the only one with real, complete, uncensored data all the way to cutoff — see module docstring) plus synthetic augmentation (`curve_train.generate_synthetic_curve`, reused unchanged from the pack-level generator).
- **Trained by**: `ml_pipeline/training/module_curve_train.py`.

### Cross-rate module SOH model (`ml_pipeline/models/module_soh_model_1_0c_to_0_3c.pkl`)

- Predicts a module's **0.3C** SOH/capacity from features extracted at **1.0C** — see [Cross-rate prediction](#cross-rate-prediction-10c--03c) above for the full pipeline and known limitations.
- **Trained by**: `ml_pipeline/training/module_soh_cross_rate_train.py`. Its docstring is the canonical record of every approach tried for this model (including several that looked good on paper but didn't survive validation) — read it before changing this model's feature set or synthetic generator.

### Disabled/legacy models — kept as documented fallback, not loaded by `app.py` by default

These are **not** part of the active pipeline (their loading code in `app.py` is commented out / hardcoded to `None`). They're kept because their training scripts still work and may be useful for comparison or as an emergency fallback — but a client training on new data should generally ignore these two and focus on the five active models above.

- **Legacy pack-level SOH model** (`ml_pipeline/models/soh_model_{0_3c,1_0c}.pkl`) — predicts pack-level SOH directly (no per-module breakdown). Trained by `ml_pipeline/training/soh_models_train.py`. To re-enable: uncomment the `soh_models`/`soh_feature_names` loading block near the top of `app.py` (search for "1. Load SOH Models").
- **Legacy pack-level curve reconstruction** (`ml_pipeline/models/reconstruction_model_{0_3C,1_0C}_v10_knee.pkl`) — reconstructs a pack-mean-voltage curve rather than a per-module one. Trained by `ml_pipeline/training/curve_train.py`. To re-enable: uncomment the `recon_model_0_3c`/`recon_model_1_0c` loading block near the top of `app.py` (search for "2. Load Reconstruction Models").

## Setup

```bash
# needs xgboost, scikit-learn, pandas, numpy, scipy, joblib, matplotlib, fastapi, uvicorn, python-multipart
pip install xgboost scikit-learn pandas numpy scipy joblib matplotlib fastapi uvicorn python-multipart
```

> Developed/tested against the `ai_guru` conda environment on this machine (the base env doesn't have `xgboost`). `app.py` is a FastAPI app served by `uvicorn` (migrated from an earlier Flask version) — `python-multipart` is required for the file-upload routes even though nothing imports it directly.

## Usage

### Run the web app

```bash
cd new_tech
python app.py
# open http://127.0.0.1:5000
```

Upload a short-test CSV (SFT/SFCT) and its corresponding full-test CSV (FFCT, used as ground truth). The dashboard shows the weakest module (predicted vs actual), its capacity/SOH at 3.2V and 2.5V (predicted vs actual), and a plot of its complete voltage curve — model-reconstructed vs the real FFT ground truth.

The **Per-Module Analysis** section shows every one of the pack's 9 modules side by side, each bar labeled with both predicted and actual **SOH% and capacity (Ah)** (e.g. `93.83% · 146.37Ah`), plus a hover tooltip naming the actual-value's source (`measured` from the FFT curve directly, or `estimated` via tiered extrapolation when that module's curve doesn't reach the target cutoff).

A separate **Cross-Rate Prediction** section further down the page takes only a 1.0C SFT (plus an optional 0.3C FFT for validation) and predicts the same per-module SOH/capacity/curve at 0.3C — see [Cross-rate prediction](#cross-rate-prediction-10c--03c) above.

### Retrain the models

Each active model is retrained independently — no ordering dependency, and each script reads directly from the raw CSVs in `clean_data_for_test/OneDrive_1_7-9-2026_CLEANED/` by default (see [Adding new training data](#adding-new-training-data) below for retraining on a different/expanded dataset).

```bash
cd new_tech/ml_pipeline/training
python module_soh_train.py            # trains module_soh_model_{0_3c,1_0c}.pkl
python module_curve_train.py          # trains module_curve_reconstruction_model_{0_3C,1_0C}.pkl
python module_soh_cross_rate_train.py # trains module_soh_model_1_0c_to_0_3c.pkl

# legacy/disabled -- only needed if you've re-enabled these in app.py (see Models above)
python curve_train.py        # trains reconstruction_model_{0_3C,1_0C}_v10_knee.pkl
python soh_models_train.py   # trains soh_model_{0_3c,1_0c}.pkl
```

Every one of the above saves its `.pkl` output(s) into `ml_pipeline/models/`. Each script prints its own Leave-One-Pack-Out validation as it runs — check that output against the numbers documented in this README / the script's own docstring before trusting a retrain. **`app.py` only loads models at process startup — restart it after any retrain, or it will keep silently serving the old model from memory.**

### Run validation gates

```bash
cd new_tech/ml_pipeline/diagnostics
python validate_module_extrapolation.py       # tiered capacity-extrapolation accuracy
python validate_synthetic_module_generator.py # synthetic generator: distributional sanity,
                                                # synthetic-only generalization, blended LOPO
python visualize_synthetic_vs_real.py          # saves synthetic_vs_real_{0_3C,1_0C}.png to
                                                # ../generated_outputs/ -- visual sanity check
                                                # the statistical gates can miss
```

### Diagnostics

```bash
cd new_tech/ml_pipeline/diagnostics
python get_actual_c_rate.py   # edit FOLDER_PATH at the top of the file first --
                               # reports every CSV's ACTUAL C-rate (from measured
                               # current) vs its filename label, saved to a CSV
```

## Adding new training data

### One-command pipeline (recommended for non-technical users)

`run_full_pipeline.py` (project root of this app, `new_tech/`) does the whole thing in one go: cleaning + retraining all 3 active models, with no arguments to remember and no other files to touch.

```bash
cd new_tech
python run_full_pipeline.py
```

It asks you one question — the path to your raw dataset folder — then:
1. Cleans every CSV in that folder into the active training data folder (skips and reports any file it can't clean, e.g. a bad filename or a file that isn't really a discharge test, rather than stopping the whole run).
2. Retrains all 3 active models one after another. Each training script already applies its own synthetic data augmentation automatically as part of training — there's no separate augmentation step.
3. Prints a summary at the end: which files were cleaned/skipped and why, which models trained successfully (with the exact `.pkl` filenames it confirmed were written), and how long the whole run took. If one training script fails, the other two still run — the summary tells you which one needs attention.

It does **not** restart the app automatically — the summary's last line always tells you to do that yourself (`python app.py`), since that's the step that actually puts the new models into use.

Everything below this describes the same pipeline broken into its individual manual steps — useful if you want to clean a folder without retraining, retrain without adding new data, or run one specific script.

### Manual steps — where files go and what to run

```bash
# 1. Put your new raw CSV(s) in new_tech/raw_uploads/ (create the folder if it doesn't exist yet),
#    then clean them -- this trims each file down to just the real discharge window (drops the
#    pre-test charge/rest and post-test rest sections) and writes the result straight into the
#    active training data folder:
cd new_tech/ml_pipeline/core
python raw_file_cleaner.py ../../raw_uploads   # cleans every CSV in raw_uploads/ into DATA_FOLDER
# (pass a single file path instead of a folder to clean just one file)

# 2. Retrain the 3 active models, in any order:
cd ../training
python module_soh_train.py            # trains module_soh_model_{0_3c,1_0c}.pkl
python module_curve_train.py          # trains module_curve_reconstruction_model_{0_3C,1_0C}.pkl
python module_soh_cross_rate_train.py # trains module_soh_model_1_0c_to_0_3c.pkl

# 3. Restart the app so it picks up the new .pkl files:
cd ../..
python app.py
```

Step by step, with the detail behind each part above:

1. **Where the raw file goes**: drop new raw CSVs (still-unclean exports — see below) into `new_tech/raw_uploads/`, a scratch drop zone for files awaiting cleaning (not read by any training script directly). If a file is *already* cleaned (see step 3), you can skip the drop zone and copy it straight into `new_tech/clean_data_for_test/OneDrive_1_7-9-2026_CLEANED/` — that's the one folder every training script actually reads from (via `DATA_FOLDER` in `ml_pipeline/core/build_module_dataset.py`; change it there once to use a different folder entirely — every script imports it from that single place).
2. **Filename convention**: `curve_utils.parse_pack_and_crate` reads pack ID and C-rate straight from the filename, so it must contain:
   - `pkN` — the pack ID (e.g. `pk7` for a brand-new pack — new pack IDs work with no code changes, see step 5 below), and
   - a C-rate token like `1.0C` / `0.3C` / `1C` somewhere in the name, and
   - `FFCT` or `FFT` for a full/characterization test (ground truth, full curve to cutoff), **or** `SFT`/`SFCT` for a short test (the field-input-shaped file).
   - Example: `pk7-45-01012027-FFCT-0.3C 202701011200 Characterisation Test.csv`
3. **Raw vs. cleaned — what "clean" means**: compare any file in `BS_Data/` (raw) against the same pack/test in `new_tech/clean_data_for_test/OneDrive_1_7-9-2026_CLEANED/` (cleaned) to see the difference directly. A raw export bundles the pre-test charge, a rest period, the real discharge, and a post-test rest/recharge all in one file; "cleaning" means keeping **only the rows where the pack is actually discharging** (`LoadUnitCurrent` below a small negative threshold, for the longest contiguous run — see `find_discharge_block` in the script below) and adding the `min_v` column (row-wise min across all 108 `Cell NNN` columns) that every cleaned file already carries. If a file's `LoadUnitCurrent` column is already all-negative (no charge/rest sections), it's already clean — cleaning it again is harmless (it becomes a no-op trim, `min_v` gets added if missing).
4. **How to clean it**: `ml_pipeline/core/raw_file_cleaner.py` does this — run it against a single raw file or a whole folder (e.g. `raw_uploads/`, cleaning every CSV in it in one pass) and it writes the cleaned CSV(s) straight into `DATA_FOLDER` by default:
   ```bash
   cd new_tech/ml_pipeline/core
   python raw_file_cleaner.py path/to/one_raw_file.csv     # single file
   python raw_file_cleaner.py path/to/a_folder_of_raw/      # every *.csv in that folder
   python raw_file_cleaner.py ../../raw_uploads --output-dir some/other/folder   # override the output location
   ```
   (`new_raw_file.py` at the project root is kept only as a backward-compatible wrapper around this same module — prefer `raw_file_cleaner.py` directly.)

   The same cleaning is also reachable as an API route, `POST /upload_training_data` (`app.py`) — accepts one or more files, cleans each with this same module, and reports per-file results as JSON. Not currently exposed anywhere in the web UI (CLI is the supported path for now); it's there for scripting/automation if needed.
5. **Which scripts to run — all three, every time**: run all three active training scripts in [Retrain the models](#retrain-the-models) (order doesn't matter, no dependency between them). A brand-new `pkN` is automatically picked up as a new Leave-One-Pack-Out fold — nothing else to configure. Check each script's printed LOPO output against the numbers documented in this README/the script's own docstring before trusting the retrain.
6. **Restart required**: `app.py` only loads `.pkl` models once at startup, so it must be restarted after any retrain — otherwise it keeps serving the old models from memory even though the files on disk changed.

You do **not** need to touch the two legacy/disabled training scripts (`curve_train.py`, `soh_models_train.py`) unless you've specifically re-enabled those models in `app.py` (see [Disabled/legacy models](#disabledlegacy-models--kept-as-documented-fallback-not-loaded-by-apppy-by-default) above).

## Known limitations

- **Data scarcity**: only 6 real packs. A shared, shallow (regularized) tree model has limited capacity to fit one pack's narrow SOH range without affecting predictions for other packs nearby in feature space — reweighting one pack's data to fix its calibration reliably costs a little accuracy on its closest real neighbor. Fixing this for real needs either more real packs or a two-stage architecture (separate ranking model from absolute-level model), not further data-reweighting. This also limits how much a per-session feature (temperature, actual measured current) can help even when well-motivated — see the cross-rate model's docstring for two documented attempts that didn't survive validation for exactly this reason.
- **Ranking is only as reliable as the true margin**: for a pack whose weakest and 2nd-weakest modules are genuinely nearly tied in real SOH (sub-1-point margins, occasionally exact ties given the extrapolation method's own precision), which one a model calls "weakest" is close to a coin flip — and for the cross-rate model specifically, can even flip between two different real test *sessions* of the same physical pack (see pk4 in the cross-rate section above). This is a property of the pack's actual physical balance and test-to-test noise, not a model defect.
- **Curve-shape fidelity ≠ prediction accuracy**: a synthetic-generator fix that measurably improved how closely synthetic curves visually/statistically match real ones did not automatically improve live SOH-prediction accuracy in testing — the two are related but distinct things to validate.
- **LOPO alone isn't sufficient validation**: for the cross-rate model specifically, a "pooled" LOPO metric (averaged across every training slice) has repeatedly diverged from the "realistic" sub-metric (only the exact whole-file slice production actually sends) — always check both, and prefer the realistic one when they disagree.
