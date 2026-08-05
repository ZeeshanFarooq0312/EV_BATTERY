# Architecture & Technical Reference

This is the technical/developer companion to [README.md](README.md) — how the system works internally, why it's built this way, and details you'll need if you're changing the code (not just running it). If you just want to run the app or retrain on new data, README.md is all you need.

## Why this exists

A full characterization test (**FFCT** — Full Function/Characterisation Test) takes hours and fully discharges the pack to measure its true capacity. That's impractical to run often in the field. A **short test** (**SFT/SFCT** — Short Function/Characterisation Test) only captures the tail end of a discharge and takes minutes.

This project trains models to go from a short test to:
1. Which of the pack's 9 modules is **weakest** (in a series-connected pack, the whole pack's capacity is set by its weakest module — not an average across modules),
2. That module's **SOH / capacity at fixed voltage cutoffs** (3.25V and 2.5V), and
3. Its reconstructed **complete voltage-vs-capacity discharge curve**, including the **discharge knee** — the sharp voltage drop near end-of-discharge.

...and, via a separate dedicated model, the same weakest-module SOH/capacity/curve reconstruction **at 0.3C from a 1.0C-only short test** (see [Cross-rate prediction](#cross-rate-prediction-10c--03c) below).

## The core insight: module-level, not pack-level

A "pack" is just a container — 9 modules × 12 cells = 108 cells, wired in series. The real unit of prediction is the **module**: each of a pack's 9 modules degrades independently and has its own SOH. Six real packs therefore give ~54 real module-level training examples per C-rate, not 6. Every model in this pipeline treats rows at the module level, and `pack_id` is explicitly excluded from the features fed to any model — it's only used for grouping (Leave-One-Pack-Out validation) and for computing sibling features (a module's voltage relative to its own pack's other 8 modules), never as a model input. This is what lets the models generalize to a pack they've never seen.

Because a pack's discharge always stops when its single weakest cell crosses the low safety cutoff, a full-curve (FFCT) file directly gives the *real, measured* capacity of whichever module owns that weakest cell — the other 8 modules' true capacity has to be estimated via tiered extrapolation (see `module_capacity_extrapolation.py`).

## How it works (pipeline)

> **UI status**: the Same-Rate Analysis section described immediately below is currently hidden in `templates/index.html` (wrapped in a `<div style="display:none">` right after the page header, with a comment marking exactly where it starts/ends) — only [Cross-Rate Prediction](#cross-rate-prediction-10c--03c) is visible on the page right now. The model, the `/analyze` route in `app.py`, and all the code below are unaffected and still fully working — only the HTML section is hidden. To bring it back: open `templates/index.html`, find the `<!-- Same-Rate Analysis: temporarily hidden. ... -->` comment, and remove that wrapper `<div style="display:none">` and its matching closing `</div>` (marked `<!-- end Same-Rate Analysis (display:none wrapper) -->`, right before the "Cross-Rate Tool" section divider).

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
│ extrapolation.py)           │  → capacity/SOH at 3.25V AND 2.5V, from the SFT
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

**Capacity/SOH at fixed voltage cutoffs (3.25V, 2.5V), per module** (`analyze_modules_cross_rate()` in `app.py`): same idea as the same-rate route's `target_cutoffs`, but the "predicted" side can't reuse that route's tiered-extrapolation method — there's no 0.3C voltage trace at all at inference time (only the 1.0C SFT), the same constraint that makes the plot above need shape transfer in the first place. So "predicted @ cutoff" here means: build that same shape-transferred 0.3C curve for THIS module (`reconstruct_cross_rate_module_curve`, applied per-module rather than just for the weakest one), then read off where it crosses each target voltage (`_interp_target_from_reconstructed_curve` — plain interpolation, no tiered fallback needed since a shape-transferred curve already spans the full 0–100% range by construction). "Actual @ cutoff" (when a 0.3C FFT is uploaded) still uses the real tiered-extrapolation method (`estimate_module_capacity_at_targets`) against that FFT directly, exactly like the same-rate route, since that ground truth IS a real 0.3C curve.

## Key concepts

- **SOH formula**: `SOH % = (capacity_Ah / NOMINAL_CAPACITY) * 100`, where `NOMINAL_CAPACITY = 156.0 Ah`.
- **Weakest module = pack bottleneck**: in a series string, pack-level `AHDischarge` numerically equals the weakest module's own capacity. This is why the module SOH model's headline "pack SOH" is `min()` across its 9 module predictions, not an average.
- **Two voltage cutoffs, one pass**: 3.25V (a partial/usable-range cutoff) and 2.5V (near-total depletion, essentially where the FFT test itself stops) are both computed from a single tiered-extrapolation call — no need to pick one upfront.
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
│   │   ├── compute_device.py           # Picks GPU (CUDA) if usable, else CPU with every core —
│   │   │                                #   single source of truth, used by all 3 active training scripts
│   │   ├── model_versioning.py         # Resolves which models/vN/ folder app.py should load from
│   │   │                                #   (latest, or a pinned version), and hands training scripts
│   │   │                                #   a fresh version folder to save into — see "Model versioning"
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
│   ├── models/                         # Versioned trained *.pkl artifacts — see Models below and
│   │   │                                # "Model versioning"
│   │   ├── ACTIVE_VERSION              # "latest" (default, auto-picks the highest vN) or a pinned
│   │   │                                #   version name like "v1" — see model_versioning.py
│   │   ├── v1/                         # One full set of the 5 active models, from one training run
│   │   └── v2/                         # ... each retrain adds a new vN folder; older ones untouched
│   ├── models_legacy/                  # Disabled/legacy *.pkl artifacts, kept out of models/ so they
│   │                                    #   never get mixed up with what app.py actually loads
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

All five models below are **active** — loaded by `app.py` at startup and used live. Each is retrained independently (no ordering dependency between them); see [README.md's retrain instructions](README.md#add-new-data--retrain-the-models).

### Model versioning

Every full retrain (`run_full_pipeline.py`, or the 3 training scripts run together by hand) writes its output into a new folder, `ml_pipeline/models/v1/`, `v2/`, `v3/`, ... — never overwriting a previous version's files. This means a retrain that turns out worse than the previous one doesn't destroy anything; the old version is still sitting there.

- **`app.py` picks the highest-numbered version automatically** every time it starts (see `resolve_active_models_dir()` in `ml_pipeline/core/model_versioning.py`), printing which one it picked (`Using models version: v2 (...)`) as it starts up.
- **To pin it to a specific version instead** (e.g. roll back after a bad retrain, or A/B compare two versions), put that version's name — just `v1`, nothing else — into `ml_pipeline/models/ACTIVE_VERSION`, then restart the app. It stays pinned to that version, even through later retrains adding `v3`, `v4`, etc., until `ACTIVE_VERSION` is changed back to `latest` (or the file is deleted).
- **Running a training script standalone** (not via `run_full_pipeline.py`) also creates its own new version folder, via `get_training_output_dir()` — unless the `EV_MODEL_OUTPUT_DIR` environment variable is set, which is how `run_full_pipeline.py` makes all 3 scripts in one pipeline run land in the *same* new version folder instead of 3 separate ones.
- Each version folder also gets a `TRAINED_AT.txt` recording when it was created.

Below, model paths are written without the version folder (e.g. `ml_pipeline/models/module_soh_model_{0_3c,1_0c}.pkl`) since the filenames themselves don't change between versions — only which `vN/` folder they live in.

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

These are **not** part of the active pipeline (their loading code in `app.py` is commented out / hardcoded to `None`). They're kept because their training scripts still work and may be useful for comparison or as an emergency fallback — but training on new data should generally ignore these two and focus on the five active models above.

- **Legacy pack-level SOH model** (`ml_pipeline/models_legacy/soh_model_{0_3c,1_0c}.pkl`) — predicts pack-level SOH directly (no per-module breakdown). Trained by `ml_pipeline/training/soh_models_train.py` (saves into `models_legacy/`, not `models/`). To re-enable: uncomment the `soh_models`/`soh_feature_names` loading block near the top of `app.py` (search for "1. Load SOH Models") — it already points at `MODELS_LEGACY_DIR`.
- **Legacy pack-level curve reconstruction** (`ml_pipeline/models_legacy/reconstruction_model_{0_3C,1_0C}_v10_knee.pkl`) — reconstructs a pack-mean-voltage curve rather than a per-module one. Trained by `ml_pipeline/training/curve_train.py` (saves into `models_legacy/`, not `models/`). To re-enable: uncomment the `recon_model_0_3c`/`recon_model_1_0c` loading block near the top of `app.py` (search for "2. Load Reconstruction Models") — it already points at `MODELS_LEGACY_DIR`.

## GPU vs CPU for training

All 3 active training scripts automatically use a CUDA GPU if one is present and actually usable, falling back to CPU (using every available core) otherwise — no flag or config needed either way. This is handled once, centrally, by `ml_pipeline/core/compute_device.py`: it does a real (tiny) trial fit rather than just checking whether a GPU is *visible*, since a driver/CUDA/xgboost-build mismatch can leave a GPU visible but not actually usable by XGBoost — you'll see `[compute_device] GPU (CUDA) detected and usable` or `[compute_device] No usable GPU (...) -- falling back to CPU with N core(s)` printed once near the start of each training run either way.

Training may use the GPU, but every **saved** `.pkl` model is explicitly reset to `device='cpu'` before being written to disk (see the `set_params(device='cpu')` / `est.set_params(device='cpu')` calls at the end of each training script) — so `app.py` never needs a GPU to load or serve predictions from these models, only training benefits from one being present.

## Retraining manually (step by step)

`run_full_pipeline.py` (documented in README.md) is the normal way to retrain — it does everything below in one command. This section is for retraining one specific model, or cleaning data without retraining.

### Retrain one specific model

```bash
cd new_tech/ml_pipeline/training
python module_soh_train.py            # trains module_soh_model_{0_3c,1_0c}.pkl
python module_curve_train.py          # trains module_curve_reconstruction_model_{0_3C,1_0C}.pkl
python module_soh_cross_rate_train.py # trains module_soh_model_1_0c_to_0_3c.pkl

# legacy/disabled -- only needed if you've re-enabled these in app.py (see Models above)
python curve_train.py        # trains reconstruction_model_{0_3C,1_0C}_v10_knee.pkl
python soh_models_train.py   # trains soh_model_{0_3c,1_0c}.pkl
```

The 3 active scripts each save their `.pkl` output(s) into a **new** `ml_pipeline/models/vN/` folder (see "Model versioning" above) — run standalone like this (not via `run_full_pipeline.py`), each one gets its *own* new version folder rather than sharing one, so a version may end up with only 1 or 2 of the 5 active model files in it if you only run one script. The legacy scripts save into `ml_pipeline/models_legacy/` as before (not versioned). Each script prints its own Leave-One-Pack-Out validation as it runs — check that output against the numbers documented in this file / the script's own docstring before trusting a retrain. **`app.py` only loads models at process startup — restart it after any retrain, or it will keep silently serving the old model from memory.**

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

### Cleaning raw data manually

```bash
# 1. Put your new raw CSV(s) in new_tech/raw_uploads/ (create the folder if it doesn't exist yet),
#    then clean them -- this trims each file down to just the real discharge window (drops the
#    pre-test charge/rest and post-test rest sections) and writes the result straight into the
#    active training data folder:
cd new_tech/ml_pipeline/core
python raw_file_cleaner.py ../../raw_uploads   # cleans every CSV in raw_uploads/ into DATA_FOLDER
# (pass a single file path instead of a folder to clean just one file)

python raw_file_cleaner.py path/to/one_raw_file.csv     # single file
python raw_file_cleaner.py path/to/a_folder_of_raw/      # every *.csv in that folder
python raw_file_cleaner.py ../../raw_uploads --output-dir some/other/folder   # override the output location
```

(`new_raw_file.py` at the project root is kept only as a backward-compatible wrapper around this same module — prefer `raw_file_cleaner.py` directly.)

The same cleaning is also reachable as an API route, `POST /upload_training_data` (`app.py`) — accepts one or more files, cleans each with this same module, and reports per-file results as JSON. Not currently exposed anywhere in the web UI; it's there for scripting/automation if needed.

**Filename convention**: `curve_utils.parse_pack_and_crate` reads pack ID and C-rate straight from the filename, so it must contain:
- `pkN` — the pack ID (e.g. `pk7` for a brand-new pack — new pack IDs work with no code changes), and
- a C-rate token like `1.0C` / `0.3C` / `1C` somewhere in the name, and
- `FFCT` or `FFT` for a full/characterization test (ground truth, full curve to cutoff), **or** `SFT`/`SFCT` for a short test (the field-input-shaped file).
- Example: `pk7-45-01012027-FFCT-0.3C 202701011200 Characterisation Test.csv`

**What "clean" means**: compare any file in `BS_Data/` (raw) against the same pack/test in `new_tech/clean_data_for_test/OneDrive_1_7-9-2026_CLEANED/` (cleaned) to see the difference directly. A raw export bundles the pre-test charge, a rest period, the real discharge, and a post-test rest/recharge all in one file; "cleaning" means keeping **only the rows where the pack is actually discharging** (`LoadUnitCurrent` below a small negative threshold, for the longest contiguous run — see `find_discharge_block` in `raw_file_cleaner.py`) and adding the `min_v` column (row-wise min across all 108 `Cell NNN` columns) that every cleaned file already carries. If a file's `LoadUnitCurrent` column is already all-negative (no charge/rest sections), it's already clean — cleaning it again is harmless (it becomes a no-op trim, `min_v` gets added if missing).

## API reference (Postman / curl)

Every route below is a plain HTTP endpoint on the running app (`http://127.0.0.1:5000`) — test any of them directly in Postman (or curl) without going through the web page. File inputs use `multipart/form-data` (Postman: Body → form-data → set the field's type to **File**); the two training routes use a plain JSON body (Postman: Body → raw → JSON).

**Viewing a plot**: every route that generates one returns it two ways — `plot`, a base64-encoded PNG you can render inline (in Postman, open the response's **Visualize** tab and paste `<img src="data:image/png;base64,{{response.plot}}">`), and `plot_path`, the absolute path where that same image was also saved as a real `.png` file on the server, under `generated_plots/`. Easiest option: just open `plot_path` directly in an image viewer/file browser on the machine running the app — nothing to decode.

### Inference

**`POST /analyze`** — same-rate: upload an SFT and an FFT captured at the *same* C-rate.
- Form-data fields: `sft_file` (file), `fft_file` (file)
- Response: `{ soh, capacity, actual_soh, actual_capacity, weakest_module_predicted, weakest_module_actual, modules: [...], mae, pred_knee_ah, actual_knee_ah, plot, plot_path }` — `modules` is a list of 9 entries (`module_idx`, `predicted_soh`, `actual_soh`, `predicted_capacity`, `actual_capacity`, `label_source`).
- Errors: `400` (missing/invalid files, no model for that C-rate), `500` (`{ error }`)

**`POST /analyze_cross_rate`** — cross-rate: upload a 1.0C SFT + a 0.3C FFT (ground truth, optional-in-spirit but currently required by this route).
- Form-data fields: `sft_file` (file), `fft_file` (file)
- Response: `{ soh, capacity, actual_soh, actual_capacity, weakest_module_predicted, weakest_module_actual, modules: [...], plot, plot_path }` — each entry in `modules` also carries `targets` (predicted/actual capacity + SOH at 3.25V and 2.5V).
- Errors: `400`, `500`

**`POST /analyze_cross_rate_weakest_module`** — cross-rate, SFT-only: upload just a 1.0C SFT to predict every module's 0.3C SOH/capacity, no 0.3C FFT ground-truth file needed.
- Form-data field: `sft_file` (file)
- Response: `{ soh, capacity, weakest_module_predicted, modules: [...], plot, plot_path }` — predictions only, no `actual_*`/ground-truth fields at all (nothing to validate against without an FFT, so they're left out rather than sent as `null`). Each entry in `modules` is `{ module_idx, predicted_soh, predicted_capacity, targets: {"3.25": {capacity_ah, soh}, "2.5": {capacity_ah, soh}} }`.
- Errors: `400`, `500`

**`POST /analyze_weakest_module`** — same-rate, SFT-only: no ground-truth file needed.
- Form-data field: `sft_file` (file)
- Response: `{ weakest_module, predicted_soh_by_module: {"1": 96.2, ...}, results: {"3.25": {...}, "2.5": {...}}, template_used, plot, plot_path }`
- Errors: `400`, `500`

**`POST /upload_training_data`** — cleans one or more raw CSVs into the training data folder (does **not** retrain).
- Form-data field: `files` (one or more files)
- Response: `{ results: [{filename, status, raw_rows, cleaned_rows, pack_id, c_rate}, ...], cleaned_count, total_count, data_folder }`

### Training

**`POST /train_models`** — starts a full retrain (clean + train all 3 active models) as a background job; returns immediately instead of blocking for the 5-30 minutes it actually takes.
- JSON body: `{ "raw_folder": "<path to a raw dataset folder on the server>" }`
- Response: `{ job_id, status: "running", poll_at: "/train_status/<job_id>" }`
- Errors: `400` if `raw_folder` doesn't exist on the server

**`GET /train_status/{job_id}`** — poll this with the `job_id` from above.
- Response while running: `{ job_id, status: "running", started_at, raw_folder, log_tail }` (`log_tail` is the last ~4000 characters of live progress output)
- Response when done: adds `finished_at`, `elapsed_minutes`, `returncode`, and `status` becomes `"completed"` or `"failed"` (plus `error` if it crashed outright)
- Errors: `404` if the job_id is unknown (the job list is in-memory and resets if the app restarts — the training run itself is unaffected, only status polling is lost)
- After a successful run, restart the app to start serving the new model version (it's picked up automatically — see "Retraining manually" above).

## Known limitations

- **Data scarcity**: only 6 real packs. A shared, shallow (regularized) tree model has limited capacity to fit one pack's narrow SOH range without affecting predictions for other packs nearby in feature space — reweighting one pack's data to fix its calibration reliably costs a little accuracy on its closest real neighbor. Fixing this for real needs either more real packs or a two-stage architecture (separate ranking model from absolute-level model), not further data-reweighting. This also limits how much a per-session feature (temperature, actual measured current) can help even when well-motivated — see the cross-rate model's docstring for two documented attempts that didn't survive validation for exactly this reason.
- **Ranking is only as reliable as the true margin**: for a pack whose weakest and 2nd-weakest modules are genuinely nearly tied in real SOH (sub-1-point margins, occasionally exact ties given the extrapolation method's own precision), which one a model calls "weakest" is close to a coin flip — and for the cross-rate model specifically, can even flip between two different real test *sessions* of the same physical pack (see pk4 in the cross-rate section above). This is a property of the pack's actual physical balance and test-to-test noise, not a model defect.
- **Curve-shape fidelity ≠ prediction accuracy**: a synthetic-generator fix that measurably improved how closely synthetic curves visually/statistically match real ones did not automatically improve live SOH-prediction accuracy in testing — the two are related but distinct things to validate.
- **LOPO alone isn't sufficient validation**: for the cross-rate model specifically, a "pooled" LOPO metric (averaged across every training slice) has repeatedly diverged from the "realistic" sub-metric (only the exact whole-file slice production actually sends) — always check both, and prefer the realistic one when they disagree.
