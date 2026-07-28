# EV Battery SOH & Discharge Curve Reconstruction

Machine-learning pipeline that estimates the **State of Health (SOH)** of an EV battery pack at the **module level**, identifies its **weakest module** (the one that determines the whole pack's usable capacity), and **reconstructs its complete discharge voltage curve** (0–100% of capacity) — all from a short field test (SFT/SFCT), without needing to run a full multi-hour characterization test (FFCT/FFT).

## Why this exists

A full characterization test (**FFCT** — Full Function/Characterisation Test) takes hours and fully discharges the pack to measure its true capacity. That's impractical to run often in the field. A **short test** (**SFT/SFCT** — Short Function/Characterisation Test) only captures the tail end of a discharge and takes minutes.

This project trains models to go from a short test to:
1. Which of the pack's 9 modules is **weakest** (in a series-connected pack, the whole pack's capacity is set by its weakest module — not an average across modules),
2. That module's **SOH / capacity at fixed voltage cutoffs** (3.2V and 2.5V), and
3. Its reconstructed **complete voltage-vs-capacity discharge curve**, including the **discharge knee** — the sharp voltage drop near end-of-discharge.

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
│ Curve reconstruction       │  predicted head (0 → SFT start) + observed SFT
│ (reconstruct_full_curve,   │  tail + extrapolated tail past the SFT, stitched
│  in app.py)                 │  into one complete 0→depleted curve
└─────────────┬───────────────┘
              ▼
   Weakest-module voltage curve plot: predicted vs actual, both cutoffs marked
```

Two **separate module SOH models are trained per C-rate** (0.3C and 1.0C), because discharge current strongly shapes the curve (IR-drop sag scales with current — see `physics_calibration.py`).

## Key concepts

- **SOH formula**: `SOH % = (capacity_Ah / NOMINAL_CAPACITY) * 100`, where `NOMINAL_CAPACITY = 156.0 Ah`.
- **Weakest module = pack bottleneck**: in a series string, pack-level `AHDischarge` numerically equals the weakest module's own capacity. This is why the module SOH model's headline "pack SOH" is `min()` across its 9 module predictions, not an average.
- **Two voltage cutoffs, one pass**: 3.2V (a partial/usable-range cutoff) and 2.5V (near-total depletion, essentially where the FFT test itself stops) are both computed from a single tiered-extrapolation call — no need to pick one upfront.
- **SFT Ah-axis offset**: SFT files are tail-only captures — their `AHDischarge` is zeroed at wherever that short test happened to start, not at full charge. Every module-level SFT computation anchors on that module's own predicted capacity (`ah_offset = predicted_capacity - sft_local_span`) to align the local axis to the true global one.
- **IR-drop/relaxation transient**: SFT tests start from a rested state, producing a brief, real (not noise) sharp voltage sag that settles into the normal gentle plateau slope. `curve_utils.detect_settle_index` trims this before using SFT data as a curve-reconstruction anchor, so the predicted/observed join doesn't show an artificial spike.
- **Tail-protected smoothing**: `app.py`'s `_smooth_checkpoints` median-filters the reconstructed curve to remove noise, but excludes the last `tail_protect` (default 5) checkpoints from that filter. The steep discharge knee near end-of-life is a genuine feature, not noise — median-filtering it flattened the knee into a "step then plunge" artifact, so the tail is now left untouched.
- **Synthetic augmentation is physics-informed, not template-copying**: `synthetic_module_generator.py` builds independently-sampled *new* 108-cell virtual-pack curves (own capacity, knee timing, IR-sag, cell imbalance — all drawn from ranges measured in `physics_calibration.py`) rather than perturbing one real template's feature row. Every physical constant it samples from (plateau voltage, voltage span, IR-sag magnitude, knee-fraction range, cell imbalance) is measured from the real data, not hardcoded.

## Directory structure

```
new_tech/
├── app.py                              # Flask web app (the interactive demo) — the only entry point
│                                        # that matters for live serving
├── templates/index.html                # Web UI
├── clean_data_for_test/OneDrive_.../   # THE active training data folder (all 6 packs, both C-rates)
├── raw_dataset/                        # Newly-added raw files awaiting cleaning (see new_raw_file.py)
├── uploads/                            # Flask upload scratch folder
│
├── TEST_CASE/                          # All training/validation code + trained model artifacts
│   ├── curve_utils.py                  # Shared low-level utilities (curve extraction, feature
│   │                                    # extraction, knee/settle detection) — imported everywhere
│   ├── module_capacity_extrapolation.py# Tiered (measured/self-extrapolated/cross-module) capacity
│   │                                    # estimation at arbitrary target voltages
│   ├── build_module_dataset.py         # Real per-module capacity/SOH ground-truth table
│   ├── physics_calibration.py          # Single source of truth for every real-data-measured
│   │                                    # constant the synthetic generator uses
│   │
│   ├── module_soh_train.py             # ★ Trains module_soh_model_{0_3c,1_0c}.pkl — the PRIMARY,
│   │                                    #   currently-deployed per-module SOH model
│   ├── synthetic_module_generator.py   # Physics-informed synthetic module data (default augmentation)
│   ├── curve_train.py                  # Trains reconstruction_model_{0_3C,1_0C}_v10_knee.pkl
│   ├── SOH_MODELS_TRAIN.PY             # Trains soh_model_{0_3c,1_0c}.pkl — legacy pack-level
│   │                                    #   fallback, used only if module analysis is unavailable
│   │
│   ├── validate_module_extrapolation.py     # Acceptance gate for the tiered capacity method
│   ├── validate_synthetic_module_generator.py # Acceptance gate for the synthetic generator
│   │                                             # (distributional sanity + generalization + LOPO)
│   ├── visualize_synthetic_vs_real.py  # Visual overlay: synthetic vs real discharge curves,
│   │                                    # per C-rate — catches shape/level issues the statistical
│   │                                    # gate alone can miss
│   │
│   └── *.pkl                           # Trained model artifacts (see Models below)
│
└── new_raw_file.py (project root)      # Cleans a raw pack CSV down to just the real discharge
                                         # window (drops pre-test charge/rest and post-test rest)
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

### Module SOH model (`TEST_CASE/module_soh_model_{0_3c,1_0c}.pkl`) — primary

- **Algorithm**: XGBRegressor (300 trees, depth 3, L1/L2 regularized), one independent model per C-rate.
- **Input**: per-module voltage/shape features (start/end voltage, voltage drop, cell imbalance, dV/dAh mean/std/min, `ah_per_voltage_drop`) plus sibling features (`rel_end_voltage`, `sibling_rank` — this module vs. its own pack's other 8) plus C-rate/slice metadata. `pack_id` and `module_idx` are never features.
- **Training data**: real module rows (every real file, sliced at multiple depths including the exact whole-file slice the live app sends) + synthetic module rows (`synthetic_module_generator.py`, physics-informed, real weight 1.0 vs synthetic weight 0.4).
- **Validation**: Leave-One-Pack-Out (all 9 of a pack's modules held out together) + a live end-to-end pipeline test (upload real SFT+FFT pairs through the actual Flask route) — LOPO alone has repeatedly missed real regressions in this project (e.g. a train/serve feature mismatch that LOPO's in-distribution sampling couldn't see), so both are required before deploying a retrain.
- **Trained by**: `TEST_CASE/module_soh_train.py`.

### Curve reconstruction (`TEST_CASE/reconstruction_model_{0_3C,1_0C}_v10_knee.pkl`)

- Reconstructs the complete 0→100%-capacity mean-cell-voltage curve from a short-test segment, used for the weakest module's predicted-head visualization and pack-level knee detection.
- **Trained by**: `TEST_CASE/curve_train.py`.

### Legacy pack-level SOH model (`TEST_CASE/soh_model_{0_3c,1_0c}.pkl`) — fallback only

- Used only when an uploaded SFT file is too short to extract all 9 modules' features (module analysis unavailable). Predicts pack-level SOH directly rather than per-module.
- **Trained by**: `TEST_CASE/SOH_MODELS_TRAIN.PY`.

## Setup

```bash
# needs xgboost, scikit-learn, pandas, numpy, scipy, joblib, flask, matplotlib
pip install xgboost scikit-learn pandas numpy scipy joblib flask matplotlib
```

> Developed/tested against the `ai_guru` conda environment on this machine (the base env doesn't have `xgboost`).

## Usage

### Run the web app

```bash
cd new_tech
python app.py
# open http://127.0.0.1:5000
```

Upload a short-test CSV (SFT/SFCT) and its corresponding full-test CSV (FFCT, used as ground truth). The dashboard shows the weakest module (predicted vs actual), its capacity/SOH at 3.2V and 2.5V (predicted vs actual), and a plot of its voltage curve — predicted head + observed SFT + extrapolated tail, against the real FFT ground truth.

The **Per-Module Analysis** section shows every one of the pack's 9 modules side by side, each bar labeled with both predicted and actual **SOH% and capacity (Ah)** (e.g. `93.83% · 146.37Ah`), plus a hover tooltip naming the actual-value's source (`measured` from the FFT curve directly, or `estimated` via tiered extrapolation when that module's curve doesn't reach the target cutoff).

### Retrain the models

```bash
cd new_tech/TEST_CASE
python module_soh_train.py   # trains module_soh_model_{0_3c,1_0c}.pkl -- the primary model
python curve_train.py        # trains reconstruction_model_{0_3C,1_0C}_v10_knee.pkl
python SOH_MODELS_TRAIN.PY   # trains soh_model_{0_3c,1_0c}.pkl -- legacy fallback only
```

`module_soh_train.py` reads from `clean_data_for_test/OneDrive_1_7-9-2026_CLEANED/` by default. **The Flask app only loads models at process startup — restart `app.py` after any retrain, or it will keep silently serving the old model from memory.**

### Run validation gates

```bash
cd new_tech/TEST_CASE
python validate_module_extrapolation.py       # tiered capacity-extrapolation accuracy
python validate_synthetic_module_generator.py # synthetic generator: distributional sanity,
                                                # synthetic-only generalization, blended LOPO
python visualize_synthetic_vs_real.py          # saves synthetic_vs_real_{0_3C,1_0C}.png --
                                                # visual sanity check the statistical gates can miss
```

## Known limitations

- **Data scarcity**: only 6 real packs. A shared, shallow (regularized) tree model has limited capacity to fit one pack's narrow SOH range without affecting predictions for other packs nearby in feature space — reweighting one pack's data to fix its calibration reliably costs a little accuracy on its closest real neighbor. Fixing this for real needs either more real packs or a two-stage architecture (separate ranking model from absolute-level model), not further data-reweighting.
- **Ranking is only as reliable as the true margin**: for a pack whose weakest and 2nd-weakest modules are genuinely nearly tied in real SOH (sub-0.1-point margins, occasionally exact ties given the extrapolation method's own precision), which one a model calls "weakest" is close to a coin flip — this is a property of the pack's actual physical balance, not a model defect.
- **Curve-shape fidelity ≠ prediction accuracy**: a synthetic-generator fix that measurably improved how closely synthetic curves visually/statistically match real ones did not automatically improve live SOH-prediction accuracy in testing — the two are related but distinct things to validate.
