# EV Battery SOH & Discharge Curve Reconstruction

Machine-learning pipeline that estimates the **State of Health (SOH)** of an EV battery pack and **reconstructs its complete discharge voltage curve** (0–100% of capacity) from a short field test — without needing to run a full multi-hour characterization test.

## Why this exists

A full characterization test (**FFCT** — Full Function/Characterisation Test) takes hours and fully discharges the pack to measure its true capacity. That's impractical to run often in the field. A **short test** (**SFT/SFCT** — Short Function/Characterisation Test) only captures the tail end of a discharge and takes minutes.

This project trains models to go from a short test to:
1. An estimated **SOH / capacity**, and
2. A reconstructed **complete voltage-vs-capacity discharge curve**, including the **discharge knee** — the sharp voltage drop near end-of-discharge.

## How it works (two-part pipeline)

```
Short test (SFT/SFCT file)
        │
        ▼
┌───────────────────────┐
│ Part 1: SOH model     │  shape features (voltage drop, cell imbalance,
│ (soh_model_*.pkl)     │  temperature, dV/dAh stats, C-rate) → predicted SOH%
└───────────┬───────────┘
            │  predicted capacity = (SOH / 100) * 156 Ah
            ▼
┌───────────────────────┐
│ Part 2: Reconstruction│  observed short-test voltage samples + shape
│ model (reconstruction_│  features + predicted SOH → full discharge curve
│ model_*_v10_knee.pkl) │  (57 checkpoints spanning 0-100% of capacity)
└───────────┬───────────┘
            │
            ▼
   Complete global discharge curve + detected knee point
```

Two **separate models are trained per C-rate** (0.3C and 1.0C) for both parts, because discharge current strongly shapes the curve (see [Key findings](#key-findings-from-the-data) below).

## Key concepts

- **SOH formula**: `SOH % = (actual_capacity_delivered_Ah / NOMINAL_CAPACITY) * 100`, where `NOMINAL_CAPACITY = 156.0 Ah`. Actual capacity is the pack's measured `AHDischarge` at the end of a full discharge. This replaced an earlier hardcoded per-pack lookup table.
- **The "complete global curve"**: the model predicts voltage at 57 fixed checkpoints spanning the *entire* discharge (0–100% of total capacity), not just the portion missing from a short test. Earlier versions only reconstructed the missing head segment and spliced it onto the real short-test data.
- **The discharge knee**: the point of sharpest voltage drop near the end of discharge. Detected via a chord-distance method — the point in the last half of the curve with maximum perpendicular distance from the straight line connecting that window's endpoints (a standard "elbow detection" technique).
- **Pack configuration**: 108 series-connected cells, 16 temperature sensors (see raw CSV columns below).

## Key findings from the data

These were empirically measured from the real pack data and directly shaped the model design:

| Finding | Value | Where it's used |
|---|---|---|
| Initial IR-drop at start of discharge (first 1% of capacity) | ~30mV @ 0.3C vs ~130–150mV @ 1.0C (4–5x sharper under load) | Synthetic curve generation (`EXTRA_IR_DROP_PER_SOH_POINT`) |
| Discharge knee location | ~83–92% of total capacity, for both C-rates | Checkpoint/sampling density allocation, knee-region diagnostics |
| Real pack SOH range | ~86.7%–95.9% across 5 packs | Synthetic data extrapolates down to an 80% SOH floor |

## Directory structure

```
new_tech/
├── app.py                        # Flask web app (the interactive demo)
├── templates/index.html          # Web UI
├── curve_train.py (TEST_CASE/)   # Trains the curve-reconstruction models
├── SOH_MODELS_TRAIN.PY (TEST_CASE/) # Trains the SOH/capacity models
├── daignos.py (TEST_CASE/)       # Same-session diagnostic: per-checkpoint error table
├── curve_test.py                 # Standalone end-to-end validation script (plots + MAE)
├── soh_calculate.py               # One-off: computes actual SOH per file from FFCT capacity
├── plot_fft.py, side_by_side.py, data_overview.py  # Ad-hoc plotting/exploration utilities
├── fft_raw_data/                 # Real FULL discharge tests (FFCT) — training ground truth
├── stf_raw_data/                 # Real SHORT discharge tests (SFT/SFCT) — simulated field input
├── raw_dataset/                  # Combined raw copy of fft_raw_data + stf_raw_data
├── clean_data_for_test/          # Newer cleaned data batch (includes pk6, not yet in fft/stf split)
├── TEST_CASE/                    # Trained model artifacts + their training/eval scripts
│   ├── soh_model_{0_3c,1_0c}.pkl
│   ├── feature_names_{0_3c,1_0c}.pkl
│   └── reconstruction_model_{0_3C,1_0C}_v10_knee.pkl
└── uploads/                      # Flask upload scratch folder
```

## Data format

Each raw CSV is a time series from a pack cycler with columns:

```
Timestamp, Cell 001 ... Cell 108, Temperature 001 ... Temperature 016,
AhCharge, AHDischarge, WhCharge, WhDischarge,
LoadUnitCurrent, LoadUnitVoltage, LoadUnitPower, SoC, PackCurrent,
isolation_resistance, min_v
```

Filenames encode pack ID, test type, and C-rate, e.g.:
`pk1-62-08062021-FFCT-0.3C 202605151215 Characterisation Test.csv`
`pk4-60pc-29052021-SFCT-1.0C 202606110707 Characterisation Test.csv`

- `pkN` — pack identifier
- `FFCT`/`FFT` — full characterization test (used as ground truth / full-curve training target)
- `SFT`/`SFCT` — short (function) test (simulates the field input)
- `0.3C` / `0.95C` / `1.0C` — discharge C-rate (0.95C is treated as 1.0C throughout)

All curve extraction filters to the valid discharge window (`2.0V < mean cell voltage < 4.15V`) and re-zeros `AHDischarge` to start at 0 for each curve.

## Models

### Part 1 — SOH / capacity model (`TEST_CASE/soh_model_{0_3c,1_0c}.pkl`)

- **Algorithm**: XGBRegressor (300 trees, depth 3), one independent model per C-rate.
- **Input** (19 features, listed in `feature_names_{0_3c,1_0c}.pkl`): start/end voltage, voltage drop (raw and C-rate normalized), cell-to-cell imbalance stats, temperature mean/rise, `delta_Ah` (raw and normalized), dV/dAh mean/std/min, C-rate, `is_sfct` flag, slice start percentage.
- **Output**: predicted SOH (%). Capacity = `(SOH / 100) * 156`.
- **Validation**: Leave-One-Pack-Out cross-validation (trains on 4 packs, tests on the 5th, rotated).
- **Trained by**: `TEST_CASE/SOH_MODELS_TRAIN.PY`.

### Part 2 — Curve reconstruction model (`TEST_CASE/reconstruction_model_{0_3C,1_0C}_v10_knee.pkl`)

- **Algorithm**: `MultiOutputRegressor` wrapping XGBRegressor (600 trees, depth 5, lr 0.015) — effectively **57 independent regressors**, one per output checkpoint.
- **Input** (69 features): 53 real voltage samples of the observed short-test segment (non-uniformly spaced — original ~2.5% density kept everywhere, doubled density in the last ~34% of the segment where the knee usually falls) + SOH + slope/curvature/plateau/imbalance summary stats + explicit knee-location features (`tail_knee_pct`, `tail_knee_slope`) + the Ah offset where the short-test segment sits within the full curve.
- **Output** (57 checkpoints): voltage at fixed percentages of total capacity (0–100%), non-uniformly spaced the same way — original 2.5% density everywhere, extra density at the very start (initial IR-drop) and from 71.25–100% (discharge knee).
- **Trained by**: `TEST_CASE/curve_train.py`.

### Training data construction

For each real FFCT (full curve), the training script simulates a short test by cutting the curve at 12 different points (20%–75% of capacity) — everything after the cut is the "observed" input, the full curve (at the 57 checkpoints, spanning 0–100%) is the prediction target. On top of the ~20 real curves, synthetic curves are generated by rescaling a real curve to a lower target SOH (down to an 80% floor) and adding a C-rate-specific extra IR sag proportional to how far below the template's own SOH we're extrapolating — giving ~800 training rows per C-rate.

## Setup

```bash
# needs xgboost, scikit-learn, pandas, numpy, scipy, joblib, flask, matplotlib
pip install xgboost scikit-learn pandas numpy scipy joblib flask matplotlib
```

> This project was developed/tested against the `ai_guru` conda environment on this machine (the default `base` env doesn't have `xgboost` installed).

## Usage

### Run the web app

```bash
cd new_tech
python app.py
# open http://127.0.0.1:5000
```

Upload a short-test CSV (SFT/SFCT) and its corresponding full-test CSV (FFCT, used only for ground-truth comparison in the UI), pick a voltage cutoff, and click **Analyze Battery**. The dashboard shows:

- Predicted vs actual SOH and capacity
- Reconstruction MAE (mV)
- Capacity remaining at the chosen voltage cutoff
- **Predicted vs actual discharge knee point** (Ah), with the location error, and both marked with stars on the curve plot
- The full overlay plot: real FFCT curve vs. the ML-reconstructed complete global curve

### Retrain the models

```bash
cd new_tech/TEST_CASE
python SOH_MODELS_TRAIN.PY   # trains soh_model_{0_3c,1_0c}.pkl
python curve_train.py        # trains reconstruction_model_{0_3C,1_0C}_v10_knee.pkl
```

Both read from `../fft_raw_data` by default (only FFCT/FFT-labeled files are used for curve training; the SOH trainer filters similarly).

### Run diagnostics

```bash
cd new_tech/TEST_CASE
python daignos.py
```

Slices every real FFCT file at a fixed 50% cutoff and reports per-checkpoint average/max error (mV), plus a knee-region (72–100%) summary. This measures accuracy under training-like conditions (the "tail" comes from the same curve being predicted).

```bash
cd new_tech
python curve_test.py
```

A harder, more realistic end-to-end test: predicts SOH from a **real, independently-recorded** short-test file, reconstructs the full curve, and validates against that pack's **real** FFCT file. Displays an overlay plot and prints MAE/RMSE.

## Known limitation: cross-session generalization

The reconstruction model is trained by slicing **one** real FFCT curve into an "observed tail" + "target," so both halves always come from the same recording. In production, the observed short test and the full test it's validated against are **two independently-run sessions** on the same pack — which can differ slightly in starting rest voltage, temperature, and other conditions that a same-curve-slice training setup never exposes the model to.

Testing against genuine SFT-vs-FFCT file pairs (rather than same-curve slices) shows this gap: errors are noisier and occasionally much larger (tens of mV) than the same-session diagnostic suggests, for both old and current model versions. This is a separate, deeper issue from checkpoint/knee resolution — fixing it would mean training on real matched short-test/full-test pairs (only a handful exist per C-rate today) blended in alongside the simulated slices, rather than relying on simulated slices alone.

## Model version history

| Version | Change |
|---|---|
| v7 | Predicts only the missing head segment (percentage-of-head checkpoints), splices onto the real observed tail |
| v8 | Predicts the complete global curve (checkpoints as % of total capacity) instead of just the head; SOH switched from a hardcoded per-pack table to the `capacity/156*100` formula; synthetic data added down to an 80% SOH floor with C-rate-specific IR-drop modeling |
| v9 | *(superseded — reallocating checkpoint density away from the mid-curve plateau toward the knee region regressed cross-session accuracy; not shipped)* |
| v10 | Fixed a bug where the short-test input samples were a straight line between the segment's first/last voltage instead of the actual observed curve; added explicit knee-location features (`tail_knee_pct`, `tail_knee_slope`); added knee-region resolution **additively** on top of the original density, rather than reallocating it |
