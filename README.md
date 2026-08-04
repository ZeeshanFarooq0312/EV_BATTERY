# EV Battery SOH & Discharge Curve Reconstruction

This tool predicts the health of an EV battery pack from a short test, instead of requiring a full multi-hour test. Given a short test file, it finds the weakest module in the pack, estimates its remaining capacity and State of Health (SOH), and reconstructs its full discharge curve.

For a deeper technical explanation of how it works, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup

Install the required packages once:

```bash
pip install xgboost scikit-learn pandas numpy scipy joblib matplotlib fastapi uvicorn python-multipart
```

## Running the app

```bash
cd new_tech
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

The **Cross-Rate Prediction** section is what's currently shown on the page: upload only a 1.0C short-test file (SFT/SFCT) to predict every module's SOH/capacity at 0.3C, using a dedicated cross-rate model. Optionally add a 0.3C full-test file (FFCT/FFT) as well to validate the prediction against real ground truth. You'll see:
- The weakest module in the pack
- Its predicted capacity/SOH compared to the actual value (if a ground-truth file was uploaded)
- Its full discharge curve, predicted vs. actual

The **Same-Rate Analysis** tool (same idea, but SFT and FFT captured at the same C-rate) is currently hidden from the page, though the underlying model and code are still there — see [ARCHITECTURE.md](ARCHITECTURE.md#how-it-works-pipeline) for what it does and how to bring it back.

## Add new data & retrain the models

If you have new test data and want the models to learn from it, run:

```bash
cd new_tech
python run_full_pipeline.py
```

This will:
1. Ask for the folder containing your new raw data files.
2. Clean the files automatically.
3. Retrain all the models on the updated data.
4. Print a summary when finished, showing what was done and whether it succeeded.

It automatically installs any missing packages, and automatically uses your computer's GPU for faster training if one is available (otherwise it uses the CPU).

**After it finishes, restart the app** (`python app.py`) so it starts using the newly trained models.

## Project structure

```
new_tech/
├── app.py                  # The web app
├── run_full_pipeline.py    # One command: clean new data + retrain everything
├── templates/index.html    # Web page
├── clean_data_for_test/    # Training data currently in use
├── raw_uploads/            # Drop new raw data files here before cleaning
└── ml_pipeline/            # Training scripts, saved models, and shared code
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full breakdown of every file and how the models work.
