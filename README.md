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
- For every module: predicted vs. actual capacity/SOH at the two standard voltage cutoffs, **3.25V and 2.5V**

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

Each run saves its models into a new, numbered folder (`ml_pipeline/models/v2/`, `v3/`, ...) instead of overwriting the previous ones, so nothing is ever lost. **After it finishes, restart the app** (`python app.py`) — it always picks up the newest version automatically.

If a new retrain turns out worse, you don't need to retrain again to undo it: open `ml_pipeline/models/ACTIVE_VERSION` and put the older version's name in it (e.g. `v1`), then restart the app. It'll keep using that version until you change it back to `latest`.

## Testing the API directly (Postman)

The app can also be tested endpoint-by-endpoint in Postman (or curl) instead of through the web page — useful for checking a specific result without uploading files by hand each time. The full list of endpoints, what to send, and what comes back is in [ARCHITECTURE.md](ARCHITECTURE.md#api-reference-postman--curl).

## Project structure

```
new_tech/
├── app.py                  # The web app
├── run_full_pipeline.py    # One command: clean new data + retrain everything
├── templates/index.html    # Web page
├── clean_data_for_test/    # Training data currently in use
├── raw_uploads/            # Drop new raw data files here before cleaning
├── generated_plots/        # Every plot any API route generates is saved here as a .png
└── ml_pipeline/            # Training scripts, saved models, and shared code
    └── models/
        ├── ACTIVE_VERSION  # Which version to use -- "latest" (default) or a specific one like "v1"
        ├── v1/             # Each retrain adds a new version folder like this one; older ones are kept
        └── v2/
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full breakdown of every file and how the models work.
