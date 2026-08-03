"""One-command training pipeline for non-technical users.

Given a folder of raw pack CSV exports, this:
  1. Cleans every file in it (see ml_pipeline/core/raw_file_cleaner.py --
     keeps only the rows where the pack is actually discharging) into the
     active training data folder.
  2. Retrains all 3 active models. Each training script already applies its
     own physics-informed synthetic data augmentation internally (see the
     README's "Models" section) -- there is no separate augmentation step
     to run here, it happens automatically as part of training.

Files that fail to clean (bad filename, not really a discharge test, etc.)
are skipped and reported at the end rather than stopping the whole run; the
3 training scripts are independent of each other, so if one fails the other
two still run, and the failure is reported in the final summary.

Does NOT restart app.py automatically -- restart it yourself afterward
(printed as the final instruction) to actually start serving the new
models; app.py only loads models once at startup.

Usage (from new_tech/):
    python run_full_pipeline.py
"""

import os
import subprocess
import sys
import time

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ML_PIPELINE_DIR = os.path.join(APP_DIR, 'ml_pipeline')
CORE_DIR = os.path.join(ML_PIPELINE_DIR, 'core')
TRAINING_DIR = os.path.join(ML_PIPELINE_DIR, 'training')
MODELS_DIR = os.path.join(ML_PIPELINE_DIR, 'models')

sys.path.insert(0, CORE_DIR)
from raw_file_cleaner import clean_all_uploads, DATA_FOLDER  # noqa: E402

# (training script filename, model files it should produce in MODELS_DIR)
TRAINING_SCRIPTS = [
    ('module_soh_train.py', [
        'module_soh_model_0_3c.pkl', 'module_soh_model_1_0c.pkl',
    ]),
    ('module_curve_train.py', [
        'module_curve_reconstruction_model_0_3C.pkl', 'module_curve_reconstruction_model_1_0C.pkl',
    ]),
    ('module_soh_cross_rate_train.py', [
        'module_soh_model_1_0c_to_0_3c.pkl',
    ]),
]


def prompt_for_raw_folder(max_attempts=3):
    for _ in range(max_attempts):
        raw = input("Enter the path to your raw dataset folder: ").strip().strip('"').strip("'")
        if not raw:
            print("  Please enter a path.\n")
            continue
        path = os.path.expanduser(raw)
        if not os.path.isdir(path):
            print(f"  '{raw}' is not a folder that exists -- check the path and try again.\n")
            continue
        return path
    print("Too many invalid attempts -- exiting.")
    sys.exit(1)


def run_cleaning_step(raw_folder):
    print("\n" + "=" * 70)
    print("STEP 1/2: Cleaning raw files")
    print("=" * 70)
    results = clean_all_uploads(input_dir=raw_folder, output_dir=DATA_FOLDER)
    cleaned = [r for r in results if r[1] == 'ok']
    skipped = [r for r in results if r[1] != 'ok']
    return cleaned, skipped


def run_training_step():
    print("\n" + "=" * 70)
    print("STEP 2/2: Training models (each script applies its own synthetic")
    print("data augmentation automatically -- no separate step needed)")
    print("=" * 70)
    outcomes = []
    for script_name, expected_outputs in TRAINING_SCRIPTS:
        print(f"\n--- Running {script_name} ---")
        script_path = os.path.join(TRAINING_DIR, script_name)
        result = subprocess.run([sys.executable, script_path], cwd=TRAINING_DIR)
        ok = result.returncode == 0
        outcomes.append((script_name, ok, expected_outputs))
        if not ok:
            print(f"!! {script_name} exited with an error (code {result.returncode}) -- "
                  f"see output above. Continuing with the next script.")
    return outcomes


def main():
    print("EV Battery Model Training Pipeline")
    print("This cleans your raw dataset and retrains all 3 active models in one go.\n")

    raw_folder = prompt_for_raw_folder()
    start_time = time.time()

    cleaned, skipped = run_cleaning_step(raw_folder)
    if not cleaned:
        print("\nNo files were cleaned successfully -- nothing to train on. Stopping.")
        sys.exit(1)

    outcomes = run_training_step()
    elapsed_min = (time.time() - start_time) / 60

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Raw dataset folder: {raw_folder}")
    print(f"Files cleaned into training data ({DATA_FOLDER}): {len(cleaned)}")
    if skipped:
        print(f"Files skipped ({len(skipped)}):")
        for fname, status in skipped:
            print(f"  - {fname}: {status}")

    print("\nModel training:")
    any_failed = False
    for script_name, ok, expected_outputs in outcomes:
        print(f"  [{'OK' if ok else 'FAILED'}] {script_name}")
        if not ok:
            any_failed = True
            continue
        for out_file in expected_outputs:
            exists = os.path.exists(os.path.join(MODELS_DIR, out_file))
            print(f"      -> {out_file}: {'saved' if exists else 'MISSING -- check output above'}")

    print(f"\nTotal time: {elapsed_min:.1f} minutes")

    if any_failed:
        print("\nOne or more training scripts failed -- scroll up for the error, fix it, and re-run this script.")
    else:
        print("\nAll models retrained successfully.")

    print("\nNEXT STEP -- restart the app so it picks up the new models:")
    print("  cd new_tech")
    print("  python app.py")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(1)
