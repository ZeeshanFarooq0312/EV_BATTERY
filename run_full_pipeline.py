"""One-command training pipeline.

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

Each run saves its models into a new, numbered version folder
(ml_pipeline/models/v2/, v3/, ...) instead of overwriting the previous
one, so an older working set of models is never lost. app.py always uses
the newest version automatically once restarted -- see
ml_pipeline/core/model_versioning.py to pin it to an older version instead.

Does NOT restart app.py automatically -- restart it yourself afterward
(printed as the final instruction) to actually start serving the new
models; app.py only loads models once at startup.

Before doing anything else, this also checks that every package the
pipeline needs (numpy, pandas, xgboost, joblib, scikit-learn) is installed,
and pip-installs whichever ones are missing automatically -- no need to know
what a "ModuleNotFoundError" is or run pip by hand. If a package
fails to auto-install (e.g. no internet connection), that's reported clearly
and the script stops instead of crashing later with a confusing traceback
deep inside some other file.

Usage (from new_tech/):
    python run_full_pipeline.py
"""

import importlib
import os
import subprocess
import sys
import time
import traceback

# (import name, pip install name) -- only what THIS pipeline (cleaning +
# the 3 active training scripts) needs; app.py has its own extra
# dependencies (fastapi, uvicorn, matplotlib, ...) checked when it starts.
REQUIRED_PACKAGES = [
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('xgboost', 'xgboost'),
    ('joblib', 'joblib'),
    ('sklearn', 'scikit-learn'),
]


def ensure_packages_installed():
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append((import_name, pip_name))

    if not missing:
        return

    print("Some required packages are missing -- installing them now "
          "(this only needs to happen once):")
    for import_name, pip_name in missing:
        print(f"  Installing {pip_name} ...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', pip_name],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"\nERROR: couldn't automatically install '{pip_name}'.")
            print("pip's error output:")
            print(result.stderr.strip()[-2000:])
            print(f"\nFix this (check your internet connection / permissions), then either:")
            print(f"  - re-run this script, or")
            print(f"  - install it yourself: {sys.executable} -m pip install {pip_name}")
            sys.exit(1)
        print(f"  {pip_name} installed successfully.")
    print("All required packages are now installed.\n")


ensure_packages_installed()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ML_PIPELINE_DIR = os.path.join(APP_DIR, 'ml_pipeline')
CORE_DIR = os.path.join(ML_PIPELINE_DIR, 'core')
TRAINING_DIR = os.path.join(ML_PIPELINE_DIR, 'training')

sys.path.insert(0, CORE_DIR)
from raw_file_cleaner import clean_all_uploads, DATA_FOLDER  # noqa: E402
from model_versioning import create_new_version_dir  # noqa: E402

# (training script filename, model files it should produce in this run's version folder)
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

    # One new version folder for this whole run -- all 3 scripts save into
    # it (via the EV_MODEL_OUTPUT_DIR env var) instead of each creating
    # their own. The previous version(s) are left untouched, so a bad
    # retrain never destroys a known-good set of models.
    version_dir, version_name = create_new_version_dir()
    print(f"Saving this run's models into a new version: {version_name} ({version_dir})")
    env = dict(os.environ, EV_MODEL_OUTPUT_DIR=version_dir)

    outcomes = []
    for script_name, expected_outputs in TRAINING_SCRIPTS:
        print(f"\n--- Running {script_name} ---")
        script_path = os.path.join(TRAINING_DIR, script_name)
        result = subprocess.run([sys.executable, script_path], cwd=TRAINING_DIR, env=env)
        ok = result.returncode == 0
        outcomes.append((script_name, ok, expected_outputs))
        if not ok:
            print(f"!! {script_name} exited with an error (code {result.returncode}) -- "
                  f"see output above. Continuing with the next script.")
    return outcomes, version_dir, version_name


def main():
    print("EV Battery Model Training Pipeline")
    print("This cleans your raw dataset and retrains all 3 active models in one go.\n")

    raw_folder = prompt_for_raw_folder()
    start_time = time.time()

    cleaned, skipped = run_cleaning_step(raw_folder)
    if not cleaned:
        print("\nNo files were cleaned successfully -- nothing to train on. Stopping.")
        sys.exit(1)

    outcomes, version_dir, version_name = run_training_step()
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

    print(f"\nModel training (saved into version {version_name}: {version_dir}):")
    any_failed = False
    for script_name, ok, expected_outputs in outcomes:
        print(f"  [{'OK' if ok else 'FAILED'}] {script_name}")
        if not ok:
            any_failed = True
            continue
        for out_file in expected_outputs:
            exists = os.path.exists(os.path.join(version_dir, out_file))
            print(f"      -> {out_file}: {'saved' if exists else 'MISSING -- check output above'}")

    print(f"\nTotal time: {elapsed_min:.1f} minutes")

    if any_failed:
        print("\nOne or more training scripts failed -- scroll up for the error, fix it, and re-run this script.")
        print(f"(Whatever DID save landed in {version_name} -- your previous, already-working version is untouched.)")
    else:
        print("\nAll models retrained successfully.")

    print(f"\nNEXT STEP -- restart the app so it picks up the new models ({version_name}):")
    print("  cd new_tech")
    print("  python app.py")
    print(f"\nThe app always uses the newest version automatically, so it will pick up {version_name} on its own.")
    print("If the new version turns out worse, roll back without retraining anything: put the old")
    print("version's name (e.g. \"v1\") into ml_pipeline/models/ACTIVE_VERSION and restart the app.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(1)
    except Exception as e:
        # Anything unexpected (disk full, unwritable folder, a real bug,
        # etc.) -- surface a clear headline before the traceback instead of
        # letting a bare Python stack trace be the only thing shown.
        print(f"\n\nPIPELINE FAILED: {e.__class__.__name__}: {e}")
        print("Full error details:")
        traceback.print_exc()
        sys.exit(1)
