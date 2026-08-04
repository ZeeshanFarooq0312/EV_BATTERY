"""Model version management.

Every full retrain writes its output models into their own numbered folder
(ml_pipeline/models/v1/, v2/, v3/, ...) instead of overwriting the previous
set. That means an older, already-working set of models is never lost just
because a new retrain happened -- it's still sitting there in its own
folder if the new one turns out worse.

app.py picks the highest-numbered version automatically every time it
starts. To pin it to a specific version instead (e.g. roll back to v2),
edit ml_pipeline/models/ACTIVE_VERSION and put that version's name in it
(just "v2", nothing else) -- app.py will keep using that exact version,
even after new retrains add v3, v4, etc., until ACTIVE_VERSION is changed
back to "latest" (or deleted).
"""
import os
import re
import time

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(os.path.dirname(CORE_DIR), 'models')
ACTIVE_VERSION_FILE = os.path.join(MODELS_ROOT, 'ACTIVE_VERSION')
_VERSION_RE = re.compile(r'^v(\d+)$')


def _existing_versions():
    if not os.path.isdir(MODELS_ROOT):
        return []
    versions = []
    for name in os.listdir(MODELS_ROOT):
        m = _VERSION_RE.match(name)
        if m and os.path.isdir(os.path.join(MODELS_ROOT, name)):
            versions.append(int(m.group(1)))
    return sorted(versions)


def resolve_active_models_dir():
    """Directory app.py should load models from: (path, version_name).

    version_name is None only in the fallback case where no versioned
    folders exist at all yet (a pre-versioning checkout) -- then it falls
    back to the flat models/ folder itself.
    """
    versions = _existing_versions()

    pinned = None
    if os.path.isfile(ACTIVE_VERSION_FILE):
        with open(ACTIVE_VERSION_FILE) as f:
            pinned = f.read().strip()

    if pinned and pinned.lower() != 'latest':
        pinned_dir = os.path.join(MODELS_ROOT, pinned)
        if os.path.isdir(pinned_dir):
            return pinned_dir, pinned
        print(f"[model_versioning] ACTIVE_VERSION names '{pinned}', but "
              f"ml_pipeline/models/{pinned} doesn't exist -- falling back to the latest version.")

    if versions:
        latest = f'v{versions[-1]}'
        return os.path.join(MODELS_ROOT, latest), latest

    return MODELS_ROOT, None


def create_new_version_dir():
    """A fresh, empty folder for a new training run's output: (path, version_name)."""
    versions = _existing_versions()
    next_n = (versions[-1] + 1) if versions else 1
    version_name = f'v{next_n}'
    version_dir = os.path.join(MODELS_ROOT, version_name)
    os.makedirs(version_dir, exist_ok=True)
    with open(os.path.join(version_dir, 'TRAINED_AT.txt'), 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
    return version_dir, version_name


def get_training_output_dir():
    """Directory a training script should save its output model(s) into.

    When run_full_pipeline.py trains all 3 active scripts together, they
    need to land in the SAME new version folder rather than 3 separate
    ones -- run_full_pipeline.py creates the version dir once up front and
    passes it down to each training subprocess via the EV_MODEL_OUTPUT_DIR
    environment variable. Running a training script by hand (that env var
    unset) just creates its own fresh version instead.
    """
    env_dir = os.environ.get('EV_MODEL_OUTPUT_DIR')
    if env_dir:
        os.makedirs(env_dir, exist_ok=True)
        return env_dir
    version_dir, _ = create_new_version_dir()
    return version_dir
