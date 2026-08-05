"""Picks the XGBoost compute device once per training run: GPU (CUDA) if
one is actually usable, otherwise CPU with every available core. Single
source of truth for this so every training script gets identical
GPU-detection behavior instead of five separate copies drifting apart.

Detection does a real (tiny) fit rather than just checking whether a GPU is
*visible* (e.g. via nvidia-smi) -- a driver/CUDA/xgboost-build mismatch can
leave a GPU visible but unusable by XGBoost specifically, and the only way
to catch that is to actually try training on it.
"""

import os

import numpy as np
import xgboost as xgb

_cached_device = None


def get_xgb_device_params():
    """Returns XGBRegressor kwargs for whichever device this machine can
    actually train on right now: {'device': 'cuda'} if a real CUDA GPU is
    detected AND XGBoost can fit on it, else {'device': 'cpu', 'n_jobs': N}
    with N = every CPU core available, for the fastest CPU fallback."""
    global _cached_device
    if _cached_device is None:
        _cached_device = _detect_device()
    if _cached_device == 'cuda':
        return {'device': 'cuda'}
    return {'device': 'cpu', 'n_jobs': os.cpu_count() or 1}


def _detect_device():
    try:
        probe = xgb.XGBRegressor(device='cuda', tree_method='hist', n_estimators=2)
        probe.fit(np.zeros((4, 2)), np.zeros(4))
        print("[compute_device] GPU (CUDA) detected and usable -- training will use the GPU.")
        return 'cuda'
    except Exception as e:
        print(f"[compute_device] No usable GPU ({e.__class__.__name__}: {e}) -- "
              f"falling back to CPU with {os.cpu_count()} core(s).")
        return 'cpu'
