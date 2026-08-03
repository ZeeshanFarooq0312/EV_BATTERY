"""Visual cross-check for synthetic_module_generator.py: overlays real
pack-mean discharge curves against a batch of synthetic virtual-pack curves,
per C-rate bucket. Complements validate_synthetic_module_generator.py's
distributional_sanity() (which only compares per-feature RANGES) -- a
generator can pass every range check and still produce a curve SHAPE no real
pack has, which is only obvious by eye.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ML_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ML_PIPELINE_DIR, 'core'))
sys.path.insert(0, os.path.join(_ML_PIPELINE_DIR, 'training'))

from curve_utils import list_full_curve_files, extract_and_resample_curve, get_cell_columns
from build_module_dataset import DATA_FOLDER
from synthetic_module_generator import generate_virtual_pack_dataframe

N_SYNTHETIC_CURVES = 15
OUT_DIR = os.path.join(_ML_PIPELINE_DIR, 'generated_outputs')


def _c_rate_bucket(c_rate):
    return '1.0C' if c_rate >= 0.9 else '0.3C'


def plot_bucket(bucket, data_folder=DATA_FOLDER, seed=7):
    rng = np.random.default_rng(seed)
    plt.figure(figsize=(10, 6))

    # Synthetic first (thin, semi-transparent) so real curves draw on top.
    for i in range(N_SYNTHETIC_CURVES):
        df, module_soh, _ = generate_virtual_pack_dataframe(bucket, rng, data_folder)
        cell_cols = get_cell_columns(df)
        pack_v = df[cell_cols].mean(axis=1).values
        ah = df['AHDischarge'].values
        plt.plot(ah, pack_v, color='#dc2626', alpha=0.25, linewidth=1,
                  label='Synthetic (physics generator)' if i == 0 else None)

    # Real curves (bold, one per pack).
    for entry in list_full_curve_files(data_folder):
        if _c_rate_bucket(entry['c_rate']) != bucket:
            continue
        ah, v, _cell_std = extract_and_resample_curve(entry['path'])
        if ah is None:
            continue
        plt.plot(ah, v, linewidth=2.2, label=f"Real: {entry['pack_id']}")

    plt.title(f'Synthetic vs Real Pack-Mean Discharge Curves — {bucket}', fontsize=13, fontweight='bold')
    plt.xlabel('Capacity Delivered (Ah)')
    plt.ylabel('Mean Cell Voltage (V)')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(1.8, 4.3)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f'synthetic_vs_real_{bucket.replace(".", "_")}.png')
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"Saved {out_path}")
    return out_path


if __name__ == '__main__':
    for bucket in ('0.3C', '1.0C'):
        plot_bucket(bucket)
