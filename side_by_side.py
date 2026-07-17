import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Set matplotlib backend for saving files (change to 'Qt5Agg' or comment out if running interactively)
matplotlib.use("Agg")


def extract_true_fft(df):
    """Extract discharge curve from FFT file (your custom logic)"""
    cell_cols = [
        c for c in df.columns if "Cell" in c and "Temperature" not in c
    ]
    df["Mean_Cell_Voltage"] = df[cell_cols].mean(axis=1)

    # Filter for discharge phase
    discharge_df = df[
        (df["Mean_Cell_Voltage"] < 4.2) & (df["Mean_Cell_Voltage"] > 1.5)
    ].copy()

    if "AHDischarge" in discharge_df.columns and discharge_df[
        "AHDischarge"
    ].max() > 0:
        discharge_df = discharge_df.sort_values("AHDischarge")
        discharge_df["Ah_Relative"] = (
            discharge_df["AHDischarge"] - discharge_df["AHDischarge"].iloc[0]
        )
    else:
        discharge_df["Ah_Relative"] = np.arange(len(discharge_df))

    return discharge_df["Ah_Relative"].values, discharge_df["Mean_Cell_Voltage"].values


def compare_discharge_rates(file_03c, file_10c, output_image="c_rate_comparison.png"):
    """Loads, processes, and plots 0.3C vs 1.0C discharge curves."""
    print("Loading datasets...")
    df_03c = pd.read_csv(file_03c)
    df_10c = pd.read_csv(file_10c)

    # Extract curve data
    ah_03, v_03 = extract_true_fft(df_03c)
    ah_10, v_10 = extract_true_fft(df_10c)

    # Calculate metrics
    cap_03, min_v_03 = ah_03[-1] if len(ah_03) else 0, v_03.min() if len(v_03) else 0
    cap_10, min_v_10 = ah_10[-1] if len(ah_10) else 0, v_10.min() if len(v_10) else 0

    print(f"\n--- 0.3C Curve Summary ---")
    print(f"Points: {len(ah_03)} | Capacity: {cap_03:.2f} Ah | Min Voltage: {min_v_03:.2f}V")
    print(f"\n--- 1.0C Curve Summary ---")
    print(f"Points: {len(ah_10)} | Capacity: {cap_10:.2f} Ah | Min Voltage: {min_v_10:.2f}V")

    # Set up a 1x2 subplot figure (Side-by-Side and Overlaid)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Side-by-Side Subplots ---
    ax1 = axes[0]
    ax1.plot(ah_03, v_03, linewidth=2.5, color="#10b981", label="0.3C Discharge")
    ax1.plot(ah_10, v_10, linewidth=2.5, color="#ef4444", label="1.0C Discharge")
    ax1.set_title("Side-by-Side Visual Comparison", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Capacity Delivered (Ah)", fontsize=12)
    ax1.set_ylabel("Mean Cell Voltage (V)", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_ylim(1.8, 4.3)
    ax1.legend(loc="lower left", framealpha=0.9)

    # --- Plot 2: Detailed Overlaid Analysis with Metrics ---
    ax2 = axes[1]
    ax2.plot(ah_03, v_03, linewidth=2.5, color="#10b981", label="0.3C")
    ax2.plot(ah_10, v_10, linewidth=2.5, color="#ef4444", label="1.0C")
    ax2.set_title("Overlaid discharge Comparison", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Capacity Delivered (Ah)", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_ylim(1.8, 4.3)

    # Adding Info Box
    info_text = (
        f"0.3C Cap: {cap_03:.2f} Ah\n"
        f"1.0C Cap: {cap_10:.2f} Ah\n"
        f"Voltage Sag at Start:\n"
        f"  0.3C Max: {v_03.max():.2f}V\n"
        f"  1.0C Max: {v_10.max():.2f}V"
    )
    ax2.text(
        0.05,
        0.05,
        info_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", alpha=0.9, edgecolor="#cbd5e1"),
    )
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.suptitle("EV Battery C-Rate Discharge Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save the output visualization
    plt.savefig(output_image, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Plot successfully saved to: {os.path.abspath(output_image)}")


if __name__ == "__main__":
    # Update these paths to the exact locations of your two CSV files
    FILE_03C = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/clean_data_for_test/OneDrive_1_7-9-2026_CLEANED/pk2-enyaq-23072021-FFCT-0.3C 202605141057 Characterisation Test (1).csv"
    FILE_10C = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/clean_data_for_test/OneDrive_1_7-9-2026_CLEANED/pk2-enyaq-23072021-FFCT-0.95C 202605221145 Characterisation Test (1).csv"

    # Quick check before running
    if os.path.exists(FILE_03C) and os.path.exists(FILE_10C):
        compare_discharge_rates(FILE_03C, FILE_10C, "ev_c_rate_comparison.png")
    else:
        print("Error: Make sure to update FILE_03C and FILE_10C paths to point directly to your files.")