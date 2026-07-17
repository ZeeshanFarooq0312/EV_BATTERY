import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for saving files
import matplotlib
matplotlib.use('Agg')

def extract_true_fft(df):
    """Extract discharge curve from FFT file"""
    cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
    df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)
    
    # Filter for discharge phase (capture full range down to 2.0V)
    discharge_df = df[(df['Mean_Cell_Voltage'] < 4.2) & (df['Mean_Cell_Voltage'] > 1.5)].copy()
    
    if 'AHDischarge' in discharge_df.columns and discharge_df['AHDischarge'].max() > 0:
        discharge_df = discharge_df.sort_values('AHDischarge')
        discharge_df['Ah_Relative'] = discharge_df['AHDischarge'] - discharge_df['AHDischarge'].iloc[0]
    else:
        discharge_df['Ah_Relative'] = np.arange(len(discharge_df))
    
    return discharge_df['Ah_Relative'].values, discharge_df['Mean_Cell_Voltage'].values

def plot_fft_files(data_folder, output_folder='fft_plots'):
    """Plot all FFT files and save as images"""
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Scanning folder: {data_folder}")
    print(f"Output folder: {output_folder}\n")
    
    fft_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    print(f"Found {len(fft_files)} CSV files\n")
    
    summary = []
    
    for idx, filename in enumerate(fft_files, 1):
        print(f"[{idx}/{len(fft_files)}] Processing: {filename}")
        
        file_path = os.path.join(data_folder, filename)
        df = pd.read_csv(file_path)
        
        # Extract discharge curve
        ah, v = extract_true_fft(df)
        
        if len(ah) == 0:
            print(f"  ⚠️  No discharge data found!\n")
            continue
        
        # Get statistics
        min_v = v.min()
        max_v = v.max()
        total_cap = ah[-1]
        
        print(f"  - Points: {len(ah)}")
        print(f"  - Voltage range: {min_v:.2f}V to {max_v:.2f}V")
        print(f"  - Total capacity: {total_cap:.2f} Ah\n")
        
        summary.append({
            'filename': filename,
            'points': len(ah),
            'min_voltage': min_v,
            'max_voltage': max_v,
            'total_capacity': total_cap
        })
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(ah, v, linewidth=2, color='#2563eb')
        
        plt.title(f'FFT Discharge Curve\n{filename}', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Capacity Delivered (Ah)', fontsize=12)
        plt.ylabel('Mean Cell Voltage (V)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.ylim(1.8, 4.3)
        
        # Add annotations
        plt.text(0.02, 0.95, f'Total Capacity: {total_cap:.2f} Ah', 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.text(0.02, 0.88, f'Voltage Range: {min_v:.2f}V - {max_v:.2f}V', 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        output_filename = filename.replace('.csv', '_fft_curve.png')
        output_path = os.path.join(output_folder, output_filename)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {output_path}\n")
    
    # Print summary table
    print("="*80)
    print("SUMMARY OF ALL FFT FILES")
    print("="*80)
    print(f"{'Filename':<50} {'Capacity (Ah)':<15} {'Voltage Range':<20}")
    print("-"*80)
    
    for item in summary:
        filename_short = item['filename'][:47] + '...' if len(item['filename']) > 50 else item['filename']
        voltage_range = f"{item['min_voltage']:.2f}V - {item['max_voltage']:.2f}V"
        print(f"{filename_short:<50} {item['total_capacity']:<15.2f} {voltage_range:<20}")
    
    print("="*80)
    print(f"\n✅ All plots saved to: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    # Update this path to your FFT folder
    FFT_FOLDER = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/fft_raw_data"
    
    if os.path.exists(FFT_FOLDER):
        plot_fft_files(FFT_FOLDER)
    else:
        print(f"Error: Folder not found: {FFT_FOLDER}")