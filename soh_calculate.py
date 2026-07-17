import pandas as pd
import os
import re

NOMINAL_CAPACITY = 156.0

def calculate_actual_soh_from_fft(data_folder):
    """Calculate actual SOH for each file based on its FFT capacity"""
    
    results = []
    
    for file in os.listdir(data_folder):
        if not file.endswith('.csv'): continue
        if 'FFCT' not in file.upper() and 'FFT' not in file.upper(): continue
        
        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path)
        
        # Extract metadata
        pack_match = re.search(r'(pk\d+)', file)
        pack_id = pack_match.group(1) if pack_match else 'unknown'
        
        c_rate_match = re.search(r'(\d+\.\d+)C', file)
        c_rate = float(c_rate_match.group(1)) if c_rate_match else 0.3
        
        # Extract discharge curve
        cell_cols = [c for c in df.columns if 'Cell' in c and 'Temperature' not in c]
        if len(cell_cols) == 0: continue
        
        df['Mean_Cell_Voltage'] = df[cell_cols].mean(axis=1)
        discharge_df = df[(df['Mean_Cell_Voltage'] < 4.2) & (df['Mean_Cell_Voltage'] > 1.5)].copy()
        
        if 'AHDischarge' in discharge_df.columns:
            actual_capacity = discharge_df['AHDischarge'].max()
        else:
            continue
        
        # Calculate actual SOH
        actual_soh = (actual_capacity / NOMINAL_CAPACITY) * 100
        
        results.append({
            'file': file,
            'pack_id': pack_id,
            'c_rate': c_rate,
            'actual_capacity': actual_capacity,
            'actual_soh': actual_soh
        })
    
    # Create DataFrame and show summary
    results_df = pd.DataFrame(results)
    
    print("="*80)
    print("ACTUAL SOH CALCULATED FROM FFT CAPACITY")
    print("="*80)
    print(f"{'Pack':<10} {'C-Rate':<10} {'Capacity (Ah)':<20} {'Actual SOH':<15}")
    print("-"*80)
    
    # Group by pack and c_rate to show averages
    grouped = results_df.groupby(['pack_id', 'c_rate']).agg({
        'actual_capacity': 'mean',
        'actual_soh': 'mean'
    }).round(2)
    
    for (pack_id, c_rate), row in grouped.iterrows():
        print(f"{pack_id:<10} {c_rate:<10} {row['actual_capacity']:<20} {row['actual_soh']:<15.2f}%")
    
    print("="*80)
    
    # Save to CSV
    results_df.to_csv('actual_soh_by_crate.csv', index=False)
    print("\n✅ Saved detailed results to 'actual_soh_by_crate.csv'")
    
    return results_df

if __name__ == "__main__":
    FFT_FOLDER = r"/home/ioptime/Desktop/zeeshan_farooq/ev_battery_version_2 (2)/ev_battery_version_2/new_tech/fft_raw_data"
    if os.path.exists(FFT_FOLDER):
        calculate_actual_soh_from_fft(FFT_FOLDER)
    else:
        print("Error: Folder not found")