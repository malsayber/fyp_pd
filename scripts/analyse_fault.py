# scripts/analyse_fault.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import NoReturn

def analyze_station_faults(station_id: int) -> None:
    """
    Loads data for a station, finds measurements with fault annotations,
    and saves plots of a sample of the fault signals.

    Args:
        station_id: The ID of the station to analyze.
    """
    # --- Build robust paths for all files ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'inferred_annotation.csv')
    data_folder_path = os.path.join(project_root, 'data')

    try:
        df = pd.read_csv(csv_path, delimiter=',')
        print(f"Successfully loaded {csv_path}")
    except FileNotFoundError:
        print(f"Error: Could not find 'inferred_annotation.csv' at path: {csv_path}")
        print("Please ensure the CSV file is in the 'data' directory.")
        return

    # Filter for the specific station and for faults
    station_df = df[df['idStation'] == station_id]
    fault_df = station_df[station_df['faultAnnotation'] == 1]

    if fault_df.empty:
        print(f"No faults found for station {station_id}. Exiting.")
        return

    print(f"Found {len(fault_df)} faults for station {station_id}. Sampling up to 100 to plot...")

    # Create output directory in the project root
    output_dir = os.path.join(project_root, 'data', f"fault_analysis_station_{station_id}")
    os.makedirs(output_dir, exist_ok=True)

    num_samples = min(100, len(fault_df))

    for _, row in fault_df.sample(num_samples).iterrows():
        measurement_id = row['idMeasurement']
        file_path = os.path.join(data_folder_path, str(station_id), f'{measurement_id}.npy')

        try:
            signal = np.load(file_path)
            plt.figure(figsize=(10, 4))
            plt.plot(signal)
            plt.title(f'Station {station_id}, Measurement {measurement_id} (Fault)')
            plt.xlabel('Sample')
            plt.ylabel('Signal Amplitude')
            plt.grid(True)

            save_path = os.path.join(output_dir, f"{measurement_id}.png")
            plt.savefig(save_path)
            plt.close()

        except FileNotFoundError:
            print(f"Warning: Could not find signal file: {file_path}. Skipping.")

    print(f"Done. Plotted {num_samples} faults in the '{output_dir}' directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze and plot fault signals for a given station.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('station_id', type=int, help='The ID of the station to analyze.')
    args = parser.parse_args()

    analyze_station_faults(args.station_id)