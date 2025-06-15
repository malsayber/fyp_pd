# scripts/analyse_fault.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def analyze_station_faults(station_id):
    """
    Loads data for a station, finds measurements with fault annotations,
    and saves plots of a sample of the fault signals.
    """
    # --- Build robust paths for all files ---
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (the project root)
    project_root = os.path.dirname(script_dir)

    # Create the full path to the CSV file inside the 'data' folder
    csv_path = os.path.join(project_root, 'data', 'inferred_annotation.csv')
    # Create the path to the main data folder for signals
    data_folder_path = os.path.join(project_root, 'data')

    try:
        df = pd.read_csv(csv_path, delimiter=',')
        print(f"Successfully loaded {csv_path}")
    except FileNotFoundError:
        print(f"Error: Could not find 'inferred_annotation.csv' at path: {csv_path}")
        print("Please ensure the CSV file is in the 'data' directory.")
        return

    # Filter for the specific station and for faults
    station_df = df[df['idStation'] == int(station_id)]
    fault_df = station_df[station_df['faultAnnotation'] == 1]

    if fault_df.empty:
        print(f"No faults found for station {station_id}. Exiting.")
        return

    print(f"Found {len(fault_df)} faults for station {station_id}. Sampling up to 100 to plot...")

    # Create output directory in the project root
    output_dir = os.path.join(project_root, 'data', f"fault_analysis_station_{station_id}")
    os.makedirs(output_dir, exist_ok=True)

    num_samples = min(100, len(fault_df))

    for index, row in fault_df.sample(num_samples).iterrows():
        measurement_id = row['idMeasurement']
        # Build the full path to the .npy file
        file_path = os.path.join(data_folder_path, str(station_id), f'{measurement_id}.npy')

        try:
            signal = np.load(file_path)
            plt.figure(figsize=(10, 4))
            plt.plot(signal)
            plt.title(f'Station {station_id}, Measurement {measurement_id} (Fault)')
            plt.xlabel('Sample')
            plt.ylabel('Signal Amplitude')
            plt.grid(True)

            # Save the plot as a PNG file in the correct output directory
            save_path = os.path.join(output_dir, f"{measurement_id}.png")
            plt.savefig(save_path)
            plt.close() # Use plt.close() to free up memory

        except FileNotFoundError:
            print(f"Warning: Could not find signal file: {file_path}. Skipping.")

    print(f"Done. Plotted {num_samples} faults in the '{output_dir}' directory.")


if __name__ == '__main__':
    # Check if a station ID was provided
    if len(sys.argv) < 2:
        print("\nUsage: python analyse_fault.py <station_id>")
        print("This script requires a station ID to be provided when you run it.")
        print("Example: python scripts/analyse_fault.py 52010")
    else:
        # Run the analysis with the provided station ID
        analyze_station_faults(sys.argv[1])
