# scripts/analyze_station.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from typing import NoReturn

def analyze_station(station_id: int, calculate_stats: bool = False) -> None:
    """
    Performs general analysis on a station: plots sample signals and
    optionally calculates overall statistics.

    Args:
        station_id: The ID of the station to analyze.
        calculate_stats: Whether to calculate and display overall statistics.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'inferred_annotation.csv')
    data_folder_path = os.path.join(project_root, 'data')

    try:
        df = pd.read_csv(csv_path, delimiter=',')
    except FileNotFoundError:
        print(f"Error: Could not find 'inferred_annotation.csv' at path: {csv_path}")
        return

    station_df = df[df['idStation'] == station_id].copy()
    if station_df.empty:
        print(f"No data found for station {station_id}.")
        return

    output_dir = os.path.join(project_root, f"station_analysis_station_{station_id}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Plotting a random sample of signals to '{output_dir}'...")

    for _, row in station_df.sample(n=min(10, len(station_df))).iterrows():
        measurement_id = row['idMeasurement']
        file_path = os.path.join(data_folder_path, str(station_id), f'{measurement_id}.npy')
        try:
            signal = np.load(file_path)
            plt.figure(figsize=(10, 4))
            plt.plot(signal)
            plt.title(f'Station {station_id}, Random Sample: {measurement_id}')
            plt.savefig(os.path.join(output_dir, f"random_{measurement_id}.png"))
            plt.close()
        except FileNotFoundError:
            pass

    fault_df = station_df[station_df['faultAnnotation'] == 1]
    if not fault_df.empty:
        for _, row in fault_df.sample(n=min(10, len(fault_df))).iterrows():
            measurement_id = row['idMeasurement']
            file_path = os.path.join(data_folder_path, str(station_id), f'{measurement_id}.npy')
            try:
                signal = np.load(file_path)
                plt.figure(figsize=(10, 4))
                plt.plot(signal)
                plt.title(f'Station {station_id}, Fault Sample: {measurement_id}')
                plt.savefig(os.path.join(output_dir, f"fault_{measurement_id}.png"))
                plt.close()
            except FileNotFoundError:
                pass

    print("Finished plotting samples.")

    if calculate_stats:
        print("\nCalculating overall statistics... (This may take a while)")

        all_signals = []
        for mid in tqdm(station_df['idMeasurement'], desc="Loading Signals"):
            file_path = os.path.join(data_folder_path, str(station_id), f'{mid}.npy')
            try:
                signal = np.load(file_path).astype(np.float32)
                all_signals.append(signal)
            except FileNotFoundError:
                continue

        if not all_signals:
            print("Could not load any signal data for statistics.")
            return

        full_station_signal = np.concatenate(all_signals)

        mean_val = np.mean(full_station_signal)
        std_val = np.std(full_station_signal)

        print("\n--- Overall Station Statistics ---")
        print(f"Mean Amplitude: {mean_val:.4f}")
        print(f"Standard Deviation: {std_val:.4f}")
        print(f"Min Amplitude: {np.min(full_station_signal):.4f}")
        print(f"Max Amplitude: {np.max(full_station_signal):.4f}")

        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val
        outliers = np.sum((full_station_signal < lower_bound) | (full_station_signal > upper_bound))
        outlier_percent = (outliers / len(full_station_signal)) * 100

        print(f"Data points outside 3 std devs (outliers): {outliers} ({outlier_percent:.2f}%)")
        print("----------------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Perform general analysis on a station.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('station_id', type=int, help='The ID of the station to analyze.')
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Optional: Add this flag to calculate overall statistics.'
    )
    args = parser.parse_args()

    analyze_station(args.station_id, calculate_stats=args.stats)