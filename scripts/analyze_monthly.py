# scripts/analyze_monthly.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from typing import NoReturn

def analyze_monthly_trends(station_id: int) -> None:
    """
    For a given station, calculates and plots the monthly trends of
    various signal statistics.

    Args:
        station_id: The ID of the station to analyze.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'inferred_annotation.csv')
    data_folder_path = os.path.join(project_root, 'data')

    try:
        df = pd.read_csv(csv_path, delimiter=',')
        print(f"Successfully loaded {csv_path}")
    except FileNotFoundError:
        print(f"Error: Could not find 'inferred_annotation.csv' at path: {csv_path}")
        return

    station_df = df[df['idStation'] == station_id].copy()
    if station_df.empty:
        print(f"No data found for station {station_id}.")
        return

    station_df['timeStamp'] = pd.to_datetime(station_df['timeStamp'])
    station_df['year_month'] = station_df['timeStamp'].dt.to_period('M')

    output_dir = os.path.join(project_root, f"monthly_analysis_station_{station_id}")
    os.makedirs(output_dir, exist_ok=True)

    print("Calculating statistics for each measurement... (This may take a while)")

    def get_stats(measurement_id: int) -> pd.Series:
        file_path = os.path.join(data_folder_path, str(station_id), f'{measurement_id}.npy')
        try:
            signal = np.load(file_path)
            return pd.Series([
                np.mean(signal), np.std(signal), np.min(signal), np.max(signal),
                np.mean(np.abs(signal)), np.std(np.abs(signal))
            ])
        except FileNotFoundError:
            return pd.Series([np.nan] * 6)

    tqdm.pandas(desc="Processing Signals")
    stats_df = station_df['idMeasurement'].progress_apply(get_stats)
    stats_df.columns = ['mean', 'std', 'min', 'max', 'abs_mean', 'abs_std']

    station_df = pd.concat([station_df, stats_df], axis=1).dropna()
    if station_df.empty:
        print("Could not calculate stats for any signals. Check file paths.")
        return

    monthly_stats = station_df.groupby('year_month').agg({
        'mean': 'mean', 'std': 'mean', 'min': 'mean',
        'max': 'mean', 'abs_mean': 'mean', 'abs_std': 'mean',
        'idMeasurement': 'count'
    }).rename(columns={'idMeasurement': 'measurement_count'})

    print(f"Saving plots to '{output_dir}' directory...")
    for column in monthly_stats.columns:
        plt.figure(figsize=(12, 6))
        monthly_stats[column].plot(kind='bar')
        plt.title(f'Station {station_id}: Monthly Average of {column.capitalize()}')
        plt.xlabel('Year-Month')
        plt.ylabel(f'Average {column.capitalize()}')
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"monthly_{column}.png"))
        plt.close()

    print("Analysis complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate and plot monthly trends of signal statistics for a given station.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('station_id', type=int, help='The ID of the station to analyze.')
    args = parser.parse_args()

    analyze_monthly_trends(args.station_id)