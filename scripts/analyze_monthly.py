# scripts/analyze_monthly.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

def analyze_monthly_trends(station_id):
    """
    For a given station, calculates and plots the monthly trends of
    various signal statistics.
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
        return

    # Filter for the specific station
    station_df = df[df['idStation'] == int(station_id)].copy()
    if station_df.empty:
        print(f"No data found for station {station_id}.")
        return

    # Convert timestamp and extract year/month
    station_df['timeStamp'] = pd.to_datetime(station_df['timeStamp'])
    station_df['year_month'] = station_df['timeStamp'].dt.to_period('M')

    # Create output directory
    output_dir = os.path.join(project_root, f"monthly_analysis_station_{station_id}")
    os.makedirs(output_dir, exist_ok=True)

    print("Calculating statistics for each measurement... (This may take a while)")

    # --- Function to apply with a progress bar ---
    def get_stats(measurement_id):
        file_path = os.path.join(data_folder_path, str(station_id), f'{measurement_id}.npy')
        try:
            signal = np.load(file_path)
            return pd.Series([
                np.mean(signal), np.std(signal), np.min(signal), np.max(signal),
                np.mean(np.abs(signal)), np.std(np.abs(signal))
            ])
        except FileNotFoundError:
            return pd.Series([np.nan] * 6)

    # Use tqdm for a progress bar
    tqdm.pandas(desc="Processing Signals")
    stats_df = station_df['idMeasurement'].progress_apply(get_stats)
    stats_df.columns = ['mean', 'std', 'min', 'max', 'abs_mean', 'abs_std']

    # Combine with original dataframe and drop rows with errors
    station_df = pd.concat([station_df, stats_df], axis=1).dropna()
    if station_df.empty:
        print("Could not calculate stats for any signals. Check file paths.")
        return

    # Group by month and calculate the mean of the statistics
    monthly_stats = station_df.groupby('year_month').agg({
        'mean': 'mean', 'std': 'mean', 'min': 'mean',
        'max': 'mean', 'abs_mean': 'mean', 'abs_std': 'mean',
        'idMeasurement': 'count'
    }).rename(columns={'idMeasurement': 'measurement_count'})

    # --- Plotting ---
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
    if len(sys.argv) < 2:
        print("\nUsage: python analyze_monthly.py <station_id>")
        print("Example: python scripts/analyze_monthly.py 52010")
    else:
        analyze_monthly_trends(sys.argv[1])
