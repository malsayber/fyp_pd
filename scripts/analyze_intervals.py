# scripts/analyze_intervals.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

def analyze_and_visualize_intervals():
    """
    Identifies and visualizes contiguous time intervals from the dataset.
    This helps understand data completeness.
    """
    # --- FIX: Build a robust path to the CSV file ---
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (the project root)
    project_root = os.path.dirname(script_dir)
    # Create the full path to the CSV file inside the 'data' folder
    csv_path = os.path.join(project_root, 'data', 'inferred_annotation.csv')

    try:
        df = pd.read_csv(csv_path, sep=',')
        print(f"Successfully loaded {csv_path}")
    except FileNotFoundError:
        print(f"Error: Could not find 'inferred_annotation.csv' at path: {csv_path}")
        print("Please ensure the CSV file is in the 'data' directory.")
        return

    df['timeStamp'] = pd.to_datetime(df['timeStamp'])
    df = df.sort_values(['idStation', 'timeStamp'])

    all_intervals = []
    for station, group in df.groupby('idStation'):
        if group.empty:
            continue
        start = group.iloc[0]['timeStamp']
        end = group.iloc[0]['timeStamp']

        for i in range(1, len(group)):
            # Assuming an 8-hour interval between measurements
            expected_timestamp = end + pd.Timedelta(hours=8)
            actual_timestamp = group.iloc[i]['timeStamp']

            if actual_timestamp > expected_timestamp:
                all_intervals.append({'idStation': station, 'start': start, 'end': end})
                start = actual_timestamp
            end = actual_timestamp

        all_intervals.append({'idStation': station, 'start': start, 'end': end})

    if not all_intervals:
        print("No intervals found.")
        return

    intervals = pd.DataFrame(all_intervals)
    intervals_csv_path = os.path.join(project_root, 'data_intervals.csv')
    intervals.to_csv(intervals_csv_path, index=False)
    print(f"Saved data intervals to {intervals_csv_path}")

    # Visualization
    unique_stations = intervals['idStation'].unique()
    nrows = int(np.ceil(len(unique_stations) / 2))
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(15, 4 * nrows), squeeze=False)
    axs = axs.flatten()

    for i, (station, group) in enumerate(intervals.groupby('idStation')):
        start_times = pd.to_datetime(group['start'])
        end_times = pd.to_datetime(group['end'])

        # Using matplotlib's date functionality directly
        relative_start = mdates.date2num(start_times)
        relative_end = mdates.date2num(end_times)

        durations = relative_end - relative_start

        axs[i].barh(y=np.zeros(len(start_times)), width=durations, left=relative_start, height=0.5)
        axs[i].xaxis_date()
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[i].set_title(f'Data Intervals for Station {station}')
        axs[i].get_yaxis().set_visible(False)
        fig.autofmt_xdate()

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    visualization_path = os.path.join(project_root, 'intervals_visualization.png')
    plt.savefig(visualization_path)
    print(f"Saved interval visualization to {visualization_path}")
    plt.show()


if __name__ == '__main__':
    analyze_and_visualize_intervals()
