# src/features/build_features.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data_processing import load_data

def calculate_statistical_features(signal):
    """Calculates a set of statistical features from a signal."""
    if signal is None or len(signal) == 0:
        return [np.nan] * 6 # Return NaNs for missing signals

    features = [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.mean(np.abs(signal)),
        np.std(np.abs(signal))
    ]
    return features

def create_feature_dataset(metadata):
    """
    Iterates through metadata, loads signals, calculates features,
    and returns a feature DataFrame.
    """
    feature_list = []

    # Use tqdm for a progress bar
    for index, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc="Building Features"):
        station_id = row['idStation']
        measurement_id = row['idMeasurement']

        signal = load_data.load_signal(station_id, measurement_id)

        features = calculate_statistical_features(signal)

        # Add metadata back in
        feature_list.append([station_id, measurement_id] + features + [row['faultAnnotation']])

    # Define column names for the new DataFrame
    columns = [
        'idStation', 'idMeasurement', 'mean', 'std', 'min', 'max',
        'abs_mean', 'abs_std', 'faultAnnotation'
    ]

    feature_df = pd.DataFrame(feature_list, columns=columns)

    # Drop rows where features could not be calculated
    feature_df.dropna(inplace=True)

    return feature_df