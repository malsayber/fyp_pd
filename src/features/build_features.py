# src/features/build_features.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Optional

from src.data_processing import load_data

def calculate_statistical_features(signal: Optional[np.ndarray]) -> List[float]:
    """
    Calculates a set of statistical features from a signal.

    Args:
        signal: A NumPy array representing the signal.

    Returns:
        A list of statistical features.
    """
    if signal is None or len(signal) == 0:
        return [np.nan] * 6

    features = [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.mean(np.abs(signal)),
        np.std(np.abs(signal))
    ]
    return features

def create_feature_dataset(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Iterates through metadata, loads signals, calculates features,
    and returns a feature DataFrame.

    Args:
        metadata: A pandas DataFrame with metadata about the signals.

    Returns:
        A pandas DataFrame with calculated features.
    """
    feature_list = []

    for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc="Building Features"):
        station_id = row['idStation']
        measurement_id = row['idMeasurement']

        signal = load_data.load_signal(station_id, measurement_id)
        features = calculate_statistical_features(signal)

        feature_list.append([station_id, measurement_id] + features + [row['faultAnnotation']])

    columns = [
        'idStation', 'idMeasurement', 'mean', 'std', 'min', 'max',
        'abs_mean', 'abs_std', 'faultAnnotation'
    ]
    feature_df = pd.DataFrame(feature_list, columns=columns)
    feature_df.dropna(inplace=True)

    return feature_df