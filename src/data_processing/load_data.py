# src/data_processing/load_data.py

import pandas as pd
import numpy as np
import os
from typing import Optional

try:
    from src.config import PROJECT_ROOT
except (ModuleNotFoundError, ImportError):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))


def load_metadata(filename: str = 'inferred_annotation.csv') -> Optional[pd.DataFrame]:
    """
    Loads the main metadata file from the 'data' directory in the project root.

    Args:
        filename: The name of the metadata file.

    Returns:
        A pandas DataFrame containing the metadata, or None if the file is not found.
    """
    filepath = os.path.join(PROJECT_ROOT, 'data', filename)
    try:
        metadata = pd.read_csv(filepath)
        print(f"Metadata loaded successfully from {filepath}")
        return metadata
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {filepath}")
        print("Please ensure the file exists in the 'data' directory.")
        return None

def load_signal(station_id: int, measurement_id: int) -> Optional[np.ndarray]:
    """
    Loads a single signal .npy file from the correct data subfolder.

    Args:
        station_id: The ID of the station.
        measurement_id: The ID of the measurement.

    Returns:
        A NumPy array containing the signal data, or None if the file is not found.
    """
    data_folder = os.path.join(PROJECT_ROOT, 'data')
    path = os.path.join(data_folder, str(station_id), f"{measurement_id}.npy")
    try:
        signal = np.load(path)
        return signal
    except FileNotFoundError:
        return None