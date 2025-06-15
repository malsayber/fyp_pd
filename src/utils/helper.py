# src/utils/helper.py

import pandas as pd
from typing import List

def get_station_measurements(station_id: int, metadata_path: str = 'metadata.csv') -> List[int]:
    """
    Return a list of measurement IDs for a given station.

    Args:
        station_id: The ID of the station.
        metadata_path: The path to the metadata CSV file.

    Returns:
        A list of measurement IDs.
    """
    try:
        metadata = pd.read_csv(metadata_path, header=None, names=["idStation", "idMeasurement", "faultAnnotation", "timeStamp"])
        measurements = metadata.loc[metadata['idStation'] == station_id, 'idMeasurement'].tolist()
        return measurements
    except FileNotFoundError:
        print(f"Warning: Metadata file not found at {metadata_path}")
        return []

def get_fault_annotations(station_id: int, measurement_id: int, metadata_path: str = 'metadata.csv') -> List[int]:
    """
    Return the fault annotations for a given station and measurement.

    Args:
        station_id: The ID of the station.
        measurement_id: The ID of the measurement.
        metadata_path: The path to the metadata CSV file.

    Returns:
        A list of fault annotations.
    """
    try:
        metadata = pd.read_csv(metadata_path, header=None, names=["idStation", "idMeasurement", "faultAnnotation", "timeStamp"])
        annotations = metadata.loc[
            (metadata['idStation'] == station_id) & (metadata['idMeasurement'] == measurement_id),
            'faultAnnotation'
        ].tolist()
        return annotations
    except FileNotFoundError:
        print(f"Warning: Metadata file not found at {metadata_path}")
        return []