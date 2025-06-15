# src/utils/helpers.py

import pandas as pd
import numpy as np
import os

def get_station_measurements(station_id, metadata_path='metadata.csv'):
    """Return a list of measurement IDs for a given station."""
    try:
        metadata = pd.read_csv(metadata_path, header=None, names=["idStation", "idMeasurement", "faultAnnotation", "timeStamp"])
        measurements = metadata.loc[metadata['idStation'] == station_id, 'idMeasurement'].tolist()
        return measurements
    except FileNotFoundError:
        print(f"Warning: Metadata file not found at {metadata_path}")
        return []

def get_fault_annotations(station_id, measurement_id, metadata_path='metadata.csv'):
    """Return the fault annotations for a given station and measurement."""
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