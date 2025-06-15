# src/data_processing/load_data.py

import pandas as pd
import numpy as np
import os

# --- Determine the project root directory dynamically ---
# This makes the script work regardless of where you run it from.
try:
    # This works when run as a module (e.g., python -m src.models.train_model)
    from src.config import PROJECT_ROOT
except (ModuleNotFoundError, ImportError):
    # This is a fallback for running scripts directly
    # It assumes this file is in src/data_processing/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))


def load_metadata(filename='inferred_annotation.csv'):
    """
    Loads the main metadata file from the 'data' directory
    in the project root.
    """
    # Build a full, robust path to the metadata file
    filepath = os.path.join(PROJECT_ROOT, 'data', filename)
    try:
        metadata = pd.read_csv(filepath)
        print(f"Metadata loaded successfully from {filepath}")
        return metadata
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {filepath}")
        print("Please ensure the file exists in the 'data' directory.")
        return None

def load_signal(station_id, measurement_id):
    """
    Loads a single signal .npy file from the correct data subfolder.
    """
    # Build a full, robust path to the signal file
    data_folder = os.path.join(PROJECT_ROOT, 'data')
    path = os.path.join(data_folder, str(station_id), f"{measurement_id}.npy")
    try:
        signal = np.load(path)
        return signal
    except FileNotFoundError:
        # This warning can be noisy, so it's commented out by default.
        # print(f"Warning: Signal file not found: {path}")
        return None

# --- To make this work, we need a simple config file ---
# You should also create a file named `src/config.py` with this content:
#
# import os
# # Defines the absolute path to the project's root directory
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#
