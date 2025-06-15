# src/models/predict.py

import numpy as np
import pandas as pd
import joblib
import os
import argparse
from typing import Optional

# Make sure the script can find other modules in the 'src' directory
try:
    from src.features import build_features
    from src.config import PROJECT_ROOT
    from src.data_processing import load_data
except (ModuleNotFoundError, ImportError):
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(os.path.dirname(script_dir))
    sys.path.append(project_root_dir)
    from src.features import build_features
    from src.config import PROJECT_ROOT
    from src.data_processing import load_data

def predict_from_signal(signal_data: np.ndarray, model_name: str) -> Optional[int]:
    """
    Makes a fault prediction from a raw signal array using a specified trained model.

    Args:
        signal_data (np.ndarray): A 1D NumPy array representing the signal.
        model_name (str): The name of the model to use for prediction (e.g., 'gradient_boosting').

    Returns:
        The prediction (1 for fault, 0 for no fault), or None if the model is not found.
    """
    model_path = os.path.join(PROJECT_ROOT, 'trained_models', f'{model_name}_model.joblib')

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print(f"Please run the training script first: python -m src.models.train_model --model {model_name}")
        return None

    model_pipeline = joblib.load(model_path)
    features = build_features.calculate_statistical_features(signal_data)
    feature_names = ['mean', 'std', 'min', 'max', 'abs_mean', 'abs_std']
    feature_df = pd.DataFrame([features], columns=feature_names)
    prediction = model_pipeline.predict(feature_df)

    return prediction[0]

if __name__ == '__main__':
    # --- CONFIGURATION FOR IDE AND TERMINAL ---
    # Set the default model and signal to use for prediction.
    # This is used when running from an IDE.
    DEFAULT_MODEL_TO_USE = 'svc'
    DEFAULT_STATION_ID = 52010      # <--- Change this to a real station ID
    DEFAULT_MEASUREMENT_ID = 810780 # <--- Change this to a real measurement ID
    # ------------------------------------------

    parser = argparse.ArgumentParser(
        description="Predict a fault from a real signal file using a trained model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    model_choices = ['random_forest', 'logistic_regression', 'svc', 'gradient_boosting', 'k_neighbors']
    parser.add_argument(
        '--model', type=str, default=DEFAULT_MODEL_TO_USE, choices=model_choices,
        help=f"The trained model to use for prediction.\nDefaults to '{DEFAULT_MODEL_TO_USE}'."
    )
    parser.add_argument('--station', type=int, default=DEFAULT_STATION_ID, help="The ID of the station for the signal.")
    parser.add_argument('--measurement', type=int, default=DEFAULT_MEASUREMENT_ID, help="The ID of the measurement signal to predict.")
    args = parser.parse_args()

    print(f"--- Using '{args.model}' model to predict signal from Station {args.station}, Measurement {args.measurement} ---")

    # 1. Load the real signal data
    signal_data = load_data.load_signal(args.station, args.measurement)

    if signal_data is None:
        print(f"\nError: Could not load signal file for Station {args.station}, Measurement {args.measurement}.")
        print("Please ensure the data exists in the 'data/' directory following the correct structure.")
    else:
        # 2. Get the prediction
        prediction = predict_from_signal(signal_data, model_name=args.model)

        if prediction is not None:
            # 3. Get the true label for comparison
            metadata = load_data.load_metadata()
            true_label = "Unknown"
            true_label_val = -1
            if metadata is not None:
                record = metadata[
                    (metadata['idStation'] == args.station) &
                    (metadata['idMeasurement'] == args.measurement)
                    ]
                if not record.empty:
                    true_label_val = record.iloc[0]['faultAnnotation']
                    true_label = 'FAULT' if true_label_val == 1 else 'NORMAL'

            # 4. Print results
            print(f"\nModel Prediction: {'FAULT' if prediction == 1 else 'NORMAL'}")
            print(f"Actual Label:     {true_label}")

            if true_label_val != -1:
                if prediction == true_label_val:
                    print("\nResult: Prediction was CORRECT.")
                else:
                    print("\nResult: Prediction was INCORRECT.")
