# src/models/predict.py

import numpy as np
import pandas as pd
import joblib
import os
from typing import Optional

from src.features import build_features
from src.config import PROJECT_ROOT

def predict_from_signal(signal_data: np.ndarray) -> Optional[int]:
    """
    Makes a fault prediction from a raw signal array using the trained model.

    Args:
        signal_data: A 1D NumPy array representing the signal.

    Returns:
        The prediction (1 for fault, 0 for no fault), or None if the model is not found.
    """
    model_path = os.path.join(PROJECT_ROOT, 'trained_models', 'partial_discharge_model.joblib')

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run the training script first: python -m src.models.train_model")
        return None

    model = joblib.load(model_path)
    features = build_features.calculate_statistical_features(signal_data)
    feature_names = ['mean', 'std', 'min', 'max', 'abs_mean', 'abs_std']
    feature_df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(feature_df)

    return prediction[0]

if __name__ == '__main__':
    print("--- Testing with a sample NORMAL signal ---")
    sample_normal_signal = np.random.randn(800000) * 10
    prediction_1 = predict_from_signal(sample_normal_signal)

    if prediction_1 is not None:
        print(f"Prediction: {'FAULT' if prediction_1 == 1 else 'NORMAL'}")

    print("\n--- Testing with a sample FAULT signal ---")
    sample_fault_signal = np.random.randn(800000) * 30
    prediction_2 = predict_from_signal(sample_fault_signal)

    if prediction_2 is not None:
        print(f"Prediction: {'FAULT' if prediction_2 == 1 else 'NORMAL'}")