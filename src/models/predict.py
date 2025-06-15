# src/models/predict.py

import numpy as np
import pandas as pd
import joblib
import os

# It's good practice to import from your own project modules
from src.features import build_features
from src.config import PROJECT_ROOT

def predict_from_signal(signal_data):
    """
    Makes a fault prediction from a raw signal array using the trained model.
    This version uses a DataFrame to ensure feature names match the training data.

    Args:
        signal_data (np.array): A 1D NumPy array representing the signal.

    Returns:
        int: The prediction (1 for fault, 0 for no fault).
    """
    model_path = os.path.join(PROJECT_ROOT, 'trained_models', 'partial_discharge_model.joblib')

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run the training script first: python -m src.models.train_model")
        return None

    # Load the trained model
    model = joblib.load(model_path)

    # 1. Calculate features for the new signal
    features = build_features.calculate_statistical_features(signal_data)

    # --- FIX: Create a pandas DataFrame with the correct feature names ---
    # The column names must match the ones used during training exactly.
    feature_names = ['mean', 'std', 'min', 'max', 'abs_mean', 'abs_std']
    feature_df = pd.DataFrame([features], columns=feature_names)

    # 3. Make a prediction using the DataFrame
    # This will prevent the UserWarning and ensure correctness.
    prediction = model.predict(feature_df)

    return prediction[0]

if __name__ == '__main__':
    # --- Example Usage ---
    # Create a sample signal for demonstration. In a real case, you would load this.
    print("--- Testing with a sample NORMAL signal ---")
    sample_normal_signal = np.random.randn(800000) * 10 # Lower amplitude
    prediction_1 = predict_from_signal(sample_normal_signal)

    if prediction_1 is not None:
        print(f"Prediction: {'FAULT' if prediction_1 == 1 else 'NORMAL'}")
        # We expect this to be NORMAL

    print("\n--- Testing with a sample FAULT signal ---")
    # A signal with higher amplitude and standard deviation should be a fault
    sample_fault_signal = np.random.randn(800000) * 30
    prediction_2 = predict_from_signal(sample_fault_signal)

    if prediction_2 is not None:
        print(f"Prediction: {'FAULT' if prediction_2 == 1 else 'NORMAL'}")
        # We expect this to be FAULT
