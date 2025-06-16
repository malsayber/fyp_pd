# scripts/calculate_metrics.py

import numpy as np
import pandas as pd
import joblib
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from typing import NoReturn

# Make sure the script can find other modules in the 'src' directory
try:
    from src.data_processing import load_data
    from src.features import build_features
    from src.config import PROJECT_ROOT
except (ModuleNotFoundError, ImportError):
    # This block allows the script to be run from the root directory (e.g., `python scripts/calculate_metrics.py`)
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(script_dir)
    sys.path.append(project_root_dir)
    from src.data_processing import load_data
    from src.features import build_features
    from src.config import PROJECT_ROOT


def calculate_and_print_metrics(tp: int, fp: int, fn: int, tn: int) -> None:
    """
    Calculates and prints key classification performance metrics from confusion matrix values.

    Args:
        tp (int): True Positives - Correctly predicted positive cases.
        fp (int): False Positives - Incorrectly predicted positive cases.
        fn (int): False Negatives - Incorrectly predicted negative cases.
        tn (int): True Negatives - Correctly predicted negative cases.
    """
    print("--- Model Performance Metrics ---")
    print(f"Based on: TP={tp}, FP={fp}, FN={fn}, TN={tn}\n")

    total_population = tp + fp + fn + tn
    accuracy = (tp + tn) / total_population if total_population > 0 else 0
    print(f"1. Accuracy: {accuracy:.4f}")
    print("   - Meaning: Of all the predictions made, this percentage was correct.\n")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"2. Precision (for 'Fault' class): {precision:.4f}")
    print("   - Meaning: Of all signals the model flagged as 'Fault', this percentage were actually faults.\n")

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"3. Recall (Sensitivity, for 'Fault' class): {recall:.4f}")
    print("   - Meaning: Of all actual 'Faults', this percentage was correctly identified by the model.\n")

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"4. F1-Score (for 'Fault' class): {f1_score:.4f}")
    print("   - Meaning: A single score that balances the trade-off between precision and recall.\n")

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"5. Specificity (for 'Normal' class): {specificity:.4f}")
    print("   - Meaning: Of all actual 'Normal' signals, this percentage was correctly identified.\n")

    print("--- End of Report ---")


def main(model_name: str):
    """
    Loads a trained model, evaluates it on the test set, and prints performance metrics.
    """
    print(f"Analyzing performance for model: '{model_name}'")

    # 1. Load the trained model
    model_path = os.path.join(PROJECT_ROOT, 'trained_models', f'{model_name}_model.joblib')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print(f"Please train the model first by running: python -m src.models.train_model --model {model_name}")
        return

    model = joblib.load(model_path)
    print("Model loaded successfully.")

    # 2. Load and prepare the same dataset used for training
    print("Loading and preparing test data...")
    metadata = load_data.load_metadata()
    if metadata is None: return

    # We use a sample here to match the training process, ensuring consistency.
    sample_size = min(50000, len(metadata))
    metadata_sample = metadata.sample(n=sample_size, random_state=42)

    feature_df = build_features.create_feature_dataset(metadata_sample)
    feature_df.dropna(inplace=True)

    if feature_df.empty:
        print("Could not generate features from the data. Exiting.")
        return

    X = feature_df[['mean', 'std', 'min', 'max', 'abs_mean', 'abs_std']]
    y = feature_df['faultAnnotation']

    # CRITICAL: Use the exact same train_test_split to get the same test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Test set created with {len(X_test)} samples.")

    # 3. Make predictions
    y_pred = model.predict(X_test)

    # 4. Calculate confusion matrix and extract values
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # In a binary confusion matrix, .ravel() unfolds the matrix into [TN, FP, FN, TP]
    tn, fp, fn, tp = cm.ravel()

    # 5. Print the detailed metrics report
    calculate_and_print_metrics(tp=tp, fp=fp, fn=fn, tn=tn)


if __name__ == '__main__':
    # --- CONFIGURATION FOR IDE AND TERMINAL ---
    # Set the default model to analyze. This is used when running from an IDE.
    DEFAULT_MODEL_TO_ANALYZE = 'k_neighbors'
    # ------------------------------------------

    parser = argparse.ArgumentParser(
        description="Calculate and display detailed performance metrics for a trained model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    model_choices = ['random_forest', 'logistic_regression', 'svc', 'gradient_boosting', 'k_neighbors']
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL_TO_ANALYZE,
        choices=model_choices,
        help=f"The name of the trained model to evaluate. Defaults to '{DEFAULT_MODEL_TO_ANALYZE}'."
    )
    args = parser.parse_args()

    main(model_name=args.model)
