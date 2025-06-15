# src/models/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from typing import NoReturn, Optional

from src.data_processing import load_data
from src.features import build_features
from src.config import PROJECT_ROOT

def train(model_name: str, output_dir_override: Optional[str] = None) -> None:
    """
    Main function to orchestrate the model training process for a specified model.
    Generates a comprehensive performance plot including Confusion Matrix, ROC, and PR curves.

    Args:
        model_name (str): The name of the model to train.
        output_dir_override (Optional[str]): A path to an alternative output directory.
    """
    print(f"--- Starting Training for Model: {model_name.replace('_', ' ').title()} ---")

    print("1. Loading metadata...")
    metadata = load_data.load_metadata()
    if metadata is None: return

    sample_size = min(50000, len(metadata))
    metadata_sample = metadata.sample(n=sample_size, random_state=42)

    print(f"2. Building features for {sample_size} samples...")
    feature_df = build_features.create_feature_dataset(metadata_sample)
    feature_df.dropna(inplace=True)

    fault_count = feature_df['faultAnnotation'].sum()
    print(f"Feature dataset created with {len(feature_df)} samples, including {fault_count} faults.")

    if feature_df.empty or fault_count < 2:
        print("Not enough fault samples to train. Need at least 2. Exiting.")
        return

    print("3. Splitting data...")
    X = feature_df[['mean', 'std', 'min', 'max', 'abs_mean', 'abs_std']]
    y = feature_df['faultAnnotation']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        'logistic_regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'svc': SVC(random_state=42, class_weight='balanced', probability=True),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'k_neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }
    if model_name not in models:
        print(f"Error: Model '{model_name}' not recognized. Available models: {list(models.keys())}")
        return

    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', models[model_name])])

    print(f"4. Training the {model_name.replace('_', ' ').title()} model...")
    pipeline.fit(X_train, y_train)

    print("5. Evaluating the model...")
    y_pred = pipeline.predict(X_test)
    y_scores = pipeline.predict_proba(X_test)[:, 1] # Probabilities for the 'Fault' class

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # --- Create a consolidated performance plot ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f'Performance Analysis for {model_name.replace("_", " ").title()}', fontsize=18)

    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticklabels(['Normal', 'Fault'])
    axes[0].set_yticklabels(['Normal', 'Fault'])

    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Receiver Operating Characteristic (ROC)')
    axes[1].legend(loc="lower right")

    # Plot 3: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)
    axes[2].plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AP = {avg_precision:.2f})')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_title('Precision-Recall Curve')
    axes[2].legend(loc="upper right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Determine output directory and save files
    output_dir = output_dir_override if output_dir_override else os.path.join(PROJECT_ROOT, 'trained_models')
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, f'{model_name}_performance_plots.png')
    plt.savefig(plot_path)
    print(f"\nPerformance plots saved to {plot_path}")
    plt.close()

    print("\n6. Saving the trained model...")
    model_path = os.path.join(output_dir, f'{model_name}_model.joblib')
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    print("--- Training Complete ---")

if __name__ == '__main__':

    # --- CONFIGURATION FOR IDE AND TERMINAL ---
    # Set the default model to train. This is used when running from an IDE
    # or when no --model argument is given in the terminal.
    # Your options are: 'random_forest', 'gradient_boosting', 'svc', 'logistic_regression', 'k_neighbors'
    DEFAULT_MODEL_FOR_IDE = 'k_neighbors'

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    model_choices = ['random_forest', 'logistic_regression', 'svc', 'gradient_boosting', 'k_neighbors']
    parser.add_argument(
        '--model', type=str, default=DEFAULT_MODEL_FOR_IDE, choices=model_choices,
        help=f"The type of model to train. Defaults to '{DEFAULT_MODEL_FOR_IDE}'."
    )
    args = parser.parse_args()

    print(f"Preparing to train the '{args.model}' model...")
    train(model_name=args.model)
