# src/models/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# It's good practice to import from your own project modules
from src.data_processing import load_data
from src.features import build_features
from src.config import PROJECT_ROOT

def train():
    """
    Main function to orchestrate the model training process.
    This version includes a fix for class imbalance.
    """
    print("1. Loading metadata...")
    metadata = load_data.load_metadata()
    if metadata is None:
        return

    # --- Use a larger sample of the data to get more fault examples ---
    # Let's use 50,000 samples if available, or the whole dataset if it's smaller.
    sample_size = min(50000, len(metadata))
    metadata_sample = metadata.sample(n=sample_size, random_state=42)

    print(f"2. Building features for {sample_size} samples... (This may take a while)")
    feature_df = build_features.create_feature_dataset(metadata_sample)

    # Filter out rows where signals might have been missing
    feature_df.dropna(inplace=True)

    fault_count = feature_df['faultAnnotation'].sum()
    print(f"Feature dataset created with {len(feature_df)} samples, including {fault_count} faults.")

    if feature_df.empty or fault_count == 0:
        print("No fault samples found in the dataset subset. Please try increasing the sample size. Exiting.")
        return

    print("3. Splitting data into training and testing sets...")
    X = feature_df[['mean', 'std', 'min', 'max', 'abs_mean', 'abs_std']]
    y = feature_df['faultAnnotation']

    # stratify=y is very important here! It ensures that the train and test sets
    # have the same proportion of fault samples as the original dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    # --- FIX: Use class_weight='balanced' to handle imbalanced data ---
    # This automatically gives more weight to the minority class (faults).
    print("4. Training the model (Random Forest with balanced class weights)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)

    print("5. Evaluating the model...")
    y_pred = model.predict(X_test)

    print("\n--- Classification Report ---")
    # Set zero_division=0 to avoid the warnings
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("-----------------------------\n")

    # --- ADDITION: Visualize the results with a Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fault'], yticklabels=['Normal', 'Fault'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save the confusion matrix plot
    output_dir = os.path.join(PROJECT_ROOT, 'trained_models')
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.show()


    print("\n6. Saving the trained model...")
    model_path = os.path.join(output_dir, 'partial_discharge_model.joblib')
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train()
