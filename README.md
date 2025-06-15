# Partial Discharge Detection using Machine Learning

This project aims to detect partial discharge in electrical equipment using traditional machine learning models.

## Project Structure

- `data/`: Contains raw and processed data (Obtain Separately).
- `scripts/`: Holds standalone scripts for analysis and visualization.
- `src/`: Contains the main source code for the project.
    - `data_processing/`: Scripts for loading and preparing data.
    - `features/`: Scripts for feature engineering.
    - `models/`: Scripts for training and running models.
    - `utils/`: Helper functions.
- `tests/`: Unit tests for the codebase.
- `requirements.txt`: Project dependencies.
- `trained_models/`: Stores trained machine learning models.

## Usage

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train the model:

The `train_model.py` script allows you to choose from several different machine learning models to find the one that performs best on your data. The available models are:
* `random_forest`
* `gradient_boosting`
* `svc`
* `logistic_regression`
* `k_neighbors`

There are two ways to select a model:

**Option A: From the Terminal (Recommended)**

You can specify the model using the `--model` command-line argument. This is the most flexible method.

```bash
# Example: Train a Gradient Boosting model
python -m src.models.train_model --model gradient_boosting

# Example: Train a Support Vector Classifier (SVC)
python -m src.models.train_model --model svc
```

If you don't provide the `--model` flag, it will use the default model specified in the script (`gradient_boosting`).

**Option B: For IDE Users (Without Terminal)**

If you are running the script directly from an IDE (like PyCharm or VS Code) by clicking a "Run" button, you can set the model directly in the code.

1.  Open the file `src/models/train_model.py`.
2.  Scroll to the bottom of the file to the `if __name__ == '__main__':` block.
3.  Change the value of the `DEFAULT_MODEL_FOR_IDE` variable to the model you want to train.

```python
# --- CONFIGURATION FOR IDE AND TERMINAL ---
# ...
DEFAULT_MODEL_FOR_IDE = 'svc' # Change this line
# ------------------------------------------
```
Then, simply run the file from your IDE.

### 3. Run analysis scripts:
For example, to analyze faults for a specific station:
```bash
python scripts/analyse_fault.py <station_id>
```

### 4. Running Tests:
To ensure the reliability of the project, you can run the included unit tests. From the root directory of the project, run:
```bash
python -m unittest discover tests
```