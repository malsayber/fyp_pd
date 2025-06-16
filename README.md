# Partial Discharge Detection using Machine Learning
This project aims to detect partial discharge in electrical equipment using traditional machine learning models.

## Project Structure

-   `data/`: Contains raw and processed data (Obtain Separately).
-   `scripts/`: Holds standalone scripts for analysis and visualization.
-   `src/`: Contains the main source code for the project.
  -   `data_processing/`: Scripts for loading and preparing data.
  -   `features/`: Scripts for feature engineering.
  -   `models/`: Scripts for training and running models.
  -   `utils/`: Helper functions.
-   `tests/`: Unit tests for the codebase.
-   `requirements.txt`: Project dependencies.
-   `trained_models/`: Stores trained machine learning models.

## Getting the Datasets

The datasets for this project can be obtained from Figshare at the following location:

-   **Link:** [https://figshare.com/collections/A_data_set_of_Signals_from_an_Antenna_for_Detection_of_Partial_Discharges_in_Overhead_Insulated_Power_Line/6628553](https://figshare.com/collections/A_data_set_of_Signals_from_an_Antenna_for_Detection_of_Partial_Discharges_in_Overhead_Insulated_Power_Line/6628553)

After downloading, please place the files into the `data/` directory.

## Usage

### 1. Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Train a Model:

The `train_model.py` script allows you to choose from several different machine learning models to find the one that performs best on your data. The available models are:

-   `random_forest`
-   `gradient_boosting`
-   `svc`
-   `logistic_regression`
-   `k_neighbors`

There are two ways to select a model:

**Option A: From the Terminal (Recommended)**

You can specify the model using the `--model` command-line argument. This is the most flexible method.

```bash
# Example: Train a Gradient Boosting model
python -m src.models.train_model --model gradient_boosting

# Example: Train a Support Vector Classifier (SVC)
python -m src.models.train_model --model svc
```

**Option B: For IDE Users (Without Terminal)**

If you are running the script directly from an IDE, you can set the model directly in the code.

1.  Open the file `src/models/train_model.py`.
2.  Scroll to the bottom of the file to the `if __name__ == '__main__':` block.
3.  Change the value of the `DEFAULT_MODEL_FOR_IDE` variable to the model you want to train.

### 3. Make a Prediction on a Real Signal:

After training a model, you can use `predict.py` to make a prediction on a single, real signal from your dataset and see if the model was correct.

**Option A: From the Terminal**

Use the `--model`, `--station`, and `--measurement` flags to specify which model to use and which signal to test.

```bash
# Example: Use the 'svc' model to predict on measurement 58972 from station 25
python -m src.models.predict --model svc --station 25 --measurement 58972
```

**Option B: For IDE Users**

1.  Open the file `src/models/predict.py`.
2.  Scroll to the bottom to the `if __name__ == '__main__':` block.
3.  Change the default variables for the model, station, and measurement you want to test.

```python
# --- CONFIGURATION FOR IDE AND TERMINAL ---
DEFAULT_MODEL_TO_USE = 'gradient_boosting'
DEFAULT_STATION_ID = 52010      # <--- Change this
DEFAULT_MEASUREMENT_ID = 12345 # <--- Change this
# ------------------------------------------
```

Then, simply run the file from your IDE.

### 4. Run Analysis Scripts:

For example, to analyze faults for a specific station:

```bash
python scripts/analyse_fault.py <station_id>
```

### 5. Running Tests:

To ensure the reliability of the project, you can run the included unit tests. From the root directory of the project, run:

```bash
python -m unittest discover tests
```