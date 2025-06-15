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

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the model:**
    ```bash
    python -m src.models.train_model
    ```

3.  **Run analysis scripts:**
    For example, to analyze faults for a specific station:
    ```bash
    python scripts/analyse_fault.py <station_id>
    ```

4.  **Running Tests:**
    To ensure the reliability of the project, you can run the included unit tests. From the root directory of the project, run:
    ```bash
    python -m unittest discover tests
    ```
