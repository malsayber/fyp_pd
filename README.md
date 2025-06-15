# Partial Discharge Detection using Machine Learning

This project aims to detect partial discharge in electrical equipment using traditional machine learning models.

## Project Structure

- `data/`: Contains raw and processed data.(Obtain Separately)
- `scripts/`: Holds standalone scripts for analysis and visualization.
- `src/`: Contains the main source code for the project.
    - `data_processing/`: Scripts for loading and preparing data.
    - `features/`: Scripts for feature engineering.
    - `models/`: Scripts for training and running models.
    - `utils/`: Helper functions.
- `tests/`: Unit tests for the codebase.
- `requirements.txt`: Project dependencies.

## Usage

1.  **Install dependencies:**
    `pip install -r requirements.txt`

2.  **Train the model:**
    `python src/models/train_model.py`

3.  **Run analysis scripts:**
    `python scripts/analyze_faults.py <station_id>`"# fyp_pd" 
