# tests/test_train_model.py

import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock

# Import the module and config to make patching easier
from src.models import train_model
from src.config import PROJECT_ROOT

class TestTrainModel(unittest.TestCase):

    @patch('src.data_processing.load_data.load_metadata')
    @patch('src.features.build_features.create_feature_dataset')
    @patch('src.models.train_model.Pipeline')
    @patch('joblib.dump')
    @patch('src.models.train_model.plt.savefig')
    def test_train_pipeline(self, mock_savefig, mock_joblib_dump, mock_pipeline, mock_create_features, mock_load_meta):
        """
        Tests the training pipeline against the existing train_model.py script
        without passing any special arguments.
        """
        # --- Mock Inputs ---
        mock_load_meta.return_value = pd.DataFrame({
            'idStation': [1] * 10,
            'idMeasurement': range(10),
            'faultAnnotation': [0, 1] * 5
        })

        mock_create_features.return_value = pd.DataFrame({
            'mean': np.random.rand(10), 'std': np.random.rand(10),
            'min': np.random.rand(10), 'max': np.random.rand(10),
            'abs_mean': np.random.rand(10), 'abs_std': np.random.rand(10),
            'faultAnnotation': [0, 1] * 5
        })

        # --- Configure the Mock Pipeline ---
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.predict.return_value = np.array([0, 1, 0])

        # --- Run the Function ---
        # Call the train function without the 'output_dir_override' argument
        # to match the user's existing train_model.py script.
        with patch('matplotlib.pyplot.show'):
            train_model.train(model_name='gradient_boosting')

        # --- Assertions ---
        # Verify that the key steps were called
        mock_load_meta.assert_called_once()
        mock_create_features.assert_called_once()
        mock_pipeline.assert_called_once()
        mock_pipeline_instance.fit.assert_called_once()
        mock_savefig.assert_called_once()  # Check that savefig was called
        mock_joblib_dump.assert_called_once() # Check that joblib.dump was called


if __name__ == '__main__':
    unittest.main()
