import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock

# Import the module to be tested
from src.models import train_model

class TestTrainModel(unittest.TestCase):
    """
    Test suite for the model training pipeline.
    This test now performs an integration check to ensure the plot is saved.
    """

    @patch('src.models.train_model.load_data.load_metadata')
    @patch('src.models.train_model.build_features.create_feature_dataset')
    @patch('src.models.train_model.Pipeline')
    @patch('src.models.train_model.joblib.dump')
    def test_training_pipeline_saves_artifacts(self, mock_joblib_dump,
                                               mock_pipeline, mock_create_features, mock_load_meta):
        """
        Tests the main training function, ensuring it saves the model and the
        performance plot file to the disk.
        """
        # --- Setup: Mock data and objects ---
        mock_load_meta.return_value = pd.DataFrame({'faultAnnotation': [0, 1] * 10})
        mock_create_features.return_value = pd.DataFrame({
            'mean': np.random.rand(20), 'std': np.random.rand(20),
            'min': np.random.rand(20), 'max': np.random.rand(20),
            'abs_mean': np.random.rand(20), 'abs_std': np.random.rand(20),
            'faultAnnotation': [0, 1] * 10
        }).dropna()

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_pipeline_instance.predict_proba.return_value = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]
        ])
        mock_pipeline.return_value = mock_pipeline_instance

        # FINAL FIX: Define the model_name variable to be used throughout the test
        model_name = 'gradient_boosting'

        # Define expected output paths
        output_dir = os.path.join("trained_models")
        expected_plot_path = os.path.join(output_dir, f'{model_name}_performance_plots.png')
        expected_model_path = os.path.join(output_dir, f'{model_name}_model.joblib')

        # --- Execute: Run the training function ---
        train_model.train(model_name=model_name)

        # --- Assertions ---
        # 1. Check that the model saving logic was called correctly
        mock_joblib_dump.assert_called_once()
        dump_args = mock_joblib_dump.call_args[0]
        actual_model_path = dump_args[1]
        absolute_expected_model_path = os.path.abspath(expected_model_path)

        self.assertEqual(dump_args[0], mock_pipeline_instance)
        self.assertEqual(actual_model_path, absolute_expected_model_path)

        # 2. Check that the plot file was actually created on disk
        self.assertTrue(os.path.exists(expected_plot_path), "Performance plot file was not saved.")

        # --- Cleanup ---
        # Remove the created plot file to keep the directory clean
        if os.path.exists(expected_plot_path):
            os.remove(expected_plot_path)

if __name__ == '__main__':
    unittest.main()