# tests/test_predict.py

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.models import predict

class TestPredict(unittest.TestCase):
    """
    Test suite for prediction functions.
    """

    @patch('src.models.predict.build_features.calculate_statistical_features')
    @patch('src.models.predict.joblib.load')
    @patch('src.models.predict.os.path.exists', return_value=True)
    def test_predict_from_signal_success(self, mock_exists, mock_joblib_load, mock_calculate_features):
        """
        Tests the prediction pipeline for a single signal, ensuring features
        are calculated and passed to a loaded model correctly.
        """
        # --- Setup ---
        # 1. Mock the loaded model and its prediction
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])  # Predict 'FAULT'
        mock_joblib_load.return_value = mock_model

        # 2. Mock the output of the feature calculation step
        mock_features = [0.1, 0.2, -0.5, 0.5, 0.3, 0.15]
        mock_calculate_features.return_value = mock_features

        # 3. Define a sample input signal and model name
        input_signal = np.random.randn(1000)
        model_name_to_test = 'svc'

        # --- Execute ---
        prediction = predict.predict_from_signal(input_signal, model_name=model_name_to_test)

        # --- Assert ---
        # 1. Check that the model file path was checked for existence
        mock_exists.assert_called_once()
        self.assertTrue(mock_exists.call_args[0][0].endswith(f'{model_name_to_test}_model.joblib'))

        # 2. Check that the model was loaded from the correct path
        mock_joblib_load.assert_called_once()

        # 3. Check that feature calculation was called with the raw input signal
        mock_calculate_features.assert_called_once()
        np.testing.assert_array_equal(mock_calculate_features.call_args[0][0], input_signal)

        # 4. Check that the model's predict method was called with a DataFrame
        #    containing the exact features from the previous step.
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args[0][0] # The DataFrame passed to predict()
        self.assertIsInstance(call_args, pd.DataFrame)
        self.assertEqual(list(call_args.iloc[0]), mock_features)

        # 5. Check that the final prediction result is correct
        self.assertEqual(prediction, 1)

    @patch('src.models.predict.os.path.exists', return_value=False)
    def test_predict_from_signal_model_not_found(self, mock_exists):
        """
        Tests that predict_from_signal returns None when the model file does not exist.
        """
        # --- Execute ---
        prediction = predict.predict_from_signal(np.random.rand(100), model_name='non_existent_model')

        # --- Assert ---
        mock_exists.assert_called_once()
        self.assertIsNone(prediction)

if __name__ == '__main__':
    unittest.main()