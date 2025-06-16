# tests/test_predict.py

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from src.models import predict

class TestPredict(unittest.TestCase):

    @patch('joblib.load')
    def test_predict_from_signal(self, mock_joblib_load):
        """Test prediction from a signal."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_joblib_load.return_value = mock_model

        with patch('os.path.exists', return_value=True):
            prediction = predict.predict_from_signal(np.random.rand(100))
            self.assertEqual(prediction, 1)
            mock_model.predict.assert_called_once()

    @patch('os.path.exists', return_value=False)
    def test_predict_model_not_found(self, mock_exists):
        """Test behavior when the model file is not found."""
        prediction = predict.predict_from_signal(np.random.rand(100))
        self.assertIsNone(prediction)

if __name__ == '__main__':
    unittest.main()