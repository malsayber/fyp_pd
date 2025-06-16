# tests/test_load_data.py

import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, mock_open

from src.data_processing import load_data

class TestLoadData(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_load_metadata_success(self, mock_read_csv):
        """Test that metadata is loaded correctly."""
        mock_df = pd.DataFrame({'idStation': [1], 'idMeasurement': [101]})
        mock_read_csv.return_value = mock_df
        with patch('os.path.exists', return_value=True):
            df = load_data.load_metadata('dummy.csv')
            self.assertIsNotNone(df)
            self.assertEqual(len(df), 1)

    @patch('pandas.read_csv', side_effect=FileNotFoundError)
    def test_load_metadata_not_found(self, mock_read_csv):
        """Test that None is returned when metadata file is not found."""
        df = load_data.load_metadata('non_existent.csv')
        self.assertIsNone(df)

    @patch('numpy.load')
    def test_load_signal_success(self, mock_load):
        """Test that a signal is loaded correctly."""
        mock_signal = np.array([1, 2, 3])
        mock_load.return_value = mock_signal
        with patch('os.path.exists', return_value=True):
            signal = load_data.load_signal(1, 101)
            self.assertIsNotNone(signal)
            self.assertEqual(len(signal), 3)

    @patch('numpy.load', side_effect=FileNotFoundError)
    def test_load_signal_not_found(self, mock_load):
        """Test that None is returned when signal file is not found."""
        signal = load_data.load_signal(99, 999)
        self.assertIsNone(signal)

if __name__ == '__main__':
    unittest.main()