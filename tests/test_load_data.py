# tests/test_load_data.py

import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch

# Correctly import the module to be tested
from src.data_processing import load_data

class TestLoadData(unittest.TestCase):
    """
    Test suite for data loading utilities.
    """

    @patch('src.data_processing.load_data.pd.read_csv')
    def test_load_metadata_success(self, mock_read_csv):
        """
        Tests that metadata is loaded into a DataFrame when the file exists.
        """
        # --- Setup ---
        # Configure the mock to return a sample DataFrame
        mock_df = pd.DataFrame({'idStation': [1, 2], 'idMeasurement': [101, 102]})
        mock_read_csv.return_value = mock_df

        # --- Execute ---
        df = load_data.load_metadata('dummy_path.csv')

        # --- Assert ---
        # Verify that read_csv was called with a path inside the 'data' directory
        mock_read_csv.assert_called_once()
        call_path = mock_read_csv.call_args[0][0]
        self.assertTrue(call_path.endswith(os.path.join('data', 'dummy_path.csv')))

        # Verify the returned DataFrame is the one from the mock
        self.assertIsNotNone(df)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(len(df), 2)

    @patch('src.data_processing.load_data.pd.read_csv', side_effect=FileNotFoundError)
    def test_load_metadata_not_found(self, mock_read_csv):
        """
        Tests that the function returns None when the metadata file is not found.
        """
        # --- Execute ---
        df = load_data.load_metadata('non_existent.csv')

        # --- Assert ---
        mock_read_csv.assert_called_once()
        self.assertIsNone(df)

    @patch('src.data_processing.load_data.np.load')
    def test_load_signal_success(self, mock_np_load):
        """
        Tests that a signal is loaded correctly when the file exists.
        """
        # --- Setup ---
        mock_signal = np.array([1, 2, 3])
        mock_np_load.return_value = mock_signal

        # --- Execute ---
        signal = load_data.load_signal(station_id=1, measurement_id=101)

        # --- Assert ---
        # Verify that np.load was called with the correct path structure
        mock_np_load.assert_called_once()
        call_path = mock_np_load.call_args[0][0]
        self.assertTrue(call_path.endswith(os.path.join('data', '1', '101.npy')))

        # Verify the returned signal is the one from the mock
        self.assertIsNotNone(signal)
        np.testing.assert_array_equal(signal, mock_signal)

    @patch('src.data_processing.load_data.np.load', side_effect=FileNotFoundError)
    def test_load_signal_not_found(self, mock_np_load):
        """
        Tests that the function returns None when a signal file is not found.
        """
        # --- Execute ---
        signal = load_data.load_signal(station_id=99, measurement_id=999)

        # --- Assert ---
        mock_np_load.assert_called_once()
        self.assertIsNone(signal)

if __name__ == '__main__':
    unittest.main()