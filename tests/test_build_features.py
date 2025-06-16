# tests/test_build_features.py

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.features import build_features

class TestBuildFeatures(unittest.TestCase):
    """
    Test suite for feature engineering functions.
    """

    def test_calculate_statistical_features_with_valid_signal(self):
        """
        Tests that all statistical features are calculated correctly for a valid signal.
        """
        # --- Setup ---
        signal = np.array([1, 2, 3, 4, 5])
        expected_features = [
            np.mean(signal),
            np.std(signal),
            np.min(signal),
            np.max(signal),
            np.mean(np.abs(signal)),
            np.std(np.abs(signal))
        ]

        # --- Execute ---
        features = build_features.calculate_statistical_features(signal)

        # --- Assert ---
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 6)
        for i in range(6):
            self.assertAlmostEqual(features[i], expected_features[i])

    def test_calculate_statistical_features_with_empty_signal(self):
        """
        Tests that NaN values are returned for an empty or None signal.
        """
        # Test with an empty array
        empty_features = build_features.calculate_statistical_features(np.array([]))
        self.assertTrue(all(np.isnan(f) for f in empty_features))

        # Test with None
        none_features = build_features.calculate_statistical_features(None)
        self.assertTrue(all(np.isnan(f) for f in none_features))


    @patch('src.features.build_features.load_data.load_signal')
    def test_create_feature_dataset_with_mock_data(self, mock_load_signal):
        """
        Tests creating a feature dataset from metadata, correctly handling missing signals.
        """
        # --- Setup ---
        metadata = pd.DataFrame({
            'idStation': [1, 1, 2],
            'idMeasurement': [101, 102, 201],
            'faultAnnotation': [0, 1, 0]
        })
        # Mock the return values for load_signal for each row
        mock_load_signal.side_effect = [
            np.array([1, 2, 3]),   # Valid signal
            np.array([10, 20, 30]),# Valid signal
            None                   # Simulate a missing signal file
        ]

        # --- Execute ---
        feature_df = build_features.create_feature_dataset(metadata)

        # --- Assert ---
        # The row with the missing signal (None) should be dropped automatically
        self.assertEqual(len(feature_df), 2)
        self.assertIn('mean', feature_df.columns)
        self.assertIn('faultAnnotation', feature_df.columns)

        # Check calculated features for the first valid signal
        self.assertAlmostEqual(feature_df.iloc[0]['mean'], 2.0)
        self.assertEqual(feature_df.iloc[0]['faultAnnotation'], 0)

        # Check calculated features for the second valid signal
        self.assertAlmostEqual(feature_df.iloc[1]['mean'], 20.0)
        self.assertEqual(feature_df.iloc[1]['faultAnnotation'], 1)

        # Verify load_signal was called for all three rows
        self.assertEqual(mock_load_signal.call_count, 3)

if __name__ == '__main__':
    unittest.main()