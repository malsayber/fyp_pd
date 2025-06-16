# tests/test_build_features.py

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.features import build_features

class TestBuildFeatures(unittest.TestCase):

    def test_calculate_statistical_features(self):
        """Test the statistical feature calculation."""
        signal = np.array([1, 2, 3, 4, 5])
        features = build_features.calculate_statistical_features(signal)
        self.assertAlmostEqual(features[0], 3.0)  # mean
        self.assertAlmostEqual(features[1], np.std(signal))  # std

    def test_calculate_statistical_features_empty(self):
        """Test with an empty signal."""
        features = build_features.calculate_statistical_features(np.array([]))
        self.assertTrue(all(np.isnan(f) for f in features))

    @patch('src.data_processing.load_data.load_signal')
    def test_create_feature_dataset(self, mock_load_signal):
        """Test the creation of a feature dataset."""
        metadata = pd.DataFrame({
            'idStation': [1, 1],
            'idMeasurement': [101, 102],
            'faultAnnotation': [0, 1]
        })
        mock_load_signal.side_effect = [np.array([1, 2, 3]), np.array([10, 20, 30])]
        feature_df = build_features.create_feature_dataset(metadata)
        self.assertEqual(len(feature_df), 2)
        self.assertIn('mean', feature_df.columns)
        self.assertAlmostEqual(feature_df.iloc[0]['mean'], 2.0)
        self.assertAlmostEqual(feature_df.iloc[1]['mean'], 20.0)

if __name__ == '__main__':
    unittest.main()