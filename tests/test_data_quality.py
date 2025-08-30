import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import json

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import validate_data_quality, generate_data_hash, setup_logging

class TestDataQuality(unittest.TestCase):
    """Test suite for data quality validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample valid data
        self.valid_X = pd.DataFrame({
            'feature1': [0.1, 0.5, 0.8, 0.3],
            'feature2': [0.9, 0.2, 0.6, 0.4],
            'feature3': [0.0, 1.0, 0.5, 0.7]
        })
        
        self.valid_y = pd.Series([0, 1, 2, 0])
        
        self.expected_features = ['feature1', 'feature2', 'feature3']
        
        # Setup minimal logging for tests
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_config = {
                'logging': {
                    'level': 'WARNING',  # Reduce noise in tests
                    'format': '%(levelname)s - %(message)s',
                    'file': 'logs/test.log'
                }
            }
            json.dump(test_config, f)
            self.config_file = f.name
        
        self.logger = setup_logging(self.config_file)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.config_file):
            os.unlink(self.config_file)
    
    def test_valid_data_passes(self):
        """Test that valid data passes all checks"""
        result = validate_data_quality(
            self.valid_X, self.valid_y, self.expected_features, self.logger
        )
        self.assertTrue(result, "Valid data should pass validation")
    
    def test_shape_mismatch_fails(self):
        """Test that shape mismatch is detected"""
        invalid_y = pd.Series([0, 1])  # Wrong length
        result = validate_data_quality(
            self.valid_X, invalid_y, self.expected_features, self.logger
        )
        self.assertFalse(result, "Shape mismatch should fail validation")
    
    def test_missing_features_fails(self):
        """Test that missing features are detected"""
        incomplete_X = self.valid_X[['feature1', 'feature2']]  # Missing feature3
        result = validate_data_quality(
            incomplete_X, self.valid_y, self.expected_features, self.logger
        )
        self.assertFalse(result, "Missing features should fail validation")
    
    def test_missing_values_fails(self):
        """Test that missing values are detected"""
        invalid_X = self.valid_X.copy()
        invalid_X.loc[0, 'feature1'] = np.nan
        
        result = validate_data_quality(
            invalid_X, self.valid_y, self.expected_features, self.logger
        )
        self.assertFalse(result, "Missing values should fail validation")
    
    def test_invalid_target_values_fails(self):
        """Test that invalid target values are detected"""
        invalid_y = pd.Series([0, 1, 3, 0])  # 3 is not a valid target
        result = validate_data_quality(
            self.valid_X, invalid_y, self.expected_features, self.logger
        )
        self.assertFalse(result, "Invalid target values should fail validation")
    
    def test_unnormalized_features_warning(self):
        """Test that unnormalized features generate warnings but don't fail"""
        unnorm_X = self.valid_X.copy()
        unnorm_X['feature1'] = [-0.1, 1.5, 0.5, 0.3]  # Outside [0,1] range
        
        # This should pass but generate warnings
        result = validate_data_quality(
            unnorm_X, self.valid_y, self.expected_features, self.logger
        )
        # Note: We expect this to pass because normalization warnings don't fail validation
        self.assertTrue(result, "Unnormalized features should warn but not fail")

class TestDataVersioning(unittest.TestCase):
    """Test suite for data versioning"""
    
    def test_data_hash_consistency(self):
        """Test that same data produces same hash"""
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        hash1 = generate_data_hash(df1)
        hash2 = generate_data_hash(df2)
        
        self.assertEqual(hash1, hash2, "Same data should produce same hash")
    
    def test_data_hash_different(self):
        """Test that different data produces different hash"""
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df2 = pd.DataFrame({'A': [1, 2, 4], 'B': [4, 5, 6]})  # Different data
        
        hash1 = generate_data_hash(df1)
        hash2 = generate_data_hash(df2)
        
        self.assertNotEqual(hash1, hash2, "Different data should produce different hash")

class TestMLDataIntegrity(unittest.TestCase):
    """Integration tests for ML data preparation"""
    
    def setUp(self):
        """Set up test data that mimics actual ML dataset structure"""
        self.ml_data = pd.DataFrame({
            'form_diff_normalized': [0.6, 0.3, 0.8, 0.4, 0.2],
            'elo_diff_normalized': [0.7, 0.4, 0.9, 0.5, 0.1],
            'h2h_score': [0.5, 0.5, 0.6, 0.4, 0.5],
            'home_advantage': [1, 1, 1, 1, 1],
            'matchday_normalized': [0.1, 0.2, 0.3, 0.4, 0.5],
            'season_period_numeric': [0, 0, 1, 1, 2],
            'shots_diff_normalized': [0.6, 0.4, 0.8, 0.3, 0.7],
            'corners_diff_normalized': [0.5, 0.6, 0.4, 0.8, 0.2],
            'points_diff_normalized': [0.7, 0.3, 0.9, 0.1, 0.5],
            'position_diff_normalized': [0.8, 0.2, 0.6, 0.4, 0.3],
            'FullTimeResult': ['H', 'D', 'A', 'H', 'A']
        })
    
    def test_feature_extraction(self):
        """Test that features are correctly extracted"""
        expected_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'home_advantage', 'matchday_normalized', 'season_period_numeric',
            'shots_diff_normalized', 'corners_diff_normalized',
            'points_diff_normalized', 'position_diff_normalized'
        ]
        
        X = self.ml_data[expected_features]
        
        self.assertEqual(list(X.columns), expected_features)
        self.assertEqual(len(X), len(self.ml_data))
    
    def test_target_encoding(self):
        """Test that targets are correctly encoded"""
        label_mapping = {'H': 0, 'D': 1, 'A': 2}
        encoded_targets = self.ml_data['FullTimeResult'].map(label_mapping)
        
        expected = pd.Series([0, 1, 2, 0, 2])
        pd.testing.assert_series_equal(encoded_targets, expected, check_names=False)
    
    def test_no_data_leakage(self):
        """Test that non-ML columns are properly excluded"""
        ml_columns = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'home_advantage', 'matchday_normalized', 'season_period_numeric',
            'shots_diff_normalized', 'corners_diff_normalized',
            'points_diff_normalized', 'position_diff_normalized'
        ]
        
        # These should NOT be in ML dataset
        excluded_columns = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult']
        
        X = self.ml_data[ml_columns]
        
        for col in excluded_columns:
            if col in self.ml_data.columns:
                self.assertNotIn(col, X.columns, f"{col} should not be in features")

if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(loader.loadTestsFromTestCase(TestDataQuality))
    suite.addTest(loader.loadTestsFromTestCase(TestDataVersioning))
    suite.addTest(loader.loadTestsFromTestCase(TestMLDataIntegrity))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)