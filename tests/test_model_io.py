import unittest
import numpy as np
import json
import os
import sys
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.linear_regression import LinearRegressionScratch
from src.preprocessing import StandardScaler
from src.model_io import save_model, load_model

class TestModelIO(unittest.TestCase):
    
    def test_save_and_load(self):
        """Test that saving and loading preserves model parameters."""
        model = LinearRegressionScratch(learning_rate=0.01, n_epochs=100)
        model.weights = np.array([1.5, -2.3, 0.7])
        model.bias = 0.5
        
        scaler_X = StandardScaler()
        scaler_X.mean = np.array([10.0, 20.0, 30.0])
        scaler_X.std = np.array([2.0, 3.0, 4.0])
        
        feature_names = ['f1', 'f2', 'f3']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            save_model(model, scaler_X, None, feature_names, filepath)
            loaded_model, loaded_scaler_X, loaded_scaler_y, loaded_features, _ = load_model(filepath)
            
            np.testing.assert_array_almost_equal(loaded_model.weights, model.weights)
            self.assertAlmostEqual(loaded_model.bias, model.bias)
            np.testing.assert_array_almost_equal(loaded_scaler_X.mean, scaler_X.mean)
            np.testing.assert_array_almost_equal(loaded_scaler_X.std, scaler_X.std)
            self.assertEqual(loaded_features, feature_names)
        finally:
            os.unlink(filepath)
    
    def test_predictions_after_reload(self):
        """Test that predictions are consistent after save/load."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = 3 * X[:, 0] - 2 * X[:, 1] + 1
        
        model = LinearRegressionScratch(learning_rate=0.05, n_epochs=200)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y, verbose=False)
        
        pred_before = model.predict(X_scaled)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            save_model(model, scaler, None, ['f1', 'f2'], filepath)
            loaded_model, loaded_scaler, _, _, _ = load_model(filepath)
            
            X_loaded_scaled = loaded_scaler.transform(X)
            pred_after = loaded_model.predict(X_loaded_scaled)
            
            np.testing.assert_array_almost_equal(pred_before, pred_after, decimal=5)
        finally:
            os.unlink(filepath)

if __name__ == '__main__':
    unittest.main()
