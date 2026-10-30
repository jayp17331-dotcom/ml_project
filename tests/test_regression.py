import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.linear_regression import LinearRegressionScratch
from src.preprocessing import StandardScaler

class TestLinearRegression(unittest.TestCase):
    
    def test_prediction_with_known_weights(self):
        """Test prediction with manually set weights."""
        model = LinearRegressionScratch()
        model.weights = np.array([2.0, 3.0])
        model.bias = 1.0
        X = np.array([[1, 2], [3, 4]])
        predictions = model.predict(X)
        expected = np.array([9.0, 19.0])  # [1*2+2*3+1, 3*2+4*3+1]
        np.testing.assert_array_almost_equal(predictions, expected)
    
    def test_training_simple_linear(self):
        """Test that model can learn a simple linear relationship."""
        np.random.seed(42)
        X = np.random.randn(100, 1) * 10
        y = 3 * X.squeeze() + 5 + np.random.randn(100) * 0.1
        model = LinearRegressionScratch(learning_rate=0.01, n_epochs=1000)
        # Need to standardize for stable training
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y, verbose=False)
        # Check that loss decreased
        self.assertLess(model.training_loss_history[-1], model.training_loss_history[0])
    
    def test_loss_reduction(self):
        """Test that loss decreases during training."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + 1
        model = LinearRegressionScratch(learning_rate=0.01, n_epochs=500)
        model.fit(X, y, verbose=False)
        self.assertLess(model.training_loss_history[-1], model.training_loss_history[0])
    
    def test_output_dimensions(self):
        """Test that predictions have correct shape."""
        model = LinearRegressionScratch()
        model.weights = np.array([1.0, 2.0, 3.0])
        model.bias = 0.0
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        pred = model.predict(X)
        self.assertEqual(pred.shape, (3,))
    
    def test_invalid_input(self):
        """Test handling of invalid inputs."""
        model = LinearRegressionScratch()
        with self.assertRaises((ValueError, TypeError)):
            model.fit(np.array([]), np.array([]))
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        X = np.random.RandomState(42).randn(50, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + 1
        
        model1 = LinearRegressionScratch(learning_rate=0.01, n_epochs=100)
        model1.fit(X, y, verbose=False)
        
        model2 = LinearRegressionScratch(learning_rate=0.01, n_epochs=100)
        model2.fit(X, y, verbose=False)
        
        np.testing.assert_array_almost_equal(model1.weights, model2.weights)
    
    def test_l2_regularization(self):
        """Test that L2 regularization produces smaller weights."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.5
        
        model_no_reg = LinearRegressionScratch(learning_rate=0.01, n_epochs=500, l2_lambda=0.0)
        model_no_reg.fit(X, y, verbose=False)
        
        model_reg = LinearRegressionScratch(learning_rate=0.01, n_epochs=500, l2_lambda=1.0)
        model_reg.fit(X, y, verbose=False)
        
        # Regularized model should have smaller weight magnitudes
        self.assertLess(np.sum(model_reg.weights**2), np.sum(model_no_reg.weights**2))

class TestStandardScaler(unittest.TestCase):
    
    def test_correct_means(self):
        """Test that fit computes correct means."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        scaler.fit(X)
        np.testing.assert_array_almost_equal(scaler.mean, np.array([3.0, 4.0]))
    
    def test_correct_std(self):
        """Test that fit computes correct standard deviations."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        scaler.fit(X)
        expected_std = np.std(X, axis=0)
        np.testing.assert_array_almost_equal(scaler.std, expected_std)
    
    def test_zero_variance(self):
        """Test handling of zero-variance features."""
        X = np.array([[1, 5], [1, 5], [1, 5]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Should not produce NaN or inf
        self.assertFalse(np.any(np.isnan(X_scaled)))
        self.assertFalse(np.any(np.isinf(X_scaled)))
    
    def test_consistent_transform(self):
        """Test that transform is consistent."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        X1 = scaler.fit_transform(X)
        X2 = scaler.transform(X)
        np.testing.assert_array_almost_equal(X1, X2)
    
    def test_inverse_transform(self):
        """Test that inverse_transform recovers original data."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X_recovered, X)

if __name__ == '__main__':
    unittest.main()
