import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.metrics import (
    mean_absolute_error, mean_squared_error, root_mean_squared_error,
    r_squared, classification_metrics, tolerance_accuracy, evaluate_all
)

class TestMetrics(unittest.TestCase):
    
    def test_classification_metrics(self):
        """Test accuracy, precision, recall, and f1 on manually verified data."""
        y_true = np.array([100, 200, 300, 400])
        y_pred = np.array([120, 190, 350, 280])
        # Threshold at 250:
        # y_true_binary = [0, 0, 1, 1]
        # y_pred_binary = [0, 0, 1, 1] (since 280 >= 250)
        res = classification_metrics(y_true, y_pred, threshold=250)
        self.assertEqual(res['Accuracy'], 1.0)
        self.assertEqual(res['Precision'], 1.0)
        self.assertEqual(res['Recall'], 1.0)
        self.assertEqual(res['F1'], 1.0)

    def test_tolerance_accuracy(self):
        """Test tolerance accuracy calculation."""
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([105.0, 240.0, 310.0, 500.0])
        # Relative errors: 5/100 = 0.05 (<=10%), 40/200 = 0.20 (>10%), 10/300 = 0.033 (<=10%), 100/400 = 0.25 (>10%)
        # 2 out of 4 <= 0.10 -> 0.50
        tol_10 = tolerance_accuracy(y_true, y_pred, tolerance=0.10)
        self.assertAlmostEqual(tol_10, 0.50)
    
    def test_mae_manual(self):
        """Test MAE with manually calculated example."""
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        # |0.5| + |0.5| + |0| + |1| = 2.0, /4 = 0.5
        self.assertAlmostEqual(mean_absolute_error(y_true, y_pred), 0.5)
    
    def test_mse_manual(self):
        """Test MSE with manually calculated example."""
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        # 0.25 + 0.25 + 0 + 1 = 1.5, /4 = 0.375
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 0.375)
    
    def test_rmse_manual(self):
        """Test RMSE with manually calculated example."""
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        expected = np.sqrt(0.375)
        self.assertAlmostEqual(root_mean_squared_error(y_true, y_pred), expected)
    
    def test_r2_perfect(self):
        """Test R² for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        self.assertAlmostEqual(r_squared(y_true, y_pred), 1.0)
    
    def test_r2_constant_baseline(self):
        """Test R² when predicting the mean (should be 0)."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.full(5, np.mean(y_true))
        self.assertAlmostEqual(r_squared(y_true, y_pred), 0.0)
    
    def test_r2_constant_target(self):
        """Test R² handling when target is constant."""
        y_true = np.array([5, 5, 5, 5])
        y_pred = np.array([5, 5, 5, 5])
        # When target is constant and predictions are perfect, R² should be 1.0 or handled gracefully
        result = r_squared(y_true, y_pred)
        self.assertTrue(np.isfinite(result))
    
    def test_dimension_mismatch(self):
        """Test that mismatched dimensions raise an error."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])
        with self.assertRaises(ValueError):
            mean_absolute_error(y_true, y_pred)
    
    def test_empty_input(self):
        """Test that empty inputs raise an error."""
        with self.assertRaises(ValueError):
            mean_absolute_error(np.array([]), np.array([]))

if __name__ == '__main__':
    unittest.main()
