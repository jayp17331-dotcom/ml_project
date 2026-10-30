"""
metrics.py — Regression Evaluation Metrics from Scratch
========================================================

Implements standard regression metrics using only NumPy.

Metrics
-------
MAE   = (1/m) Σ|yᵢ − ŷᵢ|
MSE   = (1/m) Σ(yᵢ − ŷᵢ)²
RMSE  = √MSE
R²    = 1 − SS_res / SS_tot
        where  SS_res = Σ(yᵢ − ŷᵢ)²
               SS_tot = Σ(yᵢ − ȳ)²
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────

def _validate_inputs(y_true, y_pred):
    """
    Validate that y_true and y_pred are compatible 1-D arrays.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    y_true, y_pred : np.ndarray
        Validated 1-D float64 arrays.

    Raises
    ------
    ValueError
        If inputs are empty or have mismatched lengths.
    """
    y_true = np.array(y_true, dtype=np.float64).ravel()
    y_pred = np.array(y_pred, dtype=np.float64).ravel()

    if y_true.shape[0] == 0:
        raise ValueError("y_true is empty.")
    if y_pred.shape[0] == 0:
        raise ValueError("y_pred is empty.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true has {y_true.shape[0]} elements, "
            f"y_pred has {y_pred.shape[0]}."
        )

    return y_true, y_pred


# ──────────────────────────────────────────────────────────────────
# Mean Absolute Error
# ──────────────────────────────────────────────────────────────────

def mean_absolute_error(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE).

    MAE = (1/m) Σ|yᵢ − ŷᵢ|

    Measures the average magnitude of errors without considering
    direction.  Lower is better; units match the target variable.

    Parameters
    ----------
    y_true : array-like of shape (m,)
        True target values.
    y_pred : array-like of shape (m,)
        Predicted values.

    Returns
    -------
    float
        The MAE value.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


# ──────────────────────────────────────────────────────────────────
# Mean Squared Error
# ──────────────────────────────────────────────────────────────────

def mean_squared_error(y_true, y_pred):
    """
    Compute Mean Squared Error (MSE).

    MSE = (1/m) Σ(yᵢ − ŷᵢ)²

    Penalises larger errors more heavily than MAE due to squaring.
    Units are the square of the target variable.

    Parameters
    ----------
    y_true : array-like of shape (m,)
        True target values.
    y_pred : array-like of shape (m,)
        Predicted values.

    Returns
    -------
    float
        The MSE value.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


# ──────────────────────────────────────────────────────────────────
# Root Mean Squared Error
# ──────────────────────────────────────────────────────────────────

def root_mean_squared_error(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE).

    RMSE = √MSE = √[ (1/m) Σ(yᵢ − ŷᵢ)² ]

    Same units as the target variable; sensitive to outliers.

    Parameters
    ----------
    y_true : array-like of shape (m,)
        True target values.
    y_pred : array-like of shape (m,)
        Predicted values.

    Returns
    -------
    float
        The RMSE value.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ──────────────────────────────────────────────────────────────────
# R-Squared (Coefficient of Determination)
# ──────────────────────────────────────────────────────────────────

def r_squared(y_true, y_pred):
    """
    Compute the Coefficient of Determination (R²).

    R² = 1 − SS_res / SS_tot

    where:
        SS_res = Σ(yᵢ − ŷᵢ)²     (residual sum of squares)
        SS_tot = Σ(yᵢ − ȳ)²      (total sum of squares)

    R² = 1 means perfect predictions.
    R² = 0 means the model is as good as predicting the mean.
    R² < 0 means the model is worse than predicting the mean.

    If the target is constant (SS_tot = 0), returns 0.0 to avoid
    division by zero.

    Parameters
    ----------
    y_true : array-like of shape (m,)
        True target values.
    y_pred : array-like of shape (m,)
        Predicted values.

    Returns
    -------
    float
        The R² score.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Handle constant target (zero variance)
    if ss_tot == 0.0:
        # If predictions also match, perfect score; otherwise 0
        if ss_res == 0.0:
            return 1.0
        return 0.0

    return float(1.0 - ss_res / ss_tot)


# ──────────────────────────────────────────────────────────────────
# Tolerance-Based Accuracy (Regression)
# ──────────────────────────────────────────────────────────────────

def tolerance_accuracy(y_true, y_pred, tolerance=0.15):
    """
    Compute tolerance accuracy: percentage of predictions within +/- tolerance of true values.

    Formula:
        Accuracy_tol = (1/m) sum( |y_true - y_pred| / y_true <= tolerance )

    Parameters
    ----------
    y_true : array-like of shape (m,)
        True target values.
    y_pred : array-like of shape (m,)
        Predicted values.
    tolerance : float, default=0.15
        Acceptable relative error threshold (e.g., 0.15 for +/- 15%).

    Returns
    -------
    float
        Fraction of predictions within the specified tolerance window (0.0 to 1.0).
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    # Avoid division by zero
    non_zero = y_true != 0.0
    if not np.any(non_zero):
        return 0.0
    relative_error = np.abs(y_true[non_zero] - y_pred[non_zero]) / np.abs(y_true[non_zero])
    return float(np.mean(relative_error <= tolerance))


# ──────────────────────────────────────────────────────────────────
# Classification Metrics (Binary Thresholding)
# ──────────────────────────────────────────────────────────────────

def classification_metrics(y_true, y_pred, threshold=None):
    """
    Compute classification metrics by converting continuous values into binary labels.

    Label Definition:
        Class 1 (High Price): Value >= threshold
        Class 0 (Affordable): Value < threshold

    Formulas:
        Accuracy  = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)

    Parameters
    ----------
    y_true : array-like of shape (m,)
        True target values.
    y_pred : array-like of shape (m,)
        Predicted values.
    threshold : float or None, default=None
        Threshold separating class 1 and class 0. If None, median of y_true is used.

    Returns
    -------
    dict
        Dictionary containing 'Accuracy', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'TN', 'FN', 'Threshold'.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if threshold is None:
        threshold = float(np.median(y_true))

    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
    fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
    tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
    fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1': float(f1),
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'Threshold': float(threshold),
    }


# ──────────────────────────────────────────────────────────────────
# Evaluate All Metrics
# ──────────────────────────────────────────────────────────────────

def evaluate_all(y_true, y_pred, threshold=None):
    """
    Compute all regression, tolerance, and classification metrics.

    Parameters
    ----------
    y_true : array-like of shape (m,)
        True target values.
    y_pred : array-like of shape (m,)
        Predicted values.
    threshold : float or None
        Optional threshold for binary classification (defaults to median of y_true).

    Returns
    -------
    dict
        Combined dictionary of all metrics.
    """
    mae_val = mean_absolute_error(y_true, y_pred)
    mse_val = mean_squared_error(y_true, y_pred)
    rmse_val = root_mean_squared_error(y_true, y_pred)
    r2_val = r_squared(y_true, y_pred)

    cls_res = classification_metrics(y_true, y_pred, threshold=threshold)
    tol_10 = tolerance_accuracy(y_true, y_pred, tolerance=0.10)
    tol_15 = tolerance_accuracy(y_true, y_pred, tolerance=0.15)
    tol_20 = tolerance_accuracy(y_true, y_pred, tolerance=0.20)

    return {
        'MAE': mae_val,
        'MSE': mse_val,
        'RMSE': rmse_val,
        'R2': r2_val,
        'mae': mae_val,
        'mse': mse_val,
        'rmse': rmse_val,
        'r2': r2_val,
        'Accuracy': cls_res['Accuracy'],
        'Precision': cls_res['Precision'],
        'Recall': cls_res['Recall'],
        'F1': cls_res['F1'],
        'Tolerance_10': tol_10,
        'Tolerance_15': tol_15,
        'Tolerance_20': tol_20,
        'TP': cls_res['TP'],
        'FP': cls_res['FP'],
        'TN': cls_res['TN'],
        'FN': cls_res['FN'],
        'Threshold': cls_res['Threshold'],
    }


# ──────────────────────────────────────────────────────────────────
# Pretty Print
# ──────────────────────────────────────────────────────────────────

def print_metrics(metrics_dict):
    """
    Pretty-print a comprehensive metrics dictionary.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metric names -> values (from evaluate_all).
    """
    print("=" * 50)
    print("           REGRESSION METRICS")
    print("=" * 50)
    for name in ['MAE', 'MSE', 'RMSE', 'R2']:
        if name in metrics_dict:
            value = metrics_dict[name]
            if name == 'R2':
                print(f"  {name:12s} : {value:.6f} ({value*100:.2f}%)")
            else:
                print(f"  {name:12s} : {value:,.4f}")

    if 'Accuracy' in metrics_dict:
        print("\n" + "-" * 50)
        print("  CLASSIFICATION METRICS (High vs Affordable)")
        print("-" * 50)
        thresh = metrics_dict.get('Threshold', None)
        if thresh is not None:
            print(f"  Threshold    : ${thresh:,.2f} (Median)")
        print(f"  Accuracy     : {metrics_dict['Accuracy']:.4f} ({metrics_dict['Accuracy']*100:.2f}%)")
        print(f"  Precision    : {metrics_dict['Precision']:.4f} ({metrics_dict['Precision']*100:.2f}%)")
        print(f"  Recall       : {metrics_dict['Recall']:.4f} ({metrics_dict['Recall']*100:.2f}%)")
        print(f"  F1-Score     : {metrics_dict['F1']:.4f}")

    if 'Tolerance_15' in metrics_dict:
        print("\n" + "-" * 50)
        print("         TOLERANCE ACCURACY")
        print("-" * 50)
        if 'Tolerance_10' in metrics_dict:
            print(f"  Within +/-10%: {metrics_dict['Tolerance_10']*100:.2f}%")
        print(f"  Within +/-15%: {metrics_dict['Tolerance_15']*100:.2f}%")
        if 'Tolerance_20' in metrics_dict:
            print(f"  Within +/-20%: {metrics_dict['Tolerance_20']*100:.2f}%")
    print("=" * 50)
