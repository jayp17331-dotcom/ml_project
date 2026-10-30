"""
model_io.py — Model Serialization and Deserialization
======================================================

Save and load trained LinearRegressionScratch models along with
their StandardScaler parameters as JSON files.  This avoids any
dependency on pickle and ensures the saved artefacts are
human-readable and portable.

Saved JSON structure
--------------------
{
    "weights": [...],
    "bias": float,
    "feature_names": [...],
    "scaler_X": {"mean": [...], "std": [...]},
    "scaler_y": {"mean": [...], "std": [...]} | null,
    "hyperparameters": {"learning_rate": ..., "n_epochs": ..., "l2_lambda": ...},
    "config": {...} | null
}
"""

import json
import os
import numpy as np

from src.linear_regression import LinearRegressionScratch
from src.preprocessing import StandardScaler


def save_model(model, scaler_X, scaler_y, feature_names, filepath,
               config=None):
    """
    Save a trained model and its associated scalers to a JSON file.

    All numpy arrays are converted to Python lists for JSON
    serialization.

    Parameters
    ----------
    model : LinearRegressionScratch
        The trained model.
    scaler_X : StandardScaler
        Scaler fitted on the input features.
    scaler_y : StandardScaler or None
        Scaler fitted on the target variable (may be None).
    feature_names : list of str
        Ordered list of feature names used during training.
    filepath : str
        Destination path for the JSON file.
    config : dict or None
        Optional dictionary with extra configuration metadata
        (e.g. dataset path, split ratios, notes).

    Raises
    ------
    RuntimeError
        If the model or scaler_X has not been fitted.
    """
    if model.weights is None:
        raise RuntimeError("Cannot save an unfitted model.")
    if not scaler_X.is_fitted:
        raise RuntimeError("Cannot save an unfitted feature scaler.")

    # Build payload
    payload = {
        'weights': _to_list(model.weights),
        'bias': float(model.bias),
        'feature_names': list(feature_names),
        'n_features': len(feature_names),
        'scaler_X': {
            'mean': _to_list(scaler_X.mean_),
            'std': _to_list(scaler_X.std_),
        },
        'scaler_y': None,
        'hyperparameters': {
            'learning_rate': model.learning_rate,
            'n_epochs': model.n_epochs,
            'l2_lambda': model.l2_lambda,
        },
        'training_loss_final': (
            model.training_loss_history[-1]
            if model.training_loss_history else None
        ),
        'config': config,
    }

    if scaler_y is not None and scaler_y.is_fitted:
        payload['scaler_y'] = {
            'mean': _to_list(scaler_y.mean_),
            'std': _to_list(scaler_y.std_),
        }

    # Write JSON
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"[SUCCESS] Model saved to: {filepath}")
    print(f"   Weights : {len(payload['weights'])} features")
    print(f"   Bias    : {payload['bias']:.6f}")


def load_model(filepath):
    """
    Load a model, scalers, and feature names from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to the JSON model file.

    Returns
    -------
    model : LinearRegressionScratch
        Reconstructed model with weights and bias set.
    scaler_X : StandardScaler
        Reconstructed feature scaler.
    scaler_y : StandardScaler or None
        Reconstructed target scaler, or None if not saved.
    feature_names : list of str
        Feature names in the correct order.
    config : dict or None
        Any extra configuration that was saved.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    with open(filepath, 'r') as f:
        payload = json.load(f)

    # ── Reconstruct model ─────────────────────────────────────────
    hp = payload.get('hyperparameters', {})
    model = LinearRegressionScratch(
        learning_rate=hp.get('learning_rate', 0.01),
        n_epochs=hp.get('n_epochs', 1000),
        l2_lambda=hp.get('l2_lambda', 0.0),
    )
    model.weights = np.array(payload['weights'], dtype=np.float64)
    model.bias = float(payload['bias'])

    # ── Reconstruct feature scaler ────────────────────────────────
    scaler_X = StandardScaler()
    sx = payload['scaler_X']
    scaler_X.mean_ = np.array(sx['mean'], dtype=np.float64)
    scaler_X.std_ = np.array(sx['std'], dtype=np.float64)
    scaler_X.n_features_ = len(sx['mean'])
    scaler_X.is_fitted = True

    # ── Reconstruct target scaler (optional) ──────────────────────
    scaler_y = None
    if payload.get('scaler_y') is not None:
        sy = payload['scaler_y']
        scaler_y = StandardScaler()
        scaler_y.mean_ = np.array(sy['mean'], dtype=np.float64)
        scaler_y.std_ = np.array(sy['std'], dtype=np.float64)
        scaler_y.n_features_ = len(sy['mean'])
        scaler_y.is_fitted = True

    feature_names = payload.get('feature_names', [])
    config = payload.get('config', None)

    print(f"[SUCCESS] Model loaded from: {filepath}")
    print(f"   Features : {feature_names}")
    print(f"   Weights  : {len(model.weights)}")

    return model, scaler_X, scaler_y, feature_names, config


# ──────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────

def _to_list(arr):
    """
    Convert a numpy array (or scalar) to a JSON-serialisable Python list.
    """
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if isinstance(arr, (np.floating, np.integer)):
        return float(arr)
    return arr
