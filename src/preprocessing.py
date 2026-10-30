"""
preprocessing.py — Data Preprocessing and Feature Scaling
==========================================================

Provides a custom StandardScaler implementation (z-score normalization)
built from scratch using NumPy, plus utility functions for feature
selection, missing-value handling, and numerical validation.

StandardScaler Formula:
    z = (x - μ) / σ
where μ is the column mean and σ is the column standard deviation.
Columns with zero variance have σ set to 1.0 to avoid division by zero.
"""

import numpy as np
import pandas as pd


class StandardScaler:
    """
    Standard Scaler (Z-Score Normalization) implemented from scratch.

    Transforms features by removing the mean and scaling to unit variance:
        z = (x - mean) / std

    Attributes
    ----------
    mean_ : np.ndarray or None
        Per-feature mean computed during fit.
    std_ : np.ndarray or None
        Per-feature standard deviation computed during fit.
        Zero-variance features have std set to 1.0.
    is_fitted : bool
        Whether the scaler has been fitted on data.
    n_features_ : int or None
        Number of features seen during fit.
    """

    def __init__(self):
        """Initialize the StandardScaler with unfitted state."""
        self.mean_ = None
        self.std_ = None
        self.is_fitted = False
        self.n_features_ = None

    @property
    def mean(self):
        """Return the mean of features."""
        return self.mean_

    @mean.setter
    def mean(self, value):
        self.mean_ = np.array(value, dtype=np.float64) if value is not None else None
        if self.mean_ is not None and self.std_ is not None:
            self.is_fitted = True
            self.n_features_ = len(self.mean_)

    @property
    def std(self):
        """Return the standard deviation of features."""
        return self.std_

    @std.setter
    def std(self, value):
        self.std_ = np.array(value, dtype=np.float64) if value is not None else None
        if self.mean_ is not None and self.std_ is not None:
            self.is_fitted = True
            self.n_features_ = len(self.std_)

    def fit(self, X):
        """
        Compute the mean and standard deviation from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data to compute scaling parameters from.

        Returns
        -------
        self
            The fitted scaler instance.

        Raises
        ------
        ValueError
            If X contains NaN or infinite values, or is not 2D.
        """
        X = self._validate_input(X)

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # Handle zero-variance features: set std to 1.0 to avoid division by zero
        zero_var_mask = self.std_ == 0.0
        if np.any(zero_var_mask):
            n_zero = np.sum(zero_var_mask)
            print(f"[WARNING] {n_zero} feature(s) have zero variance. "
                  f"Setting their std to 1.0.")
            self.std_[zero_var_mask] = 1.0

        self.n_features_ = X.shape[1]
        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Apply z-score normalization: z = (x - mean) / std.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        np.ndarray
            Scaled data with zero mean and unit variance.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        ValueError
            If X has a different number of features than training data.
        """
        if not self.is_fitted:
            raise RuntimeError("StandardScaler has not been fitted yet. "
                               "Call fit() first.")

        X = self._validate_input(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}."
            )

        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """
        Fit the scaler on X and then transform X.

        Equivalent to calling fit(X) followed by transform(X), but
        slightly more efficient.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to fit and transform.

        Returns
        -------
        np.ndarray
            Scaled data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """
        Reverse the z-score normalization: x = z * std + mean.

        Parameters
        ----------
        X_scaled : np.ndarray of shape (n_samples, n_features)
            Scaled data to reverse.

        Returns
        -------
        np.ndarray
            Data in original scale.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError("StandardScaler has not been fitted yet. "
                               "Call fit() first.")

        X_scaled = np.array(X_scaled, dtype=np.float64)

        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)

        if X_scaled.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X_scaled.shape[1]}."
            )

        return X_scaled * self.std_ + self.mean_

    def _validate_input(self, X):
        """
        Validate input array for correct type, dimensions, NaN, and inf.

        Parameters
        ----------
        X : array-like
            Input to validate.

        Returns
        -------
        np.ndarray
            Validated 2D float64 array.

        Raises
        ------
        ValueError
            If input is invalid.
        """
        X = np.array(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array.")

        if X.shape[0] == 0:
            raise ValueError("Input array has 0 samples.")

        if np.any(np.isnan(X)):
            raise ValueError("Input contains NaN values. "
                             "Handle missing values before scaling.")

        if np.any(np.isinf(X)):
            raise ValueError("Input contains infinite values. "
                             "Remove or replace them before scaling.")

        return X


def select_features(df, feature_names):
    """
    Select specified features from a DataFrame and return as numpy array.

    Parameters
    ----------
    df : pd.DataFrame
        The source DataFrame.
    feature_names : list of str
        Column names to select.

    Returns
    -------
    np.ndarray
        2D array of selected features, shape (n_samples, n_features).

    Raises
    ------
    KeyError
        If any feature name is not found in the DataFrame.
    """
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Features not found in DataFrame: {missing_cols}")

    X = df[feature_names].values.astype(np.float64)
    print(f"[INFO] Selected {len(feature_names)} features: {feature_names}")
    print(f"   Shape: {X.shape}")
    return X


def handle_missing_values(df, strategy='median'):
    """
    Fill missing values in numeric columns using the specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potential missing values.
    strategy : str, default='median'
        Strategy for filling missing values: 'median' or 'mean'.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        - The DataFrame with missing values filled.
        - A dictionary mapping column names to their fill values.

    Raises
    ------
    ValueError
        If an unsupported strategy is specified.
    """
    if strategy not in ('median', 'mean'):
        raise ValueError(f"Unsupported strategy '{strategy}'. "
                         f"Use 'median' or 'mean'.")

    df_filled = df.copy()
    fill_values = {}

    numeric_cols = df_filled.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        if df_filled[col].isnull().any():
            if strategy == 'median':
                fill_val = df_filled[col].median()
            else:
                fill_val = df_filled[col].mean()

            df_filled[col].fillna(fill_val, inplace=True)
            fill_values[col] = fill_val

    if fill_values:
        print(f"[INFO] Filled missing values using '{strategy}' strategy:")
        for col, val in fill_values.items():
            print(f"   {col}: filled with {val:.4f}")
    else:
        print("[INFO] No missing values found in numeric columns.")

    return df_filled, fill_values


def validate_numerical(X):
    """
    Check if a numpy array contains NaN or infinite values.

    Parameters
    ----------
    X : np.ndarray
        Array to validate.

    Returns
    -------
    bool
        True if the array is clean (no NaN or inf), False otherwise.
    """
    X = np.array(X, dtype=np.float64)

    has_nan = np.any(np.isnan(X))
    has_inf = np.any(np.isinf(X))

    if has_nan:
        nan_count = np.sum(np.isnan(X))
        print(f"[WARNING] Found {nan_count} NaN value(s).")
    if has_inf:
        inf_count = np.sum(np.isinf(X))
        print(f"[WARNING] Found {inf_count} infinite value(s).")

    is_clean = not has_nan and not has_inf

    if is_clean:
        print("[INFO] Array is clean: no NaN or inf values.")

    return is_clean


def remove_missing_target(df, target_col):
    """
    Drop rows where the target column has missing values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to clean.
    target_col : str
        Name of the target column.

    Returns
    -------
    pd.DataFrame
        DataFrame with rows containing missing target values removed.

    Raises
    ------
    KeyError
        If the target column is not in the DataFrame.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    n_before = len(df)
    df_clean = df.dropna(subset=[target_col]).copy()
    n_after = len(df_clean)
    n_dropped = n_before - n_after

    if n_dropped > 0:
        print(f"[WARNING] Dropped {n_dropped} rows with missing '{target_col}' values.")
    else:
        print(f"[INFO] No missing values in target column '{target_col}'.")

    return df_clean
