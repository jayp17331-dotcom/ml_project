"""
feature_engineering.py — Feature Engineering for Ames Housing Dataset
======================================================================

Creates derived features from the raw Ames Housing columns to improve
model performance. Each function adds one or more new columns to the
DataFrame and returns a copy so the original is not mutated.

Engineered features:
    - HouseAge:        Age of house at time of sale reference year
    - AreaPerBedroom:  Living area per bedroom (handles zero bedrooms)
    - TotalBathrooms:  Full baths + 0.5 × half baths
    - TotalSF:         Total square footage across all floors + basement
"""

import numpy as np
import pandas as pd


def add_house_age(df, reference_year=2010):
    """
    Create a HouseAge feature: reference_year − YearBuilt.

    The Ames Housing dataset sales span roughly 2006–2010, so a default
    reference year of 2010 is used.  A higher HouseAge indicates an
    older house.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the column 'YearBuilt'.
    reference_year : int, default=2010
        The year to measure age from.

    Returns
    -------
    pd.DataFrame
        Copy of df with a new 'HouseAge' column.

    Raises
    ------
    KeyError
        If 'YearBuilt' is not present in df.
    """
    if 'YearBuilt' not in df.columns:
        raise KeyError("Column 'YearBuilt' not found in DataFrame.")

    df = df.copy()
    df['HouseAge'] = reference_year - df['YearBuilt']

    # Clip negative ages (houses built after reference year) to 0
    df['HouseAge'] = df['HouseAge'].clip(lower=0)

    print(f"[INFO] Added 'HouseAge' (reference year={reference_year})")
    print(f"   Range: {df['HouseAge'].min()} - {df['HouseAge'].max()} years")
    return df


def add_area_per_bedroom(df):
    """
    Create an AreaPerBedroom feature: GrLivArea / BedroomAbvGr.

    When BedroomAbvGr is 0 (e.g. studio), the value is set to GrLivArea
    (i.e. the entire area counts as one open space).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'GrLivArea' and 'BedroomAbvGr'.

    Returns
    -------
    pd.DataFrame
        Copy of df with a new 'AreaPerBedroom' column.

    Raises
    ------
    KeyError
        If required columns are missing.
    """
    required = ['GrLivArea', 'BedroomAbvGr']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()
    bedrooms = df['BedroomAbvGr'].copy()

    # Handle division by zero: where bedrooms == 0, use 1 so result = GrLivArea
    safe_bedrooms = bedrooms.replace(0, 1)
    df['AreaPerBedroom'] = df['GrLivArea'] / safe_bedrooms

    n_zero = (bedrooms == 0).sum()
    if n_zero > 0:
        print(f"   [INFO] {n_zero} homes have 0 bedrooms; "
              f"AreaPerBedroom set to GrLivArea for those.")

    print(f"[INFO] Added 'AreaPerBedroom'")
    print(f"   Range: {df['AreaPerBedroom'].min():.1f} - "
          f"{df['AreaPerBedroom'].max():.1f} sq ft/bedroom")
    return df


def add_total_bathrooms(df):
    """
    Create a TotalBathrooms feature: FullBath + 0.5 * HalfBath.

    If HalfBath is not present, TotalBathrooms equals FullBath.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'FullBath'.  'HalfBath' is optional.

    Returns
    -------
    pd.DataFrame
        Copy of df with a new 'TotalBathrooms' column.
    """
    if 'FullBath' not in df.columns:
        raise KeyError("Column 'FullBath' not found in DataFrame.")

    df = df.copy()
    df['TotalBathrooms'] = df['FullBath'].astype(float)

    if 'HalfBath' in df.columns:
        df['TotalBathrooms'] += 0.5 * df['HalfBath']
        print("[INFO] Added 'TotalBathrooms' (FullBath + 0.5 * HalfBath)")
    else:
        print("[INFO] Added 'TotalBathrooms' (FullBath only; HalfBath not found)")

    print(f"   Range: {df['TotalBathrooms'].min():.1f} - "
          f"{df['TotalBathrooms'].max():.1f}")
    return df


def add_total_sf(df):
    """
    Create a TotalSF feature summing all available floor areas.

    TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF

    Each component is included only if its column exists in the
    DataFrame, so the function degrades gracefully.

    Parameters
    ----------
    df : pd.DataFrame
        Should contain some subset of 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'.

    Returns
    -------
    pd.DataFrame
        Copy of df with a new 'TotalSF' column.
    """
    sf_columns = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    available = [c for c in sf_columns if c in df.columns]

    if not available:
        print("[WARNING] None of the SF columns found. 'TotalSF' not created.")
        return df.copy()

    df = df.copy()
    df['TotalSF'] = 0.0
    for col in available:
        df['TotalSF'] += df[col].fillna(0)

    print(f"[INFO] Added 'TotalSF' = {' + '.join(available)}")
    print(f"   Range: {df['TotalSF'].min():.0f} - {df['TotalSF'].max():.0f} sq ft")
    return df


def create_feature_set(df, feature_set='baseline'):
    """
    Return a list of feature column names for a given feature-set level.

    Available sets
    --------------
    'baseline'
        Core features: GrLivArea, OverallQual, YearBuilt, BedroomAbvGr,
        FullBath, GarageCars.
    'with_age'
        Baseline + HouseAge (must be created first via add_house_age).
    'with_area_per_bed'
        Baseline + AreaPerBedroom (must be created first).
    'expanded'
        Larger feature set including engineered features and additional
        raw columns for richer modelling.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame (used to verify column existence).
    feature_set : str, default='baseline'
        Which feature set to return.

    Returns
    -------
    list of str
        Feature column names present in df for the requested set.

    Raises
    ------
    ValueError
        If an unknown feature_set name is given.
    """
    baseline_features = [
        'GrLivArea', 'OverallQual', 'YearBuilt',
        'BedroomAbvGr', 'FullBath', 'GarageCars'
    ]

    if feature_set == 'baseline':
        features = baseline_features

    elif feature_set == 'with_age':
        features = baseline_features + ['HouseAge']

    elif feature_set == 'with_area_per_bed':
        features = baseline_features + ['AreaPerBedroom']

    elif feature_set == 'expanded':
        features = baseline_features + [
            'HouseAge', 'AreaPerBedroom', 'TotalBathrooms', 'TotalSF',
            'OverallCond', 'LotArea', 'TotalBsmtSF',
            '1stFlrSF', '2ndFlrSF', 'GarageArea', 'YearRemodAdd'
        ]

    else:
        raise ValueError(
            f"Unknown feature set '{feature_set}'. Choose from: "
            f"'baseline', 'with_age', 'with_area_per_bed', 'expanded'."
        )

    # Filter to only columns actually present in the DataFrame
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    if missing:
        print(f"[WARNING] Requested features not in DataFrame (skipped): {missing}")

    print(f"[INFO] Feature set '{feature_set}': {len(available)} features")
    for f in available:
        print(f"   * {f}")

    return available


def validate_features(X):
    """
    Check a feature array for NaN and inf values, replacing them if found.

    NaN values are replaced with the column median; inf values are
    replaced with the column max (for +inf) or min (for -inf) of
    the finite values.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix to validate.

    Returns
    -------
    np.ndarray
        Cleaned feature matrix.
    """
    X = np.array(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_nan = np.sum(np.isnan(X))
    n_inf = np.sum(np.isinf(X))

    if n_nan == 0 and n_inf == 0:
        print("[INFO] Feature array is clean.")
        return X

    X_clean = X.copy()

    # Replace NaN with column median
    if n_nan > 0:
        print(f"[WARNING] Found {n_nan} NaN values - replacing with column median.")
        for col_idx in range(X_clean.shape[1]):
            col = X_clean[:, col_idx]
            nan_mask = np.isnan(col)
            if np.any(nan_mask):
                finite_vals = col[~nan_mask & ~np.isinf(col)]
                if len(finite_vals) > 0:
                    median_val = np.median(finite_vals)
                else:
                    median_val = 0.0
                col[nan_mask] = median_val

    # Replace inf with column finite max/min
    if n_inf > 0:
        print(f"[WARNING] Found {n_inf} inf values - replacing with column max/min.")
        for col_idx in range(X_clean.shape[1]):
            col = X_clean[:, col_idx]
            finite_mask = np.isfinite(col)
            if not np.all(finite_mask):
                finite_vals = col[finite_mask]
                if len(finite_vals) > 0:
                    col_max = np.max(finite_vals)
                    col_min = np.min(finite_vals)
                else:
                    col_max = 0.0
                    col_min = 0.0
                col[np.isposinf(col)] = col_max
                col[np.isneginf(col)] = col_min

    print("[INFO] Feature array cleaned.")
    return X_clean
