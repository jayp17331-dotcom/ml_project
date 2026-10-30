"""
data_split.py — Train / Validation / Test Splitting
=====================================================

Provides a function to split feature and target arrays into three
non-overlapping sets (train, validation, test) using NumPy random
permutation.  No external ML libraries are used.

Typical split ratios:
    70 % training  /  15 % validation  /  15 % test
"""

import numpy as np


def train_val_test_split(X, y,
                         train_ratio=0.70,
                         val_ratio=0.15,
                         test_ratio=0.15,
                         random_seed=42):
    """
    Split data into training, validation, and test sets.

    The data is randomly shuffled using a fixed seed for reproducibility,
    then divided according to the given ratios.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target vector.
    train_ratio : float, default=0.70
        Fraction of data for training.
    val_ratio : float, default=0.15
        Fraction of data for validation.
    test_ratio : float, default=0.15
        Fraction of data for testing.
    random_seed : int, default=42
        Seed for reproducible shuffling.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : np.ndarray
        The six split arrays.

    Raises
    ------
    ValueError
        If ratios do not sum to ~1.0, arrays have mismatched lengths,
        or arrays are empty.
    """
    # ── Input validation ──────────────────────────────────────────
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got X: {X.shape[0]}, y: {y.shape[0]}."
        )

    n_samples = X.shape[0]
    if n_samples == 0:
        raise ValueError("Cannot split an empty dataset.")

    # Verify ratios
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"Ratios must sum to 1.0. "
            f"Got {train_ratio} + {val_ratio} + {test_ratio} = {ratio_sum:.6f}."
        )

    if any(r < 0 for r in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("All ratios must be non-negative.")

    # ── Shuffle ────────────────────────────────────────────────────
    rng = np.random.RandomState(random_seed)
    indices = rng.permutation(n_samples)

    # ── Compute split boundaries ──────────────────────────────────
    n_train = int(np.floor(n_samples * train_ratio))
    n_val = int(np.floor(n_samples * val_ratio))
    # Test gets the remainder to avoid off-by-one from rounding
    n_test = n_samples - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # ── Verify no overlap ─────────────────────────────────────────
    assert len(set(train_idx) & set(val_idx)) == 0, \
        "Overlap detected between train and val sets!"
    assert len(set(train_idx) & set(test_idx)) == 0, \
        "Overlap detected between train and test sets!"
    assert len(set(val_idx) & set(test_idx)) == 0, \
        "Overlap detected between val and test sets!"
    assert len(train_idx) + len(val_idx) + len(test_idx) == n_samples, \
        "Total split sizes do not equal original dataset size!"

    # ── Slice arrays ──────────────────────────────────────────────
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # ── Report ────────────────────────────────────────────────────
    print("=" * 60)
    print("DATA SPLIT SUMMARY")
    print("=" * 60)
    print(f"  Total samples : {n_samples}")
    print(f"  Training      : {n_train:5d}  ({n_train/n_samples*100:.1f}%)")
    print(f"  Validation    : {n_val:5d}  ({n_val/n_samples*100:.1f}%)")
    print(f"  Test          : {n_test:5d}  ({n_test/n_samples*100:.1f}%)")
    print(f"  Random seed   : {random_seed}")
    print(f"  No overlap    : verified")
    print("=" * 60)

    return X_train, X_val, X_test, y_train, y_val, y_test
