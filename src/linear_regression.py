"""
linear_regression.py — Multiple Linear Regression from Scratch
===============================================================

Implements Batch Gradient Descent for multiple linear regression
with optional L2 (Ridge) regularization, using only NumPy.

Mathematical Model
------------------
    ŷ = Xw + b

Loss Function (MSE with optional L2 penalty)
---------------------------------------------
    J = (1/m) Σ(ŷᵢ − yᵢ)²  +  λ Σwⱼ²

Gradients
---------
    ∂J/∂w = (2/m) Xᵀ(ŷ − y)  +  2λw
    ∂J/∂b = (2/m) Σ(ŷ − y)
"""

import numpy as np


class LinearRegressionScratch:
    """
    Multiple Linear Regression trained with Batch Gradient Descent.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent updates.
    n_epochs : int, default=1000
        Number of complete passes over the training data.
    l2_lambda : float, default=0.0
        L2 regularization strength (0 = no regularization).

    Attributes
    ----------
    weights : np.ndarray or None
        Learned weight vector w of shape (n_features,).
    bias : float or None
        Learned bias (intercept) term b.
    training_loss_history : list of float
        MSE loss on training data at each epoch.
    validation_loss_history : list of float
        MSE loss on validation data at each epoch (if val data provided).
    """

    def __init__(self, learning_rate=0.01, n_epochs=1000, l2_lambda=0.0):
        """Initialize hyperparameters and empty model state."""
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.l2_lambda = l2_lambda

        # Model parameters (set during fit)
        self.weights = None          # w vector
        self.bias = None             # b scalar

        # Loss tracking
        self.training_loss_history = []
        self.validation_loss_history = []

    # ──────────────────────────────────────────────────────────────
    # Input Validation
    # ──────────────────────────────────────────────────────────────

    def _validate_input(self, X, y=None):
        """
        Validate input arrays for NaN, inf, and dimension mismatches.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like or None
            Target vector (optional).

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,) or None

        Raises
        ------
        ValueError
            On invalid input.
        """
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D.")
        if X.shape[0] == 0:
            raise ValueError("X has 0 samples.")
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values.")
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values.")

        if y is not None:
            y = np.array(y, dtype=np.float64).ravel()
            if y.shape[0] != X.shape[0]:
                raise ValueError(
                    f"X has {X.shape[0]} samples but y has {y.shape[0]}."
                )
            if np.any(np.isnan(y)):
                raise ValueError("y contains NaN values.")
            if np.any(np.isinf(y)):
                raise ValueError("y contains infinite values.")

        return X, y

    # ──────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Generate predictions: ŷ = Xw + b.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted target values.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If X has wrong number of features.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X, _ = self._validate_input(X)

        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Expected {self.weights.shape[0]} features, "
                f"got {X.shape[1]}."
            )

        y_hat = X @ self.weights + self.bias
        return y_hat

    # ──────────────────────────────────────────────────────────────
    # Loss Computation
    # ──────────────────────────────────────────────────────────────

    def compute_loss(self, y_true, y_pred):
        """
        Compute Mean Squared Error loss with optional L2 penalty.

        MSE = (1/m) Σ(ŷᵢ − yᵢ)²
        L2  = λ Σwⱼ²
        J   = MSE + L2

        Parameters
        ----------
        y_true : np.ndarray of shape (m,)
            Actual target values.
        y_pred : np.ndarray of shape (m,)
            Predicted target values.

        Returns
        -------
        float
            The total loss.
        """
        m = len(y_true)
        mse = (1.0 / m) * np.sum((y_pred - y_true) ** 2)

        # Add L2 regularization penalty (does not include bias)
        if self.l2_lambda > 0.0 and self.weights is not None:
            l2_penalty = self.l2_lambda * np.sum(self.weights ** 2)
            return mse + l2_penalty

        return mse

    # ──────────────────────────────────────────────────────────────
    # Gradient Computation
    # ──────────────────────────────────────────────────────────────

    def compute_gradients(self, X, y_true, y_pred):
        """
        Compute gradients of the loss w.r.t. weights and bias.

        ∂J/∂w = (2/m) Xᵀ(ŷ − y)  +  2λw
        ∂J/∂b = (2/m) Σ(ŷ − y)

        Parameters
        ----------
        X : np.ndarray of shape (m, n_features)
            Feature matrix.
        y_true : np.ndarray of shape (m,)
            Actual targets.
        y_pred : np.ndarray of shape (m,)
            Predicted targets.

        Returns
        -------
        dw : np.ndarray of shape (n_features,)
            Gradient of loss w.r.t. weights.
        db : float
            Gradient of loss w.r.t. bias.
        """
        m = X.shape[0]
        error = y_pred - y_true  # (m,)

        dw = (2.0 / m) * (X.T @ error)     # (n_features,)
        db = (2.0 / m) * np.sum(error)      # scalar

        # Add L2 regularization gradient (only for weights, not bias)
        if self.l2_lambda > 0.0:
            dw += 2.0 * self.l2_lambda * self.weights

        return dw, db

    # ──────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            verbose=True, print_every=100):
        """
        Train the model using Batch Gradient Descent.

        Steps per epoch:
            1. Forward pass  — compute predictions
            2. Loss          — compute MSE (+ L2)
            3. Gradients     — compute ∂J/∂w and ∂J/∂b
            4. Update        — w ← w − α·dw ,  b ← b − α·db

        Parameters
        ----------
        X_train : np.ndarray of shape (m, n_features)
            Training features.
        y_train : np.ndarray of shape (m,)
            Training targets.
        X_val : np.ndarray or None
            Validation features (for tracking only; not used for updates).
        y_val : np.ndarray or None
            Validation targets.
        verbose : bool, default=True
            Whether to print progress.
        print_every : int, default=100
            Print interval in epochs.

        Returns
        -------
        self
            The trained model.
        """
        # Validate training data
        X_train, y_train = self._validate_input(X_train, y_train)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val, y_val = self._validate_input(X_val, y_val)
            if X_val.shape[1] != X_train.shape[1]:
                raise ValueError(
                    f"Validation features ({X_val.shape[1]}) must match "
                    f"training features ({X_train.shape[1]})."
                )

        m, n_features = X_train.shape

        # Initialize parameters to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Reset loss histories
        self.training_loss_history = []
        self.validation_loss_history = []

        if verbose:
            print("=" * 60)
            print("TRAINING LINEAR REGRESSION")
            print("=" * 60)
            print(f"  Samples       : {m}")
            print(f"  Features      : {n_features}")
            print(f"  Learning rate : {self.learning_rate}")
            print(f"  Epochs        : {self.n_epochs}")
            print(f"  L2 lambda     : {self.l2_lambda}")
            print("-" * 60)

        for epoch in range(self.n_epochs):
            # 1. Forward pass
            y_pred = X_train @ self.weights + self.bias

            # 2. Compute training loss
            train_loss = self.compute_loss(y_train, y_pred)
            self.training_loss_history.append(train_loss)

            # 3. Compute validation loss (no gradient update)
            if has_val:
                y_val_pred = X_val @ self.weights + self.bias
                val_loss = self.compute_loss(y_val, y_val_pred)
                self.validation_loss_history.append(val_loss)

            # 4. Compute gradients
            dw, db = self.compute_gradients(X_train, y_train, y_pred)

            # 5. Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print progress
            if verbose and (epoch % print_every == 0 or epoch == self.n_epochs - 1):
                msg = f"  Epoch {epoch:5d}/{self.n_epochs}  |  Train Loss: {train_loss:.6f}"
                if has_val:
                    msg += f"  |  Val Loss: {val_loss:.6f}"
                print(msg)

        if verbose:
            print("-" * 60)
            print(f"[INFO] Training complete.")
            print(f"   Final train loss: {self.training_loss_history[-1]:.6f}")
            if has_val:
                print(f"   Final val loss  : {self.validation_loss_history[-1]:.6f}")
            print("=" * 60)

        return self

    # ──────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────

    def evaluate_loss(self, X, y):
        """
        Evaluate MSE loss on given data without updating parameters.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target vector.

        Returns
        -------
        float
            The MSE loss (with L2 penalty if applicable).
        """
        X, y = self._validate_input(X, y)
        y_pred = self.predict(X)
        return self.compute_loss(y, y_pred)
