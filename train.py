#!/usr/bin/env python
"""
HousePrice-AI Training Pipeline
================================
Complete training pipeline for house price prediction using
Multiple Linear Regression implemented from scratch.

Usage:
    python train.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dataset, display_basic_info, check_missing_values
from src.preprocessing import StandardScaler, select_features, handle_missing_values, remove_missing_target
from src.feature_engineering import create_feature_set
from src.data_split import train_val_test_split
from src.linear_regression import LinearRegressionScratch
from src.metrics import evaluate_all, print_metrics
from src.visualization import (
    plot_loss_curve, plot_train_val_loss,
    plot_actual_vs_predicted, plot_residuals,
    plot_error_distribution
)
from src.model_io import save_model


def main():
    print("=" * 60)
    print("  HousePrice-AI: Training Pipeline")
    print("=" * 60)
    
    # ============================================================
    # Configuration
    # ============================================================
    config = {
        'data_path': str(PROJECT_ROOT / 'data' / 'raw' / 'train.csv'),
        'feature_set': 'baseline',
        'learning_rate': 0.01,
        'n_epochs': 1000,
        'l2_lambda': 0.0,
        'random_seed': 42,
        'train_ratio': 0.70,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'standardize_target': True,
        'log_transform_target': False,
    }
    
    print("\n[Config]")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # ============================================================
    # Step 1: Load Dataset
    # ============================================================
    print("\n" + "=" * 60)
    print("  Step 1: Loading Dataset")
    print("=" * 60)
    
    data_path = config['data_path']
   
    
    df = load_dataset(data_path)
    display_basic_info(df)
    
    # ============================================================
    # Step 2: Select Features and Target
    # ============================================================
    print("\n" + "=" * 60)
    print("  Step 2: Feature Selection and Preprocessing")
    print("=" * 60)
    
    # Detect target column and baseline features
    if 'SalePrice' in df.columns:
        target_col = 'SalePrice'
        baseline_features = ['GrLivArea', 'OverallQual', 'YearBuilt', 'BedroomAbvGr', 'FullBath', 'GarageCars']
    elif 'Price' in df.columns:
        target_col = 'Price'
        candidate_features = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']
        baseline_features = [c for c in candidate_features if c in df.columns]
        print(f"[INFO] Detected alternative housing schema: target='{target_col}', features={baseline_features}")
    else:
        target_col = df.columns[-1]
        baseline_features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in [target_col, 'Id']][:6]
        print(f"[INFO] Using numeric columns: target='{target_col}', features={baseline_features}")
    
    # Verify columns exist
    missing_cols = [c for c in baseline_features + [target_col] if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns in dataset: {missing_cols}")
        sys.exit(1)
    
    # Remove rows with missing target
    df = remove_missing_target(df, target_col)
    
    # Handle missing values in features
    feature_df = df[baseline_features].copy()
    feature_df, fill_values = handle_missing_values(feature_df, strategy='median')
    
    print(f"\nFeatures: {baseline_features}")
    print(f"Target: {target_col}")
    print(f"Fill values used: {fill_values}")
    
    # Extract arrays
    X = feature_df.values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # ============================================================
    # Step 3: Train/Validation/Test Split
    # ============================================================
    print("\n" + "=" * 60)
    print("  Step 3: Data Splitting")
    print("=" * 60)
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y,
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio'],
        random_seed=config['random_seed']
    )
    
    # ============================================================
    # Step 4: Feature Scaling
    # ============================================================
    print("\n" + "=" * 60)
    print("  Step 4: Feature Scaling (Standardization)")
    print("=" * 60)
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    print(f"Feature means (training): {np.round(scaler_X.mean, 2)}")
    print(f"Feature stds  (training): {np.round(scaler_X.std, 2)}")
    
    # Optionally standardize target
    scaler_y = None
    y_train_input = y_train.copy()
    y_val_input = y_val.copy()
    
    if config['log_transform_target']:
        print("\nApplying log transform to target...")
        y_train_input = np.log1p(y_train_input)
        y_val_input = np.log1p(y_val_input)
    
    if config['standardize_target']:
        print("Standardizing target variable...")
        scaler_y = StandardScaler()
        y_train_input = scaler_y.fit_transform(y_train_input.reshape(-1, 1)).squeeze()
        y_val_input = scaler_y.transform(y_val_input.reshape(-1, 1)).squeeze()
        print(f"Target mean (training): {scaler_y.mean[0]:.2f}")
        print(f"Target std  (training): {scaler_y.std[0]:.2f}")
    
    # ============================================================
    # Step 5: Model Training
    # ============================================================
    print("\n" + "=" * 60)
    print("  Step 5: Training Linear Regression (Gradient Descent)")
    print("=" * 60)
    
    model = LinearRegressionScratch(
        learning_rate=config['learning_rate'],
        n_epochs=config['n_epochs'],
        l2_lambda=config['l2_lambda']
    )
    
    model.fit(
        X_train_scaled, y_train_input,
        X_val=X_val_scaled, y_val=y_val_input,
        verbose=True, print_every=100
    )
    
    print(f"\nFinal training loss: {model.training_loss_history[-1]:.6f}")
    if model.validation_loss_history:
        print(f"Final validation loss: {model.validation_loss_history[-1]:.6f}")
    
    # ============================================================
    # Step 6: Evaluation on Validation Set
    # ============================================================
    print("\n" + "=" * 60)
    print("  Step 6: Validation Set Evaluation")
    print("=" * 60)
    
    # Get predictions and reverse transforms
    y_val_pred_scaled = model.predict(X_val_scaled)
    
    if config['standardize_target'] and scaler_y is not None:
        y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).squeeze()
    else:
        y_val_pred = y_val_pred_scaled
    
    if config['log_transform_target']:
        y_val_pred = np.expm1(y_val_pred)
    
    val_metrics = evaluate_all(y_val, y_val_pred)
    print("\nValidation Metrics (original price scale):")
    print_metrics(val_metrics)
    
    # ============================================================
    # Step 7: Test Set Evaluation (Final)
    # ============================================================
    print("\n" + "=" * 60)
    print("  Step 7: Test Set Evaluation (Final)")
    print("=" * 60)
    
    y_test_pred_scaled = model.predict(X_test_scaled)
    
    if config['standardize_target'] and scaler_y is not None:
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).squeeze()
    else:
        y_test_pred = y_test_pred_scaled
    
    if config['log_transform_target']:
        y_test_pred = np.expm1(y_test_pred)
    
    test_metrics = evaluate_all(y_test, y_test_pred)
    print("\nTest Metrics (original price scale):")
    print_metrics(test_metrics)
    
    # ============================================================
    # Step 8: Generate Visualizations
    # ============================================================
    print("\n" + "=" * 60)
    print("  Step 8: Generating Visualizations")
    print("=" * 60)
    
    outputs_dir = PROJECT_ROOT / 'outputs'
    
    # Training loss curve
    plot_loss_curve(
        model.training_loss_history,
        title='Training Loss Curve',
        save_path=str(outputs_dir / 'training' / 'loss_curve.png')
    )
    print("  Saved: outputs/training/loss_curve.png")
    
    # Train vs Validation loss
    if model.validation_loss_history:
        plot_train_val_loss(
            model.training_loss_history,
            model.validation_loss_history,
            save_path=str(outputs_dir / 'training' / 'train_validation_loss.png')
        )
        print("  Saved: outputs/training/train_validation_loss.png")
    
    # Actual vs Predicted (test set)
    plot_actual_vs_predicted(
        y_test, y_test_pred,
        save_path=str(outputs_dir / 'evaluation' / 'actual_vs_predicted.png')
    )
    print("  Saved: outputs/evaluation/actual_vs_predicted.png")
    
    # Residual plot
    plot_residuals(
        y_test, y_test_pred,
        save_path=str(outputs_dir / 'evaluation' / 'residual_plot.png')
    )
    print("  Saved: outputs/evaluation/residual_plot.png")
    
    # Error distribution
    plot_error_distribution(
        y_test, y_test_pred,
        save_path=str(outputs_dir / 'evaluation' / 'error_distribution.png')
    )
    print("  Saved: outputs/evaluation/error_distribution.png")
    
    # ============================================================
    # Step 9: Save Model
    # ============================================================
    print("\n" + "=" * 60)
    print("  Step 9: Saving Model")
    print("=" * 60)
    
    model_path = str(PROJECT_ROOT / 'models' / 'linear_regression.json')
    save_model(
        model=model,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        feature_names=baseline_features,
        filepath=model_path,
        config=config
    )
    print(f"  Model saved to: {model_path}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"\n  Model: Multiple Linear Regression (from scratch)")
    print(f"  Features: {len(baseline_features)}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Epochs: {config['n_epochs']}")
    print(f"  Test RMSE:      ${test_metrics['rmse']:,.2f}")
    print(f"  Test R2:        {test_metrics['r2']:.4f} ({test_metrics['r2']*100:.2f}%)")
    if 'Accuracy' in test_metrics:
        print(f"  Test Accuracy:  {test_metrics['Accuracy']*100:.2f}% (High vs Affordable)")
        print(f"  Test Precision: {test_metrics['Precision']*100:.2f}%")
        print(f"  Test Recall:    {test_metrics['Recall']*100:.2f}%")
        print(f"  Test F1-Score:  {test_metrics['F1']:.4f}")
    if 'Tolerance_20' in test_metrics:
        print(f"  Within +/-20%:  {test_metrics['Tolerance_20']*100:.2f}%")
    print(f"\n  Model file: {model_path}")
    print()


if __name__ == '__main__':
    main()
