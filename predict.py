#!/usr/bin/env python
"""
HousePrice-AI Prediction Script
=================================
Load a trained model and predict house prices.

Usage:
    python predict.py
    python predict.py --interactive
    
Example:
    python predict.py --GrLivArea 1500 --OverallQual 7 --YearBuilt 2000 --BedroomAbvGr 3 --FullBath 2 --GarageCars 2
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_io import load_model


def predict_price(features_dict, model, scaler_X, scaler_y, feature_names, config):
    """
    Predict house price from feature dictionary.
    
    Parameters
    ----------
    features_dict : dict
        Dictionary mapping feature names to values.
    model : LinearRegressionScratch
        Trained model.
    scaler_X : StandardScaler
        Feature scaler fitted on training data.
    scaler_y : StandardScaler or None
        Target scaler (if target was standardized during training).
    feature_names : list
        Ordered list of feature names.
    config : dict
        Training configuration.
    
    Returns
    -------
    float
        Predicted house price in original scale.
    """
    # Build feature vector in correct order
    X = np.array([[features_dict[name] for name in feature_names]], dtype=np.float64)
    
    # Apply same preprocessing as training
    X_scaled = scaler_X.transform(X)
    
    # Predict
    y_pred_scaled = model.predict(X_scaled)
    
    # Reverse target scaling
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).squeeze()
    else:
        y_pred = y_pred_scaled[0] if len(y_pred_scaled.shape) > 0 else y_pred_scaled
    
    # Reverse log transform if used
    if config and config.get('log_transform_target', False):
        y_pred = np.expm1(y_pred)
    
    return float(y_pred)


def interactive_mode(model, scaler_X, scaler_y, feature_names, config):
    """Run interactive prediction mode."""
    print("\n" + "=" * 50)
    print("  HousePrice-AI: Interactive Prediction")
    print("=" * 50)
    print(f"\nRequired features: {feature_names}")
    print("Type 'quit' to exit.\n")
    
    while True:
        features = {}
        for name in feature_names:
            while True:
                val = input(f"  Enter {name}: ")
                if val.lower() == 'quit':
                    print("\nGoodbye!")
                    return
                try:
                    features[name] = float(val)
                    break
                except ValueError:
                    print(f"    Invalid number. Please enter a numeric value for {name}.")
        
        price = predict_price(features, model, scaler_X, scaler_y, feature_names, config)
        print(f"\n  {'='*40}")
        print(f"  Estimated Price: ${price:,.2f}")
        print(f"  {'='*40}\n")


def main():
    parser = argparse.ArgumentParser(description='HousePrice-AI: Predict house prices')
    parser.add_argument('--model', type=str, default='models/linear_regression.json',
                        help='Path to saved model JSON file')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    # Feature arguments
    parser.add_argument('--GrLivArea', type=float, help='Above grade living area (sq ft)')
    parser.add_argument('--OverallQual', type=float, help='Overall material and finish quality (1-10)')
    parser.add_argument('--YearBuilt', type=float, help='Year built')
    parser.add_argument('--BedroomAbvGr', type=float, help='Bedrooms above grade')
    parser.add_argument('--FullBath', type=float, help='Full bathrooms')
    parser.add_argument('--GarageCars', type=float, help='Garage car capacity')
    parser.add_argument('--Area', type=float, help='Property Area (sq ft)')
    parser.add_argument('--Bedrooms', type=float, help='Bedrooms count')
    parser.add_argument('--Bathrooms', type=float, help='Bathrooms count')
    parser.add_argument('--Floors', type=float, help='Number of floors')
    
    args = parser.parse_args()
    
    # Load model
    model_path = str(PROJECT_ROOT / args.model)
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at '{model_path}'")
        print("Run 'python train.py' first to train and save a model.")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    model, scaler_X, scaler_y, feature_names, config = load_model(model_path)
    print(f"Model loaded. Features: {feature_names}")
    
    if args.interactive:
        interactive_mode(model, scaler_X, scaler_y, feature_names, config)
        return
    
    # Check if features provided via command line
    feature_args = {
        'GrLivArea': args.GrLivArea,
        'OverallQual': args.OverallQual,
        'YearBuilt': args.YearBuilt,
        'BedroomAbvGr': args.BedroomAbvGr,
        'FullBath': args.FullBath,
        'GarageCars': args.GarageCars,
        'Area': args.Area,
        'Bedrooms': args.Bedrooms,
        'Bathrooms': args.Bathrooms,
        'Floors': args.Floors,
    }
    
    # Filter to only features the model uses
    features = {}
    missing = []
    for name in feature_names:
        if name in feature_args and feature_args[name] is not None:
            features[name] = feature_args[name]
        else:
            missing.append(name)
    
    if missing:
        # Demo prediction
        print(f"\nNo feature values provided. Running demo prediction...")
        demo_features = {
            'GrLivArea': 1500,
            'OverallQual': 7,
            'YearBuilt': 2000,
            'BedroomAbvGr': 3,
            'FullBath': 2,
            'GarageCars': 2,
            'Area': 2500,
            'Bedrooms': 3,
            'Bathrooms': 2,
            'Floors': 2,
        }
        features = {k: demo_features[k] for k in feature_names if k in demo_features}
        
        if len(features) < len(feature_names):
            print(f"ERROR: Cannot provide demo values for features: {[f for f in feature_names if f not in features]}")
            sys.exit(1)
        
        print(f"\nDemo input:")
        for k, v in features.items():
            print(f"  {k}: {v}")
    else:
        print(f"\nInput features:")
        for k, v in features.items():
            print(f"  {k}: {v}")
    
    price = predict_price(features, model, scaler_X, scaler_y, feature_names, config)
    
    print(f"\n{'='*50}")
    print(f"  Estimated Sale Price: ${price:,.2f}")
    print(f"{'='*50}")
    
    print(f"\nCommand-line usage:")
    print(f"  python predict.py --GrLivArea 1500 --OverallQual 7 --YearBuilt 2000 --BedroomAbvGr 3 --FullBath 2 --GarageCars 2")
    print(f"  python predict.py --interactive")


if __name__ == '__main__':
    main()
