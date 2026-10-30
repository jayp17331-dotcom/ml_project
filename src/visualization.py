"""
visualization.py - Plotting Utilities for the HousePrice-AI Project
=====================================================================

All functions use matplotlib with the 'Agg' (non-interactive) backend
so plots can be saved to disk on any system, including headless servers.

Every function:
    - Creates parent directories if they don't exist.
    - Saves the figure to the given path.
    - Closes the figure after saving to free memory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - must be set before pyplot import
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────

def _ensure_dir(filepath):
    """Create parent directories for *filepath* if they don't exist."""
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ──────────────────────────────────────────────────────────────────
# Loss Curves
# ──────────────────────────────────────────────────────────────────

def plot_loss_curve(loss_history, title='Training Loss Curve',
                    save_path='outputs/training/loss_curve.png'):
    """
    Plot training loss vs. epoch.

    Parameters
    ----------
    loss_history : list of float
        Loss values at each epoch.
    title : str
        Plot title.
    save_path : str
        File path to save the figure.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(loss_history)), loss_history, color='steelblue',
            linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {save_path}")


def plot_train_val_loss(train_loss, val_loss,
                        save_path='outputs/training/train_val_loss.png'):
    """
    Plot training and validation loss curves on one figure.

    Parameters
    ----------
    train_loss : list of float
        Training loss at each epoch.
    val_loss : list of float
        Validation loss at each epoch.
    save_path : str
        File path to save the figure.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(len(train_loss))
    ax.plot(epochs, train_loss, color='steelblue', linewidth=1.5,
            label='Training Loss')
    ax.plot(epochs, val_loss, color='coral', linewidth=1.5,
            label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {save_path}")


# ──────────────────────────────────────────────────────────────────
# Prediction Analysis
# ──────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_true, y_pred,
                             save_path='outputs/evaluation/actual_vs_predicted.png'):
    """
    Scatter plot of actual vs. predicted values with a diagonal
    reference line (perfect-prediction line).

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    save_path : str
        File path to save the figure.
    """
    _ensure_dir(save_path)
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue',
               edgecolors='white', linewidth=0.3)

    # Perfect-prediction diagonal
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', linewidth=1.5, label='Perfect Prediction')

    ax.set_xlabel('Actual Sale Price', fontsize=12)
    ax.set_ylabel('Predicted Sale Price', fontsize=12)
    ax.set_title('Actual vs Predicted Sale Price', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {save_path}")


def plot_residuals(y_true, y_pred,
                   save_path='outputs/evaluation/residuals.png'):
    """
    Plot residuals (y_true − y_pred) vs. predicted values.

    A well-performing model should show residuals randomly scattered
    around the zero line with no systematic pattern.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    save_path : str
        File path to save the figure.
    """
    _ensure_dir(save_path)
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, color='steelblue',
               edgecolors='white', linewidth=0.3)
    ax.axhline(y=0, color='red', linewidth=1.5, linestyle='--')

    ax.set_xlabel('Predicted Sale Price', fontsize=12)
    ax.set_ylabel('Residual (Actual − Predicted)', fontsize=12)
    ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {save_path}")


def plot_error_distribution(y_true, y_pred,
                            save_path='outputs/evaluation/error_distribution.png'):
    """
    Histogram of prediction errors (residuals).

    A roughly normal distribution centred on zero suggests unbiased
    predictions.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    save_path : str
        File path to save the figure.
    """
    _ensure_dir(save_path)
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    errors = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=0, color='red', linewidth=1.5, linestyle='--',
               label='Zero Error')
    ax.axvline(x=np.mean(errors), color='orange', linewidth=1.5,
               linestyle='--', label=f'Mean Error ({np.mean(errors):,.0f})')

    ax.set_xlabel('Prediction Error (Actual − Predicted)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Prediction Errors', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {save_path}")


# ──────────────────────────────────────────────────────────────────
# EDA Plots
# ──────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df, features,
                             save_path='outputs/eda/correlation_heatmap.png'):
    """
    Plot a correlation heatmap for selected features.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    features : list of str
        Column names to include in the heatmap.
    save_path : str
        File path to save the figure.
    """
    _ensure_dir(save_path)
    corr_matrix = df[features].corr().values
    n = len(features)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.7)))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Tick labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(features, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {save_path}")


def plot_sale_price_distribution(prices,
                                 save_path='outputs/eda/sale_price_distribution.png'):
    """
    Histogram of SalePrice values.

    Parameters
    ----------
    prices : array-like
        SalePrice values.
    save_path : str
        File path to save the figure.
    """
    _ensure_dir(save_path)
    prices = np.array(prices).ravel()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(prices, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=np.mean(prices), color='red', linewidth=1.5, linestyle='--',
               label=f'Mean (${np.mean(prices):,.0f})')
    ax.axvline(x=np.median(prices), color='orange', linewidth=1.5,
               linestyle='--',
               label=f'Median (${np.median(prices):,.0f})')

    ax.set_xlabel('Sale Price ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Sale Prices', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {save_path}")


def plot_feature_vs_target(feature, target, feature_name='Feature',
                           save_path='outputs/eda/feature_vs_target.png'):
    """
    Scatter plot of a single feature vs. the target variable.

    Parameters
    ----------
    feature : array-like
        Feature values.
    target : array-like
        Target values.
    feature_name : str
        Name of the feature for axis label.
    save_path : str
        File path to save the figure.
    """
    _ensure_dir(save_path)
    feature = np.array(feature).ravel()
    target = np.array(target).ravel()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(feature, target, alpha=0.5, s=20, color='steelblue',
               edgecolors='white', linewidth=0.3)
    ax.set_xlabel(feature_name, fontsize=12)
    ax.set_ylabel('Sale Price ($)', fontsize=12)
    ax.set_title(f'{feature_name} vs Sale Price', fontsize=14,
                 fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {save_path}")


def plot_box_plot(df, column,
                  save_path='outputs/eda/box_plot.png'):
    """
    Box plot for a single numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    column : str
        Column name to plot.
    save_path : str
        File path to save the figure.
    """
    _ensure_dir(save_path)
    data = df[column].dropna().values

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, patch_artist=True, vert=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)

    ax.set_ylabel(column, fontsize=12)
    ax.set_title(f'Box Plot: {column}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels([column])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {save_path}")


def plot_quality_vs_price(df,
                          save_path='outputs/eda/quality_vs_price.png'):
    """
    Box plot of SalePrice grouped by OverallQual (1-10).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'OverallQual' and 'SalePrice'.
    save_path : str
        File path to save the figure.
    """
    _ensure_dir(save_path)

    if 'OverallQual' not in df.columns or 'SalePrice' not in df.columns:
        print("[WARNING] 'OverallQual' or 'SalePrice' not in DataFrame. "
              "Skipping plot.")
        return

    groups = []
    labels = []
    for qual in sorted(df['OverallQual'].unique()):
        group_data = df[df['OverallQual'] == qual]['SalePrice'].dropna().values
        if len(group_data) > 0:
            groups.append(group_data)
            labels.append(str(int(qual)))

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(groups, patch_artist=True, labels=labels)

    # Colour gradient
    cmap = plt.cm.Blues
    n_groups = len(groups)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(cmap(0.3 + 0.7 * i / max(n_groups - 1, 1)))
        box.set_alpha(0.8)

    ax.set_xlabel('Overall Quality', fontsize=12)
    ax.set_ylabel('Sale Price ($)', fontsize=12)
    ax.set_title('Overall Quality vs Sale Price', fontsize=14,
                 fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {save_path}")
