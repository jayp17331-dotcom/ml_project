"""
data_loader.py — Data Loading and Inspection Utilities
=======================================================

Provides functions to load the Ames Housing dataset from CSV,
display basic information, compute summary statistics, and
check for missing values and duplicate rows.

All functions operate on pandas DataFrames and are designed to
be used early in the ML pipeline for exploratory analysis.
"""

import os
import pandas as pd


def load_dataset(filepath):
    """
    Load a CSV dataset into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file is empty or cannot be parsed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file is empty: {filepath}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Could not parse the CSV file: {e}")

    if df.empty:
        raise ValueError(f"The loaded DataFrame is empty: {filepath}")

    print(f"[INFO] Dataset loaded successfully from: {filepath}")
    print(f"   Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def display_basic_info(df):
    """
    Display basic information about the DataFrame.

    Prints the shape, column names, data types, and the first/last
    few rows of the dataset for a quick overview.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    """
    print("=" * 60)
    print("DATASET BASIC INFORMATION")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"   {i+1:3d}. {col} ({df[col].dtype})")

    print("\nData Types Summary:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nLast 5 rows:")
    print(df.tail().to_string())
    print("=" * 60)


def get_summary_statistics(df):
    """
    Compute and return summary statistics for the DataFrame.

    Returns descriptive statistics (count, mean, std, min, quartiles, max)
    for all numeric columns, and value counts for categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize.

    Returns
    -------
    pd.DataFrame
        Summary statistics from pandas describe (all columns included).
    """
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Numeric summary
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric columns ({len(numeric_cols)}):")
        numeric_summary = df[numeric_cols].describe()
        print(numeric_summary.to_string())
    else:
        print("\n[INFO] No numeric columns found.")

    # Categorical summary
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        print(f"\nCategorical columns ({len(cat_cols)}):")
        for col in cat_cols:
            n_unique = df[col].nunique()
            print(f"   {col}: {n_unique} unique values")

    summary = df.describe(include='all')
    print("=" * 60)
    return summary


def check_missing_values(df):
    """
    Check for missing values in each column of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'Missing Count' and 'Missing Percentage',
        sorted by missing count in descending order, showing only columns
        that have at least one missing value.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100

    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage': missing_pct.round(2)
    })

    # Filter to only columns with missing values and sort
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    missing_df = missing_df.sort_values('Missing Count', ascending=False)

    print("=" * 60)
    print("MISSING VALUES REPORT")
    print("=" * 60)

    if len(missing_df) == 0:
        print("[INFO] No missing values found!")
    else:
        print(f"[WARNING] {len(missing_df)} columns have missing values:\n")
        print(missing_df.to_string())
        total_missing = missing_count.sum()
        total_cells = df.shape[0] * df.shape[1]
        print(f"\nTotal missing cells: {total_missing} / {total_cells} "
              f"({(total_missing/total_cells)*100:.2f}%)")

    print("=" * 60)
    return missing_df


def check_duplicates(df):
    """
    Check for duplicate rows in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.

    Returns
    -------
    int
        The number of duplicate rows found.
    """
    n_duplicates = df.duplicated().sum()

    print("=" * 60)
    print("DUPLICATE ROWS CHECK")
    print("=" * 60)

    if n_duplicates == 0:
        print("[INFO] No duplicate rows found.")
    else:
        print(f"[WARNING] Found {n_duplicates} duplicate rows "
              f"({(n_duplicates/len(df))*100:.2f}% of total)")

    print("=" * 60)
    return n_duplicates
