"""
Data Utilities for F1 Model Training

This module provides reusable functions for data preprocessing,
feature alignment, and cleaning across all F1 prediction models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def process_df(df, drop_qualifying_secs=True, drop_driver_season_round=True):
    """
    Process dataframe to prepare features for model training.

    Args:
        df: Input dataframe
        drop_qualifying_secs: Whether to drop qualifying_secs (default: True to prevent leakage)
        drop_driver_season_round: Whether to drop driver, season, round columns (default: True)

    Returns:
        X: Feature matrix
        y: Target variable (grid position)
    """
    # Extract target variable
    y = df.loc[:, 'grid']

    # Drop metadata columns
    cols_to_drop = ['grid']
    if drop_driver_season_round:
        cols_to_drop.extend(['driver', 'season', 'round'])

    X = df.drop(columns=cols_to_drop, errors='ignore')

    # Drop qualifying_secs to prevent data leakage
    if drop_qualifying_secs and 'qualifying_secs' in X.columns:
        X = X.drop(columns=['qualifying_secs'])

    # Drop non-numeric columns (circuit_id, constructor names, weather strings, etc.)
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        X = X.drop(columns=object_cols)

    # Convert boolean columns to int (0/1)
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    return X, y


def align_train_test_features(X_train, X_test, fill_value=0):
    """
    Align features between training and test sets.

    Handles cases where test set has new circuits/constructors that weren't
    in the training data (e.g., new circuits added in 2022-2023).

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix (can be single test set or list of test sets)
        fill_value: Value to fill for missing columns (default: 0)

    Returns:
        X_train_aligned: Training set with aligned columns
        X_test_aligned: Test set(s) with aligned columns (single df or list)
    """
    # Handle single test set or multiple test sets
    if isinstance(X_test, pd.DataFrame):
        X_test_list = [X_test]
        return_single = True
    else:
        X_test_list = X_test
        return_single = False

    # Get all unique columns across all sets
    all_cols = set(X_train.columns)
    for X_t in X_test_list:
        all_cols = all_cols | set(X_t.columns)

    # Sort columns for consistent order
    all_cols = sorted(all_cols)

    # Add missing columns to training set
    for col in all_cols:
        if col not in X_train.columns:
            X_train[col] = fill_value

    # Add missing columns to test sets
    X_test_aligned = []
    for X_t in X_test_list:
        for col in all_cols:
            if col not in X_t.columns:
                X_t[col] = fill_value
        # Reorder columns to match
        X_test_aligned.append(X_t[all_cols])

    # Reorder training columns
    X_train_aligned = X_train[all_cols]

    if return_single:
        return X_train_aligned, X_test_aligned[0]
    else:
        return X_train_aligned, X_test_aligned


def fill_missing_values(X_train, X_test, method='mean'):
    """
    Fill missing values in train and test sets.

    Uses training set statistics to fill both train and test sets
    to prevent data leakage.

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix (can be single test set or list)
        method: Filling method ('mean', 'median', 'zero')

    Returns:
        X_train_filled: Training set with filled values
        X_test_filled: Test set(s) with filled values (single df or list)
    """
    # Handle single test set or multiple test sets
    if isinstance(X_test, pd.DataFrame):
        X_test_list = [X_test]
        return_single = True
    else:
        X_test_list = X_test
        return_single = False

    # Calculate fill values from training set only
    if method == 'mean':
        fill_values = X_train.mean()
    elif method == 'median':
        fill_values = X_train.median()
    elif method == 'zero':
        fill_values = pd.Series(0, index=X_train.columns)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Fill training set
    X_train_filled = X_train.fillna(fill_values)

    # Fill test sets
    X_test_filled = [X_t.fillna(fill_values) for X_t in X_test_list]

    if return_single:
        return X_train_filled, X_test_filled[0]
    else:
        return X_train_filled, X_test_filled


def prepare_train_test(train_df, test_df, scale=False, drop_qualifying_secs=True):
    """
    Complete preprocessing pipeline for train and test dataframes.

    Combines processing, alignment, and filling into one function.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe (can be single df or list of dfs)
        scale: Whether to apply StandardScaler (default: False)
        drop_qualifying_secs: Whether to drop qualifying_secs (default: True)

    Returns:
        X_train: Processed training features
        y_train: Training labels
        X_test: Processed test features (single df or list)
        y_test: Test labels (single array or list)
        scaler: StandardScaler object (None if scale=False)
    """
    # Handle single test set or multiple test sets
    if isinstance(test_df, pd.DataFrame):
        test_df_list = [test_df]
        return_single = True
    else:
        test_df_list = test_df
        return_single = False

    # Process dataframes
    X_train, y_train = process_df(train_df, drop_qualifying_secs=drop_qualifying_secs)

    X_test_list = []
    y_test_list = []
    for test_d in test_df_list:
        X_t, y_t = process_df(test_d, drop_qualifying_secs=drop_qualifying_secs)
        X_test_list.append(X_t)
        y_test_list.append(y_t)

    # Align features
    X_train, X_test_list = align_train_test_features(X_train, X_test_list)

    # Fill missing values
    X_train, X_test_list = fill_missing_values(X_train, X_test_list)

    # Scale if requested
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_list = [
            pd.DataFrame(
                scaler.transform(X_t),
                columns=X_t.columns,
                index=X_t.index
            )
            for X_t in X_test_list
        ]

    if return_single:
        return X_train, y_train, X_test_list[0], y_test_list[0], scaler
    else:
        return X_train, y_train, X_test_list, y_test_list, scaler


def calculate_pole_accuracy(test_df, predictions, return_details=False):
    """
    Calculate pole position accuracy.

    Args:
        test_df: Test dataframe with 'season', 'round', 'driver', 'grid' columns
        predictions: Model predictions (array-like)
        return_details: Whether to return detailed session info (default: False)

    Returns:
        pole_accuracy: Percentage of correctly predicted poles
        correct_poles: Number of correct predictions
        total_sessions: Total number of sessions
        correct_sessions: List of (season, round, driver) for correct predictions (if return_details=True)
    """
    test_df = test_df.copy()
    test_df['prediction'] = predictions

    correct_poles = 0
    total_sessions = 0
    correct_sessions = []

    for (season, round_num), session_df in test_df.groupby(['season', 'round']):
        if len(session_df) == 0:
            continue

        # Get predicted and actual pole sitters
        predicted_order = session_df.sort_values('prediction')
        actual_order = session_df.sort_values('grid')

        if len(predicted_order) > 0 and len(actual_order) > 0:
            predicted_pole = predicted_order.iloc[0]['driver']
            actual_pole = actual_order.iloc[0]['driver']

            print(predicted_pole, actual_pole)

            if predicted_pole == actual_pole:
                correct_poles += 1
                if return_details:
                    correct_sessions.append((season, round_num, actual_pole))

        total_sessions += 1

    pole_accuracy = (correct_poles / total_sessions) * 100 if total_sessions > 0 else 0

    if return_details:
        return pole_accuracy, correct_poles, total_sessions, correct_sessions
    else:
        return pole_accuracy, correct_poles, total_sessions


def load_leak_free_datasets(train_year, test_years, data_dir='../data'):
    """
    Load leak-free training and test datasets.

    Args:
        train_year: Year to train up to (exclusive). E.g., 2022 means train on <2022
        test_years: Year or list of years to test on
        data_dir: Directory containing the CSV files

    Returns:
        train_df: Training dataframe
        test_dfs: Test dataframe(s) - single df if single year, list if multiple
    """
    import os

    # Load training dataset
    train_file = os.path.join(data_dir, f'HOLY_qualifying_v1_train{train_year}.csv')
    train_df = pd.read_csv(train_file)

    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop(columns=['Unnamed: 0'])

    # Load full dataset for test data
    full_file = os.path.join(data_dir, 'HOLY_qualifying_full_to_2023.csv')
    full_df = pd.read_csv(full_file)

    if 'Unnamed: 0' in full_df.columns:
        full_df = full_df.drop(columns=['Unnamed: 0'])

    # Extract test sets
    if isinstance(test_years, int):
        test_dfs = full_df[full_df['season'] == test_years].copy()
    else:
        test_dfs = [full_df[full_df['season'] == year].copy() for year in test_years]

    return train_df, test_dfs


# Example usage
if __name__ == "__main__":
    print("Data Utilities Module")
    print("=" * 70)
    print("\nAvailable functions:")
    print("  - process_df(): Clean and prepare features")
    print("  - align_train_test_features(): Align columns between train/test")
    print("  - fill_missing_values(): Fill NaN values")
    print("  - prepare_train_test(): Complete preprocessing pipeline")
    print("  - calculate_pole_accuracy(): Calculate pole prediction accuracy")
    print("  - load_leak_free_datasets(): Load proper train/test splits")
    print("\nExample usage:")
    print("""
    from data_utils import prepare_train_test, calculate_pole_accuracy

    # Load data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # Prepare data (with scaling for LinearRegression)
    X_train, y_train, X_test, y_test, scaler = prepare_train_test(
        train_df, test_df, scale=True
    )

    # Train model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate
    accuracy, correct, total = calculate_pole_accuracy(test_df, predictions)
    print(f"Pole Accuracy: {accuracy:.2f}% ({correct}/{total})")
    """)
