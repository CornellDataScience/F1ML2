"""
Test Models with Telemetry Features on 2022 and 2023 Data

This script tests BayesianRidge and LinearRegression models
on both 2022 and 2023 seasons using the new telemetry-enhanced dataset.

Train on 2022: 2018-2021 (with telemetry features)
Test on 2022: 2022

Train on 2023: 2018-2022 (with telemetry features)
Test on 2023: 2023
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("TESTING MODELS WITH TELEMETRY FEATURES ON 2022 & 2023")
print("=" * 70)

def process_df(df):
    """Process dataframe to prepare features"""
    y = df['grid']
    X = df.drop(columns=['grid', 'driver', 'season', 'round', 'circuit_id'], errors='ignore')

    # Drop qualifying_secs to prevent data leakage
    if 'qualifying_secs' in X.columns:
        X = X.drop(columns=['qualifying_secs'])

    # Drop non-numeric columns
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        X = X.drop(columns=object_cols)

    # Convert booleans to int
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    return X, y

def calculate_pole_accuracy(test_df, predictions):
    """Calculate pole position accuracy"""
    test_df = test_df.copy()
    test_df['prediction'] = predictions

    correct_poles = 0
    total_sessions = 0
    correct_sessions = []

    for (season, round_num), session_df in test_df.groupby(['season', 'round']):
        if len(session_df) == 0:
            continue

        predicted_order = session_df.sort_values('prediction')
        actual_order = session_df.sort_values('grid')

        if len(predicted_order) > 0 and len(actual_order) > 0:
            predicted_pole = predicted_order.iloc[0]['driver']
            actual_pole = actual_order.iloc[0]['driver']

            if predicted_pole == actual_pole:
                correct_poles += 1
                correct_sessions.append((season, round_num, actual_pole))

        total_sessions += 1

    pole_accuracy = (correct_poles / total_sessions) * 100
    return pole_accuracy, correct_poles, total_sessions, correct_sessions

def test_year(test_year, train_df, full_df):
    """Test models on a specific year"""
    print("\n" + "=" * 70)
    print(f"TESTING ON {test_year}")
    print("=" * 70)

    test_df = full_df[full_df['season'] == test_year].copy()

    print(f"Training set: {train_df.shape}")
    print(f"Test set ({test_year}): {test_df.shape}")
    print(f"Test sessions: {test_df.groupby(['season', 'round']).ngroups}")

    # Prepare data
    X_train, y_train = process_df(train_df)
    X_test, y_test = process_df(test_df)

    # Align columns
    all_cols = set(X_train.columns) | set(X_test.columns)
    for col in all_cols - set(X_train.columns):
        X_train[col] = 0
    for col in all_cols - set(X_test.columns):
        X_test[col] = 0

    X_train = X_train[sorted(all_cols)]
    X_test = X_test[sorted(all_cols)]

    # Fill missing values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())

    print(f"Features: {X_train.shape[1]}")

    # Model 1: BayesianRidge
    print("\n  BayesianRidge (no scaling)...")
    model_br = BayesianRidge()
    model_br.fit(X_train, y_train)
    predictions_br = model_br.predict(X_test)
    pole_acc_br, correct_br, total_br, sessions_br = calculate_pole_accuracy(test_df, predictions_br)
    print(f"    Pole Accuracy: {pole_acc_br:.2f}% ({correct_br}/{total_br})")

    # Model 2: LinearRegression
    print("\n  LinearRegression (with scaling)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train)
    predictions_lr = model_lr.predict(X_test_scaled)
    pole_acc_lr, correct_lr, total_lr, sessions_lr = calculate_pole_accuracy(test_df, predictions_lr)
    print(f"    Pole Accuracy: {pole_acc_lr:.2f}% ({correct_lr}/{total_lr})")

    return {
        'year': test_year,
        'br_acc': pole_acc_br,
        'br_correct': correct_br,
        'br_total': total_br,
        'br_sessions': sessions_br,
        'lr_acc': pole_acc_lr,
        'lr_correct': correct_lr,
        'lr_total': total_lr,
        'lr_sessions': sessions_lr
    }

# Load full dataset
full_df = pd.read_csv('../data/HOLY_qualifying_modern_2018_2023.csv')
print(f"\nFull dataset: {full_df.shape}")
print(f"Years: {full_df['season'].min()} - {full_df['season'].max()}")

# Check telemetry features
telemetry_cols = [col for col in full_df.columns if 'practice_' in col]
print(f"Telemetry features: {len(telemetry_cols)}")

# Test 2022
train_2022 = pd.read_csv('../data/HOLY_qualifying_modern_train2022.csv')
results_2022 = test_year(2022, train_2022, full_df)

# Test 2023
train_2023 = pd.read_csv('../data/HOLY_qualifying_modern_train2023.csv')
results_2023 = test_year(2023, train_2023, full_df)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: TELEMETRY-ENHANCED MODELS")
print("=" * 70)

summary = pd.DataFrame({
    'Year': [2022, 2023],
    'BayesianRidge': [
        f"{results_2022['br_acc']:.2f}% ({results_2022['br_correct']}/{results_2022['br_total']})",
        f"{results_2023['br_acc']:.2f}% ({results_2023['br_correct']}/{results_2023['br_total']})"
    ],
    'LinearRegression': [
        f"{results_2022['lr_acc']:.2f}% ({results_2022['lr_correct']}/{results_2022['lr_total']})",
        f"{results_2023['lr_acc']:.2f}% ({results_2023['lr_correct']}/{results_2023['lr_total']})"
    ]
})

print(summary.to_string(index=False))

# Overall statistics
print("\n" + "=" * 70)
print("OVERALL STATISTICS")
print("=" * 70)

avg_br = (results_2022['br_acc'] + results_2023['br_acc']) / 2
avg_lr = (results_2022['lr_acc'] + results_2023['lr_acc']) / 2

print(f"BayesianRidge average:      {avg_br:.2f}%")
print(f"LinearRegression average:   {avg_lr:.2f}%")
print(f"Overall average:            {(avg_br + avg_lr)/2:.2f}%")

# Best model
print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

if avg_br > avg_lr:
    print(f"🏆 BayesianRidge: {avg_br:.2f}% vs {avg_lr:.2f}%")
    print(f"   Margin: +{avg_br - avg_lr:.2f}%")
elif avg_lr > avg_br:
    print(f"🏆 LinearRegression: {avg_lr:.2f}% vs {avg_br:.2f}%")
    print(f"   Margin: +{avg_lr - avg_br:.2f}%")
else:
    print(f"🤝 TIE: Both at {avg_br:.2f}%")

print("\n" + "=" * 70)
print(f"Telemetry features used: {len(telemetry_cols)}")
print("=" * 70)
