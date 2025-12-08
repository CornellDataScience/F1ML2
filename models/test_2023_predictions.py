"""
Test Models on 2023 Data

This script tests our best models on the 2023 season to see how well
they generalize to new data.

Train on: 1983-2022
Test on: 2023
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

print("=" * 70)
print("TESTING MODELS ON 2023 DATA")
print("Using LEAK-FREE datasets")
print("=" * 70)

# Load leak-free training dataset
print("\nLoading leak-free training dataset (1983-2022)...")
train_df = pd.read_csv('../data/HOLY_qualifying_v1_train2023.csv')

if 'Unnamed: 0' in train_df.columns:
    train_df = train_df.drop(columns=['Unnamed: 0'])

print(f"Training dataset: {len(train_df)} records ({train_df['season'].min()}-{train_df['season'].max()})")

# Load full dataset for test data
print("\nLoading full dataset for test data...")
df = pd.read_csv('../data/HOLY_qualifying_full_to_2023.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"Full dataset: {len(df)} records ({df['season'].min()}-{df['season'].max()})")

def process_df(df):
    """Process dataframe to prepare features"""
    y = df.loc[:, 'grid']
    X = df.drop(columns=['grid', 'driver', 'season', 'round'], errors='ignore')

    # Drop qualifying_secs to prevent data leakage
    if 'qualifying_secs' in X.columns:
        print(f"  🔬 DROPPING 'qualifying_secs'")
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

# Split data: Train on <2023, Test on 2023
print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)
TEST_YEAR = 2023

test_df = df[df['season'] == TEST_YEAR].copy()

print(f"Training set: {len(train_df)} records (1983-2022)")
print(f"Test set: {len(test_df)} records ({TEST_YEAR})")
print(f"Test sessions: {test_df.groupby(['season', 'round']).ngroups}")

# Prepare data
print("\nPreparing features...")
X_train, y_train = process_df(train_df)
X_test, y_test = process_df(test_df)

# Align columns: add new circuits/constructors from test to train (with 0 values)
print("Aligning features between train and test sets...")
all_cols = set(X_train.columns) | set(X_test.columns)

# Add missing columns
for col in all_cols - set(X_train.columns):
    X_train[col] = 0
for col in all_cols - set(X_test.columns):
    X_test[col] = 0

# Ensure same column order
X_train = X_train[sorted(all_cols)]
X_test = X_test[sorted(all_cols)]

# Fill missing values
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

print(f"Training shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# ============================================================================
# MODEL 1: BayesianRidge WITHOUT Scaling
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 1: BayesianRidge (NO SCALING)")
print("=" * 70)

model_br = BayesianRidge()
model_br.fit(X_train, y_train)
predictions_br = model_br.predict(X_test)

pole_acc_br, correct_br, total_br, sessions_br = calculate_pole_accuracy(test_df, predictions_br)
mae_br = mean_absolute_error(y_test, predictions_br)

print(f"\nRESULTS:")
print(f"  Pole Position Accuracy: {pole_acc_br:.2f}% ({correct_br}/{total_br})")
print(f"  MAE: {mae_br:.3f} grid positions")

# ============================================================================
# MODEL 2: LinearRegression WITH Scaling
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 2: LinearRegression (WITH SCALING)")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
predictions_lr = model_lr.predict(X_test_scaled)

pole_acc_lr, correct_lr, total_lr, sessions_lr = calculate_pole_accuracy(test_df, predictions_lr)
mae_lr = mean_absolute_error(y_test, predictions_lr)

print(f"\nRESULTS:")
print(f"  Pole Position Accuracy: {pole_acc_lr:.2f}% ({correct_lr}/{total_lr})")
print(f"  MAE: {mae_lr:.3f} grid positions")

# ============================================================================
# DETAILED ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("CORRECTLY PREDICTED POLES IN 2023")
print("=" * 70)

print("\nBayesianRidge correct predictions:")
for season, round_num, driver in sessions_br:
    print(f"  Round {round_num}: {driver}")

print(f"\nLinearRegression correct predictions:")
for season, round_num, driver in sessions_lr:
    print(f"  Round {round_num}: {driver}")

# Check which model is better for 2023
print("\n" + "=" * 70)
print("WINNER FOR 2023")
print("=" * 70)

if pole_acc_br > pole_acc_lr:
    print(f"🏆 BayesianRidge wins: {pole_acc_br:.2f}% vs {pole_acc_lr:.2f}%")
    print(f"   Margin: +{pole_acc_br - pole_acc_lr:.2f}%")
elif pole_acc_lr > pole_acc_br:
    print(f"🏆 LinearRegression wins: {pole_acc_lr:.2f}% vs {pole_acc_br:.2f}%")
    print(f"   Margin: +{pole_acc_lr - pole_acc_br:.2f}%")
else:
    print(f"🤝 TIE: Both models at {pole_acc_br:.2f}%")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

avg_2023 = (pole_acc_br + pole_acc_lr) / 2
avg_mae = (mae_br + mae_lr) / 2

print(f"Pole Position Accuracy:")
print(f"  Average 2023 performance: {avg_2023:.2f}%")
print(f"  Change: {avg_2023 - 50.00:+.2f}%")

print(f"\nMean Absolute Error (MAE):")
print(f"  BayesianRidge: {mae_br:.3f} grid positions")
print(f"  LinearRegression: {mae_lr:.3f} grid positions")
print(f"  Average: {avg_mae:.3f} grid positions")

print("=" * 70)
