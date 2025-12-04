"""
Test Models with Telemetry Features on 2023 Data

This script tests BayesianRidge and LinearRegression models
on the 2023 season using the new telemetry-enhanced dataset.

Train on: 2018-2022 (with telemetry features)
Test on: 2023
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("TESTING MODELS WITH TELEMETRY FEATURES ON 2023")
print("=" * 70)

# Load datasets
print("\nLoading datasets...")
train_df = pd.read_csv('../data/HOLY_qualifying_modern_train2023.csv')
test_df_full = pd.read_csv('../data/HOLY_qualifying_modern_2018_2023.csv')
test_df = test_df_full[test_df_full['season'] == 2023].copy()

print(f"Training set (2018-2022): {train_df.shape}")
print(f"Test set (2023): {test_df.shape}")
print(f"Test sessions: {test_df.groupby(['season', 'round']).ngroups}")

# Check telemetry features
telemetry_cols = [col for col in train_df.columns if 'practice_' in col]
print(f"\nTelemetry features included: {len(telemetry_cols)}")
for col in telemetry_cols[:5]:
    print(f"  - {col}")
if len(telemetry_cols) > 5:
    print(f"  ... and {len(telemetry_cols) - 5} more")

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

# Prepare data
print("\nPreparing features...")
X_train, y_train = process_df(train_df)
X_test, y_test = process_df(test_df)

# Align columns
print("Aligning features between train and test sets...")
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

print(f"\nRESULTS:")
print(f"  Pole Position Accuracy: {pole_acc_br:.2f}% ({correct_br}/{total_br})")

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

print(f"\nRESULTS:")
print(f"  Pole Position Accuracy: {pole_acc_lr:.2f}% ({correct_lr}/{total_lr})")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS WITH TELEMETRY FEATURES")
print("=" * 70)

results = pd.DataFrame({
    'Model': ['BayesianRidge (no scaling)', 'LinearRegression (with scaling)'],
    '2023_Accuracy': [pole_acc_br, pole_acc_lr],
    'Correct/Total': [f'{correct_br}/{total_br}', f'{correct_lr}/{total_lr}']
})

print(results.to_string(index=False))

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

# Check which model is better
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

# Summary stats
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

avg_2023 = (pole_acc_br + pole_acc_lr) / 2
print(f"Average 2023 performance: {avg_2023:.2f}%")
print(f"Telemetry features used: {len(telemetry_cols)}")

print("\n" + "=" * 70)
