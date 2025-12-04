"""
Test Models on 2023 Data - Hybrid Dataset

This script tests models using the HYBRID dataset:
- Historical data (1983-2023) for long-term patterns
- Telemetry features (2018-2023) for modern races where available

Train on: 1983-2022 (with telemetry for 2018-2022)
Test on: 2023
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("TESTING MODELS ON 2023 - HYBRID DATASET")
print("Historical (1983-2022) + Telemetry (2018-2022)")
print("=" * 70)

# Load datasets
print("\nLoading hybrid dataset...")
train_df = pd.read_csv('../data/HOLY_qualifying_hybrid_train2023.csv')
full_df = pd.read_csv('../data/HOLY_qualifying_hybrid_1983_2023.csv')
test_df = full_df[full_df['season'] == 2023].copy()

print(f"Training set: {train_df.shape}")
print(f"Test set (2023): {test_df.shape}")

# Check telemetry coverage
telemetry_cols = [col for col in train_df.columns if 'practice_' in col]
train_with_telem = train_df[train_df['practice_overall_avg_speed'].notna()]
test_with_telem = test_df[test_df['practice_overall_avg_speed'].notna()]

print(f"\nTelemetry features: {len(telemetry_cols)}")
print(f"Training records with telemetry: {len(train_with_telem)}/{len(train_df)} ({100*len(train_with_telem)/len(train_df):.1f}%)")
print(f"Test records with telemetry: {len(test_with_telem)}/{len(test_df)} ({100*len(test_with_telem)/len(test_df):.1f}%)")

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
print("Aligning features...")
all_cols = set(X_train.columns) | set(X_test.columns)
for col in all_cols - set(X_train.columns):
    X_train[col] = 0
for col in all_cols - set(X_test.columns):
    X_test[col] = 0

X_train = X_train[sorted(all_cols)]
X_test = X_test[sorted(all_cols)]

# Fill missing values (telemetry NaNs for old data)
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
print("COMPARISON: HYBRID vs PURE APPROACHES")
print("=" * 70)

results = pd.DataFrame({
    'Model': ['BayesianRidge', 'LinearRegression'],
    'Hybrid (1983-2023 + Telemetry)': [
        f'{pole_acc_br:.2f}%',
        f'{pole_acc_lr:.2f}%'
    ],
    'Historical Only (from test_2023_predictions)': [
        '47.62%',
        '14.29%'
    ],
    'Telemetry Only (from test_2023_with_telemetry)': [
        '23.81%',
        '57.14%'
    ]
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

# Winner
print("\n" + "=" * 70)
print("WINNER FOR 2023 - HYBRID APPROACH")
print("=" * 70)

if pole_acc_br > pole_acc_lr:
    print(f"🏆 BayesianRidge wins: {pole_acc_br:.2f}% vs {pole_acc_lr:.2f}%")
    print(f"   Margin: +{pole_acc_br - pole_acc_lr:.2f}%")
elif pole_acc_lr > pole_acc_br:
    print(f"🏆 LinearRegression wins: {pole_acc_lr:.2f}% vs {pole_acc_br:.2f}%")
    print(f"   Margin: +{pole_acc_lr - pole_acc_br:.2f}%")
else:
    print(f"🤝 TIE: Both at {pole_acc_br:.2f}%")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

avg_hybrid = (pole_acc_br + pole_acc_lr) / 2
print(f"\nHybrid approach average: {avg_hybrid:.2f}%")
print(f"Historical only average: {(47.62 + 14.29)/2:.2f}%")
print(f"Telemetry only average: {(23.81 + 57.14)/2:.2f}%")

print("\n✓ The hybrid approach combines:")
print("  - Deep historical patterns (1983-2022)")
print("  - Modern telemetry insights (2018-2022)")
print("  - Best of both worlds!")

print("=" * 70)
