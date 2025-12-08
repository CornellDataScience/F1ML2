"""
BayesianRidge WITHOUT Scaling - Test if scaling hurts performance
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

print("=" * 70)
print("BAYESIAN RIDGE (NO SCALING) - QUALIFYING PREDICTION")
print("=" * 70)

# Load data
print("\nLoading qualifying dataset...")
df = pd.read_csv('../data/HOLY_qualifying_v1.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"Dataset loaded: {len(df)} records")

def process_df(df):
    y = df.loc[:, 'grid']
    X = df.drop(columns=['grid', 'driver', 'season', 'round'], errors='ignore')

    if 'qualifying_secs' in X.columns:
        print(f"  🔬 DROPPING 'qualifying_secs'")
        X = X.drop(columns=['qualifying_secs'])

    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        X = X.drop(columns=object_cols)

    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    return X, y

# Split data
TEST_YEAR = 2022
train_df = df[df['season'] < TEST_YEAR].copy()
test_df = df[df['season'] == TEST_YEAR].copy()

print(f"\nTraining set: {len(train_df)} records")
print(f"Test set: {len(test_df)} records")

# Prepare data WITHOUT SCALING
print("\nPreparing features (NO SCALING)...")
X_train, y_train = process_df(train_df)
X_train = X_train.fillna(X_train.mean())

# Train BayesianRidge WITHOUT scaling
print("Training BayesianRidge model (no scaling)...")
model = BayesianRidge()
model.fit(X_train, y_train)  # No scaling!

print("Model trained!")

# Evaluate
print(f"\nEVALUATING ON TEST SET ({TEST_YEAR})")
print("=" * 70)

X_test, y_test = process_df(test_df)
X_test = X_test.fillna(X_train.mean())

test_predictions = model.predict(X_test)  # No scaling!
test_df['prediction'] = test_predictions

# Calculate pole accuracy
correct_poles = 0
total_sessions = 0

for (season, round_num), session_df in test_df.groupby(['season', 'round']):
    if len(session_df) == 0:
        continue

    predicted_order = session_df.sort_values('prediction')
    actual_order = session_df.sort_values('grid')

    if len(predicted_order) > 0 and len(actual_order) > 0:
        if predicted_order.iloc[0]['driver'] == actual_order.iloc[0]['driver']:
            correct_poles += 1

    total_sessions += 1

pole_accuracy = (correct_poles / total_sessions) * 100

print(f"\nTEST SET RESULTS ({TEST_YEAR})")
print("=" * 70)
print(f"Total sessions: {total_sessions}")
print(f"Pole Position Accuracy: {pole_accuracy:.2f}% ({correct_poles}/{total_sessions})")
print("=" * 70)

print(f"\nCOMPARISON")
print("=" * 70)
print(f"BayesianRidge (NO scaling):  {pole_accuracy:.2f}% pole accuracy")
print(f"BayesianRidge (WITH scaling): 40.91% pole accuracy")
print(f"LinearRegression (WITH scaling): 50.00% pole accuracy")
print("=" * 70)
