"""
BayesianRidge Model for Qualifying Position Prediction

Based on LazyPredict results, BayesianRidge is the top-performing model.
It's a regularized linear regression with Bayesian priors that prevents overfitting.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=4)

print("=" * 70)
print("BAYESIAN RIDGE - QUALIFYING POSITION PREDICTION")
print("=" * 70)
print("(Top model from LazyPredict analysis)")

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
        print(f"  🔬 DROPPING 'qualifying_secs' to prevent data leakage")
        X = X.drop(columns=['qualifying_secs'])

    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"  Dropping {len(object_cols)} non-numeric columns")
        X = X.drop(columns=object_cols)

    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        print(f"  Converting {len(bool_cols)} boolean columns to int")
        X[bool_cols] = X[bool_cols].astype(int)

    return X, y

# Split data
TEST_YEAR = 2022
print(f"\nDATA SPLIT: Training on <{TEST_YEAR}, Testing on {TEST_YEAR}")

train_df = df[df['season'] < TEST_YEAR].copy()
test_df = df[df['season'] == TEST_YEAR].copy()

print(f"Training set: {len(train_df)} records")
print(f"Test set: {len(test_df)} records")

# Prepare data
print("\nPreparing training features...")
X_train, y_train = process_df(train_df)
X_train = X_train.fillna(X_train.mean())

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train BayesianRidge
print("\nTraining BayesianRidge model...")
model = BayesianRidge()
model.fit(X_train_scaled, y_train)

print("\nModel trained!")
print(f"Number of features: {len(model.coef_)}")

# Save model
with open('bayesian_ridge_quali_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('bayesian_ridge_quali_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Model and scaler saved!")

# Evaluate on test set
print(f"\nEVALUATING ON TEST SET ({TEST_YEAR})")
print("=" * 70)

X_test, y_test = process_df(test_df)
X_test = X_test.fillna(X_train.mean())
X_test_scaled = scaler.transform(X_test)

test_predictions = model.predict(X_test_scaled)
test_df['prediction'] = test_predictions

# Calculate metrics
correct_poles = 0
total_sessions = 0
correct_top3 = 0
correct_top3_any_order = 0

for (season, round_num), session_df in test_df.groupby(['season', 'round']):
    if len(session_df) == 0:
        continue

    predicted_order = session_df.sort_values('prediction')
    actual_order = session_df.sort_values('grid')

    # Pole position
    if len(predicted_order) > 0 and len(actual_order) > 0:
        if predicted_order.iloc[0]['driver'] == actual_order.iloc[0]['driver']:
            correct_poles += 1

    # Top 3
    if len(predicted_order) >= 3 and len(actual_order) >= 3:
        pred_top3 = predicted_order.iloc[:3]['driver'].tolist()
        actual_top3 = actual_order.iloc[:3]['driver'].tolist()

        if pred_top3 == actual_top3:
            correct_top3 += 1

        if set(pred_top3) == set(actual_top3):
            correct_top3_any_order += 1

    total_sessions += 1

pole_accuracy = (correct_poles / total_sessions) * 100
top3_order_accuracy = (correct_top3 / total_sessions) * 100
top3_any_accuracy = (correct_top3_any_order / total_sessions) * 100

print(f"\nTEST SET RESULTS ({TEST_YEAR})")
print("=" * 70)
print(f"Total qualifying sessions: {total_sessions}")
print(f"\nPole Position Accuracy: {pole_accuracy:.2f}% ({correct_poles}/{total_sessions})")
print(f"Top 3 in Exact Order: {top3_order_accuracy:.2f}% ({correct_top3}/{total_sessions})")
print(f"Top 3 Any Order: {top3_any_accuracy:.2f}% ({correct_top3_any_order}/{total_sessions})")
print("=" * 70)

print(f"\nCOMPARISON WITH OTHER MODELS")
print("=" * 70)
print(f"BayesianRidge:          {pole_accuracy:.2f}% pole accuracy")
print(f"LinearRegression:       50.00% pole accuracy")
print(f"XGBoost Ranker:         36.36% pole accuracy")
print("=" * 70)

print("\n✅ BayesianRidge model complete!")
