"""
Linear Regression Model for Qualifying Position Prediction

This model predicts qualifying positions (grid) using linear regression.
It learns continuous relationships between features and qualifying performance.

Target: grid (qualifying position, 1 = pole, 2 = 2nd, etc.)
Dataset: HOLY_qualifying_v2.csv (with practice telemetry features)

Model Architecture:
- Sklearn LinearRegression
- StandardScaler for feature normalization
- Trained on historical data, tested on unseen 2022 season

Performance Metrics:
- Pole Position Accuracy (did we predict pole correctly?)
- Top 3 Accuracy (did we get top 3 right?)
- Mean Absolute Error (average position error)
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=4)

# Load and prepare data
print("=" * 70)
print("LINEAR REGRESSION - QUALIFYING POSITION PREDICTION")
print("=" * 70)

print("\nLoading qualifying dataset...")
# Test with v1 (no telemetry) to isolate the effect of telemetry features
df = pd.read_csv('../data/HOLY_qualifying_v1.csv')

# Drop unnamed index columns if they exist
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"Dataset loaded: {len(df)} records")
print(f"Features: {len(df.columns)} columns")
print(f"Date range: {df['season'].min()}-{df['season'].max()}")

# Define process_df function to prepare features
def process_df(df):
    """
    Prepare dataframe for training
    Returns: (X, y) where y is qualifying position (grid)
    """
    y = df.loc[:, 'grid']

    # Drop columns not needed for prediction
    X = df.drop(columns=['grid', 'driver', 'season', 'round'], errors='ignore')

    # IMPORTANT: Drop qualifying_secs to prevent data leakage
    # qualifying_secs is the actual qualifying lap time, which IS the result we're predicting
    if 'qualifying_secs' in X.columns:
        print(f"  🔬 DROPPING 'qualifying_secs' to prevent data leakage")
        X = X.drop(columns=['qualifying_secs'])

    # Drop any remaining object/string columns (circuit_id, constructor, etc.)
    # Keep only numeric columns
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"  Dropping {len(object_cols)} non-numeric columns: {list(object_cols)[:5]}...")
        X = X.drop(columns=object_cols)

    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        print(f"  Converting {len(bool_cols)} boolean columns to int")
        X[bool_cols] = X[bool_cols].astype(int)

    return X, y

# Split data: Train on everything before 2022, Test on 2022
TEST_YEAR = 2022
print(f"\n{'='*70}")
print(f"DATA SPLIT: Training on <{TEST_YEAR}, Testing on {TEST_YEAR}")
print(f"{'='*70}")

train_df = df[df['season'] < TEST_YEAR].copy()
test_df = df[df['season'] == TEST_YEAR].copy()

print(f"\nTraining set: {len(train_df)} records ({train_df['season'].min()}-{train_df['season'].max()})")
print(f"Test set: {len(test_df)} records (season {TEST_YEAR})")

# Prepare training data
print("\nPreparing training features and target...")
X_train, y_train = process_df(train_df)

print(f"Training feature matrix shape: {X_train.shape}")
print(f"Training target shape: {y_train.shape}")

# Handle missing values (fill with column mean)
print("\nHandling missing values...")
X_train = X_train.fillna(X_train.mean())

# Scale features
print("Scaling features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Train the model
print("\nTraining Linear Regression model...")
print("=" * 70)
model = LinearRegression(fit_intercept=True)
model.fit(X_train_scaled, y_train)

print("\nModel trained successfully!")
print(f"Number of features used: {len(model.coef_)}")
print(f"Model intercept: {model.intercept_:.4f}")

# Display top 10 most important coefficients (by absolute value)
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': model.coef_
})
feature_importance['abs_coef'] = feature_importance['coefficient'].abs()
feature_importance = feature_importance.sort_values('abs_coef', ascending=False)

print("\nTop 10 Most Important Features:")
print("-" * 70)
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:40s}: {row['coefficient']:10.6f}")

# Save the model and scaler
print("\n" + "=" * 70)
model_filename = "linear_reg_quali_model.pkl"
scaler_filename = "linear_reg_quali_scaler.pkl"

with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to: {model_filename}")

with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to: {scaler_filename}")

# Evaluate on TEST SET (2022)
print(f"\n{'='*70}")
print(f"EVALUATING ON TEST SET ({TEST_YEAR})")
print(f"{'='*70}")

X_test, y_test = process_df(test_df)

# Handle missing values with training set means
X_test = X_test.fillna(X_train.mean())

# Scale using training scaler
X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Make predictions
test_predictions = model.predict(X_test_scaled)

# Add predictions to test dataframe
test_df['prediction'] = test_predictions

# Calculate performance metrics
correct_poles = 0
total_sessions = 0
correct_top3 = 0
correct_top3_any_order = 0
position_errors = []

# Group by session (season + round combination)
for (season, round_num), session_df in test_df.groupby(['season', 'round']):
    if len(session_df) == 0:
        continue

    # Sort by prediction (lowest = best)
    predicted_order = session_df.sort_values('prediction').reset_index(drop=True)

    # Sort by actual grid (lowest = best)
    actual_order = session_df.sort_values('grid').reset_index(drop=True)

    # Pole position accuracy
    if len(predicted_order) > 0 and len(actual_order) > 0:
        predicted_pole = predicted_order.iloc[0]['driver']
        actual_pole = actual_order.iloc[0]['driver']

        if predicted_pole == actual_pole:
            correct_poles += 1

    # Top 3 in exact order
    if len(predicted_order) >= 3 and len(actual_order) >= 3:
        pred_top3 = predicted_order.iloc[:3]['driver'].tolist()
        actual_top3 = actual_order.iloc[:3]['driver'].tolist()

        if pred_top3 == actual_top3:
            correct_top3 += 1

        # Top 3 in any order
        if set(pred_top3) == set(actual_top3):
            correct_top3_any_order += 1

    # Calculate position errors for all drivers in session
    for _, driver_row in session_df.iterrows():
        actual_pos = driver_row['grid']
        predicted_pos = driver_row['prediction']

        # Find actual position in predicted ranking
        driver_pred_pos = (predicted_order[predicted_order['driver'] == driver_row['driver']].index[0] + 1
                          if driver_row['driver'] in predicted_order['driver'].values else actual_pos)

        position_errors.append(abs(actual_pos - driver_pred_pos))

    total_sessions += 1

# Calculate metrics
pole_accuracy = (correct_poles / total_sessions) * 100 if total_sessions > 0 else 0
top3_order_accuracy = (correct_top3 / total_sessions) * 100 if total_sessions > 0 else 0
top3_any_accuracy = (correct_top3_any_order / total_sessions) * 100 if total_sessions > 0 else 0
mean_absolute_error = np.mean(position_errors) if position_errors else 0

print(f"\n{'='*70}")
print(f"TEST SET RESULTS ({TEST_YEAR})")
print(f"{'='*70}")
print(f"Total qualifying sessions: {total_sessions}")
print(f"\nPole Position Accuracy: {pole_accuracy:.2f}% ({correct_poles}/{total_sessions})")
print(f"Top 3 in Exact Order: {top3_order_accuracy:.2f}% ({correct_top3}/{total_sessions})")
print(f"Top 3 Any Order: {top3_any_accuracy:.2f}% ({correct_top3_any_order}/{total_sessions})")
print(f"Mean Absolute Position Error: {mean_absolute_error:.2f} positions")
print(f"{'='*70}")

# Save predictions to CSV
# predictions_filename = "linear_reg_quali_predictions.csv"
# predictions_df = test_df[['season', 'round', 'driver', 'grid', 'prediction']].copy()
# predictions_df = predictions_df.sort_values(['season', 'round', 'prediction'])
# predictions_df.to_csv(predictions_filename, index=False)
# print(f"\nPredictions saved to: {predictions_filename}")

print("\nTraining complete! Model ready for qualifying predictions.")
print("Use this model to predict grid positions based on historical data and practice telemetry.")

# Print comparison with baseline
print(f"\n{'='*70}")
print("BASELINE COMPARISONS")
print(f"{'='*70}")
print(f"Random guess: ~5% pole accuracy")
print(f"Championship leader heuristic: ~30-40% pole accuracy")
print(f"Linear Regression Model: {pole_accuracy:.2f}% pole accuracy")
print(f"{'='*70}")
