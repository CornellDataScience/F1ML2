"""
XGBoost Hyperparameter Tuning for Qualifying Prediction

This script tunes XGBoost Ranker specifically for qualifying position prediction.
The current model uses hyperparameters optimized for race prediction (podium).
We'll use Optuna to find better parameters for qualifying (grid position).

Target: grid (qualifying position, 1 = pole, 2 = 2nd, etc.)
Dataset: HOLY_qualifying_v2.csv
"""

import xgboost as xgb
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

print("=" * 70)
print("XGBOOST HYPERPARAMETER TUNING - QUALIFYING PREDICTION")
print("=" * 70)

# Load data
print("\nLoading qualifying dataset...")
df = pd.read_csv('../data/HOLY_qualifying_v2.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"Dataset loaded: {len(df)} records")
print(f"Features: {len(df.columns)} columns")
print(f"Date range: {df['season'].min()}-{df['season'].max()}")

# Create unique ID for each qualifying session (season + round)
id = 0
for s in set(df['season']):
    for r in set(df['round']):
        df.loc[(df['season'] == s) & (df['round'] == r), 'id'] = id
        id += 1

df.sort_values('id', inplace=True)
print(f"Created {id} unique qualifying sessions")

def process_df(df):
    """
    Prepare dataframe for training
    Returns: (X, y) where y is qualifying position (grid)
    """
    y = df.loc[:, 'grid']

    # Drop columns not needed for prediction
    X = df.drop(columns=['grid', 'driver', 'season', 'round', 'id'], errors='ignore')

    # Drop qualifying_secs to prevent data leakage
    if 'qualifying_secs' in X.columns:
        X = X.drop(columns=['qualifying_secs'])

    # Drop any remaining object/string columns
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        X = X.drop(columns=object_cols)

    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    return X, y

# Split data: Train/Validation (pre-2022), Test (2022)
# We'll use 2021 as validation set for tuning
VALIDATION_YEAR = 2021
TEST_YEAR = 2022

print(f"\n{'='*70}")
print(f"DATA SPLIT:")
print(f"  Training: <{VALIDATION_YEAR}")
print(f"  Validation: {VALIDATION_YEAR} (for hyperparameter tuning)")
print(f"  Test: {TEST_YEAR} (final evaluation)")
print(f"{'='*70}")

train_df = df[df['season'] < VALIDATION_YEAR].copy()
val_df = df[df['season'] == VALIDATION_YEAR].copy()
test_df = df[df['season'] == TEST_YEAR].copy()

print(f"\nTraining set: {len(train_df)} records ({train_df['season'].min()}-{train_df['season'].max()})")
print(f"Validation set: {len(val_df)} records (season {VALIDATION_YEAR})")
print(f"Test set: {len(test_df)} records (season {TEST_YEAR})")

# Prepare training data
x_train, y_train = process_df(train_df)
x_val, y_val = process_df(val_df)

train_groups = train_df.groupby('id').size().to_frame('size')['size'].to_numpy()
val_groups = val_df.groupby('id').size().to_frame('size')['size'].to_numpy()

print(f"\nTraining feature matrix shape: {x_train.shape}")
print(f"Validation feature matrix shape: {x_val.shape}")
print(f"Number of training sessions: {len(train_groups)}")
print(f"Number of validation sessions: {len(val_groups)}")

# Define objective function for Optuna
def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization
    Returns: Pole position accuracy on validation set
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'alpha': trial.suggest_float('alpha', 0.0, 1.0),
        'lambda': trial.suggest_float('lambda', 0.0, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 200),
        'verbosity': 0,
        'objective': 'rank:pairwise',
    }

    # Train model
    model = xgb.XGBRanker(**params)
    model.fit(x_train, y_train, group=train_groups, verbose=False)

    # Predict on validation set
    val_predictions = model.predict(x_val)
    val_df['prediction'] = val_predictions

    # Calculate pole position accuracy
    correct_poles = 0
    total_sessions = 0

    for session_id in val_df['id'].unique():
        session_df = val_df[val_df['id'] == session_id].copy()

        if len(session_df) == 0:
            continue

        # Sort by prediction (lowest = best)
        predicted_order = session_df.sort_values('prediction')

        # Sort by actual grid (lowest = best)
        actual_order = session_df.sort_values('grid')

        # Check if pole position is correct
        if len(predicted_order) > 0 and len(actual_order) > 0:
            predicted_pole = predicted_order.iloc[0]['driver']
            actual_pole = actual_order.iloc[0]['driver']

            if predicted_pole == actual_pole:
                correct_poles += 1

        total_sessions += 1

    pole_accuracy = (correct_poles / total_sessions) * 100 if total_sessions > 0 else 0

    return pole_accuracy

# Run Optuna optimization
print(f"\n{'='*70}")
print("STARTING HYPERPARAMETER OPTIMIZATION")
print(f"{'='*70}")
print(f"Optimization metric: Pole position accuracy on {VALIDATION_YEAR}")
print(f"Number of trials: 50")
print(f"This may take 5-10 minutes...\n")

study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42)
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

# Print results
print(f"\n{'='*70}")
print("OPTIMIZATION RESULTS")
print(f"{'='*70}")
print(f"Best pole accuracy on validation set: {study.best_value:.2f}%")
print(f"\nBest hyperparameters:")
print("-" * 70)
for key, value in study.best_params.items():
    print(f"  {key:20s}: {value}")

# Save best parameters
import json
with open('best_xgb_quali_params.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)
print(f"\nBest parameters saved to: best_xgb_quali_params.json")

# Train final model with best parameters on full pre-2022 data
print(f"\n{'='*70}")
print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
print(f"{'='*70}")

# Combine train and validation for final training
full_train_df = df[df['season'] < TEST_YEAR].copy()
x_full_train, y_full_train = process_df(full_train_df)
full_train_groups = full_train_df.groupby('id').size().to_frame('size')['size'].to_numpy()

print(f"Full training set: {len(full_train_df)} records ({full_train_df['season'].min()}-{full_train_df['season'].max()})")

best_params = study.best_params.copy()
best_params['verbosity'] = 1
best_params['objective'] = 'rank:pairwise'

final_model = xgb.XGBRanker(**best_params)
final_model.fit(x_full_train, y_full_train, group=full_train_groups, verbose=True)

# Save the model
final_model.save_model("xgbranker_quali_tuned_model.json")
print(f"\nTuned model saved to: xgbranker_quali_tuned_model.json")

# Evaluate on TEST SET (2022)
print(f"\n{'='*70}")
print(f"EVALUATING ON TEST SET ({TEST_YEAR})")
print(f"{'='*70}")

x_test, y_test = process_df(test_df)
test_predictions = final_model.predict(x_test)
test_df['prediction'] = test_predictions

# Calculate metrics
correct_poles = 0
total_sessions = 0
correct_top3 = 0
correct_top3_any_order = 0

for session_id in test_df['id'].unique():
    session_df = test_df[test_df['id'] == session_id].copy()

    if len(session_df) == 0:
        continue

    predicted_order = session_df.sort_values('prediction')
    actual_order = session_df.sort_values('grid')

    # Pole position accuracy
    if len(predicted_order) > 0 and len(actual_order) > 0:
        predicted_pole = predicted_order.iloc[0]['driver']
        actual_pole = actual_order.iloc[0]['driver']

        if predicted_pole == actual_pole:
            correct_poles += 1

    # Top 3 metrics
    if len(predicted_order) >= 3 and len(actual_order) >= 3:
        pred_top3 = predicted_order.iloc[:3]['driver'].tolist()
        actual_top3 = actual_order.iloc[:3]['driver'].tolist()

        if pred_top3 == actual_top3:
            correct_top3 += 1

        if set(pred_top3) == set(actual_top3):
            correct_top3_any_order += 1

    total_sessions += 1

# Calculate accuracies
pole_accuracy = (correct_poles / total_sessions) * 100 if total_sessions > 0 else 0
top3_order_accuracy = (correct_top3 / total_sessions) * 100 if total_sessions > 0 else 0
top3_any_accuracy = (correct_top3_any_order / total_sessions) * 100 if total_sessions > 0 else 0

print(f"\n{'='*70}")
print(f"TEST SET RESULTS ({TEST_YEAR})")
print(f"{'='*70}")
print(f"Total qualifying sessions: {total_sessions}")
print(f"\nPole Position Accuracy: {pole_accuracy:.2f}% ({correct_poles}/{total_sessions})")
print(f"Top 3 in Exact Order: {top3_order_accuracy:.2f}% ({correct_top3}/{total_sessions})")
print(f"Top 3 Any Order: {top3_any_accuracy:.2f}% ({correct_top3_any_order}/{total_sessions})")
print(f"{'='*70}")

# Comparison
print(f"\n{'='*70}")
print("MODEL COMPARISON")
print(f"{'='*70}")
print(f"XGBoost (race hyperparameters):    36.36% pole accuracy")
print(f"Linear Regression:                 45.45% pole accuracy")
print(f"XGBoost (tuned for qualifying):    {pole_accuracy:.2f}% pole accuracy")
print(f"{'='*70}")

print("\nHyperparameter tuning complete!")
