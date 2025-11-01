"""
XGBoost Ranker for Qualifying Position Prediction

This model predicts qualifying positions (grid) using pairwise ranking.
It learns to rank drivers in order based on their qualifying performance.

Target: grid (qualifying position, 1 = pole, 2 = 2nd, etc.)
Dataset: HOLY_qualifying_v1.csv
"""

import xgboost as xgb
import numpy as np
import pandas as pd

# Import & Split Train/Test Data
print("Loading qualifying dataset...")
df = pd.read_csv('../data/HOLY_qualifying_v1.csv')

# Drop unnamed index columns if they exist
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
    y = df.loc[:, 'grid']  # CHANGED FROM 'podium' to 'grid'

    # Drop columns not needed for prediction
    X = df.drop(columns=['grid', 'driver', 'season', 'round', 'id'], errors='ignore')

    # EXPERIMENT: Drop qualifying_secs to test model without lap time data
    if 'qualifying_secs' in X.columns:
        print(f"  🔬 EXPERIMENT: Dropping 'qualifying_secs' column")
        X = X.drop(columns=['qualifying_secs'])

    # Drop any remaining object/string columns (circuit_id, constructor, etc.)
    # Keep only numeric columns
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"  Dropping {len(object_cols)} non-numeric columns: {list(object_cols)}")
        X = X.drop(columns=object_cols)

    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        print(f"  Converting {len(bool_cols)} boolean columns to int")
        X[bool_cols] = X[bool_cols].astype(int)

    return X, y

# Create Tuned Model (using same hyperparameters as race model)
# These were optimized for race prediction but should work well for qualifying too
model = xgb.XGBRanker(**{
    'n_estimators': 169,
    'max_depth': 7,
    'learning_rate': 0.023674177607202307,
    'colsample_bytree': 0.5438701967084194,
    'subsample': 0.4532488227992929,
    'alpha': 0.10664514806429565,
    'lambda': 0.09342144391433568,
    'min_child_weight': 140.49671900290397,
    'verbosity': 1,  # Changed to 1 to see progress
    'objective': 'rank:pairwise',
})

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
x_train, y_train = process_df(train_df)

print(f"Training feature matrix shape: {x_train.shape}")
print(f"Training target shape: {y_train.shape}")

# Create groups for training (one group per qualifying session)
train_groups = train_df.groupby('id').size().to_frame('size')['size'].to_numpy()
print(f"Number of training sessions: {len(train_groups)}")

# Train the model
print("\nTraining XGBoost Ranker for qualifying prediction...")
print("=" * 70)
model.fit(x_train, y_train, group=train_groups, verbose=True)

# Save the model
model.save_model("xgbranker_quali_model.json")
print("\n" + "=" * 70)
print("Model saved to: xgbranker_quali_model.json")

# Evaluate on TEST SET (2022)
print(f"\n{'='*70}")
print(f"EVALUATING ON TEST SET ({TEST_YEAR})")
print(f"{'='*70}")

x_test, y_test = process_df(test_df)
test_predictions = model.predict(x_test)

# Add predictions to test dataframe
test_df['prediction'] = test_predictions

# Calculate pole position accuracy
correct_poles = 0
total_sessions = 0
correct_top3 = 0
correct_top3_any_order = 0

for session_id in test_df['id'].unique():
    session_df = test_df[test_df['id'] == session_id].copy()

    if len(session_df) == 0:
        continue

    # Sort by prediction (lowest = best)
    predicted_order = session_df.sort_values('prediction')

    # Sort by actual grid (lowest = best)
    actual_order = session_df.sort_values('grid')

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

print("\nTraining complete! Model ready for qualifying predictions.")
print("Use generatepredictions_quali.py to make predictions for upcoming races.")
