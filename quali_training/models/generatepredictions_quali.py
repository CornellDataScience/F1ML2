"""
Generate Qualifying Predictions using XGBoost Ranker

This script loads the trained qualifying ranker model and predicts
qualifying positions for an upcoming race.

Usage:
    1. Update the prediction CSV path below (p_df = ...)
    2. Run: python generatepredictions_quali.py
    3. Output: Predicted qualifying order from pole to last
"""

import pandas as pd
import xgboost as xgb
import numpy as np

print("=" * 70)
print("QUALIFYING POSITION PREDICTIONS - XGBoost Ranker")
print("=" * 70)

# Load Trained Model
print("\nLoading trained qualifying model...")
model = xgb.XGBRanker()
model.load_model("xgbranker_quali_model.json")
print("Model loaded: xgbranker_quali_model.json")

# Load reference dataset to match column structure
print("\nLoading reference dataset...")
df = pd.read_csv('../data/HOLY_qualifying_v1.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Create session IDs
id = 0
for s in set(df['season']):
    for r in set(df['round']):
        df.loc[(df['season'] == s) & (df['round'] == r), 'id'] = id
        id += 1
df.sort_values('id', inplace=True)

print(f"Reference dataset: {len(df)} records, {len(df.columns)} columns")

# Load Prediction DataFrame
# TODO: Update this path to your prediction CSV
# This should be a CSV with the same structure as HOLY_qualifying_v1.csv
# but for the upcoming race you want to predict
print("\nLoading prediction data...")
try:
    p_df = pd.read_csv('../../pred_df_quali_monaco.csv')  # UPDATE THIS PATH
    print(f"Prediction data loaded: {len(p_df)} drivers")
except FileNotFoundError:
    print("\nERROR: Prediction file not found!")
    print("Please create a prediction CSV for the upcoming race.")
    print("Path expected: ../../pred_df_quali_monaco.csv")
    print("\nThe prediction CSV should have the same columns as HOLY_qualifying_v1.csv")
    print("but with qualifying results (grid) missing (what we're predicting).")
    exit(1)

# Ensure columns match
print("\nAligning columns with training data...")
p_df = p_df.reindex(columns=df.columns, fill_value=0)

# Make predictions
print("\nGenerating qualifying predictions...")
feature_cols_to_drop = ['grid', 'driver', 'season', 'round', 'id']
X_pred = p_df.drop(columns=feature_cols_to_drop, errors='ignore')

predictions = model.predict(X_pred)
p_df['prediction_score'] = predictions

# Sort by prediction score (lower = better qualifying position)
p_df = p_df.sort_values('prediction_score')

# Display results
print("\n" + "=" * 70)
print("PREDICTED QUALIFYING ORDER")
print("=" * 70)
print(f"{'Position':<12} {'Driver':<20} {'Prediction Score':<20}")
print("-" * 70)

for idx, (index, row) in enumerate(p_df.iterrows(), 1):
    driver_name = row['driver'] if 'driver' in row else f"Driver {idx}"
    pred_score = row['prediction_score']
    position_str = f"P{idx}"

    # Highlight pole position
    if idx == 1:
        print(f"{position_str:<12} {driver_name:<20} {pred_score:<20.6f}  🏆 POLE")
    elif idx <= 3:
        print(f"{position_str:<12} {driver_name:<20} {pred_score:<20.6f}  ⭐")
    elif idx <= 10:
        print(f"{position_str:<12} {driver_name:<20} {pred_score:<20.6f}  (Q3)")
    else:
        print(f"{position_str:<12} {driver_name:<20} {pred_score:<20.6f}")

print("=" * 70)

# Save predictions to CSV
output_file = "qualifying_predictions.csv"
p_df[['driver', 'prediction_score']].to_csv(output_file, index=False)
print(f"\nPredictions saved to: {output_file}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Predicted Pole Position: {p_df.iloc[0]['driver']}")
print(f"Predicted P2: {p_df.iloc[1]['driver']}")
print(f"Predicted P3: {p_df.iloc[2]['driver']}")
print("\nPredicted Q3 qualifiers (Top 10):")
for idx in range(min(10, len(p_df))):
    print(f"  {idx + 1}. {p_df.iloc[idx]['driver']}")

print("=" * 70)
