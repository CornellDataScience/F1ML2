"""
LazyPredict WITHOUT Scaling - Test all models on unscaled features
"""

import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor

print("=" * 70)
print("LAZYPREDICT (NO SCALING) - TESTING ALL MODELS")
print("=" * 70)

# Load data
print("\nLoading qualifying dataset...")
df = pd.read_csv('../data/HOLY_qualifying_v1.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"Dataset loaded: {len(df)} records")

# Prepare features
print("\nPreparing features...")
y = df['grid']
X = df.drop(columns=['grid', 'driver', 'season', 'round', 'circuit_id'], errors='ignore')

if 'qualifying_secs' in X.columns:
    print(f"  🔬 DROPPING 'qualifying_secs'")
    X = X.drop(columns=['qualifying_secs'])

object_cols = X.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    print(f"  Dropping {len(object_cols)} non-numeric columns")
    X = X.drop(columns=object_cols)

bool_cols = X.select_dtypes(include=['bool']).columns
if len(bool_cols) > 0:
    print(f"  Converting {len(bool_cols)} boolean columns to int")
    X[bool_cols] = X[bool_cols].astype(int)

print(f"Feature matrix shape: {X.shape}")

# Split by season
TEST_YEAR = 2022
print(f"\nDATA SPLIT: Training on <{TEST_YEAR}, Testing on {TEST_YEAR}")

train_mask = df['season'] < TEST_YEAR
X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[~train_mask]
y_test = y[~train_mask]

print(f"Training set: {len(X_train)} records")
print(f"Test set: {len(X_test)} records")

# Handle missing values (NO SCALING)
print("\nHandling missing values (NO SCALING)...")
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

# Run LazyPredict
print(f"\n{'='*70}")
print("RUNNING LAZYPREDICT WITHOUT SCALING...")
print(f"{'='*70}")
print("Training ~40 models (this may take a few minutes)...\n")

reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Display results
print(f"\n{'='*70}")
print("RESULTS - TOP 20 MODELS (NO SCALING)")
print(f"{'='*70}")
print("\nSorted by R-Squared:\n")
print(models.head(20))

# Save results
results_file = "lazypredict_no_scaling_results.csv"
models.to_csv(results_file)
print(f"\nFull results saved to: {results_file}")

# Top 5
print(f"\n{'='*70}")
print("TOP 5 MODELS (NO SCALING)")
print(f"{'='*70}")

top5 = models.head(5)
for i, (model_name, row) in enumerate(top5.iterrows(), 1):
    print(f"\n{i}. {model_name}")
    print(f"   R-Squared: {row['R-Squared']:.4f}")
    print(f"   RMSE: {row['RMSE']:.4f}")
    print(f"   Time: {row['Time Taken']:.2f}s")

# Check specific models
print(f"\n{'='*70}")
print("KEY MODELS COMPARISON")
print(f"{'='*70}")

for model_name in ['BayesianRidge', 'LinearRegression', 'Ridge', 'Lasso']:
    if model_name in models.index:
        rank = models.index.get_loc(model_name) + 1
        row = models.loc[model_name]
        print(f"\n{model_name}:")
        print(f"  Rank: {rank}/{len(models)}")
        print(f"  R²: {row['R-Squared']:.4f}")
        print(f"  RMSE: {row['RMSE']:.4f}")

print(f"\n{'='*70}")
print("NOTE: POLE POSITION ACCURACY")
print(f"{'='*70}")
print("LazyPredict returns R² and RMSE metrics, not actual predictions.")
print("For pole position accuracy, see: test_top_models_pole_accuracy.py")
print(f"{'='*70}")

print(f"\n{'='*70}")
print("SUMMARY (NO SCALING)")
print(f"{'='*70}")
print(f"Models tested: {len(models)}")
print(f"Best R² score: {models.iloc[0]['R-Squared']:.4f} ({models.index[0]})")
print(f"Best RMSE: {models.iloc[0]['RMSE']:.4f} ({models.index[0]})")
print(f"\nFor pole position accuracy results:")
print(f"  - BayesianRidge (no scaling): 50.00%")
print(f"  - Ridge (no scaling): 45.45%")
print(f"  - See test_top_models_pole_accuracy.py for full rankings")
print(f"{'='*70}")

print(f"\n{'='*70}")
print("COMPARISON: WITH vs WITHOUT SCALING")
print(f"{'='*70}")
print("\nBest R² scores:")
print(f"  WITH scaling:    0.4712 (BayesianRidge)")
print(f"  WITHOUT scaling: {models.iloc[0]['R-Squared']:.4f} ({models.index[0]})")
print(f"{'='*70}")

print("\n✅ LazyPredict (no scaling) complete!")
