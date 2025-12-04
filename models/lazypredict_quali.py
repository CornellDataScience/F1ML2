"""
LazyPredict - Test Multiple Regression Models for Qualifying Prediction

This script uses LazyPredict to automatically train and evaluate dozens of
regression models to find the best one for qualifying position prediction.

LazyPredict trains: Linear Regression, Ridge, Lasso, ElasticNet, Decision Trees,
Random Forest, XGBoost, LightGBM, SVR, KNN, and many more!
"""

import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("LAZYPREDICT - TESTING ALL REGRESSION MODELS")
print("=" * 70)

# Load data
print("\nLoading qualifying dataset...")
df = pd.read_csv('../data/HOLY_qualifying_v1.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"Dataset loaded: {len(df)} records")
print(f"Features: {len(df.columns)} columns")
print(f"Date range: {df['season'].min()}-{df['season'].max()}")

# Prepare features
print("\nPreparing features...")
y = df['grid']

# Drop columns not needed for prediction
X = df.drop(columns=['grid', 'driver', 'season', 'round'], errors='ignore')

# IMPORTANT: Drop qualifying_secs to prevent data leakage
# TEMPORARILY COMMENTED OUT TO TEST DATA LEAKAGE THEORY
# if 'qualifying_secs' in X.columns:
#     print(f"  🔬 DROPPING 'qualifying_secs' to prevent data leakage")
#     X = X.drop(columns=['qualifying_secs'])

# Drop any remaining object/string columns
object_cols = X.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    print(f"  Dropping {len(object_cols)} non-numeric columns: {list(object_cols)[:5]}...")
    X = X.drop(columns=object_cols)

# Convert boolean columns to int
bool_cols = X.select_dtypes(include=['bool']).columns
if len(bool_cols) > 0:
    print(f"  Converting {len(bool_cols)} boolean columns to int")
    X[bool_cols] = X[bool_cols].astype(int)

print(f"Feature matrix shape: {X.shape}")

# Split data: Train on <2022, Test on 2022
TEST_YEAR = 2022
print(f"\n{'='*70}")
print(f"DATA SPLIT: Training on <{TEST_YEAR}, Testing on {TEST_YEAR}")
print(f"{'='*70}")

train_mask = df['season'] < TEST_YEAR
X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[~train_mask]
y_test = y[~train_mask]

print(f"\nTraining set: {len(X_train)} records")
print(f"Test set: {len(X_test)} records")

# Handle missing values
print("\nHandling missing values...")
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())  # Use training means for test set

# Scale features (important for some models)
print("Scaling features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for LazyPredict
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

# Run LazyPredict
print(f"\n{'='*70}")
print("RUNNING LAZYPREDICT - THIS MAY TAKE A FEW MINUTES...")
print(f"{'='*70}")
print("Training and evaluating ~40 regression models...\n")

reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train_scaled, X_test_scaled, y_train, y_test)

# Display results
print(f"\n{'='*70}")
print("LAZYPREDICT RESULTS")
print(f"{'='*70}")
print("\nSorted by R-Squared (higher is better):\n")
print(models)

# Save full results
results_file = "lazypredict_results.csv"
models.to_csv(results_file)
print(f"\nFull results saved to: {results_file}")

# Highlight top models
print(f"\n{'='*70}")
print("TOP 5 MODELS")
print(f"{'='*70}")

top5 = models.head(5)
for i, (model_name, row) in enumerate(top5.iterrows(), 1):
    print(f"\n{i}. {model_name}")
    print(f"   R-Squared: {row['R-Squared']:.4f}")
    print(f"   RMSE: {row['RMSE']:.4f}")
    print(f"   Time Taken: {row['Time Taken']:.2f}s")

# Check if Linear Regression is in top 5
print(f"\n{'='*70}")
print("LINEAR REGRESSION PERFORMANCE")
print(f"{'='*70}")

if 'LinearRegression' in models.index:
    lr_row = models.loc['LinearRegression']
    rank = models.index.get_loc('LinearRegression') + 1
    print(f"Rank: {rank}/{len(models)}")
    print(f"R-Squared: {lr_row['R-Squared']:.4f}")
    print(f"RMSE: {lr_row['RMSE']:.4f}")
    print(f"Time Taken: {lr_row['Time Taken']:.2f}s")
else:
    print("LinearRegression not found in results")

print(f"\n{'='*70}")
print("NOTE: POLE POSITION ACCURACY")
print(f"{'='*70}")
print("LazyPredict returns R² and RMSE metrics, not actual predictions.")
print("For pole position accuracy, see: test_top_models_pole_accuracy.py")
print(f"{'='*70}")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Models tested: {len(models)}")
print(f"Best R² score: {models.iloc[0]['R-Squared']:.4f} ({models.index[0]})")
print(f"Best RMSE: {models.iloc[0]['RMSE']:.4f} ({models.index[0]})")
print(f"\nFor pole position accuracy results:")
print(f"  - BayesianRidge (no scaling): 50.00%")
print(f"  - LinearRegression (with scaling): 50.00%")
print(f"  - See test_top_models_pole_accuracy.py for full rankings")
print(f"{'='*70}")

print("\n✅ LazyPredict analysis complete!")
print("Review lazypredict_results.csv for detailed model comparisons")
