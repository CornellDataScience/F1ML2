"""
Compare v1 vs v2 for overall grid position prediction (R², RMSE) using 40+ regression models.
Tests which models benefit most from telemetry features. Train 2018-2022, test 2023.
"""

import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("\nComparing v1 vs v2 with LazyPredict")
print("Train 2018-2022, test 2023\n")

def prepare_dataset(df_path, dataset_name, filter_telemetry=False):
    """Load and prepare dataset."""
    print(f"Loading {dataset_name}...")
    
    df = pd.read_csv(df_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    df = df[df['season'] >= 2018].copy()
    
    if filter_telemetry:
        telemetry_cols = ['driver_avg_corner_speed', 'driver_top_speed', 'track_total_corners']
        df = df.dropna(subset=telemetry_cols)
    
    print(f"  {len(df)} records")
    
    y = df['grid']
    X = df.drop(columns=['grid', 'driver', 'season', 'round', 'qualifying_secs'], errors='ignore')
    
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        X = X.drop(columns=object_cols)
    
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
    
    print(f"  {X.shape[1]} features")
    
    return X, y, df['season']


def evaluate_models(X, y, season, test_year=2023):
    """Split by year and run LazyPredict."""
    
    train_mask = season < test_year
    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    X_test = X[~train_mask].copy()
    y_test = y[~train_mask].copy()
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Handle NaN
    X_train = X_train.dropna(axis=1, how='all')
    X_test = X_test[X_train.columns]
    X_train = X_train.fillna(X_train.mean()).fillna(0)
    X_test = X_test.fillna(X_train.mean()).fillna(0)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns, index=X_test.index)
    
    print(f"  Running LazyPredict...")
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = reg.fit(X_train_scaled, X_test_scaled, y_train, y_test)
    
    return models


# Test v1
X_v1, y_v1, season_v1 = prepare_dataset(
    df_path='../data/HOLY_qualifying_v1.csv',
    dataset_name='v1 (baseline)',
    filter_telemetry=False
)
models_v1 = evaluate_models(X_v1, y_v1, season_v1, test_year=2023)

print("\nv1 top 10:")
print(models_v1.head(10))

# Test v2
print()
X_v2, y_v2, season_v2 = prepare_dataset(
    df_path='../data/HOLY_qualifying_v2.csv',
    dataset_name='v2 (with telemetry)',
    filter_telemetry=True
)
models_v2 = evaluate_models(X_v2, y_v2, season_v2, test_year=2023)

print("\nv2 top 10:")
print(models_v2.head(10))

# Comparison
print("\nComparison:")
best_v1 = models_v1.iloc[0]
best_v2 = models_v2.iloc[0]

print(f"\nBest v1: {models_v1.index[0]}")
print(f"  R²: {best_v1['R-Squared']:.4f}, RMSE: {best_v1['RMSE']:.4f}")

print(f"\nBest v2: {models_v2.index[0]}")
print(f"  R²: {best_v2['R-Squared']:.4f}, RMSE: {best_v2['RMSE']:.4f}")

if best_v2['R-Squared'] > best_v1['R-Squared']:
    improvement = (best_v2['R-Squared'] - best_v1['R-Squared']) / abs(best_v1['R-Squared']) * 100
    print(f"\nv2 better by {improvement:.2f}%")
elif best_v1['R-Squared'] > best_v2['R-Squared']:
    decline = (best_v1['R-Squared'] - best_v2['R-Squared']) / abs(best_v1['R-Squared']) * 100
    print(f"\nv1 better by {decline:.2f}%")

# BayesianRidge comparison
if 'BayesianRidge' in models_v1.index and 'BayesianRidge' in models_v2.index:
    br_v1 = models_v1.loc['BayesianRidge']
    br_v2 = models_v2.loc['BayesianRidge']
    print(f"\nBayesianRidge: v1 R²={br_v1['R-Squared']:.4f}, v2 R²={br_v2['R-Squared']:.4f}")

# Save
models_v1.to_csv('lazypredict_v1_results.csv')
models_v2.to_csv('lazypredict_v2_results.csv')
print("\nSaved results to lazypredict_v1_results.csv and lazypredict_v2_results.csv")
print("Done.")

