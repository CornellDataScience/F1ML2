"""
Compare v1 (baseline) vs v3 (with historical driver profiles).
v3 uses year-1 telemetry profiles - NO data leakage.
Train on 2019-2022, test on 2023.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent

print("\nComparing v1 vs v3 - BayesianRidge")
print("v3 uses historical driver profiles (year-1) - no data leakage")
print("Train 2019-2022, test 2023\n")


def prepare_dataset(df_path, dataset_name, min_year=2019):
    """Load and prepare dataset."""
    print(f"Loading {dataset_name}...")
    
    df = pd.read_csv(df_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Filter to years with potential telemetry (2019+ uses 2018+ profiles)
    df = df[df['season'] >= min_year].copy()
    
    print(f"  {len(df)} records")
    return df


def process_df(df):
    """Extract features and target from dataframe."""
    y = df['grid']
    X = df.drop(columns=['grid', 'driver', 'season', 'round'], errors='ignore')
    
    # Drop qualifying_secs to prevent data leakage
    if 'qualifying_secs' in X.columns:
        X = X.drop(columns=['qualifying_secs'])
    
    # Drop non-numeric columns
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        X = X.drop(columns=object_cols)
    
    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
    
    return X, y


def calculate_accuracies(test_df):
    """Calculate pole position and top 3 accuracies."""
    correct_poles = 0
    correct_top3_any = 0
    total_sessions = 0
    
    for (season, round_num), session_df in test_df.groupby(['season', 'round']):
        if len(session_df) == 0:
            continue
        
        predicted_order = session_df.sort_values('prediction')
        actual_order = session_df.sort_values('grid')
        
        if len(predicted_order) > 0 and len(actual_order) > 0:
            if predicted_order.iloc[0]['driver'] == actual_order.iloc[0]['driver']:
                correct_poles += 1
        
        if len(predicted_order) >= 3 and len(actual_order) >= 3:
            pred_top3 = set(predicted_order.iloc[:3]['driver'].tolist())
            actual_top3 = set(actual_order.iloc[:3]['driver'].tolist())
            if pred_top3 == actual_top3:
                correct_top3_any += 1
        
        total_sessions += 1
    
    return {
        'pole_accuracy': (correct_poles / total_sessions) * 100 if total_sessions > 0 else 0,
        'top3_any_accuracy': (correct_top3_any / total_sessions) * 100 if total_sessions > 0 else 0,
        'correct_poles': correct_poles,
        'total_sessions': total_sessions
    }


def train_and_evaluate(df, dataset_name, test_year=2023):
    """Train and evaluate model."""
    train_df = df[df['season'] < test_year].copy()
    test_df = df[df['season'] == test_year].copy()
    
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    
    X_train, y_train = process_df(train_df)
    X_train = X_train.dropna(axis=1, how='all')
    X_train = X_train.fillna(X_train.mean()).fillna(0)
    
    print(f"  Features: {X_train.shape[1]}")
    
    model = BayesianRidge()
    model.fit(X_train, y_train)
    
    X_test, y_test = process_df(test_df)
    X_test = X_test[X_train.columns]
    X_test = X_test.fillna(X_train.mean()).fillna(0)
    
    test_predictions = model.predict(X_test)
    test_df['prediction'] = test_predictions
    
    r2 = r2_score(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    
    accuracies = calculate_accuracies(test_df)
    
    return {
        'r2': r2,
        'rmse': rmse,
        **accuracies,
        'num_features': X_train.shape[1],
        'train_size': len(train_df),
        'test_size': len(test_df)
    }


# Run comparison
df_v1 = prepare_dataset(PROJECT_ROOT / 'data' / 'HOLY_qualifying_v1.csv', 'v1 (baseline)')
results_v1 = train_and_evaluate(df_v1, 'v1')
print(f"v1: R²={results_v1['r2']:.4f}, RMSE={results_v1['rmse']:.2f}, Pole={results_v1['pole_accuracy']:.1f}% ({results_v1['correct_poles']}/{results_v1['total_sessions']})")

print()

df_v3 = prepare_dataset(PROJECT_ROOT / 'data' / 'HOLY_qualifying_v3.csv', 'v3 (with profiles)')
results_v3 = train_and_evaluate(df_v3, 'v3')
print(f"v3: R²={results_v3['r2']:.4f}, RMSE={results_v3['rmse']:.2f}, Pole={results_v3['pole_accuracy']:.1f}% ({results_v3['correct_poles']}/{results_v3['total_sessions']})")

# Summary
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"{'Metric':<25} {'v1 (baseline)':<15} {'v3 (profiles)':<15} {'Change':<15}")
print("-"*70)

r2_diff = results_v3['r2'] - results_v1['r2']
rmse_diff = results_v3['rmse'] - results_v1['rmse']
pole_diff = results_v3['pole_accuracy'] - results_v1['pole_accuracy']

print(f"{'R²':<25} {results_v1['r2']:<15.4f} {results_v3['r2']:<15.4f} {r2_diff:+.4f}")
print(f"{'RMSE':<25} {results_v1['rmse']:<15.2f} {results_v3['rmse']:<15.2f} {rmse_diff:+.2f}")
print(f"{'Features':<25} {results_v1['num_features']:<15} {results_v3['num_features']:<15}")
print(f"{'Pole Accuracy %':<25} {results_v1['pole_accuracy']:<15.1f} {results_v3['pole_accuracy']:<15.1f} {pole_diff:+.1f}")
print(f"{'Top 3 Accuracy %':<25} {results_v1['top3_any_accuracy']:<15.1f} {results_v3['top3_any_accuracy']:<15.1f}")

print("\nNote: v3 uses year-1 driver profiles (no data leakage)")
print("      Train: 2019-2022, Test: 2023")

