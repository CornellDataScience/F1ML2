"""
Compare v1 vs v3 using ALL years (1983-2022 train, 2023 test).
Uses Option 2: Fill telemetry with 0 + add has_telemetry flag.
This lets the model learn when to use telemetry features.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent

print("\nComparing v1 vs v3 - ALL YEARS")
print("Option 2: Fill telemetry with 0 + has_telemetry flag")
print("Testing on 2021, 2022, 2023\n")

TEST_YEARS = [2021, 2022, 2023]

# Telemetry columns that only exist for 2019+
TELEMETRY_COLS = [
    'driver_avg_corner_speed', 'driver_min_corner_speed', 'driver_corner_throttle',
    'driver_corner_brake', 'driver_corner_rpm', 'driver_avg_straight_speed',
    'driver_top_speed', 'driver_straight_throttle', 'driver_drs_usage',
    'circuits_sampled'
]


def prepare_dataset_v1(df_path):
    """Load v1 baseline dataset."""
    print("Loading v1 (baseline)...")
    df = pd.read_csv(df_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    print(f"  {len(df)} records")
    return df


def prepare_dataset_v3_option2(df_path):
    """Load v3 and apply Option 2: fill telemetry with 0 + add flag."""
    print("Loading v3 (with Option 2 processing)...")
    df = pd.read_csv(df_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Add has_telemetry flag (1 if any telemetry column has data)
    telemetry_exists = df[TELEMETRY_COLS].notna().any(axis=1)
    df['has_telemetry'] = telemetry_exists.astype(int)
    
    # Fill telemetry NaN with 0
    df[TELEMETRY_COLS] = df[TELEMETRY_COLS].fillna(0)
    
    rows_with_telemetry = df['has_telemetry'].sum()
    print(f"  {len(df)} records")
    print(f"  {rows_with_telemetry} rows with telemetry ({100*rows_with_telemetry/len(df):.1f}%)")
    
    return df


def process_df(df):
    """Extract features and target."""
    y = df['grid']
    X = df.drop(columns=['grid', 'driver', 'season', 'round'], errors='ignore')
    
    if 'qualifying_secs' in X.columns:
        X = X.drop(columns=['qualifying_secs'])
    
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        X = X.drop(columns=object_cols)
    
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
    
    return X, y


def calculate_accuracies(test_df):
    """Calculate pole and top 3 accuracies."""
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
    """Train on all years before test_year, evaluate on test_year."""
    train_df = df[df['season'] < test_year].copy()
    test_df = df[df['season'] == test_year].copy()
    
    print(f"  Train: {len(train_df)} ({train_df['season'].min()}-{train_df['season'].max()})")
    print(f"  Test: {len(test_df)} ({test_year})")
    
    X_train, y_train = process_df(train_df)
    X_train = X_train.dropna(axis=1, how='all')
    X_train = X_train.fillna(X_train.mean()).fillna(0)
    
    print(f"  Features: {X_train.shape[1]}")
    
    model = BayesianRidge()
    model.fit(X_train, y_train)
    
    X_test, y_test = process_df(test_df)
    X_test = X_test[X_train.columns]
    X_test = X_test.fillna(X_train.mean()).fillna(0)
    
    predictions = model.predict(X_test)
    test_df['prediction'] = predictions
    
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    accuracies = calculate_accuracies(test_df)
    
    return {
        'r2': r2,
        'rmse': rmse,
        **accuracies,
        'num_features': X_train.shape[1],
        'train_size': len(train_df),
        'test_size': len(test_df)
    }


# Load datasets once
df_v1 = prepare_dataset_v1(PROJECT_ROOT / 'data' / 'HOLY_qualifying_v1.csv')
print()
df_v3 = prepare_dataset_v3_option2(PROJECT_ROOT / 'data' / 'HOLY_qualifying_v3.csv')

# Store results for each year
all_results_v1 = []
all_results_v3 = []

print("\n" + "="*70)
print("RESULTS BY TEST YEAR")
print("="*70)

for test_year in TEST_YEARS:
    print(f"\n--- Test Year: {test_year} ---")
    
    results_v1 = train_and_evaluate(df_v1, 'v1', test_year=test_year)
    print(f"v1: R²={results_v1['r2']:.4f}, RMSE={results_v1['rmse']:.2f}, Pole={results_v1['pole_accuracy']:.1f}% ({results_v1['correct_poles']}/{results_v1['total_sessions']})")
    all_results_v1.append(results_v1)
    
    results_v3 = train_and_evaluate(df_v3, 'v3', test_year=test_year)
    print(f"v3: R²={results_v3['r2']:.4f}, RMSE={results_v3['rmse']:.2f}, Pole={results_v3['pole_accuracy']:.1f}% ({results_v3['correct_poles']}/{results_v3['total_sessions']})")
    all_results_v3.append(results_v3)

# Aggregate results
total_poles_v1 = sum(r['correct_poles'] for r in all_results_v1)
total_poles_v3 = sum(r['correct_poles'] for r in all_results_v3)
total_sessions = sum(r['total_sessions'] for r in all_results_v1)
avg_r2_v1 = np.mean([r['r2'] for r in all_results_v1])
avg_r2_v3 = np.mean([r['r2'] for r in all_results_v3])
avg_rmse_v1 = np.mean([r['rmse'] for r in all_results_v1])
avg_rmse_v3 = np.mean([r['rmse'] for r in all_results_v3])

# Summary
print("\n" + "="*70)
print(f"AGGREGATE RESULTS ({TEST_YEARS[0]}-{TEST_YEARS[-1]})")
print("="*70)
print(f"{'Metric':<25} {'v1 (baseline)':<15} {'v3 (Option 2)':<15} {'Change':<15}")
print("-"*70)

print(f"{'Avg R²':<25} {avg_r2_v1:<15.4f} {avg_r2_v3:<15.4f} {avg_r2_v3-avg_r2_v1:+.4f}")
print(f"{'Avg RMSE':<25} {avg_rmse_v1:<15.2f} {avg_rmse_v3:<15.2f} {avg_rmse_v3-avg_rmse_v1:+.2f}")
print(f"{'Total Poles Correct':<25} {total_poles_v1:<15} {total_poles_v3:<15} {total_poles_v3-total_poles_v1:+d}")
print(f"{'Pole Accuracy %':<25} {100*total_poles_v1/total_sessions:<15.1f} {100*total_poles_v3/total_sessions:<15.1f} {100*(total_poles_v3-total_poles_v1)/total_sessions:+.1f}")
print(f"{'Total Sessions':<25} {total_sessions:<15}")

print("\nOption 2: Telemetry filled with 0 + has_telemetry flag")
print("Model learns to use telemetry when available, ignore when not")

