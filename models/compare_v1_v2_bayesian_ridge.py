"""
Compare v1 vs v2 for grid positions (R², RMSE), pole accuracy, and top 3 predictions.
Uses recent data only (train 2018-2022, test 2023), same circuits for fair comparison.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("\nComparing v1 vs v2 - BayesianRidge (no scaling)")
print("Train 2018-2022, test 2023\n")

def prepare_dataset(df_path, dataset_name, filter_telemetry=False, filter_to_circuits=None):
    """Load and filter dataset."""
    print(f"Loading {dataset_name}...")
    
    df = pd.read_csv(df_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    df = df[df['season'] >= 2018].copy()
    
    if filter_telemetry:
        telemetry_cols = ['driver_avg_corner_speed', 'driver_top_speed', 'track_total_corners']
        df = df.dropna(subset=telemetry_cols)
    
    if filter_to_circuits is not None:
        df = df[df.set_index(['season', 'round', 'circuit_id']).index.isin(filter_to_circuits)]
    
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
    """
    Calculate pole position and top 3 accuracies from predictions.
    """
    correct_poles = 0
    correct_top3 = 0
    correct_top3_any_order = 0
    total_sessions = 0
    
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
    
    return {
        'pole_accuracy': pole_accuracy,
        'top3_order_accuracy': top3_order_accuracy,
        'top3_any_accuracy': top3_any_accuracy,
        'correct_poles': correct_poles,
        'correct_top3': correct_top3,
        'correct_top3_any': correct_top3_any_order,
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
    
    # Use same columns as training
    X_test = X_test[X_train.columns]
    X_test = X_test.fillna(X_train.mean()).fillna(0)
    
    # Make predictions
    test_predictions = model.predict(X_test)
    test_df['prediction'] = test_predictions
    
    # Calculate R² and RMSE
    r2 = r2_score(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    
    # Calculate pole position and top 3 accuracies
    accuracies = calculate_accuracies(test_df)
    
    # Store results
    results = {
        'r2': r2,
        'rmse': rmse,
        **accuracies,
        'num_features': X_train.shape[1],
        'train_size': len(train_df),
        'test_size': len(test_df)
    }
    
    return results, model


# Get circuits with telemetry for fair comparison
print("Identifying circuits with telemetry...")
df_v2_full = pd.read_csv('../data/HOLY_qualifying_v2.csv')
if 'Unnamed: 0' in df_v2_full.columns:
    df_v2_full = df_v2_full.drop(columns=['Unnamed: 0'])

df_v2_full = df_v2_full[df_v2_full['season'] >= 2018].copy()
telemetry_cols = ['driver_avg_corner_speed', 'driver_top_speed', 'track_total_corners']
df_v2_filtered = df_v2_full.dropna(subset=telemetry_cols)

circuits_with_telemetry = set(df_v2_filtered.set_index(['season', 'round', 'circuit_id']).index)
unique_circuits = df_v2_filtered['circuit_id'].nunique()
print(f"  {unique_circuits} circuits with telemetry\n")

df_v1 = prepare_dataset(
    df_path='../data/HOLY_qualifying_v1.csv',
    dataset_name='v1 (baseline)',
    filter_telemetry=False,
    filter_to_circuits=circuits_with_telemetry
)
results_v1, model_v1 = train_and_evaluate(df_v1, 'v1')

print(f"v1 results:")
print(f"  R²: {results_v1['r2']:.4f}, RMSE: {results_v1['rmse']:.4f}")
print(f"  Pole: {results_v1['pole_accuracy']:.1f}% ({results_v1['correct_poles']}/{results_v1['total_sessions']})")
print(f"  Top 3: {results_v1['top3_any_accuracy']:.1f}%")

print()
df_v2 = prepare_dataset(
    df_path='../data/HOLY_qualifying_v2.csv',
    dataset_name='v2 (with telemetry)',
    filter_telemetry=True,
    filter_to_circuits=None
)
results_v2, model_v2 = train_and_evaluate(df_v2, 'v2')

print(f"v2 results:")
print(f"  R²: {results_v2['r2']:.4f}, RMSE: {results_v2['rmse']:.4f}")
print(f"  Pole: {results_v2['pole_accuracy']:.1f}% ({results_v2['correct_poles']}/{results_v2['total_sessions']})")
print(f"  Top 3: {results_v2['top3_any_accuracy']:.1f}%")


# Summary
print("\nComparison:")
print(f"{'Metric':<25} {'v1':<15} {'v2':<15} {'Change':<15}")
print("-" * 70)

r2_diff = results_v2['r2'] - results_v1['r2']
rmse_diff = results_v2['rmse'] - results_v1['rmse']
pole_diff = results_v2['pole_accuracy'] - results_v1['pole_accuracy']

print(f"{'R²':<25} {results_v1['r2']:<15.4f} {results_v2['r2']:<15.4f} {r2_diff:+.4f}")
print(f"{'RMSE':<25} {results_v1['rmse']:<15.4f} {results_v2['rmse']:<15.4f} {rmse_diff:+.4f}")
print(f"{'Features':<25} {results_v1['num_features']:<15} {results_v2['num_features']:<15}")
print(f"{'Pole accuracy':<25} {results_v1['pole_accuracy']:<15.1f} {results_v2['pole_accuracy']:<15.1f} {pole_diff:+.1f}")
print(f"{'Top 3 (any)':<25} {results_v1['top3_any_accuracy']:<15.1f} {results_v2['top3_any_accuracy']:<15.1f}")

print(f"\nTested on same {unique_circuits} circuits (2018-2022 train, 2023 test)")
print("Done.")

