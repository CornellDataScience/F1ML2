"""
Test Models on Multiple Years

Tests BayesianRidge and LinearRegression on multiple test years.
Train on: all years before test year
Test on: specified years (default: 2021, 2022, 2023)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.preprocessing import StandardScaler

TEST_YEARS = [2021, 2022, 2023]

print("=" * 70)
print("TESTING MODELS ON MULTIPLE YEARS")
print(f"Test years: {TEST_YEARS}")
print("=" * 70)

# Load v1 dataset
print("\nLoading HOLY_qualifying_v3.csv...")
df = pd.read_csv('../data/HOLY_qualifying_v3.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"Dataset: {len(df)} records ({df['season'].min()}-{df['season'].max()})")

# Check telemetry coverage
if 'has_telemetry' in df.columns:
    telem_count = df['has_telemetry'].sum()
    print(f"Telemetry coverage: {telem_count}/{len(df)} rows ({100*telem_count/len(df):.1f}%)")
    print("  (Pre-2019 rows have has_telemetry=0, telemetry filled with 0)")


def process_df(df, verbose=True):
    """Process dataframe to prepare features"""
    y = df.loc[:, 'grid']
    X = df.drop(columns=['grid', 'driver', 'season', 'round'], errors='ignore')

    if 'qualifying_secs' in X.columns:
        if verbose:
            print("  Dropping 'qualifying_secs'")
        X = X.drop(columns=['qualifying_secs'])

    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        X = X.drop(columns=object_cols)

    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    return X, y


def calculate_pole_accuracy(test_df, predictions):
    """Calculate pole position accuracy"""
    test_df = test_df.copy()
    test_df['prediction'] = predictions

    correct_poles = 0
    total_sessions = 0

    for (season, round_num), session_df in test_df.groupby(['season', 'round']):
        if len(session_df) == 0:
            continue

        predicted_order = session_df.sort_values('prediction')
        actual_order = session_df.sort_values('grid')

        if len(predicted_order) > 0 and len(actual_order) > 0:
            predicted_pole = predicted_order.iloc[0]['driver']
            actual_pole = actual_order.iloc[0]['driver']

            if predicted_pole == actual_pole:
                correct_poles += 1

        total_sessions += 1

    pole_accuracy = (correct_poles / total_sessions) * 100 if total_sessions > 0 else 0
    return pole_accuracy, correct_poles, total_sessions


def test_year(df, test_year):
    """Test models on a single year"""
    train_df = df[df['season'] < test_year].copy()
    test_df = df[df['season'] == test_year].copy()

    if len(test_df) == 0:
        return None

    X_train, y_train = process_df(train_df, verbose=False)
    X_test, y_test = process_df(test_df, verbose=False)

    # Align columns
    all_cols = set(X_train.columns) | set(X_test.columns)
    for col in all_cols - set(X_train.columns):
        X_train[col] = 0
    for col in all_cols - set(X_test.columns):
        X_test[col] = 0

    X_train = X_train[sorted(all_cols)]
    X_test = X_test[sorted(all_cols)]

    # Fill missing values
    X_train = X_train.fillna(X_train.mean())
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(X_train.mean())
    X_test = X_test.fillna(0)

    # BayesianRidge (no scaling)
    model_br = BayesianRidge()
    model_br.fit(X_train, y_train)
    pred_br = model_br.predict(X_test)
    acc_br, correct_br, total_br = calculate_pole_accuracy(test_df, pred_br)

    # LinearRegression (with scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train)
    pred_lr = model_lr.predict(X_test_scaled)
    acc_lr, correct_lr, total_lr = calculate_pole_accuracy(test_df, pred_lr)

    return {
        'year': test_year,
        'sessions': total_br,
        'br_acc': acc_br,
        'br_correct': correct_br,
        'lr_acc': acc_lr,
        'lr_correct': correct_lr
    }


# Run tests for each year
results = []
total_br_correct = 0
total_lr_correct = 0
total_sessions = 0

for year in TEST_YEARS:
    print(f"\nTesting {year}...")
    result = test_year(df, year)
    if result:
        results.append(result)
        total_br_correct += result['br_correct']
        total_lr_correct += result['lr_correct']
        total_sessions += result['sessions']
        print(f"  BayesianRidge: {result['br_acc']:.1f}% ({result['br_correct']}/{result['sessions']})")
        print(f"  LinearRegression: {result['lr_acc']:.1f}% ({result['lr_correct']}/{result['sessions']})")

# Summary
print("\n" + "=" * 70)
print("RESULTS BY YEAR")
print("=" * 70)

print(f"\n{'Year':<8} {'Sessions':<10} {'BayesianRidge':<20} {'LinearRegression':<20}")
print("-" * 58)
for r in results:
    br_str = f"{r['br_acc']:.1f}% ({r['br_correct']}/{r['sessions']})"
    lr_str = f"{r['lr_acc']:.1f}% ({r['lr_correct']}/{r['sessions']})"
    print(f"{r['year']:<8} {r['sessions']:<10} {br_str:<20} {lr_str:<20}")

# Aggregate
print("\n" + "=" * 70)
print("AGGREGATE RESULTS")
print("=" * 70)

agg_br_acc = (total_br_correct / total_sessions) * 100 if total_sessions > 0 else 0
agg_lr_acc = (total_lr_correct / total_sessions) * 100 if total_sessions > 0 else 0

print(f"\nTotal sessions tested: {total_sessions}")
print(f"\nBayesianRidge (no scaling):")
print(f"  Pole accuracy: {agg_br_acc:.1f}% ({total_br_correct}/{total_sessions})")

print(f"\nLinearRegression (with scaling):")
print(f"  Pole accuracy: {agg_lr_acc:.1f}% ({total_lr_correct}/{total_sessions})")

print("\n" + "=" * 70)
if agg_br_acc > agg_lr_acc:
    print(f"Winner: BayesianRidge (+{agg_br_acc - agg_lr_acc:.1f}%)")
elif agg_lr_acc > agg_br_acc:
    print(f"Winner: LinearRegression (+{agg_lr_acc - agg_br_acc:.1f}%)")
else:
    print("Tie")
print("=" * 70)
