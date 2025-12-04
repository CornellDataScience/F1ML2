NOT WORKING RN



"""
Test Top LazyPredict Models for Pole Position Accuracy

Based on LazyPredict results, we'll manually train the top models
and calculate their pole position accuracy.

Tests BOTH with and without scaling!
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge, Lasso, ElasticNet, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from data_utils import prepare_train_test, load_leak_free_datasets

print("=" * 70)
print("TESTING TOP MODELS FOR POLE POSITION ACCURACY")
print("=" * 70)
print("Testing WITH and WITHOUT scaling!")

train_df, [test_2022_df, test_2023_df] = load_leak_free_datasets(
    train_year=2022,
    test_years=[2022, 2023]
)

X_train, y_train, [X_test_2022, X_test_2023], [y_test_2022, y_test_2023], _ = prepare_train_test(
    train_df,
    [test_2022_df, test_2023_df],
    scale=False
)

test_df = y_test_2022 + y_test_2023

# Define models to test (top from LazyPredict)
models_to_test = {
    'BayesianRidge': BayesianRidge(),
    'Ridge': Ridge(),
    'LGBMRegressor': lgb.LGBMRegressor(verbose=-1),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
}

print(f"\nTraining {len(models_to_test)} models...")
print(f"Training set: {len(X_train)} records")
print(f"Test set: {len(X_test_2022) + len(X_test_2023)} records\n")

# Test WITHOUT scaling
print("=" * 70)
print("TEST 1: WITHOUT SCALING")
print("=" * 70)

results_no_scaling = []

for model_name, model in models_to_test.items():
    print(f"Training {model_name} (no scaling)...")

    # Train without scaling
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Calculate pole accuracy
    test_df = df[~train_mask].copy().reset_index(drop=True)
    test_df['prediction'] = predictions

    correct_poles = 0
    total_sessions = 0

    for (season, round_num), session_df in test_df.groupby(['season', 'round']):
        if len(session_df) == 0:
            continue

        predicted_order = session_df.sort_values('prediction')
        actual_order = session_df.sort_values('grid')

        if len(predicted_order) > 0 and len(actual_order) > 0:
            if predicted_order.iloc[0]['driver'] == actual_order.iloc[0]['driver']:
                correct_poles += 1

        total_sessions += 1

    pole_accuracy = (correct_poles / total_sessions) * 100

    results_no_scaling.append({
        'Model': model_name,
        'Pole_Accuracy_No_Scaling': pole_accuracy,
        'Correct_Poles': correct_poles,
        'Total_Sessions': total_sessions
    })

    print(f"  Pole Accuracy: {pole_accuracy:.2f}% ({correct_poles}/{total_sessions})")

# Test WITH scaling
print(f"\n{'='*70}")
print("TEST 2: WITH SCALING (StandardScaler)")
print("=" * 70)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results_with_scaling = []

for model_name, model_class in models_to_test.items():
    print(f"Training {model_name} (with scaling)...")

    # Create fresh model instance
    if model_name == 'BayesianRidge':
        model = BayesianRidge()
    elif model_name == 'Ridge':
        model = Ridge()
    elif model_name == 'LGBMRegressor':
        model = lgb.LGBMRegressor(verbose=-1)
    elif model_name == 'GradientBoostingRegressor':
        model = GradientBoostingRegressor()
    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'Lasso':
        model = Lasso()
    elif model_name == 'ElasticNet':
        model = ElasticNet()

    # Train with scaling
    model.fit(X_train_scaled, y_train)

    # Predict
    predictions = model.predict(X_test_scaled)

    # Calculate pole accuracy
    test_df = df[~train_mask].copy().reset_index(drop=True)
    test_df['prediction'] = predictions

    correct_poles = 0
    total_sessions = 0

    for (season, round_num), session_df in test_df.groupby(['season', 'round']):
        if len(session_df) == 0:
            continue

        predicted_order = session_df.sort_values('prediction')
        actual_order = session_df.sort_values('grid')

        if len(predicted_order) > 0 and len(actual_order) > 0:
            if predicted_order.iloc[0]['driver'] == actual_order.iloc[0]['driver']:
                correct_poles += 1

        total_sessions += 1

    pole_accuracy = (correct_poles / total_sessions) * 100

    results_with_scaling.append({
        'Model': model_name,
        'Pole_Accuracy_With_Scaling': pole_accuracy,
        'Correct_Poles': correct_poles,
        'Total_Sessions': total_sessions
    })

    print(f"  Pole Accuracy: {pole_accuracy:.2f}% ({correct_poles}/{total_sessions})")

# Combine results
df_no_scaling = pd.DataFrame(results_no_scaling)
df_with_scaling = pd.DataFrame(results_with_scaling)

results_df = df_no_scaling.merge(
    df_with_scaling[['Model', 'Pole_Accuracy_With_Scaling']],
    on='Model'
)
results_df['Difference'] = results_df['Pole_Accuracy_No_Scaling'] - results_df['Pole_Accuracy_With_Scaling']
results_df = results_df.sort_values('Pole_Accuracy_No_Scaling', ascending=False)

print(f"\n{'='*70}")
print("RESULTS: WITH vs WITHOUT SCALING")
print(f"{'='*70}\n")
print(results_df[['Model', 'Pole_Accuracy_No_Scaling', 'Pole_Accuracy_With_Scaling', 'Difference']].to_string(index=False))

# Save results
results_df.to_csv('top_models_pole_accuracy_comparison.csv', index=False)
print(f"\nResults saved to: top_models_pole_accuracy_comparison.csv")

# Analyze which models prefer scaling
print(f"\n{'='*70}")
print("SCALING ANALYSIS")
print(f"{'='*70}")

better_without = results_df[results_df['Difference'] > 0].sort_values('Difference', ascending=False)
better_with = results_df[results_df['Difference'] < 0].sort_values('Difference')

print(f"\nModels that perform BETTER WITHOUT scaling:")
for _, row in better_without.iterrows():
    print(f"  {row['Model']:30s} {row['Difference']:+.2f}% ({row['Pole_Accuracy_No_Scaling']:.2f}% vs {row['Pole_Accuracy_With_Scaling']:.2f}%)")

print(f"\nModels that perform BETTER WITH scaling:")
for _, row in better_with.iterrows():
    print(f"  {row['Model']:30s} {row['Difference']:+.2f}% ({row['Pole_Accuracy_No_Scaling']:.2f}% vs {row['Pole_Accuracy_With_Scaling']:.2f}%)")

print(f"\n{'='*70}")
print("TOP PERFORMERS")
print(f"{'='*70}")
print(f"Best without scaling: {results_df.iloc[0]['Model']} - {results_df.iloc[0]['Pole_Accuracy_No_Scaling']:.2f}%")
best_with = results_df.loc[results_df['Pole_Accuracy_With_Scaling'].idxmax()]
print(f"Best with scaling:    {best_with['Model']} - {best_with['Pole_Accuracy_With_Scaling']:.2f}%")

print(f"\n{'='*70}")
print("COMPARISON WITH OTHER MODELS")
print(f"{'='*70}")
print(f"BayesianRidge (no scaling):         {results_df.iloc[0]['Pole_Accuracy_No_Scaling']:.2f}%")
print(f"{'='*70}")
