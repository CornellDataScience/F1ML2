"""
Test Models on 2022 and 2023 Using Data Utils

Clean version using the data_utils module for preprocessing.
"""

import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_absolute_error
from data_utils import load_leak_free_datasets, prepare_train_test, calculate_pole_accuracy

print("=" * 70)
print("TESTING MODELS ON 2022 AND 2023 (Using Data Utils)")
print("=" * 70)

# Load leak-free datasets
print("\nLoading leak-free datasets...")
train_df, [test_2022_df, test_2023_df] = load_leak_free_datasets(
    train_year=2022,
    test_years=[2022, 2023]
)

print(f"Training: {len(train_df)} records ({train_df['season'].min()}-{train_df['season'].max()})")
print(f"Test 2022: {len(test_2022_df)} records ({test_2022_df.groupby(['season', 'round']).ngroups} sessions)")
print(f"Test 2023: {len(test_2023_df)} records ({test_2023_df.groupby(['season', 'round']).ngroups} sessions)")

# ============================================================================
# MODEL 1: BayesianRidge (NO SCALING)
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 1: BayesianRidge (NO SCALING)")
print("=" * 70)

# Prepare data without scaling
X_train, y_train, [X_test_2022, X_test_2023], [y_test_2022, y_test_2023], _ = prepare_train_test(
    train_df,
    [test_2022_df, test_2023_df],
    scale=False
)

print(f"\nTraining on {len(X_train)} samples with {X_train.shape[1]} features...")
model_br = BayesianRidge()
model_br.fit(X_train, y_train)

# Test on 2022
print("\n--- Testing on 2022 ---")
preds_br_2022 = model_br.predict(X_test_2022)
acc_br_2022, correct_br_2022, total_br_2022, sessions_br_2022 = calculate_pole_accuracy(
    test_2022_df, preds_br_2022, return_details=True
)
mae_br_2022 = mean_absolute_error(y_test_2022, preds_br_2022)
print(f"Pole Accuracy: {acc_br_2022:.2f}% ({correct_br_2022}/{total_br_2022})")
print(f"MAE: {mae_br_2022:.3f} grid positions")

# Test on 2023
print("\n--- Testing on 2023 ---")
preds_br_2023 = model_br.predict(X_test_2023)
acc_br_2023, correct_br_2023, total_br_2023, sessions_br_2023 = calculate_pole_accuracy(
    test_2023_df, preds_br_2023, return_details=True
)
mae_br_2023 = mean_absolute_error(y_test_2023, preds_br_2023)
print(f"Pole Accuracy: {acc_br_2023:.2f}% ({correct_br_2023}/{total_br_2023})")
print(f"MAE: {mae_br_2023:.3f} grid positions")

# ============================================================================
# MODEL 2: LinearRegression (WITH SCALING)
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 2: LinearRegression (WITH SCALING)")
print("=" * 70)

# Prepare data with scaling
X_train_scaled, y_train, [X_test_2022_scaled, X_test_2023_scaled], [y_test_2022, y_test_2023], scaler = prepare_train_test(
    train_df,
    [test_2022_df, test_2023_df],
    scale=True
)

print(f"\nTraining on {len(X_train_scaled)} samples with {X_train_scaled.shape[1]} features...")
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)

# Test on 2022
print("\n--- Testing on 2022 ---")
preds_lr_2022 = model_lr.predict(X_test_2022_scaled)
acc_lr_2022, correct_lr_2022, total_lr_2022, sessions_lr_2022 = calculate_pole_accuracy(
    test_2022_df, preds_lr_2022, return_details=True
)
mae_lr_2022 = mean_absolute_error(y_test_2022, preds_lr_2022)
print(f"Pole Accuracy: {acc_lr_2022:.2f}% ({correct_lr_2022}/{total_lr_2022})")
print(f"MAE: {mae_lr_2022:.3f} grid positions")

# Test on 2023
print("\n--- Testing on 2023 ---")
preds_lr_2023 = model_lr.predict(X_test_2023_scaled)
acc_lr_2023, correct_lr_2023, total_lr_2023, sessions_lr_2023 = calculate_pole_accuracy(
    test_2023_df, preds_lr_2023, return_details=True
)
mae_lr_2023 = mean_absolute_error(y_test_2023, preds_lr_2023)
print(f"Pole Accuracy: {acc_lr_2023:.2f}% ({correct_lr_2023}/{total_lr_2023})")
print(f"MAE: {mae_lr_2023:.3f} grid positions")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

results = pd.DataFrame({
    'Model': ['BayesianRidge (no scaling)', 'LinearRegression (with scaling)'],
    '2022': [f"{acc_br_2022:.2f}% ({correct_br_2022}/{total_br_2022})",
             f"{acc_lr_2022:.2f}% ({correct_lr_2022}/{total_lr_2022})"],
    '2023': [f"{acc_br_2023:.2f}% ({correct_br_2023}/{total_br_2023})",
             f"{acc_lr_2023:.2f}% ({correct_lr_2023}/{total_lr_2023})"],
    'Average': [f"{(acc_br_2022+acc_br_2023)/2:.2f}%",
                f"{(acc_lr_2022+acc_lr_2023)/2:.2f}%"],
    'Consistency': [f"±{abs(acc_br_2023-acc_br_2022):.2f}%",
                    f"±{abs(acc_lr_2023-acc_lr_2022):.2f}%"]
})

print("\n" + results.to_string(index=False))

# Winner
overall_br = (acc_br_2022 + acc_br_2023) / 2
overall_lr = (acc_lr_2022 + acc_lr_2023) / 2
overall_mae_br = (mae_br_2022 + mae_br_2023) / 2
overall_mae_lr = (mae_lr_2022 + mae_lr_2023) / 2

print("\n" + "=" * 70)
print("WINNER")
print("=" * 70)

if overall_br > overall_lr:
    print(f"🏆 BayesianRidge wins: {overall_br:.2f}% vs {overall_lr:.2f}%")
    print(f"   More consistent: ±{abs(acc_br_2023-acc_br_2022):.2f}%")
elif overall_lr > overall_br:
    print(f"🏆 LinearRegression wins: {overall_lr:.2f}% vs {overall_br:.2f}%")
    print(f"   Better adaptation to dominance eras")
else:
    print(f"🤝 TIE at {overall_br:.2f}%")

print("\n" + "=" * 70)
print("MEAN ABSOLUTE ERROR (MAE) SUMMARY")
print("=" * 70)

print(f"\nBayesianRidge:")
print(f"  2022 MAE: {mae_br_2022:.3f} grid positions")
print(f"  2023 MAE: {mae_br_2023:.3f} grid positions")
print(f"  Average: {overall_mae_br:.3f} grid positions")

print(f"\nLinearRegression:")
print(f"  2022 MAE: {mae_lr_2022:.3f} grid positions")
print(f"  2023 MAE: {mae_lr_2023:.3f} grid positions")
print(f"  Average: {overall_mae_lr:.3f} grid positions")

print(f"\nOverall average MAE: {(overall_mae_br + overall_mae_lr)/2:.3f} grid positions")

print("\n✅ No data leakage - using leak-free datasets!")
print("=" * 70)
