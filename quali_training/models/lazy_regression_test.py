"""
LazyPredict Test - Compare Multiple Regression Models for Qualifying Prediction

This script tests various regression models to predict qualifying position (grid).
Unlike XGBoost Ranker which uses pairwise ranking, these models predict 
the absolute position number directly.

Regression treats grid position as a continuous number to predict.
"""

from lazypredict.Supervised import LazyRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LAZYPREDICT REGRESSION TEST - QUALIFYING PREDICTION")
print("="*70)

# Load qualifying dataset
print("\nLoading qualifying dataset...")
df = pd.read_csv('../data/HOLY_qualifying_v1.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"Dataset loaded: {len(df)} records")
print(f"Date range: {df['season'].min()}-{df['season'].max()}")

# Target variable
y = df['grid']  # Qualifying position (1, 2, 3, etc.)

# Drop columns not needed for prediction
X = df.drop(columns=['grid', 'driver', 'season', 'round'], errors='ignore')

# Drop qualifying_secs (same as the ranker model)
if 'qualifying_secs' in X.columns:
    print("\nDropping 'qualifying_secs' column (experimental)")
    X = X.drop(columns=['qualifying_secs'])

# Drop non-numeric columns
object_cols = X.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    print(f"Dropping {len(object_cols)} non-numeric columns: {list(object_cols)}")
    X = X.drop(columns=object_cols)

# Convert boolean to int
bool_cols = X.select_dtypes(include=['bool']).columns
if len(bool_cols) > 0:
    print(f"Converting {len(bool_cols)} boolean columns to int")
    X[bool_cols] = X[bool_cols].astype(int)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data: 80% train, 20% test
print("\nSplitting data: 80% train, 20% test")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Scale features (helps some models perform better)
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame (LazyPredict needs column names)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Run LazyPredict
print("\n" + "="*70)
print("RUNNING LAZYPREDICT - TESTING MULTIPLE REGRESSION MODELS")
print("="*70)
print("\nThis may take a few minutes...\n")

# Initialize LazyRegressor
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

# Fit all models
models, predictions = reg.fit(X_train_scaled, X_test_scaled, y_train, y_test)

# Display results
print("\n" + "="*70)
print("MODEL COMPARISON RESULTS")
print("="*70)
print("\nKey Metrics:")
print("  R² Score: Higher is better (1.0 = perfect, <0 = worse than baseline)")
print("  RMSE: Lower is better (average error in positions)")
print("  MAE: Lower is better (mean absolute error in positions)")
print("  Time: Training time in seconds")
print("\n")

# Sort by R² score (descending)
models_sorted = models.sort_values(by='R-Squared', ascending=False)
print(models_sorted)

print("\n" + "="*70)
print("TOP 3 MODELS")
print("="*70)
top_3 = models_sorted.head(3)
for idx, (model_name, row) in enumerate(top_3.iterrows(), 1):
    print(f"\n{idx}. {model_name}")
    print(f"   R² Score: {row['R-Squared']:.4f}")
    print(f"   RMSE: {row['RMSE']:.4f} positions")
    print(f"   MAE: {row['MAE']:.4f} positions")
    print(f"   Training Time: {row['Time Taken']:.2f}s")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
print("\nNote: XGBoost Ranker (current model) uses a different approach:")
print("  - Ranker: Learns relative ordering between drivers")
print("  - Regression: Predicts absolute position numbers")
print("\nRegression models to consider for experimentation:")
print("  1. Random Forest - Good baseline, handles non-linear relationships")
print("  2. XGBoost Regressor - Often performs well on structured data")
print("  3. Ridge/Lasso - Simple linear models with regularization")
print("\nYou can now pick the best performing models from above and")
print("create dedicated training scripts for them (like trainmodel_quali.py).")

