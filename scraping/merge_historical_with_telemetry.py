"""
Merge Historical Dataset with Telemetry Features

This script combines:
- Historical data (1983-2023) for long-term patterns
- Telemetry features (2018-2023) for modern races

Result: Full historical dataset with telemetry features where available
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'test' / 'data'

print("=" * 80)
print("MERGING HISTORICAL DATA WITH TELEMETRY FEATURES")
print("=" * 80)

# Load datasets
print("\nLoading datasets...")
historical = pd.read_csv(DATA_DIR / 'HOLY_qualifying_full_to_2023.csv')
telemetry = pd.read_csv(DATA_DIR / 'HOLY_qualifying_modern_2018_2023.csv')

print(f"Historical (1983-2023): {historical.shape}")
print(f"Telemetry (2018-2023): {telemetry.shape}")

# Get telemetry columns
telemetry_cols = [col for col in telemetry.columns if 'practice_' in col]
print(f"\nTelemetry features to add: {len(telemetry_cols)}")
for col in telemetry_cols:
    print(f"  - {col}")

# Add telemetry columns to historical dataset (initially NaN)
print("\nAdding telemetry columns to historical dataset...")
for col in telemetry_cols:
    historical[col] = np.nan

print(f"Historical dataset now: {historical.shape}")

# Merge telemetry data for 2018-2023
print("\nMerging telemetry data for 2018-2023...")

# Create merge keys
historical['merge_key'] = (
    historical['season'].astype(str) + '_' +
    historical['round'].astype(str) + '_' +
    historical['driver'].astype(str)
)

telemetry['merge_key'] = (
    telemetry['season'].astype(str) + '_' +
    telemetry['round'].astype(str) + '_' +
    telemetry['driver'].astype(str)
)

# Update telemetry values for matching records
matches = 0
for idx, row in historical.iterrows():
    merge_key = row['merge_key']
    telemetry_row = telemetry[telemetry['merge_key'] == merge_key]

    if not telemetry_row.empty:
        # Copy telemetry features
        for col in telemetry_cols:
            historical.at[idx, col] = telemetry_row.iloc[0][col]
        matches += 1

print(f"  Matched {matches} records with telemetry data")

# Drop merge key
historical = historical.drop(columns=['merge_key'])

# Check results
print("\nChecking results...")
records_with_telemetry = historical[historical['practice_overall_avg_speed'].notna()]
print(f"Records with telemetry: {len(records_with_telemetry)}")
print(f"  Years: {records_with_telemetry['season'].min()} - {records_with_telemetry['season'].max()}")

records_without_telemetry = historical[historical['practice_overall_avg_speed'].isna()]
print(f"Records without telemetry: {len(records_without_telemetry)}")
print(f"  Years: {records_without_telemetry['season'].min()} - {records_without_telemetry['season'].max()}")

# Save merged dataset
output_file = DATA_DIR / 'HOLY_qualifying_hybrid_1983_2023.csv'
print(f"\nSaving to: {output_file}")
historical.to_csv(output_file, index=False)

# Create split datasets
print("\n" + "=" * 80)
print("CREATING SPLIT DATASETS")
print("=" * 80)

# train2023 (up to 2022)
train2023 = historical[historical['season'] < 2023].copy()
output_train2023 = DATA_DIR / 'HOLY_qualifying_hybrid_train2023.csv'
train2023.to_csv(output_train2023, index=False)
print(f"\ntrain2023: {train2023.shape}")
print(f"  Saved to: {output_train2023.name}")

# Check telemetry coverage in train2023
train2023_with_telem = train2023[train2023['practice_overall_avg_speed'].notna()]
print(f"  Records with telemetry: {len(train2023_with_telem)} ({100*len(train2023_with_telem)/len(train2023):.1f}%)")

print("\n" + "=" * 80)
print("✓ DONE!")
print("=" * 80)

print("\nSummary:")
print(f"  Full dataset: {historical.shape}")
print(f"  - Historical (no telemetry): {len(records_without_telemetry)} records")
print(f"  - Modern (with telemetry): {len(records_with_telemetry)} records")
print(f"  Telemetry features: {len(telemetry_cols)}")
print(f"\nOutputs:")
print(f"  - {output_file.name}")
print(f"  - {output_train2023.name}")
print("=" * 80)
