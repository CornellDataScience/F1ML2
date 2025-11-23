"""
Merge track and telemetry features into main qualifying dataset.
Output: HOLY_qualifying_v2.csv
"""

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

def merge_enhanced_features(
    base_dataset_path: str = None,
    track_features_path: str = None,
    telemetry_features_path: str = None,
    output_path: str = None
):
    """Merge track and telemetry features into base qualifying dataset."""
    
    # Default paths
    if base_dataset_path is None:
        base_dataset_path = PROJECT_ROOT / 'data' / 'HOLY_qualifying_v1.csv'
    if track_features_path is None:
        track_features_path = PROJECT_ROOT / 'data' / 'track_characteristics.csv'
    if telemetry_features_path is None:
        telemetry_features_path = PROJECT_ROOT / 'data' / 'driver_telemetry_features.csv'
    if output_path is None:
        output_path = PROJECT_ROOT / 'data' / 'HOLY_qualifying_v2.csv'
    
    print("Merging enhanced features...")
    
    # Load datasets
    print(f"Loading base: {base_dataset_path}")
    base_df = pd.read_csv(base_dataset_path)
    print(f"  {len(base_df)} records, {len(base_df.columns)} columns")
    
    print(f"Loading track features: {track_features_path}")
    track_df = pd.read_csv(track_features_path)
    print(f"  {len(track_df)} circuits")
    
    print(f"Loading telemetry: {telemetry_features_path}")
    telemetry_df = pd.read_csv(telemetry_features_path)
    print(f"  {len(telemetry_df)} records ({telemetry_df['season'].min()}-{telemetry_df['season'].max()})")
    
    # Merge track characteristics
    enhanced_df = base_df.merge(track_df, on='circuit_id', how='left', suffixes=('', '_track'))
    track_features_added = enhanced_df['track_total_corners'].notna().sum()
    print(f"Added track features to {track_features_added}/{len(enhanced_df)} rows")
    
    # Merge driver telemetry
    enhanced_df = enhanced_df.merge(
        telemetry_df,
        on=['driver', 'circuit_id', 'season'],
        how='left',
        suffixes=('', '_telemetry')
    )
    telemetry_features_added = enhanced_df['driver_avg_corner_speed'].notna().sum()
    print(f"Added telemetry to {telemetry_features_added}/{len(enhanced_df)} rows")
    
    # Save
    enhanced_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"  {len(enhanced_df)} records, {len(enhanced_df.columns)} columns (+{len(enhanced_df.columns) - len(base_df.columns)} new)")
    
    return enhanced_df


if __name__ == '__main__':
    enhanced_df = merge_enhanced_features()
    print("\nDone. Use HOLY_qualifying_v2.csv in your models.")

