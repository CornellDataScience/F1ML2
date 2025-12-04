"""
Merge track and driver profile features into main qualifying dataset.
Uses year-1 driver profiles to avoid data leakage.

Output: HOLY_qualifying_v3.csv
"""

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


def merge_enhanced_features(
    base_dataset_path: str = None,
    track_features_path: str = None,
    driver_profile_path: str = None,
    output_path: str = None
):
    """
    Merge track and driver profile features into base qualifying dataset.
    Uses year-1 driver profiles for predictions (no data leakage).
    """
    
    # Default paths
    if base_dataset_path is None:
        base_dataset_path = PROJECT_ROOT / 'data' / 'HOLY_qualifying_v1.csv'
    if track_features_path is None:
        track_features_path = PROJECT_ROOT / 'data' / 'track_characteristics.csv'
    if driver_profile_path is None:
        driver_profile_path = PROJECT_ROOT / 'data' / 'driver_telemetry_profile.csv'
    if output_path is None:
        output_path = PROJECT_ROOT / 'data' / 'HOLY_qualifying_v3.csv'
    
    print("Merging enhanced features (Option A approach)...")
    
    # Load datasets
    print(f"Loading base: {base_dataset_path}")
    base_df = pd.read_csv(base_dataset_path)
    print(f"  {len(base_df)} records, {len(base_df.columns)} columns")
    
    print(f"Loading track features: {track_features_path}")
    track_df = pd.read_csv(track_features_path)
    print(f"  {len(track_df)} circuits")
    
    print(f"Loading driver profiles: {driver_profile_path}")
    profile_df = pd.read_csv(driver_profile_path)
    print(f"  {len(profile_df)} profiles ({profile_df['season'].min()}-{profile_df['season'].max()})")
    
    # Step 1: Merge track characteristics (static, no leakage concern)
    enhanced_df = base_df.merge(track_df, on='circuit_id', how='left', suffixes=('', '_track'))
    track_features_added = enhanced_df['track_total_corners'].notna().sum()
    print(f"Added track features to {track_features_added}/{len(enhanced_df)} rows")
    
    # Step 2: Merge driver profiles with YEAR-1 OFFSET (key fix for data leakage!)
    # Create a profile lookup where season is shifted by 1
    profile_df_shifted = profile_df.copy()
    profile_df_shifted['season'] = profile_df_shifted['season'] + 1  # 2022 profile -> used for 2023 predictions
    
    # Handle driver name format mismatch:
    # 2018-2022 use 3-letter codes (alo, ver), 2023 uses full names (alonso, verstappen)
    # Create mapping from 3-letter to full name for profiles targeting 2023+
    code_to_fullname = {
        'alo': 'alonso', 'alb': 'albon', 'bot': 'bottas', 'dev': 'de_vries',
        'gas': 'gasly', 'ham': 'hamilton', 'hul': 'hulkenberg', 'lat': 'latifi',
        'lec': 'leclerc', 'mag': 'magnussen', 'nor': 'norris', 'oco': 'ocon',
        'per': 'perez', 'pia': 'piastri', 'ric': 'ricciardo', 'rus': 'russell',
        'sai': 'sainz', 'sar': 'sargeant', 'str': 'stroll', 'tsu': 'tsunoda',
        'ver': 'verstappen', 'vet': 'vettel', 'zho': 'zhou', 'msc': 'schumacher',
        'maz': 'mazepin', 'rai': 'raikkonen', 'gio': 'giovinazzi', 'kub': 'kubica',
        'gro': 'grosjean', 'law': 'lawson',
    }
    
    # For profiles going to 2023, convert 3-letter codes to full names
    profile_2023 = profile_df_shifted[profile_df_shifted['season'] == 2023].copy()
    profile_2023['driver'] = profile_2023['driver'].map(code_to_fullname).fillna(profile_2023['driver'])
    
    # Combine with other years (which keep 3-letter codes)
    profile_other = profile_df_shifted[profile_df_shifted['season'] != 2023].copy()
    profile_df_shifted = pd.concat([profile_other, profile_2023], ignore_index=True)
    
    enhanced_df = enhanced_df.merge(
        profile_df_shifted,
        on=['driver', 'season'],  # Now 2023 rows get 2022 profiles with correct driver names
        how='left',
        suffixes=('', '_profile')
    )
    
    profile_features_added = enhanced_df['driver_avg_corner_speed'].notna().sum()
    print(f"Added driver profiles to {profile_features_added}/{len(enhanced_df)} rows")
    print(f"  (using year-1 profiles to avoid data leakage)")
    
    # Show coverage by year
    print("\nProfile coverage by year:")
    for year in sorted(enhanced_df['season'].unique()):
        if year >= 2019:  # First year with profiles is 2019 (using 2018 data)
            year_df = enhanced_df[enhanced_df['season'] == year]
            with_profile = year_df['driver_avg_corner_speed'].notna().sum()
            total = len(year_df)
            pct = 100 * with_profile / total if total > 0 else 0
            print(f"  {year}: {with_profile}/{total} rows ({pct:.1f}%)")
    
    # Save
    enhanced_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"  {len(enhanced_df)} records, {len(enhanced_df.columns)} columns")
    
    return enhanced_df


if __name__ == '__main__':
    enhanced_df = merge_enhanced_features()
    print("\nDone. Use HOLY_qualifying_v3.csv in your models.")

