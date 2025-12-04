"""
Create Qualifying Dataset WITHOUT Data Leakage

This script creates HOLY_qualifying datasets for specific years without leakage.
It only uses data available UP TO (but not including) the test year when
calculating rolling features.

Usage:
    python create_qualifying_dataset_no_leakage.py --test_year 2022
    python create_qualifying_dataset_no_leakage.py --test_year 2023
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from telemetry_features import add_practice_telemetry_features

# File paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRAPING_DIR = PROJECT_ROOT / 'scraping'
TEST_DATA_DIR = PROJECT_ROOT / 'test' / 'data'

# Input files
QUALIFYING_CSV = SCRAPING_DIR / 'qualifying.csv'
RACES_CSV = SCRAPING_DIR / 'races.csv'
DRIVER_STANDINGS_CSV = SCRAPING_DIR / 'driver_standings.csv'
CONSTRUCTOR_STANDINGS_CSV = SCRAPING_DIR / 'constructor_standings.csv'
WEATHER_CSV = SCRAPING_DIR / 'weather.csv'


def create_dataset_for_year(max_year, output_path=None):
    """
    Create qualifying dataset using only data up to max_year (exclusive).

    This prevents data leakage by ensuring that features for year X
    are calculated using only data from years < X.

    Args:
        max_year: Maximum year to include (exclusive). E.g., 2023 means include up to 2022.
        output_path: Optional output file path. If None, returns DataFrame.
    """
    print("=" * 80)
    print(f"Creating Qualifying Dataset (No Leakage)")
    print(f"Including data up to year {max_year - 1} (training on <{max_year})")
    print("=" * 80)

    # Load base data
    print("\nLoading base data...")
    quali = pd.read_csv(QUALIFYING_CSV)
    races = pd.read_csv(RACES_CSV)
    driver_standings = pd.read_csv(DRIVER_STANDINGS_CSV)
    constructor_standings = pd.read_csv(CONSTRUCTOR_STANDINGS_CSV)

    # CRITICAL: Filter to only include data before max_year
    quali = quali[quali['season'] < max_year]
    races = races[races['season'] < max_year]
    driver_standings = driver_standings[driver_standings['season'] < max_year]
    constructor_standings = constructor_standings[constructor_standings['season'] < max_year]

    print(f"  Qualifying: {len(quali)} records (up to {quali['season'].max()})")
    print(f"  Races: {len(races)} records (up to {races['season'].max()})")

    # Merge qualifying with race info
    df = quali.merge(races, on=['season', 'round'], how='left')

    # Add standings
    print("\nAdding standings...")
    df['driver'] = df['driver_name'].str.split().str[-1].str.lower()
    df['constructor'] = df['car'].str.split().str[0].str.lower()

    driver_cols = ['season', 'round', 'driver', 'driver_points', 'driver_wins', 'driver_standings_pos']
    constructor_cols = ['season', 'round', 'constructor', 'constructor_points', 'constructor_wins', 'constructor_standings_pos']

    df = df.merge(driver_standings[driver_cols], on=['season', 'round', 'driver'], how='left')
    df = df.merge(constructor_standings[constructor_cols], on=['season', 'round', 'constructor'], how='left')

    df['driver_points'] = df['driver_points'].fillna(0)
    df['driver_wins'] = df['driver_wins'].fillna(0)
    df['driver_standings_pos'] = df['driver_standings_pos'].fillna(20)
    df['constructor_points'] = df['constructor_points'].fillna(0)
    df['constructor_wins'] = df['constructor_wins'].fillna(0)
    df['constructor_standings_pos'] = df['constructor_standings_pos'].fillna(10)

    # Add weather
    print("Adding weather...")
    try:
        weather = pd.read_csv(WEATHER_CSV)
        weather = weather[weather['season'] < max_year]  # Filter weather too!
        df = df.merge(weather, on=['season', 'round'], how='left', suffixes=('', '_weather_dup'))
    except FileNotFoundError:
        df['weather_warm'] = False
        df['weather_cold'] = False
        df['weather_dry'] = True
        df['weather_wet'] = False
        df['weather_cloudy'] = False

    # Create circuit features
    print("Creating circuit features...")
    circuit_dummies = pd.get_dummies(df['circuit_id'], prefix='circuit_id')
    df = pd.concat([df, circuit_dummies], axis=1)

    # Create nationality features (placeholder)
    df['nationality_Unknown'] = 1

    # Create constructor features
    print("Creating constructor features...")
    constructor_dummies = pd.get_dummies(df['constructor'], prefix='constructor')
    df = pd.concat([df, constructor_dummies], axis=1)

    # Calculate rolling qualifying averages
    print("Calculating rolling features...")
    df = df.sort_values(['driver', 'season', 'round'])

    # Driver rolling averages (using shift to prevent leakage)
    df['driver_avg_qualifying_position'] = df.groupby('driver')['grid_position'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

    df['driver_season_avg_quali'] = df.groupby(['driver', 'season'])['grid_position'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Constructor rolling averages
    df['constructor_avg_qualifying_position'] = df.groupby('constructor')['grid_position'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

    df['qualifying_SMA_constructor'] = df.groupby('constructor')['grid_position'].transform(
        lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()
    )

    # Fill NaN
    df['driver_avg_qualifying_position'] = df['driver_avg_qualifying_position'].fillna(10.0)
    df['driver_season_avg_quali'] = df['driver_season_avg_quali'].fillna(10.0)
    df['constructor_avg_qualifying_position'] = df['constructor_avg_qualifying_position'].fillna(10.0)
    df['qualifying_SMA_constructor'] = df['qualifying_SMA_constructor'].fillna(10.0)

    # Circuit-specific history
    print("Calculating circuit-specific history...")
    df = df.sort_values(['circuit_id', 'driver', 'season', 'round'])

    df['driver_avg_quali_at_circuit'] = df.groupby(['driver', 'circuit_id'])['grid_position'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df['constructor_avg_quali_at_circuit'] = df.groupby(['constructor', 'circuit_id'])['grid_position'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df['driver_poles_at_circuit'] = df.groupby(['driver', 'circuit_id'])['grid_position'].transform(
        lambda x: (x.shift(1) == 1).cumsum()
    )

    df['driver_avg_quali_at_circuit'] = df['driver_avg_quali_at_circuit'].fillna(10.0)
    df['constructor_avg_quali_at_circuit'] = df['constructor_avg_quali_at_circuit'].fillna(10.0)
    df['driver_poles_at_circuit'] = df['driver_poles_at_circuit'].fillna(0)

    # Career statistics
    print("Calculating career statistics...")
    df = df.sort_values(['driver', 'season', 'round'])

    df['driver_career_poles'] = df.groupby('driver')['grid_position'].transform(
        lambda x: (x.shift(1) == 1).cumsum()
    )

    df['driver_avg_quali_career'] = df.groupby('driver')['grid_position'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df['driver_best_qualifying_position'] = df.groupby('driver')['grid_position'].transform(
        lambda x: x.shift(1).expanding().min()
    )

    df['constructor_career_poles'] = df.groupby('constructor')['grid_position'].transform(
        lambda x: (x.shift(1) == 1).cumsum()
    )

    df['constructor_avg_quali_season'] = df.groupby(['constructor', 'season'])['grid_position'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df['driver_career_poles'] = df['driver_career_poles'].fillna(0)
    df['driver_avg_quali_career'] = df['driver_avg_quali_career'].fillna(15.0)
    df['driver_best_qualifying_position'] = df['driver_best_qualifying_position'].fillna(20)
    df['constructor_career_poles'] = df['constructor_career_poles'].fillna(0)
    df['constructor_avg_quali_season'] = df['constructor_avg_quali_season'].fillna(10.0)

    # Add power unit manufacturer
    print("Adding power unit features...")
    engine_mapping = {
        'ferrari': 'Ferrari',
        'mercedes': 'Mercedes',
        'red_bull': 'Honda',
        'mclaren': 'Mercedes',
        'renault': 'Renault',
        'williams': 'Mercedes',
        'haas': 'Ferrari',
        'alfa': 'Ferrari',
        'alphatauri': 'Honda',
        'alpine': 'Renault',
        'red': 'Honda',  # Red Bull Racing
        'aston': 'Mercedes',  # Aston Martin
    }

    df['power_unit'] = df['constructor'].map(engine_mapping).fillna('Unknown')
    power_unit_dummies = pd.get_dummies(df['power_unit'], prefix='power_unit')
    df = pd.concat([df, power_unit_dummies], axis=1)

    # Add regulation era
    print("Adding regulation era...")
    def get_era(year):
        if year < 1989:
            return 'turbo_era'
        elif year < 1995:
            return 'post_turbo'
        elif year < 2006:
            return 'v10_era'
        elif year < 2014:
            return 'v8_era'
        else:
            return 'hybrid_era'

    df['regulation_era'] = df['season'].apply(get_era)
    era_dummies = pd.get_dummies(df['regulation_era'], prefix='era')
    df = pd.concat([df, era_dummies], axis=1)

    # Additional features
    print("Adding additional features...")
    df['rounds_completed'] = df.groupby('season').cumcount()

    # Parse qualifying time
    def parse_quali_time(time_str):
        try:
            if pd.isna(time_str) or time_str == '':
                return np.nan
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            return float(time_str)
        except:
            return np.nan

    if 'qualifying_time' in df.columns:
        df['qualifying_secs_raw'] = df['qualifying_time'].apply(parse_quali_time)
        df['qualifying_secs'] = df.groupby(['season', 'round'])['qualifying_secs_raw'].transform(
            lambda x: x - x.min()
        )

    # Add practice telemetry features if available
    print("Adding practice telemetry features...")
    df = add_practice_telemetry_features(df, max_year)

    # Clean up
    print("Cleaning and finalizing...")
    columns_to_drop = [
        'driver_name', 'car', 'qualifying_time', 'qualifying_secs_raw',
        'url', 'lat', 'long', 'country', 'date',
        'power_unit', 'regulation_era'
    ]

    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    df = df.rename(columns={'grid_position': 'grid'})

    # Reorder columns
    cols = df.columns.tolist()
    if 'grid' in cols:
        cols.remove('grid')
        driver_idx = cols.index('driver') if 'driver' in cols else 0
        cols.insert(driver_idx + 1, 'grid')
        df = df[cols]

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Date range: {df['season'].min()}-{df['season'].max()}")

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Create qualifying dataset without data leakage')
    parser.add_argument('--test_year', type=int, default=2022,
                        help='Test year (data will be filtered to < test_year)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: HOLY_qualifying_v1_train{test_year}.csv)')

    args = parser.parse_args()

    if args.output is None:
        output_path = TEST_DATA_DIR / f'HOLY_qualifying_v1_train{args.test_year}.csv'
    else:
        output_path = Path(args.output)

    print(f"\nConfiguration:")
    print(f"  Test year: {args.test_year}")
    print(f"  Training data: <{args.test_year} (up to {args.test_year - 1})")
    print(f"  Output: {output_path}\n")

    df = create_dataset_for_year(args.test_year, output_path)

    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print(f"\nDataset ready for training/testing:")
    print(f"  - Train on this dataset")
    print(f"  - Test on year {args.test_year}")
    print(f"  - No data leakage from future years!")
    print("=" * 80)


if __name__ == "__main__":
    main()
