"""
Create Qualifying Dataset with Telemetry Features (2018-2023 only)

This script creates a dataset using only modern F1 data (2018 onwards)
where FastF1 telemetry is available. This is faster than processing
all historical data.
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


def create_modern_dataset(min_year=2018, max_year=2024, output_path=None):
    """
    Create qualifying dataset using only modern data (2018+).

    Args:
        min_year: Minimum year to include (default 2018, when FastF1 data starts)
        max_year: Maximum year to include (exclusive). E.g., 2024 means include up to 2023.
        output_path: Optional output file path.
    """
    print("=" * 80)
    print(f"Creating Modern Qualifying Dataset with Telemetry Features")
    print(f"Including data from {min_year} to {max_year - 1}")
    print("=" * 80)

    # Load base data
    print("\nLoading base data...")
    quali = pd.read_csv(QUALIFYING_CSV)
    races = pd.read_csv(RACES_CSV)
    driver_standings = pd.read_csv(DRIVER_STANDINGS_CSV)
    constructor_standings = pd.read_csv(CONSTRUCTOR_STANDINGS_CSV)

    # Filter to modern era only (2018+) and before max_year
    quali = quali[(quali['season'] >= min_year) & (quali['season'] < max_year)]
    races = races[(races['season'] >= min_year) & (races['season'] < max_year)]
    driver_standings = driver_standings[(driver_standings['season'] >= min_year) & (driver_standings['season'] < max_year)]
    constructor_standings = constructor_standings[(constructor_standings['season'] >= min_year) & (constructor_standings['season'] < max_year)]

    print(f"  Qualifying: {len(quali)} records ({quali['season'].min()}-{quali['season'].max()})")
    print(f"  Races: {len(races)} records ({races['season'].min()}-{races['season'].max()})")

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
        weather = weather[(weather['season'] >= min_year) & (weather['season'] < max_year)]
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
        'red': 'Honda',
        'aston': 'Mercedes',
    }

    df['power_unit'] = df['constructor'].map(engine_mapping).fillna('Unknown')
    power_unit_dummies = pd.get_dummies(df['power_unit'], prefix='power_unit')
    df = pd.concat([df, power_unit_dummies], axis=1)

    # Add regulation era (all 2018+ is hybrid era)
    print("Adding regulation era...")
    df['regulation_era'] = 'hybrid_era'
    df['era_hybrid_era'] = 1
    df['era_post_turbo'] = 0
    df['era_turbo_era'] = 0
    df['era_v10_era'] = 0
    df['era_v8_era'] = 0

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

    # Add practice telemetry features
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
    parser = argparse.ArgumentParser(description='Create modern qualifying dataset (2018+) with telemetry')
    parser.add_argument('--min_year', type=int, default=2018,
                        help='Minimum year to include (default: 2018)')
    parser.add_argument('--max_year', type=int, default=2024,
                        help='Maximum year (exclusive, default: 2024)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')

    args = parser.parse_args()

    if args.output is None:
        output_path = TEST_DATA_DIR / f'HOLY_qualifying_modern_{args.min_year}_{args.max_year-1}.csv'
    else:
        output_path = Path(args.output)

    print(f"\nConfiguration:")
    print(f"  Date range: {args.min_year} to {args.max_year - 1}")
    print(f"  Output: {output_path}\n")

    df = create_modern_dataset(args.min_year, args.max_year, output_path)

    # Check telemetry features
    telemetry_cols = [col for col in df.columns if 'practice_' in col]
    print(f"\n✓ Added {len(telemetry_cols)} telemetry features")

    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)


if __name__ == "__main__":
    main()
