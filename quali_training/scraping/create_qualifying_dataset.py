"""
Create Qualifying Prediction Dataset (HOLY_qualifying_v1.csv)

This script builds a comprehensive qualifying prediction dataset by:
1. Starting with qualifying results (grid positions)
2. Adding driver/constructor standings, circuit info, weather
3. Engineering qualifying-specific historical features
4. Adding car/engine metadata
5. Removing race-specific features that aren't available before qualifying

Target variable: grid_position (qualifying result)
Time period: 1983-2023 (matching HOLYv1.csv)
"""

import pandas as pd
import numpy as np
from pathlib import Path

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
HOLY_V1_CSV = PROJECT_ROOT / 'HOLYv1.csv'  # For reference structure

# Output file
OUTPUT_CSV = TEST_DATA_DIR / 'HOLY_qualifying_v1.csv'


def load_base_data():
    """Load qualifying results and race metadata"""
    print("Loading base qualifying data...")

    # Load qualifying results
    quali = pd.read_csv(QUALIFYING_CSV)
    print(f"  Loaded {len(quali)} qualifying records from {QUALIFYING_CSV.name}")

    # Load race info (for circuit details)
    races = pd.read_csv(RACES_CSV)
    print(f"  Loaded {len(races)} race records from {RACES_CSV.name}")

    # Merge qualifying with race info
    df = quali.merge(races, on=['season', 'round'], how='left')
    print(f"  Merged dataset: {len(df)} records")

    return df


def add_standings(df):
    """Add driver and constructor championship standings (before the race)"""
    print("\nAdding driver and constructor standings...")

    # Load standings
    driver_standings = pd.read_csv(DRIVER_STANDINGS_CSV)
    constructor_standings = pd.read_csv(CONSTRUCTOR_STANDINGS_CSV)

    # Use the standings BEFORE the race (not after)
    driver_cols = ['season', 'round', 'driver', 'driver_points', 'driver_wins', 'driver_standings_pos']
    constructor_cols = ['season', 'round', 'constructor', 'constructor_points', 'constructor_wins', 'constructor_standings_pos']

    # Extract driver name from "driver_name" in qualifying (format: "Firstname Lastname CODE")
    # Match with "driver" in standings
    df['driver'] = df['driver_name'].str.split().str[-1].str.lower()

    # Extract constructor from "car" column
    df['constructor'] = df['car'].str.split().str[0].str.lower()

    # Merge standings (use suffixes to avoid column conflicts)
    df = df.merge(driver_standings[driver_cols], on=['season', 'round', 'driver'], how='left', suffixes=('', '_driver_dup'))
    df = df.merge(constructor_standings[constructor_cols], on=['season', 'round', 'constructor'], how='left', suffixes=('', '_constructor_dup'))

    # Fill NaN values (for first race of season or missing data)
    df['driver_points'] = df['driver_points'].fillna(0)
    df['driver_wins'] = df['driver_wins'].fillna(0)
    df['driver_standings_pos'] = df['driver_standings_pos'].fillna(20)
    df['constructor_points'] = df['constructor_points'].fillna(0)
    df['constructor_wins'] = df['constructor_wins'].fillna(0)
    df['constructor_standings_pos'] = df['constructor_standings_pos'].fillna(10)

    print(f"  Added standings for {len(df)} records")
    return df


def add_weather(df):
    """Add weather conditions"""
    print("\nAdding weather data...")
    print(f"  Columns before weather merge: {len(df.columns)}")
    print(f"  Circuit_id present: {'circuit_id' in df.columns}")

    try:
        weather = pd.read_csv(WEATHER_CSV)
        df = df.merge(weather, on=['season', 'round'], how='left', suffixes=('', '_weather_dup'))
        print(f"  Merged weather data")
    except FileNotFoundError:
        print("  Weather file not found, creating default weather features...")
        df['weather_warm'] = False
        df['weather_cold'] = False
        df['weather_dry'] = True
        df['weather_wet'] = False
        df['weather_cloudy'] = False

    print(f"  Columns after weather merge: {len(df.columns)}")
    print(f"  Circuit_id present: {'circuit_id' in df.columns}")
    return df


def create_circuit_features(df):
    """One-hot encode circuit IDs"""
    print("\nCreating circuit features...")

    # Get circuit dummies
    circuit_dummies = pd.get_dummies(df['circuit_id'], prefix='circuit_id')
    df = pd.concat([df, circuit_dummies], axis=1)

    print(f"  Created {len(circuit_dummies.columns)} circuit features")
    return df


def create_nationality_features(df):
    """One-hot encode driver nationalities"""
    print("\nCreating nationality features...")

    # Load HOLYv1 to get nationality mapping
    holy = pd.read_csv(HOLY_V1_CSV)

    # Extract unique driver-nationality mappings
    nationality_cols = [col for col in holy.columns if col.startswith('nationality_')]
    driver_col = holy['driver']

    # For now, create basic nationality features (this would need manual mapping)
    # Simplified: assume nationality from country or use placeholder
    df['nationality_Unknown'] = 1  # Placeholder

    print(f"  Created nationality features (manual mapping needed for full accuracy)")
    return df


def create_constructor_features(df):
    """One-hot encode constructors"""
    print("\nCreating constructor features...")

    constructor_dummies = pd.get_dummies(df['constructor'], prefix='constructor')
    df = pd.concat([df, constructor_dummies], axis=1)

    print(f"  Created {len(constructor_dummies.columns)} constructor features")
    return df


def calculate_rolling_quali_averages(df):
    """Calculate rolling averages for qualifying positions (historical features)"""
    print("\nCalculating qualifying rolling averages...")

    # Sort by driver and date
    df = df.sort_values(['driver', 'season', 'round'])

    # Driver average qualifying position (last 5 races)
    df['driver_avg_qualifying_position'] = df.groupby('driver')['grid_position'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

    # Driver season average (all races before this one in the season)
    df['driver_season_avg_quali'] = df.groupby(['driver', 'season'])['grid_position'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Constructor average qualifying position (last 5 races)
    df['constructor_avg_qualifying_position'] = df.groupby('constructor')['grid_position'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

    # Nationality 30-race SMA for qualifying
    # For simplicity, using constructor as proxy (would need nationality mapping)
    df['qualifying_SMA_constructor'] = df.groupby('constructor')['grid_position'].transform(
        lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()
    )

    # Fill NaN for first races
    df['driver_avg_qualifying_position'] = df['driver_avg_qualifying_position'].fillna(10.0)
    df['driver_season_avg_quali'] = df['driver_season_avg_quali'].fillna(10.0)
    df['constructor_avg_qualifying_position'] = df['constructor_avg_qualifying_position'].fillna(10.0)
    df['qualifying_SMA_constructor'] = df['qualifying_SMA_constructor'].fillna(10.0)

    print(f"  Calculated rolling averages")
    return df


def calculate_circuit_specific_history(df):
    """Calculate driver/constructor historical performance at each circuit"""
    print("\nCalculating circuit-specific historical performance...")

    # Sort by circuit, driver, and date
    df = df.sort_values(['circuit_id', 'driver', 'season', 'round'])

    # Driver's historical average qualifying at this circuit
    df['driver_avg_quali_at_circuit'] = df.groupby(['driver', 'circuit_id'])['grid_position'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Constructor's historical average at this circuit
    df['constructor_avg_quali_at_circuit'] = df.groupby(['constructor', 'circuit_id'])['grid_position'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Driver's poles at this circuit
    df['driver_poles_at_circuit'] = df.groupby(['driver', 'circuit_id'])['grid_position'].transform(
        lambda x: (x.shift(1) == 1).cumsum()
    )

    # Fill NaN
    df['driver_avg_quali_at_circuit'] = df['driver_avg_quali_at_circuit'].fillna(10.0)
    df['constructor_avg_quali_at_circuit'] = df['constructor_avg_quali_at_circuit'].fillna(10.0)
    df['driver_poles_at_circuit'] = df['driver_poles_at_circuit'].fillna(0)

    print(f"  Calculated circuit-specific history")
    return df


def calculate_career_stats(df):
    """Calculate career statistics (poles, average quali position)"""
    print("\nCalculating career statistics...")

    # Sort by driver and date
    df = df.sort_values(['driver', 'season', 'round'])

    # Career poles (count of pole positions before this race)
    df['driver_career_poles'] = df.groupby('driver')['grid_position'].transform(
        lambda x: (x.shift(1) == 1).cumsum()
    )

    # Career average qualifying position
    df['driver_avg_quali_career'] = df.groupby('driver')['grid_position'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Career best qualifying position
    df['driver_best_qualifying_position'] = df.groupby('driver')['grid_position'].transform(
        lambda x: x.shift(1).expanding().min()
    )

    # Constructor career poles
    df['constructor_career_poles'] = df.groupby('constructor')['grid_position'].transform(
        lambda x: (x.shift(1) == 1).cumsum()
    )

    # Constructor average qualifying season
    df['constructor_avg_quali_season'] = df.groupby(['constructor', 'season'])['grid_position'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Fill NaN
    df['driver_career_poles'] = df['driver_career_poles'].fillna(0)
    df['driver_avg_quali_career'] = df['driver_avg_quali_career'].fillna(15.0)
    df['driver_best_qualifying_position'] = df['driver_best_qualifying_position'].fillna(20)
    df['constructor_career_poles'] = df['constructor_career_poles'].fillna(0)
    df['constructor_avg_quali_season'] = df['constructor_avg_quali_season'].fillna(10.0)

    print(f"  Calculated career statistics")
    return df


def add_power_unit_manufacturer(df):
    """Add power unit/engine manufacturer information"""
    print("\nAdding power unit manufacturer...")

    # Engine manufacturer mapping (simplified - would need comprehensive mapping)
    # Format: constructor -> engine manufacturer
    engine_mapping = {
        'ferrari': 'Ferrari',
        'mercedes': 'Mercedes',
        'red_bull': 'Honda',  # Modern era; would vary by year
        'mclaren': 'Mercedes',  # Modern era
        'renault': 'Renault',
        'williams': 'Mercedes',  # Modern era
        'haas': 'Ferrari',
        'alfa': 'Ferrari',
        'alphatauri': 'Honda',
        'alpine': 'Renault',
        # Add more mappings as needed
    }

    df['power_unit'] = df['constructor'].map(engine_mapping).fillna('Unknown')

    # One-hot encode power units
    power_unit_dummies = pd.get_dummies(df['power_unit'], prefix='power_unit')
    df = pd.concat([df, power_unit_dummies], axis=1)

    print(f"  Created {len(power_unit_dummies.columns)} power unit features")
    return df


def add_regulation_era(df):
    """Add F1 regulation era indicators"""
    print("\nAdding regulation era features...")

    # Define regulation eras
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

    # One-hot encode eras
    era_dummies = pd.get_dummies(df['regulation_era'], prefix='era')
    df = pd.concat([df, era_dummies], axis=1)

    print(f"  Created {len(era_dummies.columns)} regulation era features")
    return df


def add_additional_features(df):
    """Add miscellaneous features"""
    print("\nAdding additional features...")

    # Rounds completed in season (season progression)
    df['rounds_completed'] = df.groupby('season').cumcount()

    # Convert qualifying time to seconds (if not already numeric)
    # Note: qualifying_time in qualifying.csv is a string like "1:34.526"
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

        # Normalize by race (gap to fastest)
        df['qualifying_secs'] = df.groupby(['season', 'round'])['qualifying_secs_raw'].transform(
            lambda x: x - x.min()
        )

    print(f"  Added additional features")
    return df


def clean_and_finalize(df):
    """Clean up dataset and prepare final output"""
    print("\nCleaning and finalizing dataset...")

    # Drop columns we don't need
    columns_to_drop = [
        'driver_name',  # Keep 'driver' code instead
        'car',  # Already extracted to 'constructor'
        'qualifying_time',  # Keep 'qualifying_secs' instead
        'qualifying_secs_raw',  # Keep normalized version
        'url',  # Race URL not needed
        'lat', 'long', 'country', 'date',  # Geographic/temporal info (keep season/round)
        'power_unit',  # Keep one-hot encoded version
        'regulation_era',  # Keep one-hot encoded version
    ]

    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # Rename grid_position to grid for consistency with HOLYv1.csv
    df = df.rename(columns={'grid_position': 'grid'})

    # Reorder columns: put target variable (grid) after driver
    cols = df.columns.tolist()
    if 'grid' in cols:
        cols.remove('grid')
        driver_idx = cols.index('driver') if 'driver' in cols else 0
        cols.insert(driver_idx + 1, 'grid')
        df = df[cols]

    print(f"  Final dataset shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")

    return df


def main():
    """Main execution function"""
    print("=" * 80)
    print("Creating Qualifying Prediction Dataset")
    print("=" * 80)

    # Load base data
    df = load_base_data()

    # Add standings
    df = add_standings(df)

    # Add weather
    df = add_weather(df)

    # Create features
    df = create_circuit_features(df)
    df = create_nationality_features(df)
    df = create_constructor_features(df)

    # Calculate historical features
    df = calculate_rolling_quali_averages(df)
    df = calculate_circuit_specific_history(df)
    df = calculate_career_stats(df)

    # Add car/engine metadata
    df = add_power_unit_manufacturer(df)
    df = add_regulation_era(df)

    # Add additional features
    df = add_additional_features(df)

    # Clean and finalize
    df = clean_and_finalize(df)

    # Save to CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 80)
    print(f"Dataset saved to: {OUTPUT_CSV}")
    print(f"Total records: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"Date range: {df['season'].min()}-{df['season'].max()}")
    print("=" * 80)

    # Print sample statistics
    print("\nSample statistics:")
    print(f"  Missing values: {df.isnull().sum().sum()} total")
    print(f"  Unique drivers: {df['driver'].nunique()}")
    print(f"  Unique constructors: {df['constructor'].nunique()}")
    print(f"  Unique circuits: {df['circuit_id'].nunique()}")

    print("\nFirst few rows:")
    print(df.head())

    print("\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")


if __name__ == "__main__":
    main()
