"""
Add FastF1 Practice Telemetry Features to Existing Qualifying Dataset

This script:
1. Loads the existing HOLY_qualifying_v1.csv
2. For each qualifying session (starting from 2018 when FastF1 data is available)
3. Extracts telemetry features from PRACTICE sessions (FP1, FP2, FP3) - NO DATA LEAKAGE
4. Merges them into the existing dataset
5. Saves as HOLY_qualifying_v2.csv with practice telemetry features
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Use existing FastF1 cache directory in project root
cache_dir = '/Users/sz/Programming/CDS/f1_ml/fastf1_cache'
fastf1.Cache.enable_cache(cache_dir)


def get_fastf1_event_name(circuit_id, year):
    """
    Map circuit_id to FastF1 event name
    FastF1 uses different naming conventions
    """
    # Common mappings
    circuit_mapping = {
        'albert_park': 'Australia',
        'bahrain': 'Bahrain',
        'shanghai': 'China',
        'baku': 'Azerbaijan',
        'catalunya': 'Spain',
        'monaco': 'Monaco',
        'villeneuve': 'Canada',
        'ricard': 'France',
        'red_bull_ring': 'Austria',
        'silverstone': 'Great Britain',
        'hockenheimring': 'Germany',
        'hungaroring': 'Hungary',
        'spa': 'Belgium',
        'monza': 'Italy',
        'marina_bay': 'Singapore',
        'sochi': 'Russia',
        'suzuka': 'Japan',
        'americas': 'United States',
        'rodriguez': 'Mexico',
        'interlagos': 'Brazil',
        'yas_marina': 'Abu Dhabi',
        'istanbul': 'Turkey',
        'portimao': 'Portugal',
        'imola': 'Emilia Romagna',
        'miami': 'Miami',
        'jeddah': 'Saudi Arabia',
        'losail': 'Qatar',
        'zandvoort': 'Netherlands',
    }

    return circuit_mapping.get(circuit_id, None)


def extract_practice_telemetry_for_driver(practice_sessions, driver_abbr):
    """
    Extract telemetry features from practice sessions (FP1, FP2, FP3)
    This avoids data leakage - we only use data available BEFORE qualifying

    Args:
        practice_sessions: List of loaded FastF1 practice sessions (FP1, FP2, FP3)
        driver_abbr: Driver abbreviation (e.g., 'VER', 'HAM')

    Returns dict of features or None if data not available
    """
    all_telemetry = []
    all_laps = []

    try:
        # Collect telemetry from all available practice sessions
        for session in practice_sessions:
            if session is None:
                continue

            try:
                driver_laps = session.laps.pick_drivers(driver_abbr)
                if len(driver_laps) == 0:
                    continue

                # Get fastest lap from this practice session
                try:
                    fastest = driver_laps.pick_fastest()
                    if pd.isna(fastest['LapTime']):
                        continue

                    # Get telemetry
                    telemetry = fastest.get_car_data()
                    all_telemetry.append(telemetry)
                    all_laps.append(fastest)

                except:
                    continue

            except:
                continue

        if len(all_telemetry) == 0:
            return None

        # Aggregate telemetry from all practice sessions
        # Take the BEST performance across practice sessions (max speed, best lap time, etc.)
        features = {
            'practice_max_speed': max([tel['Speed'].max() for tel in all_telemetry if 'Speed' in tel.columns]),
            'practice_avg_speed': np.mean([tel['Speed'].mean() for tel in all_telemetry if 'Speed' in tel.columns]),
            'practice_max_rpm': max([tel['RPM'].max() for tel in all_telemetry if 'RPM' in tel.columns]),
            'practice_avg_rpm': np.mean([tel['RPM'].mean() for tel in all_telemetry if 'RPM' in tel.columns]),
            'practice_max_throttle': max([tel['Throttle'].max() for tel in all_telemetry if 'Throttle' in tel.columns]),
            'practice_avg_throttle': np.mean([tel['Throttle'].mean() for tel in all_telemetry if 'Throttle' in tel.columns]),
            'practice_brake_usage': np.mean([(tel['Brake'] == True).sum() / len(tel) * 100 for tel in all_telemetry if 'Brake' in tel.columns]),
            'practice_best_lap_time': min([lap['LapTime'].total_seconds() for lap in all_laps if pd.notna(lap['LapTime'])]),
            'practice_sessions_count': len(all_telemetry),
        }

        return features

    except Exception as e:
        # Silently fail
        return None


def add_telemetry_to_qualifying_data(input_csv, output_csv, start_year=2018, max_sessions=20):
    """
    Main function to add practice telemetry features to qualifying dataset

    Args:
        input_csv: Path to HOLY_qualifying_v1.csv
        output_csv: Path to save HOLY_qualifying_v2.csv
        start_year: Year to start fetching telemetry (FastF1 data available from 2018)
        max_sessions: Maximum number of sessions to process (to avoid rate limits)
    """
    print("="*70)
    print("Adding FastF1 Practice Telemetry Features to Qualifying Dataset")
    print("="*70)

    # Load existing dataset
    print(f"\nLoading existing dataset: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"  Loaded {len(df)} records")
    print(f"  Date range: {df['season'].min()}-{df['season'].max()}")

    # Add practice telemetry columns (initialize with NaN)
    telemetry_columns = [
        'practice_max_speed',
        'practice_avg_speed',
        'practice_max_rpm',
        'practice_avg_rpm',
        'practice_max_throttle',
        'practice_avg_throttle',
        'practice_brake_usage',
        'practice_best_lap_time',
        'practice_sessions_count',
    ]

    for col in telemetry_columns:
        df[col] = np.nan

    # Filter to years where FastF1 has data
    df_with_telemetry = df[df['season'] >= start_year].copy()
    print(f"\n  Records from {start_year} onward: {len(df_with_telemetry)}")

    # Get unique sessions
    sessions_to_process = df_with_telemetry[['season', 'round', 'circuit_id']].drop_duplicates()
    print(f"  Unique qualifying sessions available: {len(sessions_to_process)}")

    # Limit number of sessions to process
    if max_sessions and len(sessions_to_process) > max_sessions:
        sessions_to_process = sessions_to_process.head(max_sessions)
        print(f"  Limiting to {max_sessions} sessions to avoid rate limits")

    print(f"  Sessions to process: {len(sessions_to_process)}")

    # Process each session
    processed_sessions = 0
    failed_sessions = []

    for idx, row in tqdm(sessions_to_process.iterrows(), total=len(sessions_to_process), desc="Processing sessions"):
        year = int(row['season'])
        round_num = int(row['round'])
        circuit_id = row['circuit_id']

        # Get FastF1 event name
        event_name = get_fastf1_event_name(circuit_id, year)

        if event_name is None:
            failed_sessions.append((year, round_num, circuit_id, "No mapping"))
            continue

        try:
            # Load practice sessions (FP1, FP2, FP3) - NOT qualifying!
            practice_sessions = []
            for session_name in ['FP1', 'FP2', 'FP3']:
                try:
                    fp_session = fastf1.get_session(year, event_name, session_name)
                    fp_session.load()
                    practice_sessions.append(fp_session)
                except Exception as e:
                    # Practice session might not be available
                    practice_sessions.append(None)

            # Skip if no practice sessions available
            if all(s is None for s in practice_sessions):
                failed_sessions.append((year, round_num, circuit_id, "No practice sessions"))
                continue

            # Get all drivers for this session in our dataset
            session_drivers = df_with_telemetry[
                (df_with_telemetry['season'] == year) &
                (df_with_telemetry['round'] == round_num)
            ]

            # Extract practice telemetry for each driver
            drivers_updated = 0
            for _, driver_row in session_drivers.iterrows():
                driver_code = driver_row['driver']  # e.g., 'lec', 'ver', 'ham'

                # Convert to uppercase for FastF1 (expects 'LEC', 'VER', 'HAM')
                driver_abbr = driver_code.upper()

                # Extract telemetry features from practice sessions
                features = extract_practice_telemetry_for_driver(practice_sessions, driver_abbr)

                if features is not None:
                    # Update the dataframe
                    mask = (
                        (df['season'] == year) &
                        (df['round'] == round_num) &
                        (df['driver'] == driver_code)
                    )

                    for col, value in features.items():
                        df.loc[mask, col] = value

                    drivers_updated += 1

            if drivers_updated > 0:
                processed_sessions += 1

                # Save progress after each session (incremental save)
                df.to_csv(output_csv, index=False)

        except Exception as e:
            failed_sessions.append((year, round_num, circuit_id, str(e)[:50]))
            continue

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Successfully processed: {processed_sessions}/{len(sessions_to_process)} sessions")
    print(f"Failed sessions: {len(failed_sessions)}")

    if failed_sessions:
        print("\nFailed sessions (first 10):")
        for year, round_num, circuit_id, error in failed_sessions[:10]:
            print(f"  {year} Round {round_num} ({circuit_id}): {error}")
        if len(failed_sessions) > 10:
            print(f"  ... and {len(failed_sessions) - 10} more")

    # Check how many records got practice telemetry data
    telemetry_filled = df[df['practice_max_speed'].notna()]
    print(f"\nRecords with practice telemetry data: {len(telemetry_filled)}/{len(df)}")
    print(f"Percentage coverage: {len(telemetry_filled)/len(df)*100:.2f}%")
    print(f"Coverage for {start_year}+: {len(telemetry_filled)}/{len(df_with_telemetry)} = {len(telemetry_filled)/len(df_with_telemetry)*100:.2f}%")

    # Save final enhanced dataset
    print(f"\nSaving final dataset to: {output_csv}")
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(df)} records with {len(df.columns)} columns")

    # Show sample with practice telemetry
    print("\nSample records with practice telemetry:")
    sample = df[df['practice_max_speed'].notna()][['season', 'round', 'circuit_id', 'driver', 'grid',
                                                      'practice_max_speed', 'practice_avg_speed',
                                                      'practice_best_lap_time']].head(10)
    print(sample.to_string())

    return df


if __name__ == "__main__":
    input_file = '/Users/sz/Programming/CDS/f1_ml/test/data/HOLY_qualifying_v1.csv'
    output_file = '/Users/sz/Programming/CDS/f1_ml/test/data/HOLY_qualifying_v2.csv'

    # Add practice telemetry features (starting from 2018, max 10 sessions to avoid rate limits)
    # You can run this multiple times, increasing max_sessions gradually
    df_enhanced = add_telemetry_to_qualifying_data(
        input_file,
        output_file,
        start_year=2018,
        max_sessions=20  # Process 10 sessions at a time (about half a year)
    )

    print("\n" + "="*70)
    print("Complete!")
    print("="*70)
    print("\nTo process more sessions, increase max_sessions and run again.")
