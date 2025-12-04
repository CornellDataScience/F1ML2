"""
FastF1 Data Exploration Script
Explores what car and telemetry data is available from FastF1 API
"""

import fastf1
import pandas as pd
from pathlib import Path

# Use existing FastF1 cache directory in project root
cache_dir = '/Users/sz/Programming/CDS/f1_ml/fastf1_cache'

# Enable FastF1 cache to speed up subsequent requests
fastf1.Cache.enable_cache(cache_dir)

def explore_session_data(year=2024, event='Monaco', session_type='Q'):
    """
    Explore what data is available for a qualifying session

    Args:
        year: Season year
        event: Event name or round number
        session_type: 'FP1', 'FP2', 'FP3', 'Q', 'S', 'R' (Qualifying, Sprint, Race)
    """
    print("="*70)
    print(f"Loading {year} {event} - {session_type} session...")
    print("="*70)

    # Load session
    session = fastf1.get_session(year, event, session_type)
    session.load()

    print(f"\n✓ Session loaded: {session.event['EventName']} - {session.name}")
    print(f"  Date: {session.event['EventDate']}")
    print(f"  Location: {session.event['Location']}")

    # 1. SESSION RESULTS
    print("\n" + "="*70)
    print("1. SESSION RESULTS")
    print("="*70)
    results = session.results
    print(f"\nResults DataFrame shape: {results.shape}")
    print(f"Columns ({len(results.columns)}):")
    for col in results.columns:
        print(f"  - {col}")

    print("\nSample results (top 3):")
    print(results[['DriverNumber', 'FullName', 'TeamName', 'Position', 'Q1', 'Q2', 'Q3']].head(3))

    # 2. LAPS DATA
    print("\n" + "="*70)
    print("2. LAPS DATA")
    print("="*70)
    laps = session.laps
    print(f"\nLaps DataFrame shape: {laps.shape}")
    print(f"Total laps: {len(laps)}")
    print(f"Columns ({len(laps.columns)}):")
    for col in laps.columns:
        print(f"  - {col}")

    # Get fastest lap
    fastest_lap = laps.pick_fastest()
    print(f"\nFastest lap:")
    print(f"  Driver: {fastest_lap['Driver']}")
    print(f"  Time: {fastest_lap['LapTime']}")
    print(f"  Team: {fastest_lap['Team']}")

    # 3. CAR TELEMETRY (from fastest lap)
    print("\n" + "="*70)
    print("3. CAR TELEMETRY (from fastest lap)")
    print("="*70)

    telemetry = fastest_lap.get_car_data()
    print(f"\nTelemetry DataFrame shape: {telemetry.shape}")
    print(f"Samples per lap: {len(telemetry)}")
    print(f"Columns ({len(telemetry.columns)}):")
    for col in telemetry.columns:
        print(f"  - {col}")

    print("\nSample telemetry data (first 5 rows):")
    print(telemetry.head())

    # Telemetry statistics
    print("\nTelemetry statistics:")
    print(telemetry.describe())

    # 4. WEATHER DATA
    print("\n" + "="*70)
    print("4. WEATHER DATA")
    print("="*70)

    weather = session.weather_data
    if weather is not None and len(weather) > 0:
        print(f"\nWeather DataFrame shape: {weather.shape}")
        print(f"Columns ({len(weather.columns)}):")
        for col in weather.columns:
            print(f"  - {col}")
        print("\nWeather summary:")
        print(weather.describe())
    else:
        print("\nNo weather data available for this session")

    # 5. DRIVERS
    print("\n" + "="*70)
    print("5. DRIVERS IN SESSION")
    print("="*70)

    drivers = session.drivers
    print(f"\nTotal drivers: {len(drivers)}")
    print(f"Driver numbers: {drivers}")

    # Get driver info
    print("\nDriver details:")
    for driver_num in drivers[:3]:  # Show first 3
        driver = session.get_driver(driver_num)
        print(f"\n  Driver {driver_num}:")
        print(f"    Name: {driver['FullName']}")
        print(f"    Team: {driver['TeamName']}")
        print(f"    Abbreviation: {driver['Abbreviation']}")

    return session, results, laps, telemetry, weather


def compare_two_drivers(session, driver1='VER', driver2='LEC'):
    """Compare telemetry between two drivers"""
    print("\n" + "="*70)
    print(f"COMPARING DRIVERS: {driver1} vs {driver2}")
    print("="*70)

    # Get fastest laps for each driver
    lap1 = session.laps.pick_driver(driver1).pick_fastest()
    lap2 = session.laps.pick_driver(driver2).pick_fastest()

    print(f"\n{driver1} fastest lap: {lap1['LapTime']}")
    print(f"{driver2} fastest lap: {lap2['LapTime']}")

    # Get telemetry
    tel1 = lap1.get_car_data().add_distance()
    tel2 = lap2.get_car_data().add_distance()

    print(f"\n{driver1} telemetry points: {len(tel1)}")
    print(f"{driver2} telemetry points: {len(tel2)}")

    # Speed comparison
    print(f"\n{driver1} max speed: {tel1['Speed'].max():.1f} km/h")
    print(f"{driver2} max speed: {tel2['Speed'].max():.1f} km/h")

    print(f"\n{driver1} avg speed: {tel1['Speed'].mean():.1f} km/h")
    print(f"{driver2} avg speed: {tel2['Speed'].mean():.1f} km/h")

    return tel1, tel2


def extract_car_performance_features(session):
    """Extract aggregated car performance features for ML"""
    print("\n" + "="*70)
    print("EXTRACTING CAR PERFORMANCE FEATURES FOR ML")
    print("="*70)

    features_list = []

    for driver in session.drivers:
        driver_laps = session.laps.pick_driver(driver)

        if len(driver_laps) == 0:
            continue

        # Get driver info
        driver_info = session.get_driver(driver)

        # Get fastest lap
        try:
            fastest = driver_laps.pick_fastest()
            fastest_time = fastest['LapTime'].total_seconds() if pd.notna(fastest['LapTime']) else None
        except:
            fastest_time = None

        # Calculate features
        features = {
            'driver_number': driver,
            'driver_name': driver_info['FullName'],
            'team': driver_info['TeamName'],
            'fastest_lap_time': fastest_time,
            'total_laps': len(driver_laps),
            'avg_lap_time': driver_laps['LapTime'].mean().total_seconds() if len(driver_laps) > 0 else None,
        }

        # Telemetry-based features (from fastest lap)
        if fastest_time is not None:
            try:
                telemetry = fastest.get_car_data()
                features.update({
                    'max_speed': telemetry['Speed'].max(),
                    'avg_speed': telemetry['Speed'].mean(),
                    'max_throttle': telemetry['Throttle'].max(),
                    'avg_throttle': telemetry['Throttle'].mean(),
                    'max_brake': telemetry['Brake'].max() if 'Brake' in telemetry.columns else None,
                    'max_rpm': telemetry['RPM'].max(),
                    'avg_rpm': telemetry['RPM'].mean(),
                })
            except Exception as e:
                print(f"  Warning: Could not get telemetry for {driver}: {e}")

        features_list.append(features)

    df = pd.DataFrame(features_list)

    print(f"\nExtracted features for {len(df)} drivers")
    print(f"\nFeature columns:")
    for col in df.columns:
        print(f"  - {col}")

    print(f"\nSample data:")
    print(df.head())

    return df


if __name__ == "__main__":
    print("FastF1 Data Exploration")
    print("="*70)

    # Explore a qualifying session
    session, results, laps, telemetry, weather = explore_session_data(
        year=2024,
        event='Monaco',
        session_type='Q'
    )

    # Compare two drivers
    try:
        tel1, tel2 = compare_two_drivers(session, 'VER', 'LEC')
    except Exception as e:
        print(f"\nCould not compare drivers: {e}")

    # Extract ML features
    car_features = extract_car_performance_features(session)

    # Save to CSV
    output_file = 'monaco_2024_qualifying_features.csv'
    car_features.to_csv(output_file, index=False)
    print(f"\n✓ Features saved to: {output_file}")

    print("\n" + "="*70)
    print("Exploration complete!")
    print("="*70)
