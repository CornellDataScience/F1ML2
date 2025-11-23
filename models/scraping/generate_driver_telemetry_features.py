"""
Generate driver telemetry features from qualifying sessions (2018+).
Uses rolling historical averages to avoid data leakage.
Output: data/driver_telemetry_features.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
import fastf1
from typing import List
from tqdm import tqdm
import time

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'models'))

from models.scraping.track_mapping import (
    get_fastf1_event_name,
    get_raceline_url,
    has_fastf1_data,
    has_raceline
)
from models.scraping.driver_mapping import get_fastf1_driver_code

from models.track_segmentation_utils import (
    generate_track_segments,
    aggregate_driver_by_segments
)

# Configure FastF1 cache
CACHE_DIR = PROJECT_ROOT / 'f1_cache'
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


def collect_and_aggregate_telemetry(
    year: int,
    circuit_id: str,
    driver_code: str,
) -> dict:
    """
    Collect telemetry and aggregate by track segments (corners vs straights).
    
    Uses the track segmentation approach from the notebook:
    1. Generate track segments (corners and straights)
    2. Collect driver telemetry
    3. Aggregate by segment type
    
    Returns dictionary with aggregated features or None if data unavailable.
    """
    # Get FastF1 event name and raceline URL
    fastf1_event = get_fastf1_event_name(circuit_id)
    raceline_url = get_raceline_url(circuit_id)
    
    if fastf1_event is None or raceline_url is None:
        return None
    
    try:
        # Step 1: Generate track segments
        segments = generate_track_segments(
            track_name=circuit_id,
            raceline_url=raceline_url,
            include_straights=True
        )
        
        if segments.empty:
            return None
        
        # Step 2 & 3: Collect telemetry and aggregate by segments
        segment_stats = aggregate_driver_by_segments(
            segments_df=segments,
            track_name=fastf1_event,
            year=year,
            driver=driver_code,
            sessions=('Q',),  # Qualifying only
            include_position=False
        )
        
        if segment_stats.empty:
            return None
        
        # Step 4: Group by segment type to get corner vs straight features
        corners = segment_stats[segment_stats['segment_type'] == 'corner']
        straights = segment_stats[segment_stats['segment_type'] == 'straight']
        
        features = {}
        
        # Corner features
        if not corners.empty:
            features['driver_avg_corner_speed'] = corners['avg_speed_kmh'].mean()
            features['driver_min_corner_speed'] = corners['min_speed_kmh'].min()
            features['driver_corner_throttle'] = corners['avg_throttle'].mean()
            features['driver_corner_brake'] = corners['brake_percentage'].mean()
            features['driver_corner_rpm'] = corners['avg_rpm'].mean()
        else:
            features['driver_avg_corner_speed'] = np.nan
            features['driver_min_corner_speed'] = np.nan
            features['driver_corner_throttle'] = np.nan
            features['driver_corner_brake'] = np.nan
            features['driver_corner_rpm'] = np.nan
        
        # Straight features
        if not straights.empty:
            features['driver_avg_straight_speed'] = straights['avg_speed_kmh'].mean()
            features['driver_top_speed'] = straights['max_speed_kmh'].max()
            features['driver_straight_throttle'] = straights['avg_throttle'].mean()
            features['driver_drs_usage'] = straights['drs_usage'].mean()
        else:
            features['driver_avg_straight_speed'] = np.nan
            features['driver_top_speed'] = np.nan
            features['driver_straight_throttle'] = np.nan
            features['driver_drs_usage'] = np.nan
        
        return features
        
    except Exception as e:
        # Session not available, track segmentation failed, or other error
        return None


def generate_driver_telemetry_for_year(
    year: int,
    quali_df: pd.DataFrame
) -> List[dict]:
    """
    Generate telemetry features for all driver-circuit combinations in a given year.
    
    Uses track segmentation to aggregate telemetry by corners vs straights.
    
    Args:
        year: Year to process
        quali_df: DataFrame with qualifying data (to get driver-circuit combinations)
    
    Returns:
        List of feature dictionaries
    """
    # Get unique driver-circuit combinations for this year
    year_data = quali_df[quali_df['season'] == year]
    combinations = year_data[['driver', 'circuit_id']].drop_duplicates()
    
    print(f"\nProcessing {year}: {len(combinations)} driver-circuit combinations")
    
    features_list = []
    success_count = 0
    
    for idx, row in tqdm(combinations.iterrows(), total=len(combinations), desc=f"Year {year}"):
        driver_name = row['driver']
        circuit_id = row['circuit_id']
        
        # Check if we have both FastF1 data and raceline for this circuit
        if not has_fastf1_data(circuit_id) or not has_raceline(circuit_id):
            continue
        
        # Convert driver name to FastF1 3-letter code
        # Handle both formats: '2018-2022 has 3-letter codes, 2023+ has full names
        if len(driver_name) == 3:
            # Already a 3-letter code (2018-2022 format)
            driver_code = driver_name.upper()
        else:
            # Full name, need to map (2023+ format)
            driver_code = get_fastf1_driver_code(driver_name)
            if driver_code is None:
                continue  # Skip if we don't have a mapping for this driver
        
        # Collect and aggregate telemetry using track segmentation
        features = collect_and_aggregate_telemetry(year, circuit_id, driver_code)
        
        if features is None:
            continue
        
        features['driver'] = driver_name  # Use original dataset driver name
        features['circuit_id'] = circuit_id
        features['season'] = year
        
        features_list.append(features)
        success_count += 1
        
        # Small delay to be nice to FastF1 API
        time.sleep(0.1)
    
    print(f"  Processed {success_count} combinations")
    
    return features_list


def create_rolling_historical_features(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling historical averages to avoid data leakage.
    
    For each driver-circuit-year combination, compute average of all previous years
    at that circuit (not including current year).
    """
    print("\nCreating rolling historical features...")
    
    # Sort by driver, circuit, and year
    telemetry_df = telemetry_df.sort_values(['driver', 'circuit_id', 'season']).reset_index(drop=True)
    
    # Feature columns to average
    feature_cols = [
        'driver_avg_corner_speed', 'driver_min_corner_speed', 'driver_corner_throttle',
        'driver_corner_brake', 'driver_corner_rpm', 'driver_avg_straight_speed',
        'driver_top_speed', 'driver_straight_throttle', 'driver_drs_usage'
    ]
    
    historical_features = []
    
    # Group by driver and circuit
    grouped = telemetry_df.groupby(['driver', 'circuit_id'])
    
    for (driver, circuit), group in tqdm(grouped, desc="Computing rolling averages"):
        group = group.sort_values('season')
        
        for idx in range(len(group)):
            row = group.iloc[idx]
            year = row['season']
            
            # Get all previous years at this circuit for this driver
            previous_years = group[group['season'] < year]
            
            if len(previous_years) == 0:
                # No historical data, use current year as baseline
                historical_features.append(row.to_dict())
            else:
                # Average all previous years
                hist_row = {
                    'driver': driver,
                    'circuit_id': circuit,
                    'season': year,
                }
                
                for col in feature_cols:
                    if col in previous_years.columns:
                        hist_row[col] = previous_years[col].mean()
                    else:
                        hist_row[col] = np.nan
                
                historical_features.append(hist_row)
    
    return pd.DataFrame(historical_features)


def generate_all_driver_telemetry(
    start_year: int = 2018,
    end_year: int = 2023,
    output_path: str = None
):
    """Generate driver telemetry features for all years using track segmentation."""
    if output_path is None:
        output_path = PROJECT_ROOT / 'data' / 'driver_telemetry_features.csv'
    
    print("Generating driver telemetry features...")
    
    quali_df = pd.read_csv(PROJECT_ROOT / 'data' / 'HOLY_qualifying_v1.csv')
    print(f"Loaded {len(quali_df)} records, processing {start_year}-{end_year}")
    
    # Generate telemetry features year by year
    all_features = []
    
    for year in range(start_year, end_year + 1):
        year_features = generate_driver_telemetry_for_year(year, quali_df)
        all_features.extend(year_features)
    
    if not all_features:
        print("\nWarning: No telemetry data collected - FastF1 API may be unavailable")
        print("Creating empty telemetry file...")
        telemetry_df = pd.DataFrame(columns=[
            'driver', 'circuit_id', 'season',
            'driver_avg_corner_speed', 'driver_min_corner_speed', 'driver_corner_throttle',
            'driver_corner_brake', 'driver_corner_rpm', 'driver_avg_straight_speed',
            'driver_top_speed', 'driver_straight_throttle', 'driver_drs_usage'
        ])
        telemetry_df.to_csv(output_path, index=False)
        print(f"Empty file saved to: {output_path}")
        return telemetry_df
    
    telemetry_df = pd.DataFrame(all_features)
    print(f"\nCollected {len(telemetry_df)} telemetry records")
    
    # Create rolling historical features (to avoid data leakage)
    historical_df = create_rolling_historical_features(telemetry_df)
    historical_df.to_csv(output_path, index=False)
    
    print(f"\nSaved to: {output_path}")
    print(f"  {len(historical_df)} records, {historical_df['driver'].nunique()} drivers, {historical_df['circuit_id'].nunique()} circuits")
    
    return historical_df


if __name__ == '__main__':
    # Generate telemetry for all years with FastF1 data (2018-2023)
    telemetry_df = generate_all_driver_telemetry(start_year=2018, end_year=2023)


