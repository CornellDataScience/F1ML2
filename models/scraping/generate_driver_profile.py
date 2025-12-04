"""
Generate driver telemetry profiles (Option A approach).
Aggregates each driver's telemetry across ALL circuits in a year to create a general profile.
Uses year-1 data for predictions (no data leakage).

Output: data/driver_telemetry_profile.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
import fastf1
from typing import List, Dict
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'models'))

from models.scraping.track_mapping import (
    get_fastf1_event_name,
    get_raceline_url,
    has_fastf1_data,
    has_raceline,
    CIRCUIT_TO_FASTF1
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


def collect_driver_circuit_telemetry(year: int, circuit_id: str, driver_code: str) -> dict:
    """
    Collect telemetry for one driver at one circuit in one year.
    Returns dict with telemetry features or None if unavailable.
    """
    fastf1_event = get_fastf1_event_name(circuit_id)
    raceline_url = get_raceline_url(circuit_id)
    
    if fastf1_event is None or raceline_url is None:
        return None
    
    try:
        # Generate track segments
        segments = generate_track_segments(
            track_name=circuit_id,
            raceline_url=raceline_url,
            thr_quantile=0.70,
            min_corner_length=8.0,
            merge_distance=15.0,
            include_straights=True
        )
        
        if segments.empty:
            return None
        
        # Aggregate driver telemetry by segments
        segment_stats = aggregate_driver_by_segments(
            segments_df=segments,
            track_name=fastf1_event,
            year=year,
            driver=driver_code,
            sessions=('Q',),
            include_position=False
        )
        
        if segment_stats.empty:
            return None
        
        # Group by segment type
        corners = segment_stats[segment_stats['segment_type'] == 'corner']
        straights = segment_stats[segment_stats['segment_type'] == 'straight']
        
        features = {'circuit_id': circuit_id}
        
        # Corner features
        if not corners.empty:
            features['corner_speed'] = corners['avg_speed_kmh'].mean()
            features['min_corner_speed'] = corners['min_speed_kmh'].min()
            features['corner_throttle'] = corners['avg_throttle'].mean()
            features['corner_brake'] = corners['brake_percentage'].mean()
            features['corner_rpm'] = corners['avg_rpm'].mean()
        
        # Straight features
        if not straights.empty:
            features['straight_speed'] = straights['avg_speed_kmh'].mean()
            features['top_speed'] = straights['max_speed_kmh'].max()
            features['straight_throttle'] = straights['avg_throttle'].mean()
            features['drs_usage'] = straights['drs_usage'].mean()
        
        return features
        
    except Exception as e:
        return None


def aggregate_driver_yearly_profile(driver_circuit_data: List[dict]) -> dict:
    """
    Aggregate a driver's telemetry across all circuits into a yearly profile.
    Takes the mean of all circuit-specific values.
    """
    if not driver_circuit_data:
        return None
    
    df = pd.DataFrame(driver_circuit_data)
    
    profile = {
        'circuits_sampled': len(df),
        'driver_avg_corner_speed': df['corner_speed'].mean() if 'corner_speed' in df else np.nan,
        'driver_min_corner_speed': df['min_corner_speed'].mean() if 'min_corner_speed' in df else np.nan,
        'driver_corner_throttle': df['corner_throttle'].mean() if 'corner_throttle' in df else np.nan,
        'driver_corner_brake': df['corner_brake'].mean() if 'corner_brake' in df else np.nan,
        'driver_corner_rpm': df['corner_rpm'].mean() if 'corner_rpm' in df else np.nan,
        'driver_avg_straight_speed': df['straight_speed'].mean() if 'straight_speed' in df else np.nan,
        'driver_top_speed': df['top_speed'].max() if 'top_speed' in df else np.nan,  # Max of maxes
        'driver_straight_throttle': df['straight_throttle'].mean() if 'straight_throttle' in df else np.nan,
        'driver_drs_usage': df['drs_usage'].mean() if 'drs_usage' in df else np.nan,
    }
    
    return profile


def generate_all_driver_profiles(start_year: int = 2018, end_year: int = 2023, output_path: str = None):
    """
    Generate driver telemetry profiles for all drivers across all years.
    Each row = one driver's aggregated profile for one year.
    """
    if output_path is None:
        output_path = PROJECT_ROOT / 'data' / 'driver_telemetry_profile.csv'
    
    # Load qualifying dataset to get list of drivers per year
    quali_df = pd.read_csv(PROJECT_ROOT / 'data' / 'HOLY_qualifying_v1.csv')
    
    # Get circuits with both FastF1 and raceline data
    circuits_with_data = [
        cid for cid in CIRCUIT_TO_FASTF1.keys() 
        if has_fastf1_data(cid) and has_raceline(cid)
    ]
    print(f"Circuits with both FastF1 and raceline data: {len(circuits_with_data)}")
    
    all_profiles = []
    
    for year in range(start_year, end_year + 1):
        print(f"\nProcessing {year}...")
        
        # Get drivers who raced this year
        year_df = quali_df[quali_df['season'] == year]
        drivers_this_year = year_df['driver'].unique()
        
        print(f"  {len(drivers_this_year)} drivers")
        
        for driver in tqdm(drivers_this_year, desc=f"  {year} drivers"):
            # Convert driver name to FastF1 code
            driver_code = get_fastf1_driver_code(driver)
            if driver_code is None:
                continue
            
            # Collect telemetry from all circuits
            circuit_data = []
            for circuit_id in circuits_with_data:
                # Check if this driver raced at this circuit this year
                if not ((year_df['driver'] == driver) & (year_df['circuit_id'] == circuit_id)).any():
                    continue
                
                telemetry = collect_driver_circuit_telemetry(year, circuit_id, driver_code)
                if telemetry:
                    circuit_data.append(telemetry)
            
            if circuit_data:
                # Aggregate into yearly profile
                profile = aggregate_driver_yearly_profile(circuit_data)
                if profile:
                    profile['driver'] = driver
                    profile['season'] = year
                    all_profiles.append(profile)
                    
        print(f"  Collected {len([p for p in all_profiles if p['season'] == year])} driver profiles")
    
    # Save profiles
    profiles_df = pd.DataFrame(all_profiles)
    
    # Reorder columns
    cols = ['driver', 'season', 'circuits_sampled'] + [c for c in profiles_df.columns if c not in ['driver', 'season', 'circuits_sampled']]
    profiles_df = profiles_df[cols]
    
    profiles_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(profiles_df)} driver profiles to {output_path}")
    
    return profiles_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-year', type=int, default=2018)
    parser.add_argument('--end-year', type=int, default=2023)
    parser.add_argument('--test', action='store_true', help='Quick test with one driver')
    args = parser.parse_args()
    
    if args.test:
        print("Quick test mode: testing one driver at one circuit...")
        result = collect_driver_circuit_telemetry(2023, 'bahrain', 'VER')
        print(f"Result: {result}")
    else:
        profiles_df = generate_all_driver_profiles(args.start_year, args.end_year)

