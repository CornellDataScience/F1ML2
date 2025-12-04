"""
Generate track geometry features from raceline data.
Output: data/track_characteristics.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.scraping.track_mapping import (
    get_raceline_url,
    get_circuits_with_racelines,
    CIRCUIT_TO_RACELINE_URL
)

# Import track segmentation utilities (now centralized!)
from models.track_segmentation_utils import generate_track_segments


def extract_track_features(circuit_id: str, segments_df: pd.DataFrame) -> dict:
    """Extract aggregate features from track segments."""
    corners = segments_df[segments_df['segment_type'] == 'corner']
    straights = segments_df[segments_df['segment_type'] == 'straight']
    
    total_length = segments_df['length_m'].sum()
    corner_length = corners['length_m'].sum() if len(corners) > 0 else 0
    
    features = {
        'circuit_id': circuit_id,
        'track_total_corners': len(corners),
        'track_total_straights': len(straights),
        'track_corner_percentage': (corner_length / total_length * 100) if total_length > 0 else 0,
        'track_avg_corner_length': corners['length_m'].mean() if len(corners) > 0 else 0,
        'track_avg_straight_length': straights['length_m'].mean() if len(straights) > 0 else 0,
        'track_longest_straight': straights['length_m'].max() if len(straights) > 0 else 0,
        'track_total_length': total_length,
        'track_shortest_corner': corners['length_m'].min() if len(corners) > 0 else 0,
        'track_longest_corner': corners['length_m'].max() if len(corners) > 0 else 0,
    }
    
    return features


def generate_all_track_features(output_path: str = None):
    """Generate track features for all circuits with raceline data."""
    if output_path is None:
        output_path = PROJECT_ROOT / 'data' / 'track_characteristics.csv'
    
    circuits_with_racelines = get_circuits_with_racelines()
    
    print(f"Generating track features for {len(circuits_with_racelines)} circuits...")
    
    all_features = []
    failed_circuits = []
    
    for i, circuit_id in enumerate(circuits_with_racelines, 1):
        raceline_url = get_raceline_url(circuit_id)
        
        try:
            print(f"[{i}/{len(circuits_with_racelines)}] {circuit_id}...", end=' ')
            
            segments = generate_track_segments(
                track_name=circuit_id,
                raceline_url=raceline_url,
                thr_quantile=0.70,  # Lower threshold for GeoJSON tracks (detect top 30% curvature as corners)
                min_corner_length=8.0,  # Slightly smaller min length
                merge_distance=15.0  # Smaller merge distance
            )
            
            features = extract_track_features(circuit_id, segments)
            all_features.append(features)
            
            print(f"done ({features['track_total_corners']} corners)")
            
        except Exception as e:
            print(f"failed: {str(e)}")
            failed_circuits.append((circuit_id, str(e)))
    
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(output_path, index=False)
    
    print(f"\nSaved to: {output_path}")
    print(f"Processed {len(all_features)} circuits, {len(failed_circuits)} failed")
    
    if failed_circuits:
        print("\nFailed:")
        for circuit_id, error in failed_circuits:
            print(f"  {circuit_id}: {error}")
    
    return features_df


if __name__ == '__main__':
    features_df = generate_all_track_features()


