"""
Test script to demonstrate loading F1 circuit data from GeoJSON format.
Works with bacinger/f1-circuits repository format (FeatureCollection with LineString).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'models'))

from track_segmentation_utils import load_raceline_geojson, generate_track_segments

# Example: Test with a GeoJSON URL
# Replace this with actual bacinger repository URL
test_geojson_url = "https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/ae-2009.geojson"

try:
    print(f"Loading GeoJSON from: {test_geojson_url}")
    raceline_df = load_raceline_geojson(test_geojson_url)
    print(f"  Loaded {len(raceline_df)} points")
    print(f"  X range: {raceline_df['x_m'].min():.1f} to {raceline_df['x_m'].max():.1f}")
    print(f"  Y range: {raceline_df['y_m'].min():.1f} to {raceline_df['y_m'].max():.1f}")
    
    # Generate track segments
    print("\nGenerating track segments...")
    print("  Step 1: Loading raceline...")
    segments = generate_track_segments(track_name='yas_marina', raceline_url=test_geojson_url)
    print("  Step 2: Segments generated!")
    
    # Calculate statistics from the DataFrame
    corners = segments[segments['segment_type'] == 'corner']
    straights = segments[segments['segment_type'] == 'straight']
    
    print(f"Track characteristics:")
    print(f"  Total segments: {len(segments)}")
    print(f"  Total corners: {len(corners)}")
    print(f"  Total straights: {len(straights)}")
    if len(corners) > 0:
        print(f"  Avg corner length: {corners['length_m'].mean():.1f}m")
    if len(straights) > 0:
        print(f"  Longest straight: {straights['length_m'].max():.1f}m")
        print(f"  Avg straight length: {straights['length_m'].mean():.1f}m")
    
    print("\nSuccess! GeoJSON format works perfectly.")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nNote: Replace the URL with actual bacinger repository path")

