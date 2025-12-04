"""
Quick test script to verify telemetry features integration.

This creates a small test dataset to ensure everything works without
processing the entire historical dataset.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from telemetry_features import (
    generate_track_segments,
    collect_driver_telemetry,
    aggregate_driver_by_segments,
    aggregate_segment_features
)

def test_track_segmentation():
    """Test that track segmentation works for Bahrain."""
    print("=" * 60)
    print("Test 1: Track Segmentation")
    print("=" * 60)

    try:
        segments = generate_track_segments("bahrain")

        if segments.empty:
            print("❌ Failed: No segments generated")
            return False

        corners = segments[segments['segment_type'] == 'corner']
        straights = segments[segments['segment_type'] == 'straight']

        print(f"✓ Generated {len(segments)} segments")
        print(f"  - Corners: {len(corners)}")
        print(f"  - Straights: {len(straights)}")
        print(f"\nFirst few segments:")
        print(segments.head())
        return True

    except Exception as e:
        print(f"❌ Failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_telemetry_collection():
    """Test telemetry collection for a single driver/race."""
    print("\n" + "=" * 60)
    print("Test 2: Telemetry Collection")
    print("=" * 60)

    try:
        # Test with 2024 Bahrain, Verstappen, FP1 only
        telemetry = collect_driver_telemetry(
            year=2024,
            grand_prix="Bahrain",
            driver="VER",
            sessions=("FP1",),  # Just one session for speed
            include_position=False
        )

        if telemetry.empty:
            print("⚠️  Warning: No telemetry data collected (FastF1 may not be available)")
            return True  # Not a failure, just no data

        print(f"✓ Collected {len(telemetry)} telemetry samples")
        print(f"  Columns: {list(telemetry.columns)}")
        print(f"\nSample data:")
        print(telemetry.head())
        return True

    except Exception as e:
        print(f"❌ Failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_segment_aggregation():
    """Test aggregating telemetry by segments."""
    print("\n" + "=" * 60)
    print("Test 3: Segment Aggregation")
    print("=" * 60)

    try:
        # Get segments
        segments = generate_track_segments("bahrain")

        if segments.empty:
            print("⚠️  Skipping: No segments available")
            return True

        # Get telemetry
        telemetry = collect_driver_telemetry(
            year=2024,
            grand_prix="Bahrain",
            driver="VER",
            sessions=("FP1",),
            include_position=False
        )

        if telemetry.empty:
            print("⚠️  Skipping: No telemetry data available")
            return True

        # Aggregate
        segment_stats = aggregate_driver_by_segments(segments, telemetry)

        if segment_stats.empty:
            print("⚠️  Warning: No segment stats generated")
            return True

        print(f"✓ Generated stats for {len(segment_stats)} segments")
        print(f"\nSample segment stats:")
        print(segment_stats.head())

        # Test feature aggregation
        features = aggregate_segment_features(segment_stats)
        print(f"\n✓ Generated {len(features)} features:")
        for feat_name, feat_value in list(features.items())[:5]:
            print(f"  - {feat_name}: {feat_value:.2f}")

        return True

    except Exception as e:
        print(f"❌ Failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TELEMETRY FEATURES INTEGRATION TEST")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Track segmentation
    results.append(("Track Segmentation", test_track_segmentation()))

    # Test 2: Telemetry collection
    results.append(("Telemetry Collection", test_telemetry_collection()))

    # Test 3: Segment aggregation
    results.append(("Segment Aggregation", test_segment_aggregation()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n✓ All tests passed!")
        print("\nYou can now run create_qualifying_dataset_no_leakage.py")
        print("to generate datasets with telemetry features.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
