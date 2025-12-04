"""Quick test script to verify telemetry collection works"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'models'))

from models.scraping.generate_driver_telemetry_features import collect_and_aggregate_telemetry
from models.scraping.driver_mapping import get_fastf1_driver_code

# Test parameters (using dataset format)
test_year = 2023
test_circuit = 'melbourne'  # Has both FastF1 and raceline data
test_driver_name = 'verstappen'  # Dataset format
test_driver_code = get_fastf1_driver_code(test_driver_name)

print("=" * 70)
print("QUICK TELEMETRY TEST")
print("=" * 70)
print(f"Testing: {test_driver_name} -> {test_driver_code} @ {test_circuit} ({test_year})")
print()

try:
    features = collect_and_aggregate_telemetry(
        year=test_year,
        circuit_id=test_circuit,
        driver_code=test_driver_code
    )
    
    if features is None:
        print("❌ FAILED: collect_and_aggregate_telemetry returned None")
        print("   Possible reasons:")
        print("   - Session data not available")
        print("   - Track segmentation failed")
        print("   - Driver didn't have telemetry data")
    else:
        print("✅ SUCCESS! Collected telemetry features:")
        print()
        for key, value in features.items():
            print(f"   {key:35} = {value:.2f}" if isinstance(value, (int, float)) else f"   {key:35} = {value}")
        
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)

