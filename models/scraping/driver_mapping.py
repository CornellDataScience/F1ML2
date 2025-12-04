"""
Driver Name Mapping
Maps driver names from HOLY_qualifying_v1.csv to FastF1 3-letter codes
"""

# Mapping from dataset driver names (lowercase last names) to FastF1 3-letter codes
DRIVER_NAME_TO_CODE = {
    # Current drivers (2018-2024)
    'verstappen': 'VER',
    'hamilton': 'HAM',
    'leclerc': 'LEC',
    'perez': 'PER',
    'sainz': 'SAI',
    'russell': 'RUS',
    'norris': 'NOR',
    'alonso': 'ALO',
    'ocon': 'OCO',
    'gasly': 'GAS',
    'stroll': 'STR',
    'bottas': 'BOT',
    'zhou': 'ZHO',
    'magnussen': 'MAG',
    'hulkenberg': 'HUL',
    'tsunoda': 'TSU',
    'albon': 'ALB',
    'ricciardo': 'RIC',
    'piastri': 'PIA',
    'lawson': 'LAW',
    'sargeant': 'SAR',
    'de_vries': 'DEV',
    
    # Recent drivers (2018-2022)
    'vettel': 'VET',
    'raikkonen': 'RAI',
    'grosjean': 'GRO',
    'giovinazzi': 'GIO',
    'latifi': 'LAT',
    'mazepin': 'MAZ',
    'schumacher': 'MSC',
    'kubica': 'KUB',
    'kvyat': 'KVY',
    'ericsson': 'ERI',
    'vandoorne': 'VAN',
    'hartley': 'HAR',
    'sirotkin': 'SIR',
    'leclerc': 'LEC',
    'gasly': 'GAS',
    'albon': 'ALB',
    
    # Alternative names/spellings
    'max_verstappen': 'VER',
    'lewis_hamilton': 'HAM',
    'charles_leclerc': 'LEC',
    'sergio_perez': 'PER',
    'carlos_sainz': 'SAI',
    'george_russell': 'RUS',
    'lando_norris': 'NOR',
    'fernando_alonso': 'ALO',
}

def get_fastf1_driver_code(driver_name: str) -> str:
    """
    Get FastF1 3-letter driver code from dataset driver name.
    
    Args:
        driver_name: Driver name from dataset (e.g., 'hamilton', 'verstappen', 'alo', 'VER')
    
    Returns:
        FastF1 3-letter code (e.g., 'HAM', 'VER') or None if not found
    """
    driver_str = str(driver_name).strip()
    
    # If already a 3-letter code (upper or lower), convert to uppercase and return
    if len(driver_str) == 3 and driver_str.isalpha():
        return driver_str.upper()
    
    # Otherwise look up in mapping
    driver_lower = driver_str.lower()
    return DRIVER_NAME_TO_CODE.get(driver_lower)


if __name__ == '__main__':
    """Test the mapping"""
    print("=== Driver Mapping Test ===\n")
    
    test_drivers = ['hamilton', 'verstappen', 'leclerc', 'alonso', 'norris']
    
    for driver in test_drivers:
        code = get_fastf1_driver_code(driver)
        print(f"{driver:15} -> {code if code else 'NOT FOUND'}")
    
    print(f"\nTotal mappings: {len(DRIVER_NAME_TO_CODE)}")

