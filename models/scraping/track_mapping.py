"""
Track Mapping for Circuit IDs to FastF1 Event Names

Maps circuit_id values from HOLY_qualifying_v1.csv to:
1. FastF1 event names (for telemetry collection)
2. TUM FTM Raceline URLs (for track geometry)

Note: Some circuits may not have raceline data or FastF1 data available.
"""

# Mapping from circuit_id to FastF1 event name
# FastF1 event names are case-insensitive but we use official capitalization
CIRCUIT_TO_FASTF1 = {
    # Modern F1 circuits (2018+)
    'bahrain': 'Bahrain',
    'sakhir': 'Sakhir',  # Also Bahrain (outer circuit used in 2020)
    'jeddah': 'Jeddah',
    'albert_park': 'Australia',
    'melbourne': 'Australia',
    'imola': 'Emilia Romagna',
    'miami': 'Miami',
    'monaco': 'Monaco',
    'barcelona': 'Spain',
    'catalunya': 'Spain',
    'montreal': 'Canada',
    'montréal': 'Canada',
    'villeneuve': 'Canada',
    'red_bull_ring': 'Austria',
    'spielberg': 'Austria',
    'silverstone': 'Great Britain',
    'hungaroring': 'Hungary',
    'budapest': 'Hungary',
    'spa': 'Belgium',
    'spa_francorchamps': 'Belgium',
    'zandvoort': 'Netherlands',
    'monza': 'Italy',
    'marina_bay': 'Singapore',
    'suzuka': 'Japan',
    'losail': 'Qatar',
    'lusail': 'Qatar',  # Alternative spelling used in dataset
    'austin': 'United States',
    'americas': 'United States',
    'mexico_city': 'Mexico',
    'rodriguez': 'Mexico',
    'interlagos': 'Brazil',
    'são_paulo': 'São Paulo',
    'las_vegas': 'Las Vegas',
    'yas_marina': 'Abu Dhabi',
    'baku': 'Azerbaijan',
    'shanghai': 'China',
    'portimao': 'Portugal',
    'istanbul': 'Turkey',
    'mugello': 'Tuscany',
    'sochi': 'Russia',
    
    # Historic circuits (limited/no FastF1 data)
    'hockenheimring': 'Germany',
    'nurburgring': 'Nurburgring',
    'ricard': 'France',
    'magny_cours': 'France',
    'sepang': 'Malaysia',
    'yeongam': 'Korea',
    'buddh': 'India',
    'valencia': 'Europe',
    
    # Very old circuits (pre-2000, likely no FastF1 data)
    'adelaide': None,
    'brands_hatch': None,
    'dallas': None,
    'detroit': None,
    'dijon': None,
    'donington': None,
    'estoril': None,
    'fuji': None,
    'galvez': None,
    'indianapolis': None,
    'jacarepagua': None,
    'jerez': None,
    'kyalami': None,
    'long_beach': None,
    'okayama': None,
    'phoenix': None,
    'zolder': None,
}

# Mapping from circuit_id to TUM FTM Raceline URL
# Source: https://github.com/TUMFTM/racetrack-database
CIRCUIT_TO_RACELINE_URL = {
    # Circuits with confirmed raceline data in TUM database (25 total)
    'bahrain': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Sakhir.csv',
    'sakhir': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Sakhir.csv',
    'melbourne': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Melbourne.csv',
    'albert_park': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Melbourne.csv',
    'shanghai': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Shanghai.csv',
    'barcelona': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Catalunya.csv',
    'catalunya': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Catalunya.csv',
    'montreal': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Montreal.csv',
    'montréal': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Montreal.csv',
    'villeneuve': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Montreal.csv',
    'silverstone': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Silverstone.csv',
    'hockenheimring': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Hockenheim.csv',
    'hungaroring': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Budapest.csv',
    'budapest': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Budapest.csv',
    'spa': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Spa.csv',
    'spa_francorchamps': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Spa.csv',
    'monza': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Monza.csv',
    'suzuka': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Suzuka.csv',
    'sochi': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Sochi.csv',
    'austin': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Austin.csv',
    'americas': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Austin.csv',
    'mexico_city': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/MexicoCity.csv',
    'rodriguez': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/MexicoCity.csv',
    'interlagos': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/SaoPaulo.csv',
    'são_paulo': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/SaoPaulo.csv',
    'yas_marina': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/YasMarina.csv',
    'red_bull_ring': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Spielberg.csv',
    'spielberg': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Spielberg.csv',
    'sepang': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Sepang.csv',
    'zandvoort': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Zandvoort.csv',
    'nurburgring': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/Nuerburgring.csv',
    'brands_hatch': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/BrandsHatch.csv',
    'indianapolis': 'https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/racelines/IMS.csv',
    
    # Modern circuits without raceline data in TUM database
    'monaco': None,  # Not available
    'imola': None,  # Not available
    'miami': None,  # Not available
    'baku': None,  # Not available
    'marina_bay': None,  # Singapore - Not available
    'portimao': None,  # Not available
    'istanbul': None,  # Not available
    'jeddah': None,  # Not available
    'losail': None,  # Qatar - Not available
    'lusail': None,  # Qatar (alternative spelling) - Not available
    'las_vegas': None,  # Not available
    
    # Historic circuits without raceline data
    'ricard': None,
    'magny_cours': None,
    'yeongam': None,
    'buddh': None,
    'valencia': None,
    'mugello': None,
    
    # Very old circuits - no racelines available
    'adelaide': None,
    'dallas': None,
    'detroit': None,
    'dijon': None,
    'donington': None,
    'estoril': None,
    'fuji': None,
    'galvez': None,
    'jacarepagua': None,
    'jerez': None,
    'kyalami': None,
    'long_beach': None,
    'okayama': None,
    'phoenix': None,
    'zolder': None,
}


def get_fastf1_event_name(circuit_id: str) -> str:
    """
    Get FastF1 event name for a circuit_id.
    
    Args:
        circuit_id: Circuit identifier from dataset (e.g., 'monaco', 'bahrain')
    
    Returns:
        FastF1 event name (e.g., 'Monaco', 'Bahrain') or None if not available
    """
    return CIRCUIT_TO_FASTF1.get(circuit_id.lower())


def get_raceline_url(circuit_id: str) -> str:
    """
    Get TUM FTM raceline URL for a circuit_id.
    
    Args:
        circuit_id: Circuit identifier from dataset (e.g., 'monaco', 'bahrain')
    
    Returns:
        URL to raceline CSV or None if not available
    """
    return CIRCUIT_TO_RACELINE_URL.get(circuit_id.lower())


def has_raceline(circuit_id: str) -> bool:
    """Check if circuit has raceline data available."""
    return get_raceline_url(circuit_id) is not None


def has_fastf1_data(circuit_id: str) -> bool:
    """Check if circuit has FastF1 event mapping."""
    return get_fastf1_event_name(circuit_id) is not None


def get_circuits_with_racelines():
    """Get list of all circuits that have raceline data."""
    return [cid for cid, url in CIRCUIT_TO_RACELINE_URL.items() if url is not None]


def get_circuits_with_fastf1():
    """Get list of all circuits that have FastF1 event mappings."""
    return [cid for cid, name in CIRCUIT_TO_FASTF1.items() if name is not None]


if __name__ == '__main__':
    """Test the mappings"""
    print("=== Track Mapping Test ===\n")
    
    test_circuits = ['monaco', 'bahrain', 'silverstone', 'adelaide', 'miami']
    
    for circuit in test_circuits:
        fastf1_name = get_fastf1_event_name(circuit)
        raceline_url = get_raceline_url(circuit)
        
        print(f"Circuit: {circuit}")
        print(f"  FastF1: {fastf1_name if fastf1_name else 'NOT AVAILABLE'}")
        print(f"  Raceline: {'AVAILABLE' if raceline_url else 'NOT AVAILABLE'}")
        print()
    
    print(f"\nTotal circuits with racelines: {len(get_circuits_with_racelines())}")
    print(f"Total circuits with FastF1 data: {len(get_circuits_with_fastf1())}")


