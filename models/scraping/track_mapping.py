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

# Mapping from circuit_id to  eoJSON URL
# Source: https://github.com/bacinger/f1-circuits
CIRCUIT_TO_RACELINE_URL = {
    # Modern F1 circuits (active)
    'bahrain': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/bh-2002.geojson',
    'sakhir': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/bh-2002.geojson',
    'jeddah': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/sa-2021.geojson',
    'melbourne': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/au-1953.geojson',
    'albert_park': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/au-1953.geojson',
    'baku': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/az-2016.geojson',
    'miami': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/us-2022.geojson',
    'monaco': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/mc-1929.geojson',
    'barcelona': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/es-1991.geojson',
    'catalunya': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/es-1991.geojson',
    'montreal': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/ca-1978.geojson',
    'montréal': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/ca-1978.geojson',
    'villeneuve': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/ca-1978.geojson',
    'spielberg': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/at-1969.geojson',
    'red_bull_ring': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/at-1969.geojson',
    'silverstone': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/gb-1948.geojson',
    'hungaroring': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/hu-1986.geojson',
    'budapest': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/hu-1986.geojson',
    'spa': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/be-1925.geojson',
    'spa_francorchamps': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/be-1925.geojson',
    'zandvoort': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/nl-1948.geojson',
    'monza': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/it-1922.geojson',
    'marina_bay': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/sg-2008.geojson',
    'suzuka': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/jp-1962.geojson',
    'losail': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/qa-2004.geojson',
    'lusail': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/qa-2004.geojson',
    'austin': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/us-2012.geojson',
    'americas': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/us-2012.geojson',
    'mexico_city': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/mx-1962.geojson',
    'rodriguez': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/mx-1962.geojson',
    'interlagos': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/br-1940.geojson',
    'são_paulo': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/br-1940.geojson',
    'las_vegas': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/us-2023.geojson',
    'yas_marina': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/ae-2009.geojson',
    
    # Recently retired circuits
    'portimao': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/pt-2008.geojson',
    'istanbul': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/tr-2005.geojson',
    'sochi': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/ru-2014.geojson',
    'nurburgring': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/de-1927.geojson',
    'hockenheimring': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/de-1932.geojson',
    'sepang': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/my-1999.geojson',
    'shanghai': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/cn-2004.geojson',
    
    # Historic circuits
    'imola': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/it-1953.geojson',
    'mugello': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/it-1914.geojson',
    'ricard': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/fr-1969.geojson',
    'magny_cours': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/fr-1960.geojson',
    'estoril': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/pt-1972.geojson',
    'jacarepagua': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/br-1977.geojson',
    'kyalami': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/za-1961.geojson',
    'indianapolis': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/us-1909.geojson',
    'galvez': 'https://raw.githubusercontent.com/bacinger/f1-circuits/refs/heads/master/circuits/ar-1952.geojson',
    
    # Circuits not in bacinger repository
    'brands_hatch': None,
    'yeongam': None,
    'buddh': None,
    'valencia': None,
    'adelaide': None,
    'dallas': None,
    'detroit': None,
    'dijon': None,
    'donington': None,
    'fuji': None,
    'jerez': None,
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
    Get bacinger GeoJSON raceline URL for a circuit_id.
    
    Args:
        circuit_id: Circuit identifier from dataset (e.g., 'monaco', 'bahrain')
    
    Returns:
        URL to raceline GeoJSON or None if not available
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


