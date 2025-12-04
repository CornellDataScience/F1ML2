"""
Track Segmentation Utilities

Functions extracted from track_segmentation.ipynb made by Leo.
Includes track segment generation, telemetry collection, and driver performance aggregation.
"""

import pandas as pd
import numpy as np
import requests
import io
from typing import List, Optional, Iterable
import warnings
import fastf1
from fastf1 import plotting
from fastf1.core import Laps

warnings.filterwarnings("ignore")


# ============================================================================
# Raceline Processing Functions
# ============================================================================

def load_raceline_csv(raw_url: str) -> pd.DataFrame:
    """
    Load raceline CSV and return float columns x_m, y_m.
    Handles comma or semicolon delimiters and coerces numeric types.
    """
    r = requests.get(raw_url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(
        io.StringIO(r.text),
        comment="#",
        sep=None,
        engine="python"
    )

    possible_x = [c for c in df.columns if str(c).lower() in ("x_m", "x", "xm")]
    possible_y = [c for c in df.columns if str(c).lower() in ("y_m", "y", "ym")]

    if possible_x and possible_y:
        x_col, y_col = possible_x[0], possible_y[0]
    else:
        x_col, y_col = df.columns[:2]

    df = df[[x_col, y_col]].copy()
    df.columns = ["x_m", "y_m"]
    df["x_m"] = pd.to_numeric(df["x_m"], errors="coerce")
    df["y_m"] = pd.to_numeric(df["y_m"], errors="coerce")

    df = df.dropna(subset=["x_m", "y_m"]).reset_index(drop=True)

    if (df.iloc[0] != df.iloc[-1]).any():
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    return df


def arc_length(df: pd.DataFrame) -> pd.Series:
    """Calculate cumulative arc length along the raceline."""
    dx = np.diff(df.x_m, prepend=df.x_m.iloc[0])
    dy = np.diff(df.y_m, prepend=df.y_m.iloc[0])
    return pd.Series(np.cumsum(np.hypot(dx, dy)), name="s_m")


def curvature_from_xy(x: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Calculate curvature from x, y coordinates using arc-length parameterization."""
    s = np.maximum.accumulate(s + 1e-12*np.arange(len(s)))
    x_s  = np.gradient(x, s)
    y_s  = np.gradient(y, s)
    x_ss = np.gradient(x_s, s)
    y_ss = np.gradient(y_s, s)
    kappa = np.abs(x_s * y_ss - y_s * x_ss) / np.power(x_s**2 + y_s**2, 1.5)
    kappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    return kappa


def resample_equal_arclength(df: pd.DataFrame, step_m: float = 2.0) -> pd.DataFrame:
    """Resample polyline to ~constant spacing along s, improving curvature stability."""
    s = arc_length(df).values
    s_new = np.arange(0, s[-1], step_m)
    x_new = np.interp(s_new, s, df.x_m.values)
    y_new = np.interp(s_new, s, df.y_m.values)
    out = pd.DataFrame({"s_m": s_new, "x_m": x_new, "y_m": y_new})
    if (out.iloc[0][["x_m","y_m"]].values != out.iloc[-1][["x_m","y_m"]].values).any():
        out = pd.concat([out, out.iloc[[0]]], ignore_index=True)
        out.loc[out.index[-1], "s_m"] = out["s_m"].iloc[-2] + step_m
    return out


# ============================================================================
# Corner Detection Functions
# ============================================================================

def build_improved_corner_segments(
    curv: pd.Series, 
    s: pd.Series, 
    thr_quantile: float = 0.85,
    min_corner_length: float = 10.0,
    merge_distance: float = 20.0
) -> pd.DataFrame:
    """
    Improved corner detection that merges nearby corners and enforces minimum lengths.
    """
    thr = np.quantile(curv, thr_quantile)
    is_corner = curv > thr
    
    segments = []
    run = None
    prev_s = None
    for si, flag in zip(s.values, is_corner.values):
        if flag and run is None:
            run = {"start_m": si}
        if not flag and run is not None:
            run["end_m"] = prev_s
            segments.append(run)
            run = None
        prev_s = si
    if run is not None:
        run["end_m"] = s.values[-1]
        segments.append(run)
    
    if not segments:
        return pd.DataFrame(columns=['start_m', 'end_m'])
    
    seg_df = pd.DataFrame(segments)
    seg_df['length_m'] = seg_df['end_m'] - seg_df['start_m']
    seg_df = seg_df[seg_df['length_m'] >= min_corner_length].copy()
    
    if seg_df.empty:
        return pd.DataFrame(columns=['start_m', 'end_m'])
    
    seg_df = seg_df.sort_values('start_m').reset_index(drop=True)
    
    merged_segments = []
    current_seg = seg_df.iloc[0].to_dict()
    
    for i in range(1, len(seg_df)):
        next_seg = seg_df.iloc[i].to_dict()
        gap = next_seg['start_m'] - current_seg['end_m']
        
        if gap <= merge_distance:
            current_seg['end_m'] = next_seg['end_m']
            current_seg['length_m'] = current_seg['end_m'] - current_seg['start_m']
        else:
            merged_segments.append(current_seg)
            current_seg = next_seg
    
    merged_segments.append(current_seg)
    
    result_df = pd.DataFrame(merged_segments)
    return result_df[['start_m', 'end_m']]


def create_straight_segments(corner_segments_df: pd.DataFrame, track_length: float, 
                            min_corner_length: float = 1.0) -> pd.DataFrame:
    """Identify straight segments between corners."""
    straight_segments = []
    
    corner_segments_df = corner_segments_df.copy()
    corner_segments_df['length_m'] = corner_segments_df['end_m'] - corner_segments_df['start_m']
    corners_filtered = corner_segments_df[corner_segments_df['length_m'] >= min_corner_length].copy()
    
    if corners_filtered.empty:
        return pd.DataFrame([{
            'start_m': 0,
            'end_m': track_length,
            'segment_type': 'straight'
        }])
    
    corners_sorted = corners_filtered.sort_values('start_m').reset_index(drop=True)
    
    if corners_sorted.iloc[0]['start_m'] > 0:
        straight_segments.append({
            'start_m': 0,
            'end_m': corners_sorted.iloc[0]['start_m'],
            'segment_type': 'straight'
        })
    
    for i in range(len(corners_sorted) - 1):
        current_end = corners_sorted.iloc[i]['end_m']
        next_start = corners_sorted.iloc[i + 1]['start_m']
        
        if next_start > current_end:
            straight_segments.append({
                'start_m': current_end,
                'end_m': next_start,
                'segment_type': 'straight'
            })
    
    last_corner_end = corners_sorted.iloc[-1]['end_m']
    if last_corner_end < track_length:
        straight_segments.append({
            'start_m': last_corner_end,
            'end_m': track_length,
            'segment_type': 'straight'
        })
    
    return pd.DataFrame(straight_segments)


def generate_track_segments(
    track_name: str,
    raceline_url: Optional[str] = None,
    resample_step_m: float = 2.0,
    thr_quantile: float = 0.85,
    min_corner_length: float = 10.0,
    merge_distance: float = 20.0,
    include_straights: bool = True,
    track_length: Optional[float] = None
) -> pd.DataFrame:
    """
    Generate track segments (corners and straights) from raceline data.
    
    Returns:
        DataFrame with columns: segment_id, segment_type, start_m, end_m, length_m
    """
    if raceline_url is None:
        raise ValueError(f"raceline_url is required for track '{track_name}'")
    
    raceline = load_raceline_csv(raceline_url)
    rl_eq = resample_equal_arclength(raceline, resample_step_m)
    
    kappa = curvature_from_xy(rl_eq.x_m.values, rl_eq.y_m.values, rl_eq.s_m.values)
    rl_eq["kappa"] = kappa
    
    corner_segments_df = build_improved_corner_segments(
        rl_eq["kappa"], rl_eq["s_m"],
        thr_quantile=thr_quantile,
        min_corner_length=min_corner_length,
        merge_distance=merge_distance
    )
    
    track_length = track_length or rl_eq['s_m'].max()
    
    if corner_segments_df.empty:
        if not include_straights:
            return pd.DataFrame(columns=['segment_id', 'segment_type', 'start_m', 'end_m', 'length_m'])
        return pd.DataFrame([{
            'segment_id': 0,
            'segment_type': 'straight',
            'start_m': 0.0,
            'end_m': track_length,
            'length_m': track_length
        }])
    
    corner_segments_df['segment_type'] = 'corner'
    corner_segments_df['length_m'] = corner_segments_df['end_m'] - corner_segments_df['start_m']
    corner_segments_df = corner_segments_df.reset_index(drop=True)
    corner_segments_df['segment_id'] = range(len(corner_segments_df))
    
    if not include_straights:
        return corner_segments_df[['segment_id', 'segment_type', 'start_m', 'end_m', 'length_m']]
    
    straight_segments = create_straight_segments(corner_segments_df, track_length, min_corner_length=0.0)
    straight_segments['length_m'] = straight_segments['end_m'] - straight_segments['start_m']
    straight_segments = straight_segments.reset_index(drop=True)
    straight_segments['segment_id'] = range(len(straight_segments))
    
    all_segments = pd.concat([
        corner_segments_df[['segment_id', 'segment_type', 'start_m', 'end_m', 'length_m']],
        straight_segments[['segment_id', 'segment_type', 'start_m', 'end_m', 'length_m']]
    ]).sort_values('start_m').reset_index(drop=True)
    
    all_segments['segment_id'] = range(len(all_segments))
    
    return all_segments


# ============================================================================
# Driver Telemetry Collection
# ============================================================================

def collect_driver_telemetry(
    year: int,
    grand_prix: str,
    driver: str,
    sessions: Optional[Iterable[str]] = None,
    include_position: bool = False,
    cache_dir: str = "f1_cache",
) -> pd.DataFrame:
    """
    Collect telemetry data for a driver across specified sessions.
    
    Returns:
        DataFrame with telemetry data including Distance column
    """
    if sessions is None:
        sessions = ("FP1", "FP2", "FP3", "Q", "SQ", "SS", "S", "R")

    all_chunks: List[pd.DataFrame] = []

    for sess_name in sessions:
        try:
            session = fastf1.get_session(year, grand_prix, sess_name)
            session.load()
        except Exception as e:
            warnings.warn(f"Skipping session {sess_name}: {e}")
            continue

        laps = session.laps.pick_driver(driver)
        if laps.empty:
            continue

        for _, lap in laps.iterlaps():
            try:
                car = lap.get_car_data().add_distance()
                car["Year"] = year
                car["EventName"] = session.event.EventName
                car["SessionName"] = session.name
                car["Driver"] = driver
                car["LapNumber"] = int(lap["LapNumber"])
                
                for col in ["Stint", "Compound", "TyreLife", "IsAccurate", "PitInTime", "PitOutTime", "TrackStatus", "LapTime"]:
                    if col in laps.columns:
                        car[col] = lap.get(col, pd.NA)

                if include_position:
                    pos = lap.get_pos_data().sort_values("Time").reset_index(drop=True)
                    car = car.sort_values("Time").reset_index(drop=True)
                    car = pd.merge_asof(
                        car,
                        pos[["Time", "X", "Y", "Z"]],
                        on="Time",
                        direction="nearest",
                        tolerance=pd.Timedelta("50ms"),
                    )

                all_chunks.append(car)
            except Exception as e:
                warnings.warn(f"Failed telemetry for {driver} lap {int(lap['LapNumber'])} in {session.name}: {e}")
                continue

    if not all_chunks:
        return pd.DataFrame()

    df = pd.concat(all_chunks, ignore_index=True)

    preferred_order = [
        "Year", "EventName", "SessionName", "Driver", "LapNumber",
        "Stint", "Compound", "TyreLife", "IsAccurate", "PitInTime", "PitOutTime", "TrackStatus", "LapTime",
        "Time", "Distance", "Speed", "Throttle", "Brake", "nGear", "RPM", "DRS", "Source",
        "X", "Y", "Z"
    ]
    cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    df = df[cols]

    return df


def aggregate_driver_by_segments(
    segments_df: pd.DataFrame,
    track_name: str,
    year: int,
    driver: str,
    sessions: Optional[Iterable[str]] = None,
    include_position: bool = False
) -> pd.DataFrame:
    """
    Aggregate driver telemetry data by track segments.
    
    Args:
        segments_df: DataFrame with segment definitions (from generate_track_segments)
        track_name: Name of the track (e.g., "Bahrain")
        year: Year of the race
        driver: Driver 3-letter code (e.g., "VER")
        sessions: Optional list of sessions (e.g., ("Q",) for qualifying only)
        include_position: Whether to include position data
    
    Returns:
        DataFrame with aggregated statistics per segment
    """
    telemetry_df = collect_driver_telemetry(
        year=year,
        grand_prix=track_name,
        driver=driver,
        sessions=sessions,
        include_position=include_position
    )
    
    if telemetry_df.empty:
        warnings.warn(f"No telemetry data found for {driver} at {track_name} {year}")
        return pd.DataFrame()
    
    aggregated_data = []
    
    for _, segment in segments_df.iterrows():
        start_m = segment['start_m']
        end_m = segment['end_m']
        segment_type = segment.get('segment_type', 'unknown')
        
        segment_mask = (telemetry_df['Distance'] >= start_m) & (telemetry_df['Distance'] <= end_m)
        segment_data = telemetry_df[segment_mask]
        
        if segment_data.empty:
            continue
        
        length_m = segment.get('length_m', end_m - start_m)
        
        stats = {
            'segment_id': segment['segment_id'],
            'segment_type': segment_type,
            'start_m': start_m,
            'end_m': end_m,
            'length_m': length_m,
            'sample_count': len(segment_data),
            'avg_speed_kmh': segment_data['Speed'].mean(),
            'max_speed_kmh': segment_data['Speed'].max(),
            'min_speed_kmh': segment_data['Speed'].min(),
            'avg_throttle': segment_data['Throttle'].mean(),
            'brake_percentage': segment_data['Brake'].mean() * 100,
            'avg_rpm': segment_data['RPM'].mean(),
            'max_rpm': segment_data['RPM'].max(),
            'avg_gear': segment_data['nGear'].mean(),
            'drs_usage': (segment_data['DRS'] > 0).mean() * 100,
        }
        
        if 'LapNumber' in segment_data.columns:
            stats['lap_count'] = len(segment_data['LapNumber'].unique())
        
        aggregated_data.append(stats)
    
    result_df = pd.DataFrame(aggregated_data)
    
    if not result_df.empty:
        result_df['track_name'] = track_name
        result_df['year'] = year
        result_df['driver'] = driver
    
    return result_df

