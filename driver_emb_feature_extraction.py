import os
import pandas as pd
import numpy as np
import warnings
import traceback
import osmnx as ox

# --- Import necessary functions from HelperFuncs ---
try:
    # Add safe_mean and sort_df_by_trip_time
    from HelperFuncs import (
        load_file, save_file, sort_df_by_trip_time, drop_trips_with_less_than_x_segs, 
        find_seg_start_idx, get_osrm_match_robust, haversine_np, get_path_context_from_osm,
        calculate_yaw_and_radius, query_dem_opentopo, log_issue, calculate_wind_components,
        are_distances_close, verify_altitude_consistency, verify_temperature_consistency,
        get_weather_for_route, safe_mode, validate_window_arrays, get_tqdm, get_graph_for_trip_corridor,
        save_checkpoint
        )
except ImportError:
    print("Error: Could not import from HelperFuncs. Ensure HelperFuncs.py is accessible.")

tqdm = get_tqdm()

# --- Configuration Block ---
CLEANED_DATA_PATH = r'C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\clean_df\clean_df.parquet'
SEGMENTATION_DICT_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segmentation_dict\trips_seg_dict_v7.pickle"

OUTPUT_FEATURE_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\windows_120s.pickle" 

CHECKPOINT_DIR = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\checkpoints"
CHECKPOINT_INTERVAL = 200 # Save after every 200 trips

OUTPUT_G_OSM_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\G_osm.graphml"
OUTPUT_FORMAT = 'pickle'

OUTPUT_LOG_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\feature_extraction_log.csv"

# --- Centralized Configuration Block ---
OSMNX_CACHE_FOLDER = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\osmnx_cache"
REQUEST_TIMEOUT_S = 180 # 3 minutes, generous for Overpass API


# Map Matching Config
INITIAL_MATCH_RADIUS = 25; MAX_MATCH_RADIUS = 100; MATCH_RETRY_ATTEMPTS = 4
REQUEST_TIMEOUT = 45
MATCH_DISTANCE_TOLERANCE_PCT = 10.0; GAPS_THRESHOLD_S = 300
OSRM_BASE_URL = "http://router.project-osrm.org"

MIN_MATCH_CONFIDENCE = 0.5
FINAL_DISTANCE_TOLERANCE_PCT = 10.0

MIN_SEGMENTS_PER_TRIP = 1
WIN_LEN_S = 120
WIN_HOP_S = 60
FULL_STOP = 3

ADD_REL_SPEED = True
ADD_CURVATURE = True

# MAX_SPEED_DIFF = 10.2
MAX_ALT_DIFF = 15

KMPH_TO_MPS = 5/18
DELTA_TIME_S = 1

ABS_END_GAP_TOL = 10.0     # m
REL_END_GAP_TOL = 0.01     # 1 % of route

G_CONST = 9.80665  # Standard gravity

# AUX_SOC_LOSS = True
#PROCESS_SUBSET = None # To process only x samples

# --- Standard Column Names (Must match output of preprocessing script) ---
TRIP_ID_COL = 'trip_id'
TIME_COL = 'timestamp'
SPEED_ARRAY_COL = 'speed_array'
ALT_ARRAY_COL = 'altitude_array'
CAR_MODEL_COL = "car_model"
LAT_COL = 'latitude'   
LON_COL = 'longitude' 
ODO_COL = 'current_odo'

TEMP_COL = 'outside_temp' 
BATT_HEALTH_COL = 'battery_health'
WEIGHT_COL = 'empty_weight_kg'
BATTERY_TYPE_COL = "battery_type"
BATTERY_CAP_COL = 'battery_capacity_kWh'

KINEMATIC_CHANNELS = [
    'speed_mps',
    'long_accel_mps2',
    'centripetal_accel_mps2',
    'jerk_mps3',
    'grade_corrected_accel',
    'vert_v_mps',
    'yaw_rate_dps',
    'yaw_accel_dps2',
    'speed_rel_limit_mps', # Speed relative to the speed limit
    'headwind_mps',              # Environmental Context
    'crosswind_mps',             # Environmental Context
    'propulsive_accel_mps2',      # Derived Behavioral Feature
    'propulsive_power_kw',
    'lat_accel_signed_mps2',
    'centripetal_jerk_mps3',
    'propulsive_power_kw_per_kg'
]

# --- Outlier Configuration ---
# This configuration is based on the physical limits of passenger vehicles and the
# practical realities of driving on public roads. The goal is to remove sensor
# artifacts and impossible values, not to penalize aggressive driving.

OUTLIER_CONFIGS = {
    # --- Primary Motion Channels ---
    'speed_mps': {
        # Raw: Speed must be non-negative. Max: 70 m/s (~252 kph).
        'raw_limits': (-0.5, 70.0),
    },

    'long_accel_mps2': {
        # Raw: ~[-11, 10] m/s² covers hard braking/accel in road cars.
        'raw_limits': (-11.0, 10.0),
    },

    'centripetal_accel_mps2': {
        # Raw: v^2/r >= 0. Max ~1.2 g lateral.
        'raw_limits': (0.0, 12.0),
        # Diff: sudden lateral-G step (swerve).
        'diff_limits': (-8.0, 8.0),
    },

    # --- Derivative & Rotational Channels ---
    'jerk_mps3': {
        'raw_limits': (-30.0, 30.0),
        'diff_limits': (-40.0, 40.0),
    },

    'yaw_rate_dps': {
        'raw_limits': (-120.0, 120.0),
        'diff_limits': (-80.0, 80.0),
    },

    # NEW: angular acceleration (derivative of yaw_rate)
    'yaw_accel_dps2': {
        # Conservative but generous bounds for highway/light urban driving.
        'raw_limits': (-200.0, 200.0),
        'diff_limits': (-300.0, 300.0),
    },

    # NEW: signed lateral acceleration (v * yaw_rate_rps)
    'lat_accel_signed_mps2': {
        # Allow up to ~±1.2 g lateral.
        'raw_limits': (-12.0, 12.0),
        'diff_limits': (-8.0, 8.0),
    },

    # NEW: lateral jerk (d/dt of centripetal_accel)
    'centripetal_jerk_mps3': {
        'raw_limits': (-40.0, 40.0),
        'diff_limits': (-60.0, 60.0),
    },

    # --- Environmental & Context-Normalized Behavioral Channels ---
    'grade_corrected_accel': {
        # Slightly wider than long_accel to account for grade compensation.
        'raw_limits': (-14.0, 13.0),
    },

    'vert_v_mps': {
        # Vertical speed; large magnitudes imply DEM/GPS z-noise.
        'raw_limits': (-10.0, 10.0),
        'diff_limits': (-10.0, 10.0),
    },

    'speed_rel_limit_mps': {
        # Behavioral, not physical. Generous bounds around limit compliance.
        'raw_limits': (-30.0, 40.0),
    },

    'headwind_mps': {
        # ±40 m/s (~144 km/h) covers severe storms.
        'raw_limits': (-40.0, 40.0),
        'diff_limits': (-15.0, 15.0),
    },

    'crosswind_mps': {
        # If you store absolute magnitude, keep [0, 40]. If signed, switch to (-40, 40).
        'raw_limits': (0.0, 40.0),
        'diff_limits': (-15.0, 15.0),
    },

    'propulsive_accel_mps2': {
        # Driver-commanded accel after removing grade/wind.
        'raw_limits': (-13.0, 12.0),
        'diff_limits': (-30.0, 30.0),
    },

    # NEW: propulsive power per kg (kW/kg) = a * v / 1000
    'propulsive_power_kw_per_kg': {
        # Generous bounds: ±0.25 kW/kg (~±375 kW for 1500 kg vehicle).
        # Tighten after inspecting your dataset’s percentiles.
        'raw_limits': (-0.25, 0.25),
        'diff_limits': (-0.15, 0.15),
    },
}

WEATHER_VARS_MAP = {
    # API Name -> Final Column Name
    "temperature_2m": "weather_temp_c",
    "relative_humidity_2m": "weather_humidity_pct",
    "dew_point_2m": "weather_dewpoint_c",
    "apparent_temperature": "weather_apparent_temp_c",
    "surface_pressure": "weather_pressure_hpa",
    "precipitation": "weather_precip_mm",
    "rain": "weather_rain_mm",
    "snowfall": "weather_snowfall_mm",
    "cloud_cover": "weather_cloudcover_pct",
    "shortwave_radiation": "solar_radiation",
    "wind_speed_10m": "weather_windspeed_kph",
    "wind_gusts_10m": "weather_windgust_kph",
    "wind_direction_10m": "weather_wind_dir_deg", # Fetch the raw direction
    "weather_code": "weather_code",
    "is_day": "is_day",
    "sunshine_duration": "sunshine_duration_s",
    "snow_depth": "snow_depth_m"
}

# --- This list is now DERIVED from the map, ensuring consistency ---
WEATHER_CONTEXT_COLS = list(WEATHER_VARS_MAP.values()) + [
    "weather_wind_dir_sin",  # Add derived columns here
    "weather_wind_dir_cos"
]

# Define features that are known to be sparse due to reliance on optional OSM tags.
# We will log their absence as INFO instead of WARNING.
KNOWN_SPARSE_CONTEXT_FEATURES = {
    'win_median_lanes',
    'win_intersection_count',
    'win_avg_speed_limit_kph',
    'win_surface_type',
    'win_junction_type',
    'win_smoothness_type',
    'win_lit_pct'
}

KNOWN_SPARSE_KINEMATIC_CHANNELS = {
    'speed_rel_limit_mps'
}

# -------------------------------------------------

def interp_missing_alts(
        df: pd.DataFrame, 
        trips_seg_dict: dict, 
        max_len_interp: int = 2, 
        max_diff_between_neig: int = 15,
        full_stop=FULL_STOP
        ) -> pd.DataFrame:
    """
    Linearly interpolates short runs of NaNs in per-segment altitude arrays.

    This function correctly identifies segment boundaries, concatenates altitude
    data for the entire segment, performs a safe interpolation on short NaN gaps,
    and writes the modified arrays back to the correct rows in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with trip data.
        trips_seg_dict (dict): Dictionary mapping trip_id to segmentation labels.
        alt_array_col (str): Name of the altitude array column.
        trip_id_col (str): Name of the trip ID column.
        max_len_interp (int): Maximum length of a NaN gap to interpolate.
        max_diff_between_neig (int): Max altitude diff between gap neighbors to allow interpolation.
        full_stop (int): Number of consecutive non-driving events to define a segment break.

    Returns:
        pd.DataFrame: A new DataFrame with altitude arrays interpolated.
    """
    print(f"- Max interpolation length: {max_len_interp} samples")
    print(f"- Max neighbor difference: {max_diff_between_neig} meters")

    df_copy = df.copy()

    if ALT_ARRAY_COL not in df.columns:
        warnings.warn(f"Altitude column '{ALT_ARRAY_COL}' not found. Skipping interpolation.")
        return df

    samples_per_row = df_copy[ALT_ARRAY_COL].iat[0].size     # 10

    # --- For Summary Report ---
    trips_with_fills = set()
    num_of_fills = 0

    for trip_id, trip_labels in tqdm(trips_seg_dict.items(), desc="Interpolating Altitudes"):
        
        trip_df = df[df[TRIP_ID_COL] == trip_id].copy()
        if len(trip_df) != len(trip_labels):
            warnings.warn(f"Length mismatch for trip_id {trip_id}. Skipping trip.")
            continue

        unique_seg_ids = np.unique(trip_labels[trip_labels > 0]).astype(int)
        if len(unique_seg_ids) == 0: 
            warnings.warn(f'{trip_id} Trip skipped: No valid positive segment IDs found.')
            continue 
        
        for seg_id in unique_seg_ids:

            labeled_indices = np.where(trip_labels == seg_id)[0]
            if len(labeled_indices) == 0: 
                warnings.warn(f"No samples for trip_id {trip_id} seg {seg_id}!")
                continue

            first_labeled_idx = labeled_indices[0]
            last_labeled_idx = labeled_indices[-1]

            segment_start_idx = find_seg_start_idx(trip_labels, first_labeled_idx, full_stop=full_stop)
            segment_span_indices = np.arange(segment_start_idx, last_labeled_idx + 1)

            df_seg = trip_df.iloc[segment_span_indices,:] 
            if df_seg.empty:
                warnings.warn(f"Empty df for trip_id {trip_id} seg {seg_id}!")
                continue 

            seg_alts = np.concatenate(df_seg[ALT_ARRAY_COL].values)
        
            mask   = np.isnan(seg_alts).astype(int)
            edges  = np.diff(np.concatenate(([0], mask, [0])))

            starts = np.where(edges ==  1)[0]
            ends   = np.where(edges == -1)[0] - 1
            
            seg_was_modified = False
            for s, e in zip(starts, ends):
                if (
                    s > 0 
                    and e < len(seg_alts)-1
                    and e - s < max_len_interp 
                    and abs(seg_alts[s-1] - seg_alts[e+1]) <= max_diff_between_neig
                ):
                    seg_alts[s-1:e+2] = np.linspace(seg_alts[s-1], seg_alts[e+1], num= e - s + 3)
                    num_of_fills += e-s + 1
                    seg_was_modified = True
                    trips_with_fills.add(trip_id)

            if seg_was_modified:                    
                cleaned_seg_alts_2d = seg_alts.reshape(-1, samples_per_row)
                global_segment_indices = df_seg.index

                if len(global_segment_indices) == len(cleaned_seg_alts_2d):
                    new_alt_series = pd.Series(
                        data=[row for row in cleaned_seg_alts_2d], 
                        index=global_segment_indices
                    )
                    df_copy.loc[global_segment_indices, ALT_ARRAY_COL] = new_alt_series
                else:
                    warnings.warn(f"Shape mismatch for trip {trip_id}, seg {seg_id}. Cannot write back interpolated data.")
    
    print(f"\nInterpolation complete. Filled {num_of_fills} NaN values across {len(trips_with_fills)} unique trips.")
    return df_copy                

# ------------- RECONSTRUCT 1 HZ PATH ------------- #

def reconstruct_1hz_path(df_seg, trip_id, seg_id, G_trip_wgs84, dem_query_function, osrm_base_url, min_match_confidence, issue_log):
    """
    Reconstructs a 1Hz path using a robust three-way distance validation between
    the odometer, integrated speed, and the OSRM map-matched route.
    """
    coords = list(zip(df_seg['latitude'].astype(float), df_seg['longitude'].astype(float)))

    ts = pd.to_datetime(df_seg['timestamp'], utc=True)

    if len(coords) < 2:
        warnings.warn(f"[{trip_id}-{seg_id}] Skipping: Not enough points for matching.")
        log_issue(issue_log, 'WARNING', 'reconstruct_1hz_path', 'Not enough points for matching, segment skipped.', trip_id, seg_id, details={'point_count': len(coords)})
        return None

    dist_odo_km = float(df_seg[ODO_COL].iloc[-1] - df_seg[ODO_COL].iloc[0])

    speeds_1hz_kph = np.concatenate(df_seg[SPEED_ARRAY_COL].values).astype('float32')
    dist_speed_km = np.sum(speeds_1hz_kph) / 3600.0

    match_data = get_osrm_match_robust(
        coordinates=coords,
        timestamps=ts,
        odo_distance_km=dist_odo_km,
        trip_id_debug=trip_id,
        segment_id_debug=seg_id,
        osrm_base_url=osrm_base_url,
        min_match_confidence=min_match_confidence,
        initial_radius_m=INITIAL_MATCH_RADIUS,
        max_radius_m=MAX_MATCH_RADIUS,
        max_attempts=MATCH_RETRY_ATTEMPTS,
        tolerance_pct=MATCH_DISTANCE_TOLERANCE_PCT,
        gaps_threshold_s=GAPS_THRESHOLD_S,
        request_timeout=REQUEST_TIMEOUT,
    )

    authoritative_dist_m = None
    match_status = match_data['match_status']
    dist_osrm_km = match_data['matched_distance_m'] / 1000.0 if pd.notna(match_data['matched_distance_m']) else None

    if "OK" not in match_status:
        warnings.warn(f"[{trip_id}-{seg_id}] Skipping: Match failed ({match_status}).")
        log_issue(issue_log, 'ERROR', 'reconstruct_1hz_path', 'OSRM match failed, segment skipped.', 
                  trip_id, seg_id, details={'status': match_status})
        return None

    elif match_status == "OK_InTolerance":
        # Odo and OSRM already agree. Check if speed integration forms a "triangle match".
        if are_distances_close(dist_odo_km, dist_speed_km):
            # Perfect case: all three distances agree. Trust the odometer.
            authoritative_dist_m = dist_odo_km * 1000.0
        else:
            # Speed integration is the outlier. Trust Odo/OSRM and log the warning.    
            authoritative_dist_m = dist_odo_km * 1000.0
            log_issue(issue_log, 'WARNING', 'reconstruct_1hz_path', 'TriangleMismatch: Speed integration differs from Odo/OSRM.',
                      trip_id, seg_id, details={'dist_odo_km': round(dist_odo_km, 2), 
                                                'dist_osrm_km': round(dist_osrm_km, 2), 'dist_speed_km': round(dist_speed_km, 2)})    
    
    elif match_status == "OK_ToleranceFail":
        # Odo and OSRM disagree. Use speed integration as the tie-breaker.   
        if are_distances_close(dist_odo_km, dist_speed_km):
            # Speed agrees with Odo. Trust the odometer.
            authoritative_dist_m = dist_odo_km * 1000.0
            log_issue(issue_log, 'INFO', 'reconstruct_1hz_path', 'OdoSpeedAgreement: OSRM is the outlier.',
                      trip_id, seg_id, details={'dist_odo_km': round(dist_odo_km, 2), 
                      'dist_speed_km': round(dist_speed_km, 2), 'dist_osrm_km': round(dist_osrm_km, 2)})    

        elif are_distances_close(dist_osrm_km, dist_speed_km):
            # Speed agrees with OSRM. Trust the OSRM distance.
            authoritative_dist_m = dist_osrm_km * 1000.0
            log_issue(issue_log, 'INFO', 'reconstruct_1hz_path', 'OsrmSpeedAgreement: Odometer is the outlier.',
                      trip_id, seg_id, details={'dist_odo_km': round(dist_odo_km, 2), 
                      'dist_speed_km': round(dist_speed_km, 2), 'dist_osrm_km': round(dist_osrm_km, 2)})
        else:
            # Three-way disagreement. Cannot trust any source. Skip segment.
            log_issue(issue_log, 'ERROR', 'reconstruct_1hz_path', 'ThreeWayDisagreement: Odo, Speed, and OSRM all conflict. Segment skipped.',
                      trip_id, seg_id, details={'dist_odo_km': round(dist_odo_km, 2), 'dist_speed_km': round(dist_speed_km, 2), 'dist_osrm_km': round(dist_osrm_km, 2)})
            return None

    elif match_status == "OK_NoToleranceCheck":
        # No odo distance available – trust OSRM distance
        if are_distances_close(dist_osrm_km, dist_speed_km):
            authoritative_dist_m = dist_osrm_km * 1000.0
        else:
            # Two-Way disagreement. Cannot trust any source. Skip segment.
            log_issue(issue_log, 'WARNING', 'reconstruct_1hz_path', 'TwoWayDisagreement: Speed integration differs from OSRM (no odo). Segment skipped.',
                      trip_id, seg_id, details={'dist_osrm_km': round(dist_osrm_km, 2), 'dist_speed_km': round(dist_speed_km, 2)})
            return None 

    if authoritative_dist_m is None:
        # This case handles any logic gaps or unexpected statuses.
        log_issue(issue_log, 'ERROR', 'reconstruct_1hz_path', f'Could not determine authoritative distance. Status: {match_status}. Segment skipped.',
                  trip_id, seg_id, details={'dist_odo_km': round(dist_odo_km, 2), 'dist_speed_km': round(dist_speed_km, 2), 'dist_osrm_km': round(dist_osrm_km, 2)})
        return None

    try:
        all_geom_coords = [
            coord
            for m in match_data["osrm_match_result"]["matchings"]
            for coord in m["geometry"]["coordinates"]
        ]
        geom = np.asarray(all_geom_coords, dtype=np.float64)  # [lon, lat]
    except Exception as e:
        warnings.warn(f"[{trip_id}-{seg_id}] Skipping: Could not extract geometry.")
        log_issue(issue_log, 'ERROR', 'reconstruct_1hz_path', 
                  f'Could not extract geometry from OSRM result: {e}', trip_id, seg_id)
        return None

    # geometry cumulative distance
    geom_lons, geom_lats = geom[:, 0], geom[:, 1]
    seg_dists_m = haversine_np(geom_lons[:-1], geom_lats[:-1], geom_lons[1:], geom_lats[1:]) * 1000.0
    cum_geom_m = np.concatenate([[0.0], np.cumsum(seg_dists_m)])

    # integrate vehicle distance
    dist_per_sec_m = speeds_1hz_kph / 3.6
    cum_vehicle_m_unscaled = np.concatenate([[0.0], np.cumsum(dist_per_sec_m[:-1])])

    # Rescale the vehicle's progress to match the authoritative distance
    if cum_vehicle_m_unscaled[-1] > 1e-6:
        scale_factor = authoritative_dist_m / cum_vehicle_m_unscaled[-1]
        cum_vehicle_m_scaled = cum_vehicle_m_unscaled * scale_factor
    else:
        cum_vehicle_m_scaled = cum_vehicle_m_unscaled # Avoid division by zero    
    
    lon_1hz = np.interp(cum_vehicle_m_scaled, cum_geom_m, geom_lons)
    lat_1hz = np.interp(cum_vehicle_m_scaled, cum_geom_m, geom_lats)

    start_time = pd.to_datetime(df_seg[TIME_COL].iloc[0])
    ts_1hz = pd.date_range(start=start_time, periods=len(speeds_1hz_kph), freq='s')

    df_1hz = pd.DataFrame({
        "timestamp": ts_1hz,
        "latitude":  lat_1hz.astype('float32'),
        "longitude": lon_1hz.astype('float32'),
        "speed_kph": speeds_1hz_kph,
        "speed_mps": speeds_1hz_kph * KMPH_TO_MPS,
        'osrm_match_conf': match_data.get('match_confidence', np.nan),
        'osrm_distance_m': match_data.get('matched_distance_m', np.nan)
    })

    if df_1hz['latitude'].hasnans or df_1hz['longitude'].hasnans:
        nan_count = df_1hz['latitude'].isna().sum()
        warnings.warn(f"[{trip_id}-{seg_id}] CRITICAL: NaN values detected in the final 1Hz path coordinates. Skipping segment.")
        log_issue(
            issue_log, 'ERROR', 'reconstruct_1hz_path',
            'NaNs detected in final 1Hz path coordinates, segment reconstruction failed.',
            trip_id, seg_id, details={'nan_coordinate_count': nan_count}
        )

    # OSM enrichment adding road_type and speed_limit_kph
    df_1hz = get_path_context_from_osm(df_1hz, G_trip_wgs84, trip_id, seg_id)

    # Fetch weather data for the 1Hz path
    df_1hz = get_weather_for_route(df_1hz, issue_log, trip_id, seg_id, weather_vars_map=WEATHER_VARS_MAP)
    if df_1hz is None:
        warnings.warn(f"[{trip_id}-{seg_id}] Skipping segment due to weather fetch failure.")
        # The helper function has already logged the detailed reason.
        return None

    # Perform temperature consistency check
    if not verify_temperature_consistency(df_seg, df_1hz, issue_log, trip_id, seg_id):
        warnings.warn(f"[{trip_id}-{seg_id}] Skipping segment due to failed temperature consistency check.")
        # The helper function has already logged the detailed reason.
        return None

    # DEM + grade
    try:
        elev = dem_query_function(df_1hz['latitude'].tolist(), df_1hz['longitude'].tolist())
        df_1hz['dem_elevation_m'] = np.asarray(elev, dtype=np.float32)
        lat_prev = df_1hz['latitude'].shift(fill_value=df_1hz['latitude'].iloc[0]).to_numpy()
        lon_prev = df_1hz['longitude'].shift(fill_value=df_1hz['longitude'].iloc[0]).to_numpy()
        lat_cur  = df_1hz['latitude'].to_numpy()
        lon_cur  = df_1hz['longitude'].to_numpy()

        dz = df_1hz['dem_elevation_m'].diff().to_numpy()
        ds = (haversine_np(lon_prev, lat_prev, lon_cur, lat_cur) * 1000.0)
        grade = np.divide(dz, ds, out=np.zeros_like(dz, dtype=float), where=(ds > 0.1))
        df_1hz['grade'] = pd.Series(np.clip(grade, -0.30, 0.30), index=df_1hz.index).astype('float32')
        df_1hz['grade'] = df_1hz['grade'].ffill().bfill()
    except Exception as e:
        warnings.warn(f"[{trip_id}-{seg_id}] DEM/grade failed: {e}")
        log_issue(issue_log, 'ERROR', 'reconstruct_1hz_path', f'DEM/grade calculation failed: {e}', trip_id, seg_id)
        df_1hz['dem_elevation_m'] = np.nan
        df_1hz['grade'] = 0.0

    if not verify_altitude_consistency(df_seg, df_1hz, trip_id, seg_id, issue_log):
        # The helper function has already logged the detailed reason for failure.
        # We just need to warn and skip the segment.
        warnings.warn(f"[{trip_id}-{seg_id}] Skipping segment due to failed altitude consistency check.")
        return None

    # Yaw & radius
    df_1hz = calculate_yaw_and_radius(df_1hz)

    return df_1hz

def calculate_1hz_kinematics(df_1hz, df_seg, trip_starts_with_zero=False):
    """
    Takes a 1Hz DataFrame and calculates all derivative and environmental features.
    """        
    df_1hz_out = df_1hz.copy()
    
    # 1. Calculate derivatives
    df_1hz_out['long_accel_mps2'] = np.gradient(df_1hz_out['speed_mps'], DELTA_TIME_S)
    df_1hz_out['jerk_mps3']       = np.gradient(df_1hz_out['long_accel_mps2'], DELTA_TIME_S)

    # Prefer DEM-based vertical velocity when available; fallback to vehicle altitude array
    if 'dem_elevation_m' in df_1hz_out.columns and df_1hz_out['dem_elevation_m'].notna().sum() > 2:
        df_1hz_out['vert_v_mps'] = np.gradient(df_1hz_out['dem_elevation_m'], DELTA_TIME_S)
    else:
        warnings.warn(f"[{df_seg.iloc[0]['trip_id']}-{df_seg.iloc[0]['seg_id']}] Using vehicle altitude array for vert_v_mps due to DEM failure.")
        seg_total_len = len(df_1hz_out)
        all_seg_alts_m = np.concatenate(df_seg[ALT_ARRAY_COL].values)
        x_orig = np.arange(len(all_seg_alts_m))
        x_new  = np.linspace(0, len(all_seg_alts_m) - 1, seg_total_len)
        upsampled_alts = np.interp(x_new, x_orig, all_seg_alts_m)
        df_1hz_out['vert_v_mps'] = np.gradient(upsampled_alts, DELTA_TIME_S)

    # 3. Handle initial boundary condition
    if trip_starts_with_zero:
        df_1hz_out.loc[df_1hz_out.index[0], 'long_accel_mps2'] = 0.0
        df_1hz_out.loc[df_1hz_out.index[:2], 'jerk_mps3'] = 0.0
        df_1hz_out.loc[df_1hz_out.index[0], 'vert_v_mps'] = 0.0
        trip_starts_with_zero = False

    # 4. Calculate other kinematic features
    # Calculate speed-to-limit ratio with robust division-by-zero handling
    speed_kph = df_1hz_out['speed_kph']
    limit_kph = df_1hz_out['speed_limit_kph']
    # Use np.divide to safely handle cases where the speed limit is 0
    # Where limit is 0, the ratio is set to 1.0 if speed is also 0, otherwise inf.
    ratio = np.divide(speed_kph, limit_kph, 
                      out=np.full_like(speed_kph, np.inf), 
                      where=(limit_kph != 0))
    ratio[ (speed_kph == 0) & (limit_kph == 0) ] = 1.0 # Driving 0 in a 0 zone is ratio 1.0
    df_1hz_out['speed_limit_ratio'] = ratio

    v2 = df_1hz_out['speed_mps'] ** 2
    df_1hz_out['centripetal_accel_mps2'] = np.divide(v2, df_1hz_out['radius_m'], out=np.zeros_like(v2), where=(df_1hz_out['radius_m'] > 1e-3))
    df_1hz_out['grade_corrected_accel'] = df_1hz_out['long_accel_mps2'] + (G_CONST * np.sin(np.arctan(df_1hz_out['grade'])))

    yaw_rate_rps = np.radians(df_1hz_out['yaw_rate_dps'])
    df_1hz_out['lat_accel_signed_mps2'] = df_1hz_out['speed_mps'] * yaw_rate_rps

    df_1hz_out['centripetal_jerk_mps3'] = np.gradient(df_1hz_out['centripetal_accel_mps2'], DELTA_TIME_S)

    return df_1hz_out, trip_starts_with_zero

def extract_segment_context(df_seg):
    """
    Extracts STATIC context features that are constant for the entire segment.
    """
    first_row = df_seg.iloc[0]
    start_time = pd.to_datetime(first_row.get(TIME_COL))

    context = {
        # Vehicle & Battery Properties (truly static)
        'car_model': first_row.get(CAR_MODEL_COL, 'unknown'),
        'battery_type': first_row.get(BATTERY_TYPE_COL, np.nan),
        'empty_weight_kg': first_row.get(WEIGHT_COL, np.nan),
        'battery_capacity_kWh': first_row.get(BATTERY_CAP_COL, np.nan),
        'seg_start_odo': first_row.get(ODO_COL, np.nan),

        'avg_outside_temp': df_seg[TEMP_COL].mean(),
        'seg_mean_batt_health': df_seg[BATT_HEALTH_COL].mean(),

        # Trip-level Time Properties (static for the segment)
        'start_hour_sin': np.sin(2 * np.pi * start_time.hour / 24),
        'start_hour_cos': np.cos(2 * np.pi * start_time.hour / 24),
        'day_of_week': start_time.dayofweek,
        'is_weekend': 1 if start_time.dayofweek >= 5 else 0, # Add weekend flag
        'month': start_time.month,
    }
    return context

def extract_window_context(window_df, issue_log, trip_id, seg_id, win_id, G_trip_wgs84):
    """
    Extracts DYNAMIC context features specific to a single 60-second window.
    This is the key for hard negative mining.
    """
    if window_df.empty:
        warnings.warn(f"{trip_id}-{seg_id}-{win_id} Input window DataFrame is empty. Cannot extract context.")
        log_issue(
            log_list=issue_log,
            level='ERROR',
            function_name='extract_window_context',
            reason='Input window DataFrame is empty. Cannot extract context.',
            trip_id=trip_id,
            seg_id=seg_id,
            win_id=win_id
        )
        return {}

    context = {
        #   • OSM-derived
        "win_road_type":           safe_mode(window_df.get("road_type"), "unknown"),
        "win_avg_speed_limit_kph": window_df["speed_limit_kph"].mean(),
        "win_surface_type":        safe_mode(window_df.get("surface_type"), "unknown"),
        "win_median_lanes":        window_df["lanes_count"].median()       if "lanes_count"  in window_df else np.nan,
        "win_oneway_pct":          window_df["is_oneway"].mean()*100       if "is_oneway"   in window_df else np.nan,
        "win_bridge_pct":          window_df["is_bridge"].mean()*100       if "is_bridge" in window_df else np.nan,
        "win_tunnel_pct":          window_df["is_tunnel"].mean()*100       if "is_tunnel" in window_df else np.nan,
        "win_junction_type":       safe_mode(window_df.get("junction_type"), "none"),
        "win_smoothness_type":     safe_mode(window_df.get("smoothness_type"), "unknown"),
        "win_lit_pct":             window_df["is_lit"].mean() * 100 if "is_lit" in window_df else np.nan,
        "win_traffic_signal_pct":  window_df["has_traffic_signal"].mean() * 100,
        "win_stop_sign_pct":       window_df["has_stop_sign"].mean() * 100,

        #   • DEM / geometry
        "win_avg_grade":           window_df["grade"].mean(),
        "win_elevation_change_m":  window_df["dem_elevation_m"].iat[-1] - window_df["dem_elevation_m"].iat[0],
        # bearing change ≈ route curvature, independent of speed
        "win_cum_bearing_change_deg": np.nansum(np.abs(np.diff(window_df["bearing_deg"].to_numpy()))), 

        "win_avg_headwind_mps": window_df["headwind_mps"].mean(),
        "win_max_headwind_mps": window_df["headwind_mps"].max(),
        "win_avg_crosswind_mps": window_df["crosswind_mps"].mean(),
        "win_max_crosswind_mps": window_df["crosswind_mps"].max(),
    }

    # Get the bounding box of the window
    win_bounds = window_df[['longitude', 'latitude']].agg(['min', 'max'])
    north, south = win_bounds.loc['max', 'latitude'], win_bounds.loc['min', 'latitude']
    east, west = win_bounds.loc['max', 'longitude'], win_bounds.loc['min', 'longitude']

    try:
        # Find nodes within the window's bounding box
        nodes_in_window = ox.graph_nodes_from_bbox(G_trip_wgs84, north, south, east, west)
        context['win_intersection_count'] = len(nodes_in_window)
    except:
        context['win_intersection_count'] = np.nan # Handle cases with no nodes
    
    # --- merge aggregated weather (all purely exogenous) ---
    for col in WEATHER_CONTEXT_COLS:
        if col in window_df.columns:
            if col == "weather_code":
                # Use the safe_mode helper function
                context[f"win_{col}"] = safe_mode(window_df.get(col), np.nan)
            else:
                # Use mean for all other continuous weather data
                context[f"win_{col}"] = window_df[col].mean()

    for key, value in context.items():
        if pd.isna(value):
            if key in KNOWN_SPARSE_CONTEXT_FEATURES:
                # Log as INFO without a console warning for known sparse features
                log_issue(
                    log_list=issue_log,
                    level='INFO',
                    function_name='extract_window_context',
                    reason=f"Known sparse context feature '{key}' is NaN.",
                    trip_id=trip_id,
                    seg_id=seg_id,
                    win_id=win_id
                )
            else:
                # For other features, a NaN might be unexpected, so keep the warning
                warnings.warn(f"Context feature '{key}' is entirely NaN for this window.")
                log_issue(
                    log_list=issue_log,
                    level='WARNING',
                    function_name='extract_window_context',
                    reason=f"Context feature '{key}' is entirely NaN for this window.",
                    trip_id=trip_id,
                    seg_id=seg_id,
                    win_id=win_id
                )

    return context

# --- Feature Extraction Functions ---
def get_windows_list(
    df: pd.DataFrame,
    trips_seg_dict: dict,
    issue_log: list,
    full_stop: int = 3,
    initial_windows: list = None,
    processed_trips: set = None,
    checkpoint_dir: str = None,
    checkpoint_interval: int = 200
) -> pd.DataFrame:
    """
    Generate fixed-length sliding windows of kinematic features for every
    valid driving segment in `trips_seg_dict`.

    Returns
    -------
    pd.DataFrame
        One row per window with columns:

    """
    # Initialize with pre-loaded windows if resuming
    windows = initial_windows if initial_windows is not None else []
    
    # Use a set for fast lookups
    processed_trip_ids = processed_trips if processed_trips is not None else set()

    # Use tqdm for a progress bar
    trips_to_process = {tid: labels for tid, labels in trips_seg_dict.items() if tid not in processed_trip_ids}

    print(f"Processing {len(trips_to_process)} remaining trips...")

    # We need a counter for trips processed in *this run*
    trips_processed_this_run = 0

    for trip_id, trip_labels in tqdm(trips_to_process.items(), desc="Processing Trips"):
        try: 
            trip_df = df[df[TRIP_ID_COL] == trip_id].copy()
            if len(trip_df) != len(trip_labels):
                warnings.warn(f"Length mismatch for trip_id {trip_id}. Skipping trip.")
                log_issue(issue_log, 'ERROR', 'get_windows_list', 
                        'Length mismatch between DataFrame and segmentation labels, trip skipped.', trip_id)
                continue

            trip_coords = list(
                trip_df[[LAT_COL, LON_COL]].dropna().astype(float).itertuples(index=False, name=None)
            )

            # Fetch the local WGS84 graph for THIS trip
            G_trip_wgs84 = get_graph_for_trip_corridor(trip_coords)

            if G_trip_wgs84 is None:
                warnings.warn(f"[{trip_id}] Skipping trip: Could not generate a local map corridor.")
                log_issue(issue_log, 'ERROR', 'get_windows_list', 
                        'Failed to generate local map corridor, trip skipped.', trip_id)
                continue # Move to the next trip

            unique_seg_ids = np.unique(trip_labels[trip_labels > 0]).astype(int)
            if len(unique_seg_ids) == 0: 
                warnings.warn(f'{trip_id} Trip skipped: No valid positive segment IDs found.')
                log_issue(
                    issue_log,
                    'INFO',
                    'get_windows_list',
                    'Trip skipped: No valid positive segment IDs found.',
                    trip_id=trip_id,
                    details={'unique_labels_found': np.unique(trip_labels).tolist()}
                )
                continue
            
            trip_starts_with_zero = False

            if trip_labels[0] == 1 and trip_df.iloc[0][SPEED_ARRAY_COL][0] == 0:
                trip_starts_with_zero = True        
            
            for seg_id in unique_seg_ids:

                labeled_indices = np.where(trip_labels == seg_id)[0]
                if len(labeled_indices) == 0: 
                    warnings.warn(f"No samples for trip_id {trip_id} seg {seg_id}!")
                    log_issue(issue_log, 'WARNING', 'get_windows_list', 
                            'Segment ID found but no corresponding labeled samples in trip_labels array.', trip_id, seg_id)
                    continue
                
                first_labeled_idx = labeled_indices[0]
                last_labeled_idx = labeled_indices[-1]

                segment_start_idx = find_seg_start_idx(trip_labels, first_labeled_idx, full_stop=full_stop)
                segment_span_indices = np.arange(segment_start_idx, last_labeled_idx + 1)

                df_seg = trip_df.iloc[segment_span_indices,:]  

                df_1hz = reconstruct_1hz_path(df_seg, trip_id, seg_id, G_trip_wgs84, # Pass WGS84 graph
                                            query_dem_opentopo, OSRM_BASE_URL, 
                                            MIN_MATCH_CONFIDENCE, issue_log)

                if df_1hz is None or df_1hz.empty:
                    warnings.warn(f"[{trip_id}-{seg_id}] Skipping: 1Hz path reconstruction failed.")
                    continue          

                # --- 3. Calculate Kinematics & Context ---
                df_1hz_kinematics, trip_starts_with_zero = calculate_1hz_kinematics(df_1hz, df_seg, trip_starts_with_zero)

                # --- Calculate Wind Interaction ---
                df_1hz_kinematics = calculate_wind_components(df_1hz_kinematics)

                WIND_DRAG_FACTOR = 0.05 # Ensure this constant is defined or accessible here

                # This correctly uses grade_corrected_accel, not the raw long_accel_mps2
                wind_component_accel = df_1hz_kinematics['headwind_mps'] * WIND_DRAG_FACTOR
                df_1hz_kinematics['propulsive_accel_mps2'] = df_1hz_kinematics['grade_corrected_accel'] + wind_component_accel

                # Get vehicle mass safely
                weight_value = df_seg['empty_weight_kg'].iloc[0]
                vehicle_mass_kg = pd.to_numeric(weight_value, errors='coerce')

                if pd.isna(vehicle_mass_kg):
                    # If mass is unknown, power features will be NaN
                    warnings.warn(f"Could not determine vehicle mass for trip {trip_id}, seg {seg_id}. Power features will be NaN.")
                    log_issue(issue_log, 'WARNING', 'get_windows_list', 
                              f'Could not convert vehicle weight to numeric. Power features will be NaN.', 
                              trip_id, seg_id, details={'original_value': str(weight_value)})
                    df_1hz_kinematics['propulsive_power_kw'] = np.nan
                    df_1hz_kinematics['propulsive_power_kw_per_kg'] = np.nan
                else:
                    # Calculate power features correctly
                    propulsive_force_n = vehicle_mass_kg * df_1hz_kinematics['propulsive_accel_mps2']
                    propulsive_power_watts = propulsive_force_n * df_1hz_kinematics['speed_mps']
                    df_1hz_kinematics['propulsive_power_kw'] = propulsive_power_watts / 1000.0
                    df_1hz_kinematics['propulsive_power_kw_per_kg'] = df_1hz_kinematics['propulsive_power_kw'] / vehicle_mass_kg

                # getting context: relevant info from: ['car_model', 'manufacturer', 'empty_weight_kg', 'battery_type', 'battery_health', 'trip_id', 'trip_datetime_start', 'trip_datetime_end', 'trip_date', 'trip_soc_start', 'trip_soc_end', 'trip_odo_start', 'trip_odo_end', 'trip_location_start', 'trip_location_end', 'timestamp', 'current_odo', 'current_soc', 'current_location', 'outside_temp', 'altitude_array', 'speed_array', 'latitude', 'longitude', 'battery_capacity_kWh', 'is_start_after_gap']
                # AND from df_1hz_kinematics for mining hard NEG. 
                # i think i'll just extract here, and in the next training script i'll encode, weight, vector, and store.
                segment_context = extract_segment_context(df_seg) # Static context for the whole segment

                # --- 4. Generate Sliding Windows ---
                seg_total_len = len(df_1hz_kinematics)
                win_id = 0
                for i in range(0, seg_total_len - WIN_LEN_S + 1, WIN_HOP_S): # check it
                    w = slice(i, i + WIN_LEN_S)

                    kinematics_df_window = df_1hz_kinematics.iloc[w].copy()

                    # Ensure all kinematic channels are present
                    for col in KINEMATIC_CHANNELS:
                        if col not in kinematics_df_window.columns:
                            kinematics_df_window[col] = np.nan
                            warnings.warn(f"[{trip_id}-{seg_id}-{win_id}] Kinematic channel '{col}' was missing and has been added as np.nan.")
                            log_issue(issue_log, 'WARNING', 'get_windows_list', 
                                    f"Kinematic channel '{col}' was missing, added as NaN.", trip_id, seg_id, win_id)

                    # Extract dynamic context for THIS specific window
                    window_context = extract_window_context(kinematics_df_window, issue_log, trip_id, seg_id, win_id, G_trip_wgs84)

                    # Build arrays for each kinematic channel (fixed length = WIN_LEN_S)
                    window_row = {
                        "trip_id": trip_id,
                        "seg_id":  seg_id,
                        "win_id":  win_id,
                        "start_sample": i,
                    }
                    # Static + dynamic context
                    window_row.update(segment_context)         # add segment-level metadata
                    window_row.update(window_context)          # add window-level aggregates

                    for col in KINEMATIC_CHANNELS:
                        arr = kinematics_df_window[col].to_numpy()
                        if arr.shape[0] != WIN_LEN_S:
                            warnings.warn(f"[{trip_id}-{seg_id}-{win_id}] {col} length {arr.shape[0]} != {WIN_LEN_S}, filling NaN.")
                            log_issue(issue_log, 'WARNING', 'get_windows_list', f"{col} wrong length; filled NaN.", trip_id, seg_id, win_id)
                            window_row[col] = np.full(WIN_LEN_S, np.nan, dtype='float32')
                        else:
                            window_row[col] = arr.astype('float32')

                        if np.all(np.isnan(arr)):
                            if col in KNOWN_SPARSE_KINEMATIC_CHANNELS:
                                # Log as INFO without a console warning
                                log_issue(
                                    log_list=issue_log,
                                    level='INFO',
                                    function_name='get_windows_list',
                                    reason=f"Known sparse kinematic channel '{col}' is entirely NaN.",
                                    trip_id=trip_id,
                                    seg_id=seg_id,
                                    win_id=win_id
                                )
                            else:
                                # Keep the original warning for other, more critical channels
                                warnings.warn(f"Kinematic channel '{col}' is entirely NaN")
                                log_issue(
                                    log_list=issue_log,
                                    level='WARNING',
                                    function_name='get_windows_list',
                                    reason=f"Kinematic channel '{col}' is entirely NaN for this window.",
                                    trip_id=trip_id,
                                    seg_id=seg_id,
                                    win_id=win_id
                                )

                    windows.append(window_row)
                    win_id += 1

            trips_processed_this_run += 1
            processed_trip_ids.add(trip_id) # Add the trip ID to the set of processed trips

            if checkpoint_dir and (trips_processed_this_run % checkpoint_interval == 0):
                save_checkpoint(windows, processed_trip_ids, checkpoint_dir) # Use the new helper

        except Exception as e: # <--- This is your existing block
            warnings.warn(f"CRITICAL FAILURE processing trip_id {trip_id}. Skipping. Error: {e}")
            # Log the full traceback to get the exact line number of the failure.
            log_issue(issue_log, 'CRITICAL', 'get_windows_list', 
                        f'Unhandled exception for trip, skipping. Error: {e}', trip_id,
                        details={'traceback': traceback.format_exc()}) # Add this line
            # -----------------------------
            processed_trip_ids.add(trip_id) 
            continue


    print(f"- Extracted a total of {len(windows)} windows.")
    if not windows:
        return pd.DataFrame() # Return empty if no windows were created at all
    
    df_windows = pd.DataFrame.from_records(windows)
    return df_windows

def flag_outliers(
    windows_df: pd.DataFrame,
    outlier_configs: dict,
    n_preview: int = 10
) -> pd.DataFrame:
    """
    Flags out-of-range samples in kinematic channels based on a flexible configuration.

    This function is logically hardened to handle pre-existing NaN/inf values and
    to ensure raw value outliers are removed BEFORE calculating differences,
    preventing the corruption of valid data.

    For each specified channel:
      1. Pre-existing NaN and ±inf values are identified and preserved as NaN.
      2. Raw values outside [min_raw, max_raw] are flagged.
      3. First-differences (spikes) are calculated on the cleaned data. Jumps
         outside [min_diff, max_diff] flag BOTH endpoints.

    Returns a copy of windows_df with cleaned arrays.
    """
    print("\n--- Flagging Outliers in Kinematic Channels (Hardened Logic) ---")
    df = windows_df.copy()

    channels_to_process = [ch for ch in outlier_configs.keys() if ch in df.columns]
    if not channels_to_process:
        print("Warning: No channels from the outlier configuration were found. Skipping.")
        return df

    print(f"Processing channels: {channels_to_process}")

    for ch in channels_to_process:
        cfg = outlier_configs.get(ch, {})
        try:
            # Load data and ensure it's a float type for NaN/inf handling
            X = np.vstack(df[ch].to_numpy()).astype(np.float32, copy=False)
        except Exception as e:
            print(f"Error: channel '{ch}' has inconsistent array lengths. Skipping. ({e})")
            continue

        num_wins, seq_len = X.shape

        # --- STEP 1: INITIAL SANITIZATION (Handles pre-existing NaN/±inf) ---
        # Create a mask to identify any value that is NOT a finite number.
        preexisting_invalid_mask = ~np.isfinite(X)
        
        # Reporting for pre-existing invalids
        num_preexisting_invalid = np.sum(preexisting_invalid_mask)
        if num_preexisting_invalid > 0:
            print(f"  - {ch} [INIT]: Found {num_preexisting_invalid} pre-existing NaN/inf values.")
        
        # --- STEP 2: RAW VALUE CHECK ---
        # This check is performed on the original data. np.inf will be caught by the
        # limits, and np.nan will correctly result in False.
        raw_outlier_mask = np.zeros_like(X, dtype=bool)
        if "raw_limits" in cfg:
            min_raw, max_raw = cfg["raw_limits"]
            # We only check finite values for raw limit violations.
            finite_vals_mask = ~preexisting_invalid_mask
            raw_outlier_mask[finite_vals_mask] = (X[finite_vals_mask] < min_raw) | (X[finite_vals_mask] > max_raw)

            # Reporting for raw outliers
            bad_raw = X[raw_outlier_mask]
            if bad_raw.size:
                mn, mx = np.nanmin(bad_raw), np.nanmax(bad_raw)
                print(
                    f"  - {ch} [RAW]: Found {bad_raw.size} raw value outliers "
                    f"({mn:.2f} to {mx:.2f}). limits=[{min_raw:.2f}, {max_raw:.2f}] "
                    f"Examples: {np.round(bad_raw[:n_preview], 2)}"
                )
            else:
                print(f"  - {ch} [RAW]: OK")

        # --- STEP 3: CREATE CLEAN SLATE FOR DIFFERENCE CALCULATION ---
        # Create a working copy where ALL invalid values (pre-existing or raw outliers) are NaN.
        # This is the CRITICAL step to prevent error propagation.
        X_cleaned = X.copy()
        X_cleaned[preexisting_invalid_mask | raw_outlier_mask] = np.nan

        # --- STEP 4: FIRST-DIFFERENCE (SPIKE) CHECK ---
        diff_spike_mask = np.zeros_like(X, dtype=bool)
        if "diff_limits" in cfg and seq_len >= 2:
            min_diff, max_diff = cfg["diff_limits"]
            
            # Calculate diffs on the sanitized data. Any diff involving a NaN will be NaN.
            diffs = np.diff(X_cleaned, axis=1)

            # A diff is only "bad" if it's a valid number AND outside the limits.
            diff_bad = np.zeros_like(diffs, dtype=bool)
            valid_diffs_mask = np.isfinite(diffs)
            diff_bad[valid_diffs_mask] = (diffs[valid_diffs_mask] < min_diff) | (diffs[valid_diffs_mask] > max_diff)

            if diff_bad.any():
                # Flag BOTH endpoints of each bad jump in the final mask
                diff_spike_mask[:, :-1] |= diff_bad
                diff_spike_mask[:, 1:]  |= diff_bad

                # Reporting for diff outliers
                bad_vals = diffs[diff_bad]
                mn, mx = np.nanmin(bad_vals), np.nanmax(bad_vals)
                print(
                    f"  - {ch} [DIFF]: Found {bad_vals.size} outlier jumps "
                    f"({mn:.2f} to {mx:.2f}). limits=[{min_diff:.2f}, {max_diff:.2f}] "
                    f"Examples: {np.round(bad_vals[:n_preview], 2)}"
                )
            else:
                print(f"  - {ch} [DIFF]: OK")

        # --- STEP 5: FINAL APPLICATION ---
        # Combine all masks and apply to the original data to create the final result.
        final_mask = preexisting_invalid_mask | raw_outlier_mask | diff_spike_mask
        
        if final_mask.any():
            final_X = X.copy()
            final_X[final_mask] = np.nan
            # Write back as a list of 1D arrays, ensuring they are copies
            df[ch] = [row.copy() for row in final_X]
        else:
            # If no changes, we can just write back the original (but still as copies for safety)
            df[ch] = [row.copy() for row in X]

    return df

def print_missing_values_report(
    windows_df: pd.DataFrame,
    kinematic_channels: list,
    n_preview: int = 15
) -> None:
    """
    Analyzes and prints a comprehensive report on missing values (NaNs) within
    the kinematic channel arrays of the windows DataFrame.

    The report includes:
    1.  **Sample-Level Summary:** Total count and percentage of NaN samples per channel.
    2.  **Window-Level Summary:** Total count and percentage of windows containing at least one NaN.
    3.  **Positional Summary:** The ratio of NaN-containing windows for each `win_id` (e.g., 1st, 2nd window in a segment).
    4.  **Absolute Index Summary:** The most problematic absolute sample indices across all segments, broken down per channel.

    Args:
        windows_df (pd.DataFrame): The DataFrame containing the extracted windows.
        kinematic_channels (list): A list of the kinematic channel columns to analyze.
        n_preview (int): The number of top items to show in the positional/absolute index reports.
    """
    print("\n" + "="*60)
    print("--- Comprehensive Missing Values Report for Kinematic Channels ---")
    print("="*60)

    if windows_df.empty:
        print("Warning: The windows DataFrame is empty. No report to generate.")
        return

    # 1. --- Data Preparation and Validation ---
    channels_to_process = [ch for ch in kinematic_channels if ch in windows_df.columns]
    if not channels_to_process:
        print("Warning: None of the specified kinematic channels were found in the DataFrame. Skipping report.")
        return

    try:
        # Stack all valid channel arrays into a 3D tensor for efficient analysis
        tensor = np.stack([np.vstack(windows_df[c].to_numpy()) for c in channels_to_process], axis=1)
    except ValueError as e:
        print(f"CRITICAL ERROR: Could not stack arrays. This is likely due to inconsistent lengths in one or more channels.")
        print(f"Please run `validate_window_arrays` to identify the problematic channel. Error: {e}")
        return

    n_win, n_chans, seq_len = tensor.shape
    grand_total_samples = tensor.size
    total_samples_per_chan = n_win * seq_len

    # The core of the analysis: a boolean mask of where NaNs are located
    nan_mask = np.isnan(tensor)

    # 2. --- Sample-Level NaN Summary ---
    print("\n--- 1. Summary by Individual Samples (Total NaNs) ---")
    for i, ch in enumerate(channels_to_process):
        n_missing = np.sum(nan_mask[:, i, :])
        pct_missing = (n_missing / total_samples_per_chan) * 100 if total_samples_per_chan > 0 else 0
        print(f"{ch:<25}: {n_missing:>8,} missing samples ({pct_missing:7.3f}%)")

    overall_missing = np.sum(nan_mask)
    overall_pct = (overall_missing / grand_total_samples) * 100 if grand_total_samples > 0 else 0
    print("-" * 60)
    print(f"{'TOTAL':<25}: {overall_missing:>8,} missing samples ({overall_pct:7.3f}% of all samples)")

    # 3. --- Window-Level NaN Summary ---
    print("\n--- 2. Summary by Windows (Windows with at least one NaN) ---")
    for i, ch in enumerate(channels_to_process):
        win_has_nan = nan_mask[:, i, :].any(axis=1)
        n_win_nan = np.sum(win_has_nan)
        pct_win_nan = (n_win_nan / n_win) * 100 if n_win > 0 else 0
        print(f"{ch:<25}: {n_win_nan:>8,} affected windows ({pct_win_nan:7.3f}%)")

    win_nan_any_channel = nan_mask.any(axis=(1, 2))
    n_win_nan_any = np.sum(win_nan_any_channel)
    pct_win_nan_any = (n_win_nan_any / n_win) * 100 if n_win > 0 else 0
    print("-" * 60)
    print(f"{'ANY CHANNEL':<25}: {n_win_nan_any:>8,} affected windows ({pct_win_nan_any:7.3f}%)")

    # 4. --- Per `win_id` (Positional) Summary ---
    print(f"\n--- 3. NaN Ratio by Window Position (Top {n_preview}) ---")
    if 'win_id' in windows_df.columns:
        win_stats = (
            pd.DataFrame({'win_id': windows_df["win_id"], 'has_nan': win_nan_any_channel})
            .groupby("win_id")
            .agg(n_with_nan=("has_nan", "sum"), total=("has_nan", "size"))
            .assign(ratio=lambda x: x["n_with_nan"] / x["total"])
            .sort_values("ratio", ascending=False)
        )
        if not win_stats.empty:
            for wid, row in win_stats.head(n_preview).iterrows():
                print(f"  - win_id {wid:<3}: {row.ratio:7.2%} NaN ratio ({int(row.n_with_nan):,}/{int(row.total):,} windows)")
        else:
            print("  No NaNs found to analyze by window position.")
    else:
        print("  `win_id` column not found. Skipping positional analysis.")

    # 5. --- Absolute Sample Index Summary (Per Channel) ---
    print(f"\n--- 4. Most Problematic Absolute Sample Indices (Top {n_preview} per Channel) ---")
    if 'start_sample' in windows_df.columns:
        start_samples = windows_df["start_sample"].to_numpy()
        idx_matrix = start_samples[:, None] + np.arange(seq_len)
        coverage_counts = np.bincount(idx_matrix.ravel())

        for i, ch in enumerate(channels_to_process):
            print(f"\n  Channel: {ch}")
            nan_mask_ch = nan_mask[:, i, :]
            if not np.any(nan_mask_ch):
                print("    No NaN samples found in this channel.")
                continue

            rows, offs = np.where(nan_mask_ch)
            abs_idx_nan_ch = start_samples[rows] + offs
            missing_counts_ch = np.bincount(abs_idx_nan_ch, minlength=len(coverage_counts))

            stats_ch = (
                pd.DataFrame({
                    "abs_idx": np.arange(len(coverage_counts)),
                    "n_missing": missing_counts_ch,
                    "coverage": coverage_counts
                })
                .query("coverage > 0 and n_missing > 0")
                .assign(ratio=lambda x: x.n_missing / x.coverage)
                .sort_values("ratio", ascending=False)
            )

            if not stats_ch.empty:
                for _, row in stats_ch.head(n_preview).iterrows():
                    idx = int(row.abs_idx)
                    print(f"    - Abs. Index {idx:<6}: {row.ratio:7.2%} NaN ratio ({int(row.n_missing):,}/{int(row.coverage):,} times)")
            else:
                # This case should be caught by the np.any check above, but is here for safety
                print("    No NaN samples found in this channel.")
    else:
        print("  `start_sample` column not found. Skipping absolute index analysis.")

    print("\n" + "="*60)
    print("--- End of Report ---")
    print("="*60)

def main():
    """Loads data, extracts segment features, saves results."""
    print("--- Feature Extraction Pipeline for driver embedding ---")

    print(f"\nInput cleaned data file: {CLEANED_DATA_PATH}")
    print(f"Input segmentation dict: {SEGMENTATION_DICT_PATH}")
    print(f"Output features file: {OUTPUT_FEATURE_PATH}")

    # --- Enhanced OSMnx Caching and Settings Configuration ---
    print("\n--- Configuring OSMnx Settings ---")
    # Ensure the cache folder exists before osmnx uses it
    os.makedirs(OSMNX_CACHE_FOLDER, exist_ok=True)
    
    ox.settings.use_cache = True
    ox.settings.cache_folder = OSMNX_CACHE_FOLDER
    ox.settings.timeout = REQUEST_TIMEOUT_S
    ox.settings.log_console = False           # Show query URLs and other useful logs
    ox.settings.overpass_rate_limit = True   # Automatically pause between queries to not overload the server
    print(f"OSMnx cache enabled at: {OSMNX_CACHE_FOLDER}")
    print(f"OSMnx request timeout set to: {REQUEST_TIMEOUT_S} seconds")

    # --- Initialize the central log ---
    issue_log = []

    try:
        # 1. Load Cleaned Data
        print("\n--- Loading Cleaned Data ---")
        df_clean = load_file(CLEANED_DATA_PATH)
        if not isinstance(df_clean, pd.DataFrame): raise TypeError("Cleaned data is not a DataFrame.")
        print(f"Loaded cleaned data with shape: {df_clean.shape}")

        # 2. Ensure Sorting using the Helper Function
        print(f"\n--- Sorting data by {TRIP_ID_COL}, {TIME_COL} ---")
        df_clean = sort_df_by_trip_time(df_clean, trip_id_col=TRIP_ID_COL, time_col=TIME_COL)
        print("Data sorting complete.")

        # 3. Load Segmentation Dictionary
        print(f"\n--- Loading Segmentation Dictionary ---")
        trips_seg_dict = load_file(SEGMENTATION_DICT_PATH)
        if not isinstance(trips_seg_dict, dict): raise TypeError("Segmentation data is not a dictionary.")
        print(f"Loaded segmentation dict with {len(trips_seg_dict)} trips.")

        # 4. Filter Segmentation Dict for Min Segments
        df_clean_filtered, filtered_trips_seg_dict = drop_trips_with_less_than_x_segs(
            df_clean, 
            trips_seg_dict, 
            trip_id_col=TRIP_ID_COL, 
            min_segments=MIN_SEGMENTS_PER_TRIP
        )

        print("--- RUNNING GEOGRAPHIC OUTLIER DIAGNOSTIC TEST ---")
        
        # Define a reasonable bounding box for mainland Western/Central Europe + NOR/GRC/UKR
        # We are being generous with these boundaries.
        LOGICAL_BBOX = {
            'north': 72.0,  # Northern tip of Norway
            'south': 34.0,  # South of Greece (Crete)
            'east': 41.0,   # Eastern Ukraine
            'west': -10.0   # West coast of Portugal
        }
        print(f"Checking against logical BBOX: {LOGICAL_BBOX}")

        # Find all rows that are OUTSIDE this logical bounding box
        outlier_rows = df_clean_filtered[
            (df_clean_filtered['latitude'] > LOGICAL_BBOX['north']) |
            (df_clean_filtered['latitude'] < LOGICAL_BBOX['south']) |
            (df_clean_filtered['longitude'] > LOGICAL_BBOX['east']) |
            (df_clean_filtered['longitude'] < LOGICAL_BBOX['west'])
        ]

        if outlier_rows.empty:
            print("DIAGNOSTIC PASSED: No points found outside the logical bounding box.")
        else:
            print(f"DIAGNOSTIC FAILED: Found {len(outlier_rows)} points outside the logical bounding box.")
            
            # Group the outliers by trip_id to see which trips are problematic
            outlier_trips = outlier_rows.groupby('trip_id').agg(
                outlier_points_count=('latitude', 'count'),
                min_lat=('latitude', 'min'),
                max_lat=('latitude', 'max'),
                min_lon=('longitude', 'min'),
                max_lon=('longitude', 'max')
            ).reset_index()

            print("\n--- The following trips contain outlier coordinates and MUST be investigated or removed: ---")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
                print(outlier_trips)
            
            problematic_trip_ids = outlier_trips['trip_id'].tolist()
            print("\nSUGGESTED ACTION: Manually add these trip IDs to a removal list in your script.")
            print(f"TRIP_IDS_TO_MANUALLY_REMOVE = {problematic_trip_ids}")

        # 2. Find all rows that have a duplicate combination of trip_id and timestamp
        print(f"Checking for duplicate timestamps within each '{TRIP_ID_COL}'...")
        
        # The 'keep=False' argument is crucial as it marks ALL occurrences of duplicates as True
        duplicate_mask = df_clean_filtered.duplicated(subset=[TRIP_ID_COL, TIME_COL], keep=False)
        
        # 3. Filter the DataFrame to show only the problematic rows
        duplicate_rows = df_clean_filtered[duplicate_mask]

        # 4. Report the findings
        if duplicate_rows.empty:
            print("\n✅ SUCCESS: No duplicate timestamps found within any trip.")
        else:
            print(f"\n❌ FOUND: {len(duplicate_rows)} rows with duplicate timestamps.")
            print("Displaying the conflicting rows, sorted by trip and time:")
            
            # Sort the results to group the duplicates together for easy viewing
            sorted_duplicates = duplicate_rows.sort_values(by=[TRIP_ID_COL, TIME_COL])
            
            # Define which columns to show for context
            columns_to_display = [
                TRIP_ID_COL, 
                TIME_COL, 
                'current_odo', 
                'current_soc', 
                'latitude', 
                'longitude'
            ]
            
            # Ensure all columns exist before trying to display them
            displayable_cols = [col for col in columns_to_display if col in sorted_duplicates.columns]

            # Use pandas option context to make sure we see all the rows
            with pd.option_context('display.max_rows', None, 'display.width', 120):
                print(sorted_duplicates[displayable_cols])

        # 3. Calculate the time difference between consecutive rows within each trip
        # .dt.total_seconds() gives us a number to check
        time_diffs = df_clean_filtered.groupby(TRIP_ID_COL)[TIME_COL].diff().dt.total_seconds()

        # 4. A violation occurs if the difference is zero (duplicate) or negative (backward in time)
        violation_mask = time_diffs <= 0
        
        # Get the indices of the rows where the timestamp is not strictly increasing
        problem_indices = df_clean_filtered.index[violation_mask]

        # 5. Report the findings
        if not problem_indices.any():
            print("\n✅ SUCCESS: All timestamps are strictly increasing within each trip.")
        else:
            print(f"\n❌ FOUND: {len(problem_indices)} rows where the timestamp is not strictly increasing.")
            print("Displaying the context for each violation (the problematic row and the one before it):")

            # To show context, we also need the index of the row *before* the problem
            context_indices = sorted(list(set(problem_indices.to_list() + (problem_indices - 1).to_list())))
            
            # Filter the DataFrame to show only these context rows
            context_df = df_clean_filtered.loc[context_indices]

            # Add a marker to easily spot the issues
            context_df['issue_marker'] = np.where(context_df.index.isin(problem_indices), '<-- VIOLATION', '')

            columns_to_display = [
                'issue_marker',
                TRIP_ID_COL, 
                TIME_COL, 
                'current_odo', 
                'current_soc'
            ]
            displayable_cols = [col for col in columns_to_display if col in context_df.columns]

            with pd.option_context('display.max_rows', None, 'display.width', 150):
                print(context_df[displayable_cols])

        # 5. Interpolating Missing Altitudes
        print(f"\n--- Interpolating Missing Altitudes ---")
        df_clean_filtered = interp_missing_alts(
            df_clean_filtered,
            filtered_trips_seg_dict,
            max_len_interp=2,
            max_diff_between_neig=MAX_ALT_DIFF
        )

       # --- Load from checkpoints to resume ---
        print("\n--- Checking for existing checkpoints to resume ---")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # Define checkpoint file paths
        checkpoint_windows_path = os.path.join(CHECKPOINT_DIR, "windows_checkpoint.pickle")
        checkpoint_trips_path = os.path.join(CHECKPOINT_DIR, "processed_trips_checkpoint.pickle")

        windows_df = pd.DataFrame()
        
        initial_windows = []
        processed_trip_ids = set()

        if os.path.exists(checkpoint_windows_path) and os.path.exists(checkpoint_trips_path):
            print("Found checkpoint files. Loading...")
            try:
                initial_windows = load_file(checkpoint_windows_path)
                processed_trip_ids = load_file(checkpoint_trips_path)
                if not isinstance(initial_windows, list) or not isinstance(processed_trip_ids, set):
                    print("Warning: Checkpoint files are corrupt. Starting from scratch.")
                    initial_windows, processed_trip_ids = [], set()
                else:
                    print(f"Resuming. Loaded {len(initial_windows)} windows from {len(processed_trip_ids)} processed trips.")
            except Exception as e:
                print(f"Warning: Could not load checkpoint files. Error: {e}. Starting from scratch.")
                initial_windows, processed_trip_ids = [], set()
        else:
            print("No complete checkpoint found. Starting from scratch.")

        # 7. Extract windows list
        print(f"\n--- Extracting sliding windows of kinematic features ---")
        windows_df = get_windows_list(
            df_clean_filtered,
            filtered_trips_seg_dict,
            issue_log=issue_log,
            full_stop=FULL_STOP,
            initial_windows=initial_windows, # Pass the loaded windows
            processed_trips=processed_trip_ids, # Pass the loaded set of trip IDs
            checkpoint_dir=CHECKPOINT_DIR,
            checkpoint_interval=CHECKPOINT_INTERVAL
        )

        # 8. validate windows_df arrays length
        _ = validate_window_arrays(windows_df, KINEMATIC_CHANNELS, min_len=2, strict=True)

        # 9. Flag outliers
        # print(f"\n--- Flagging outliers ---")
        # windows_df = flag_outliers(
        #     windows_df,
        #     OUTLIER_CONFIGS,
        #     n_preview= 50
        # )

        # 10. Print missing values report 
        print(f"\n--- Print missing values report ---")
        # Pass the list of channels to the function
        print_missing_values_report(
            windows_df,
            kinematic_channels=KINEMATIC_CHANNELS
        )       

        # 11. Save Results
        if not windows_df.empty:
            print(f"\n--- Saving final consolidated windows DataFrame ---")
            output_dir = os.path.dirname(OUTPUT_FEATURE_PATH)
            base_file_name = os.path.splitext(os.path.basename(OUTPUT_FEATURE_PATH))[0]
            save_file(data=windows_df, path=output_dir, file_name=base_file_name, format=OUTPUT_FORMAT)
        else:
            print("\nWarning: No windows were generated or loaded. Nothing to process or save.")

        print("\n--- Feature Extraction Pipeline Finished ---")

    except (FileNotFoundError, TypeError, ValueError, KeyError, ImportError, KeyboardInterrupt, Exception) as e:
        print(f"\n--- Feature Extraction Pipeline Failed or was Interrupted ---")
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        # --- THIS BLOCK NOW RUNS ON ANY EXIT (SUCCESS, CRASH, CTRL+C) ---
        # 1. Save the final state of windows and processed trips
        print("\n--- Final Save on Exit ---")
        # `initial_windows` now contains all windows generated during the run
        # `processed_trip_ids` contains all trips attempted
        save_checkpoint(initial_windows, processed_trip_ids, CHECKPOINT_DIR)

        # 2. Save the issue log
        print("\n--- Processing and Saving Issue Log ---")
        if issue_log:
            log_df = pd.DataFrame(issue_log)
            log_df = log_df[['timestamp', 'level', 'function', 'trip_id', 'seg_id', 'win_id', 'reason', 'details']]
            try:
                log_df.to_csv(OUTPUT_LOG_PATH, index=False)
                print(f"Successfully saved {len(log_df)} issues to: {OUTPUT_LOG_PATH}")
                print("\nIssue Summary:")
                print(log_df.groupby(['function', 'reason']).size().reset_index(name='count').sort_values('count', ascending=False))
            except Exception as e:
                print(f"CRITICAL: Failed to save issue log to {OUTPUT_LOG_PATH}. Error: {e}")
        else:
            print("No issues were logged during the pipeline execution.")

        # 3. Save the final DataFrame if it was successfully created
        if not windows_df.empty:
            print(f"\n--- Saving final consolidated windows DataFrame ---")
            output_dir = os.path.dirname(OUTPUT_FEATURE_PATH)
            base_file_name = os.path.splitext(os.path.basename(OUTPUT_FEATURE_PATH))[0]
            save_file(data=windows_df, path=output_dir, file_name=base_file_name, format=OUTPUT_FORMAT)
        else:
            print("\nWarning: Final windows DataFrame was empty. Nothing to save to final output path.")

        print("\n--- Feature Extraction Pipeline Finished ---")

if __name__ == '__main__':
    main()