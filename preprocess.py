import pandas as pd
import numpy as np
import os
import math
import warnings # To show warnings for skipped steps
import pickle # Needed by load_file for pickle format
import traceback # For detailed error printing
import geopandas as gpd
from shapely.geometry import Point as ShapelyPoint
import h3
import traceback 
from datetime import datetime, timedelta # For time handling # For meteostat time range
from itertools import chain, combinations # Used for flattening lists efficiently
from pathlib import Path
# --- Import Meteostat ---
try:
    from meteostat import Stations, Point, Hourly
    # Suppress FutureWarning messages about the deprecated 'H' usage in meteostat/pandas
    warnings.filterwarnings("ignore", message="'H' is deprecated", category=FutureWarning)
    METEOSTAT_AVAILABLE = True
except ImportError:
    print("Warning: meteostat library not found. Weather comparison step will be skipped.")
    METEOSTAT_AVAILABLE = False

# Suppress specific FutureWarning messages from pandas/meteostat if needed
warnings.filterwarnings("ignore", message="'H' is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="The behavior of DatetimeProperties.to_pydatetime is deprecated", category=FutureWarning)

# --- Import necessary functions from HelperFuncs ---
try:
    # Add safe_mean to the import list
    from HelperFuncs import (
        load_file, haversine_distance, safe_mean, h3_to_latlon, save_file, get_gadm_countries_geometry, remove_manual_trips,
        find_points_outside_target_area, plot_trip_speed_profiles, count_outliers_in_array, get_array_length,
        ratio_rule,                       # Symmetric‐ratio comparisons 
        simple_threshold_rule,            # Single‐column thresholds 
        grade_rule,                       # Grade/altitude vs. distance checks
        time_gap_rule,                    # Timestamp gap checks 
        charging_while_moving_rule,       # Charging‐while‐moving physics rule
        energy_per_100km_rule,            # Energy‐per‐100 km consumption rule 
        uphill_no_energy_rule,            # Uphill‐no‐discharge rule 
        descent_no_speed_change_rule,     # Descent‐no‐speed‐change rule
        speed_spike_while_stationary_rule,# Internal‐speed‐spike when stationary 
        odo_jump_short_time_rule,         # Odometer jump in short time 
        teleport_with_no_energy_change_rule,  # GPS jump without SoC change
        unrealistic_regen_on_descent_rule,    # Implausible regen on descent 
        analyze_feature_misalignment,    # Main misalignment engine 
        build_report,                    # Build flagged‐rows report 
        print_misalignment_report,       # Pretty‐print the report 
        MisalignmentConfig,              # Config dataclass 
        Rule,                             # Rule dataclass :contentReference[oaicite:16]{index=16}
        speed_spike_rule, # Excessive acceleration within 1-sec samples
        alt_spike_across_rows_rule # Large altitude jump between rows
) 
except ImportError:
    print("Error: Could not import from HelperFuncs. Ensure HelperFuncs.py is accessible.")
    # Define placeholders if needed for testing, but ideally fix the import
    def load_file(filepath, **kwargs): raise NotImplementedError("HelperFuncs not found")
    def save_file(data, path, file_name, format='pickle', index=False): raise NotImplementedError("HelperFuncs not found")
    def haversine_distance(lat1, lon1, lat2, lon2): raise NotImplementedError("HelperFuncs not found")
    def safe_mean(val): raise NotImplementedError("HelperFuncs not found")
    def h3_to_latlon(h3_index): raise NotImplementedError("HelperFuncs not found")
    def get_gadm_countries_geometry(h3_index): raise NotImplementedError("HelperFuncs not found")
    def find_points_outside_target_area(h3_index): raise NotImplementedError("HelperFuncs not found")

# --- Configuration Block (Moved outside main for better visibility) ---
INPUT_FILE_PATH = r'C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\df_sample\df_sample.parquet'
OUTPUT_FILE_PATH = r'C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\clean_df\clean_df.parquet'
# --- DEFINE THE PLOT OUTPUT DIRECTORY ---
SPEED_OUTLIERS_PLOT_DIR = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\speed_outliers_plots"
GADM_FOLDER_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\gadm41"

# --- Preprocessing Thresholds ---
EXPECTED_SPEED_ARRAY_SAMPLES = 60
EXPECTED_ALT_ARRAY_SAMPLES = 10

MIN_VALID_SPEED = 0
MAX_VALID_SPEED = 250
MIN_VALID_SOC = 0
MAX_VALID_SOC = 100
MIN_VALID_TEMP = -15
MAX_VALID_TEMP = 46
MIN_VALID_ALTITUDE = -300 # Allow slightly below sea level
MAX_VALID_ALTITUDE = 3375 # Reasonable max altitude
MIN_VALID_BATT_HEALTH = 50 # SOH %
MAX_VALID_BATT_HEALTH = 100 # SOH %
MAX_VALID_ODO = 1_000_000 # Max odometer reading (km)
MAX_VALID_WEIGHT = 4000 # Max empty weight (kg)
MIN_VALID_WEIGHT = 950 # Min empty weight (kg)

MAX_TIME_GAP_SECONDS = 61 
MIN_TIME_GAP_SECONDS = 59 
POS_MAX_SPEED_DIFF = 25 
NEG_MAX_SPEED_DIFF = 30
POS_MAX_ALT_DIFF = 9.5 # in 10 sec
NEG_MAX_ALT_DIFF = 10 # in 10 sec
MAX_LOCATION_DIFF = 0.08 # in km
POS_MAX_ODO_DIFF = 2.5 # in km
NEG_MAX_ODO_DIFF = 0.0001 # in km
NEG_MAX_SOC_DIFF = 3.55
POS_MAX_SOC_DIFF = 1.15
POS_MAX_TEMP_DIFF = 3.1
NEG_MAX_TEMP_DIFF = 3.6
NEG_MAX_SOH_DIFF = 0.1 

# MIN_TRIP_KM = 0.5 # Increased slightly from 0 to avoid noise trips
# MAX_TRIP_KM = 450.0 # max length

DROP_INFREQ_MODELS = True
MIN_MODEL_PROP = 0.01

FILTER_ZERO_COORDINATES = True # Filter individual (0,0) points?
FILTER_MOSTLY_ZERO_TRIPS = True # Filter trips with >75% (0,0) points?
ZERO_TRIP_THRESHOLD = 0.75 # Threshold for filtering trips

# MAX_WEATHER_STATION_DIST_KM = 35 # Max distance to nearest station for weather data to be considered reliable
MAX_TEMP_DIFFERENCE_C = 5 # Max allowed diff between vehicle temp and weather temp
FETCH_WEATHER_DATA = False # Set to False to skip weather fetching/comparison
PROCESS_SUBSET = None # Set to an integer (e.g., 5000) to process only the first N rows for testing
VALIDATE_POINTS_IN_AREA = False # Set to False to skip points in area validation 

# --- Column Rename Mapping ---
COLUMN_RENAME_MAP = {
    'FAMILY_LABEL': 'car_model','COMMERCIAL_BRAND_COMPONENT_TYP': 'manufacturer',
    'TYPE_OF_TRACTION_CHAIN': 'traction_type','EMPTY_WEIGHT_KG': 'empty_weight_kg',
    'DESTINATION_COUNTRY_CODE': 'destination_country_code','BATTERY_TYPE': 'battery_type',
    'LEV_BATT_CAPC_HEAL': 'battery_health','CYCLE_ID': 'trip_id',
    'MESS_ID_START': 'message_session_id','DATETIME_START': 'trip_datetime_start',
    'DATETIME_END': 'trip_datetime_end','dt': 'trip_date',
    'SOC_START': 'trip_soc_start','SOC_END': 'trip_soc_end',
    'ODO_START': 'trip_odo_start','ODO_END': 'trip_odo_end',
    'geoindex_10_start': 'trip_location_start','geoindex_10_end': 'trip_location_end',
    'HEAD_COLL_TIMS': 'timestamp','PERD_VEHC_LIFT_MILG': 'current_odo',
    'LEV_BATT_LOAD_LEVL': 'current_soc','geoindex_10': 'current_location',# This is the H3 index column
    'latitude': 'latitude', # Include even if not present in input, will be created
    'longitude': 'longitude', # Include even if not present in input, will be created
    'PERD_VEHC_OUTS_TEMP': 'outside_temp','PERD_ALTT': 'altitude_array',
    'PERD_VEHC_LONG_SPD': 'speed_array',
}

# --- Standard Column Names (used internally after renaming) ---
TRIP_ID_COL = 'trip_id'
TIME_COL = 'timestamp'
ODO_COL = 'current_odo'
SOC_COL = 'current_soc'
H3_COL = 'current_location' # Standard name for the H3 index column
LAT_COL = 'latitude'       # Standard name for the latitude column TO BE CREATED
LON_COL = 'longitude'      # Standard name for the longitude column TO BE CREATED
SPEED_ARRAY_COL = 'speed_array'
ALT_ARRAY_COL = 'altitude_array'
VEHICLE_TEMP_COL = 'outside_temp' # Standard name for vehicle temp
BATT_HEALTH_COL = 'battery_health'
WEIGHT_COL = 'empty_weight_kg'
MSG_SESSION_ID_COL = 'message_session_id'
CAR_MODEL_COL = "car_model"
MANUFACTURER_COL = "manufacturer"
BATTERY_TYPE_COL = "battery_type"

# Columns expected constant per trip
STATIC_TRIP_COLS = [
    'car_model', 'manufacturer', 'traction_type', 'empty_weight_kg',
    'destination_country_code', 'battery_type',
    'message_session_id', 'trip_datetime_start', 'trip_datetime_end',
    'trip_date', 'trip_soc_start', 'trip_soc_end', 'trip_odo_start',
    'trip_odo_end', 'trip_location_start', 'trip_location_end'
]
# Columns for duplicate row check
DUPLICATE_CHECK_COLS_STD = [TRIP_ID_COL, TIME_COL]
# -------------------------

def filter_inconsistent_trip_data(df, trip_id_col, cols_to_check):
    """Filters out trips where static columns have multiple unique values."""
    print(f"\n--- Checking Trip Data Consistency (Static Cols) ---")
    if trip_id_col not in df.columns:
        print(f"Warning: Trip ID column '{trip_id_col}' not found. Skipping check.")
        return df
    cols_present = [col for col in cols_to_check if col in df.columns]
    if not cols_present:
        print("Warning: None of the specified static columns exist. Skipping check.")
        return df
    print(f" - Checking consistency for: {cols_present}")
    try:
        nunique_counts = df.groupby(trip_id_col)[cols_present].nunique(dropna=True)
    except Exception as e:
         print(f"Warning: Error during groupby/nunique check. Skipping. Error: {e}")
         return df
    inconsistent_trip_mask = (nunique_counts > 1).any(axis=1)
    trip_ids_to_drop = nunique_counts.index[inconsistent_trip_mask]
    if not trip_ids_to_drop.empty:
        num_to_drop = len(trip_ids_to_drop)
        print(f"Found {num_to_drop} trips with inconsistent static values.")
        max_details_to_print = 10
        inconsistent_details = nunique_counts[inconsistent_trip_mask]
        for i, trip_id in enumerate(trip_ids_to_drop):
            if i >= max_details_to_print: print(f"   ... (details omitted for remaining trips)"); break
            problem_cols = inconsistent_details.loc[trip_id][inconsistent_details.loc[trip_id] > 1].index.tolist()
            print(f" - Trip ID: {trip_id}, Inconsistent Columns: {problem_cols}")
        df_filtered = df[~df[trip_id_col].isin(trip_ids_to_drop)].copy()
        print(f"Result shape after removing inconsistent trips: {df_filtered.shape}")
        return df_filtered
    else:
        print("No inconsistent trips found based on static columns.")
        print(f"Result shape: {df.shape}")
        return df

def remove_duplicates(df, subset_cols):
    """Remove duplicate rows based on a specified subset of standard columns."""
    print("\n--- Removing Duplicate Rows ---")
    subset_cols_present = [col for col in subset_cols if col in df.columns]
    if not subset_cols_present:
        print("Warning: No standard columns found for duplicate check. Skipping.")
        return df
    initial_rows = len(df)
    try:
        df = df.drop_duplicates(subset=subset_cols_present, keep='first')
        print(f"Result shape: {df.shape} (Removed {initial_rows - len(df)} rows based on {subset_cols_present})")
    except TypeError as e:
        print(f"Warning: TypeError during drop_duplicates. Skipping. Error: {e}")
    return df

def check_and_drop_correlated_id(df, primary_id_col, secondary_id_col):
    """Checks if secondary ID is redundant w.r.t primary ID and drops if true."""
    print(f"\n--- Checking Correlation: '{primary_id_col}' vs '{secondary_id_col}' ---")
    if not all(col in df.columns for col in [primary_id_col, secondary_id_col]):
        print(f"Warning: Missing one or both ID columns. Skipping check.")
        return df
    try:
        nunique_per_primary = df.groupby(primary_id_col)[secondary_id_col].nunique(dropna=True)
        is_correlated = (nunique_per_primary <= 1).all()
        if is_correlated:
            print(f"'{secondary_id_col}' is redundant. Dropping.")
            df_filtered = df.drop(columns=[secondary_id_col])
            print(f"Result shape: {df_filtered.shape}")
            return df_filtered
        else:
            inconsistent_groups = nunique_per_primary[nunique_per_primary > 1]
            print(f"'{secondary_id_col}' NOT redundant ({len(inconsistent_groups)} groups have >1 value). Keeping.")
            print(f"Result shape: {df.shape}")
            return df
    except Exception as e:
        print(f"Warning: Error during correlation check. Skipping drop. Error: {e}")
        return df

def remove_constant_cols(df):
    """Remove constant or all-NaN columns."""
    const_cols_to_drop = []
    print("\n--- Checking for Constant/All-NaN Columns ---")
    for col in df.columns:
        # Skip array/list columns
        if df[col].iloc[0:min(100, len(df))].apply(lambda x: isinstance(x, (list, np.ndarray))).any(): # Check sample
            # print(f" - Skipping constant check for array/list column '{col}'.")
            continue
        try:
            num_unique = df[col].nunique(dropna=False)
            if num_unique == 1:
                first_valid_index = df[col].first_valid_index()
                if first_valid_index is not None: const_value = df[col].loc[first_valid_index]
                elif df[col].isnull().all(): const_value = "All NaN"
                else: const_value = "Unknown Constant"
                const_value_str = str(const_value);
                if len(const_value_str) > 70: const_value_str = const_value_str[:67] + '...'
                print(f" - Found constant '{col}': {const_value_str}")
                const_cols_to_drop.append(col)
        except TypeError: pass # print(f" - Skipping constant check for column '{col}' due to TypeError.")
        except Exception as e: print(f" - Error checking uniqueness for column '{col}': {e}. Skipping.")
    if const_cols_to_drop:
        print(f"Removing constant/all-NaN columns: {const_cols_to_drop}")
        df = df.drop(columns=const_cols_to_drop)
    else: print("No constant value or all-NaN columns found/checked to remove.")
    print(f"Result shape: {df.shape}")
    return df

def validate_and_filter_latlon(df, lat_col='latitude', lon_col='longitude', trip_id_col='trip_id',
                               filter_zeros=True, zero_tolerance=1e-3,
                               filter_mostly_zero_trips=True, zero_trip_threshold=0.95,
                               max_trips_to_print=10):
    """
    Performs sanity checks and filtering on latitude and longitude columns:
    1. Checks for NaN values.
    2. Filters rows with values outside valid ranges [-90, 90] for lat, [-180, 180] for lon.
    3. Optionally analyzes and filters rows where both lat/lon are close to zero.
    4. Optionally filters *entire trips* where the proportion of near-(0,0) points
       exceeds a threshold.

    Args:
        df (pd.DataFrame): Input DataFrame.
        lat_col (str): Name of the latitude column.
        lon_col (str): Name of the longitude column.
        trip_id_col (str): Name of the trip identifier column.
        filter_zeros (bool): If True, filter out individual rows where both lat/lon are near zero.
        zero_tolerance (float): Tolerance for checking if coordinates are close to zero.
        filter_mostly_zero_trips (bool): If True, filter entire trips consisting mostly of zero points.
        zero_trip_threshold (float): Proportion (0 to 1) of zero points required to filter a trip.
        max_trips_to_print (int): Max number of trip IDs with issues to print.

    Returns:
        pd.DataFrame: DataFrame with invalid rows and/or trips potentially removed.
    """
    print(f"\n--- Validating and Filtering Lat/Lon ('{lat_col}', '{lon_col}') ---")
    required_cols = [lat_col, lon_col, trip_id_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns ({required_cols}). Skipping validation.")
        return df

    initial_rows = len(df)
    df_filtered = df.copy() # Work on a copy

    # --- 1. Check for NaNs ---
    nan_lat_count = df_filtered[lat_col].isna().sum()
    nan_lon_count = df_filtered[lon_col].isna().sum()
    if nan_lat_count > 0 or nan_lon_count > 0:
        print(f" - Found {nan_lat_count} NaN values in '{lat_col}'.")
        print(f" - Found {nan_lon_count} NaN values in '{lon_col}'.")
        # Drop NaNs before range/zero checks as they interfere
        df_filtered = df_filtered.dropna(subset=[lat_col, lon_col])
        print(f"   Dropped {initial_rows - len(df_filtered)} rows with NaN lat/lon.")
        if df_filtered.empty:
             print("DataFrame empty after dropping NaN lat/lon.")
             return df_filtered

    # Ensure columns are numeric after dropping NaNs
    df_filtered[lat_col] = pd.to_numeric(df_filtered[lat_col], errors='coerce')
    df_filtered[lon_col] = pd.to_numeric(df_filtered[lon_col], errors='coerce')
    # Check again if coercion created NaNs (shouldn't if previous dropna worked)
    df_filtered = df_filtered.dropna(subset=[lat_col, lon_col])
    if df_filtered.empty:
        print("DataFrame empty after numeric conversion/dropna.")
        return df_filtered

    # --- 2. Check Valid Range ---
    invalid_lat_mask = (df_filtered[lat_col] < -90) | (df_filtered[lat_col] > 90)
    invalid_lon_mask = (df_filtered[lon_col] < -180) | (df_filtered[lon_col] > 180)
    invalid_range_mask = invalid_lat_mask | invalid_lon_mask
    num_invalid_range = invalid_range_mask.sum()
    if num_invalid_range > 0:
        print(f" - Found {num_invalid_range} rows with lat/lon outside valid ranges.")
        df_filtered = df_filtered[~invalid_range_mask] # Filter out invalid ranges
        print(f"   Dropped {num_invalid_range} rows due to invalid ranges.")
    else:
        print(" - All lat/lon values are within valid ranges.")

    if df_filtered.empty:
        print("DataFrame empty after range filtering.")
        return df_filtered

    # --- 3. Analyze and Optionally Filter Near-(0,0) Points/Trips ---
    if filter_zeros or filter_mostly_zero_trips:
        zero_lat_mask = np.isclose(df_filtered[lat_col], 0.0, atol=zero_tolerance)
        zero_lon_mask = np.isclose(df_filtered[lon_col], 0.0, atol=zero_tolerance)
        zero_zero_mask = zero_lat_mask & zero_lon_mask
        num_zero_zero = zero_zero_mask.sum()

        if num_zero_zero > 0:
            print(f"\n - Found {num_zero_zero} rows with lat/lon close to (0, 0) (tolerance={zero_tolerance}). Analyzing distribution...")
            zero_points_df = df_filtered.loc[zero_zero_mask, [trip_id_col]]
            unique_trips_with_zeros = zero_points_df[trip_id_col].unique()
            num_unique_trips_with_zeros = len(unique_trips_with_zeros)
            print(f"   These points belong to {num_unique_trips_with_zeros} unique trip(s).")

            # --- Check for trips to filter entirely ---
            trips_to_drop_entirely = set()
            if filter_mostly_zero_trips:
                total_points_per_trip = df_filtered[trip_id_col].value_counts()
                zero_points_per_trip = zero_points_df[trip_id_col].value_counts()
                trip_counts = pd.merge(
                    total_points_per_trip.rename('total_count'),
                    zero_points_per_trip.rename('zero_count'),
                    left_index=True, right_index=True, how='left'
                ).fillna({'zero_count': 0})
                trip_counts['zero_ratio'] = trip_counts['zero_count'] / trip_counts['total_count']

                mostly_zero_mask = trip_counts['zero_ratio'] > zero_trip_threshold
                trips_to_drop_entirely = set(trip_counts.index[mostly_zero_mask])
                num_trips_mostly_zeros = len(trips_to_drop_entirely)

                if num_trips_mostly_zeros > 0:
                    print(f"   Identified {num_trips_mostly_zeros} trip(s) where >{zero_trip_threshold*100:.0f}% of points are near (0,0). These trips will be removed entirely.")
                    #print(f"     Example Trip IDs (up to {max_trips_to_print}): {list(trips_to_drop_entirely)[:max_trips_to_print]}" + ("..." if num_trips_mostly_zeros > max_trips_to_print else ""))
                    # Filter out these trips
                    df_filtered = df_filtered[~df_filtered[trip_id_col].isin(trips_to_drop_entirely)]
                    print(f"   Shape after removing mostly-zero trips: {df_filtered.shape}")
                else:
                    print(f"   No trips found exceeding the {zero_trip_threshold*100:.0f}% threshold for near-(0,0) points.")

            # --- Filter individual zero points (if enabled and not already dropped via trip filter) ---
            if filter_zeros:
                # Identify zero points *remaining* after potential trip filtering
                zero_lat_mask_remaining = np.isclose(df_filtered[lat_col], 0.0, atol=zero_tolerance)
                zero_lon_mask_remaining = np.isclose(df_filtered[lon_col], 0.0, atol=zero_tolerance)
                zero_zero_mask_remaining = zero_lat_mask_remaining & zero_lon_mask_remaining
                num_zero_zero_remaining = zero_zero_mask_remaining.sum()

                if num_zero_zero_remaining > 0:
                    # Only filter if not already handled by trip removal
                    if not filter_mostly_zero_trips or num_trips_mostly_zeros == 0:
                         print(f"   Filtering {num_zero_zero_remaining} individual rows near (0, 0).")
                         df_filtered = df_filtered[~zero_zero_mask_remaining]
                    else:
                         # Check if any zero points remain *after* trip filtering
                         # This can happen if filter_mostly_zero_trips is True but some trips had < threshold % zeros
                         zero_points_in_kept_trips = df_filtered.loc[zero_zero_mask_remaining, trip_id_col].nunique()
                         if zero_points_in_kept_trips > 0:
                             print(f"   Filtering {num_zero_zero_remaining} individual near-(0,0) rows from trips below the {zero_trip_threshold*100:.0f}% threshold.")
                             df_filtered = df_filtered[~zero_zero_mask_remaining]
                         else:
                              print("   All near-(0,0) points were part of the removed trips.")

                else:
                    print("   No individual near-(0,0) rows remaining to filter.")
            else:
                 print("   Skipping filtering of individual near-(0,0) rows.")

        else: # num_zero_zero == 0 initially
            print(f" - No rows found with lat/lon close to (0, 0) (tolerance={zero_tolerance}).")
    else:
         print(" - Skipping checks related to (0, 0) coordinates.")


    rows_dropped_total = initial_rows - len(df_filtered)
    print(f"\nResult shape after Lat/Lon validation: {df_filtered.shape} (Removed {rows_dropped_total} rows total)")
    return df_filtered

# --- Weather Fetching Function ---
def get_weather_for_timestamp(row, lat_col='latitude', lon_col='longitude', time_col='timestamp', h3_col='current_location', max_station_dist_km=50.0):
    """
    Fetches hourly weather data for a specific row's timestamp and location,
    adds prefixes to weather columns, and includes nearest station info.

    Returns a dictionary suitable for use with apply(..., result_type='expand').
    Keys are prefixed with 'weather_' or 'station_'.
    """
    # Initialize result with NaNs for all potential weather/station columns
    result = {
        'weather_temp': np.nan, 'weather_dwpt': np.nan, 'weather_rhum': np.nan,
        'weather_prcp': np.nan, 'weather_snow': np.nan, 'weather_wdir': np.nan,
        'weather_wspd': np.nan, 'weather_wpgt': np.nan, 'weather_pres': np.nan,
        'weather_tsun': np.nan, 'weather_coco': np.nan,
        'station_id': None, 'station_dist_km': np.nan
    }

    # --- Get Lat/Lon ---
    lat, lon = np.nan, np.nan
    # Prioritize existing lat/lon columns if they exist
    if lat_col in row.index and lon_col in row.index and not pd.isna(row[lat_col]) and not pd.isna(row[lon_col]):
         lat, lon = row[lat_col], row[lon_col]
    # Fallback to H3 conversion if lat/lon are missing/NaN but H3 exists
    elif h3_col in row.index and not pd.isna(row[h3_col]):
        try:
            lat, lon = h3_to_latlon(row[h3_col]) # Use the H3 conversion helper
        except Exception as e:
            # warnings.warn(f"H3 conversion failed for index {row.name}: {e}")
            pass # Keep lat/lon as NaN

    # --- Get Timestamp ---
    timestamp = pd.NaT
    if time_col in row.index and not pd.isna(row[time_col]):
        timestamp = pd.to_datetime(row[time_col], errors='coerce')

    if pd.isna(lat) or pd.isna(lon) or pd.isna(timestamp):
        return pd.Series(result) # Return Series of NaNs if essential info missing

    # --- Get Nearest Station Info (Optional but recommended) ---
    try:
        # Ensure lat/lon are float for Meteostat
        lat, lon = float(lat), float(lon)
        pt = Point(lat, lon)
        # Find nearby stations (increase count slightly in case closest has no data)
        stations = Stations().nearby(lat, lon)
        station_df = stations.fetch(1) # Fetch the closest one

        if not station_df.empty:
            station_info = station_df.iloc[0]
            station_id = station_info.name
            station_lat = station_info['latitude']
            station_lon = station_info['longitude']
            # Use try-except for haversine, just in case
            try:
                distance = haversine_distance(lat, lon, station_lat, station_lon)
                result['station_id'] = station_id
                result['station_dist_km'] = distance
                # Optional check: Skip if station too far?
                # if distance > max_station_dist_km: return pd.Series(result)
            except Exception as e_dist:
                 print(f"Warning: Error calculating distance for index {row.name}: {e_dist}")

        # else: print(f"Warning: No nearby station found for index {row.name}")

    except Exception as e_stat:
        print(f"Warning: Error finding nearest station for index {row.name}: {e_stat}")

    # --- Fetch Hourly Weather Data ---
    try:
        start_hour = timestamp.floor('H')
        end_hour = start_hour # Fetch data for just that hour

        # Fetch data using the Point defined earlier
        weather_df = Hourly(pt, start_hour, end_hour).fetch()

        if not weather_df.empty:
            weather_data = weather_df.iloc[0] # Get the data for the hour
            # Populate result dict, adding prefix
            for col in result.keys():
                 meteo_col = col.replace('weather_', '')
                 if meteo_col in weather_data and pd.notna(weather_data[meteo_col]):
                      result[col] = weather_data[meteo_col]
        # else: print(f"Warning: No hourly weather data found for index {row.name} at {start_hour}")

    except Exception as e_fetch:
        print(f"Warning: Error fetching hourly weather for index {row.name} at {start_hour}: {e_fetch}")

    return pd.Series(result) # Return as a Series for apply


# --- Temperature Comparison Function ---
def flag_suspicious_temperatures(df, vehicle_temp_col='outside_temp', weather_temp_col='weather_temp',
                                 station_dist_col='station_dist_km', # New argument
                                 max_difference_c=15.0, flag_col_name='temp_suspicious_flag'):
    """
    Compares vehicle temperature to external weather temperature and flags large discrepancies.
    Includes station distance in the output for flagged rows.

    Args:
        df (pd.DataFrame): Input DataFrame (must contain temperature and distance columns).
        vehicle_temp_col (str): Name of the vehicle's temperature column.
        weather_temp_col (str): Name of the fetched weather temperature column.
        station_dist_col (str): Name of the column containing distance to the nearest station (km).
        max_difference_c (float): Maximum allowable absolute difference in Celsius.
        flag_col_name (str): Name for the new boolean flag column.

    Returns:
        pd.DataFrame: DataFrame with the added flag column.
    """
    print(f"\n--- Flagging Suspicious Temperatures (|Diff| > {max_difference_c}°C) ---")
    # Check required temperature columns
    required_temp_cols = [vehicle_temp_col, weather_temp_col]
    if not all(col in df.columns for col in required_temp_cols):
        print(f"Warning: Missing required temperature columns ({required_temp_cols}). Skipping check.")
        df[flag_col_name] = False
        return df

    # Check if station distance column exists for reporting (optional)
    has_station_dist = station_dist_col in df.columns
    if not has_station_dist:
        print(f"Warning: Station distance column '{station_dist_col}' not found. Discrepancy examples will not include distance.")

    # Ensure temps are numeric
    df[vehicle_temp_col] = pd.to_numeric(df[vehicle_temp_col], errors='coerce')
    df[weather_temp_col] = pd.to_numeric(df[weather_temp_col], errors='coerce')
    if has_station_dist:
        df[station_dist_col] = pd.to_numeric(df[station_dist_col], errors='coerce')


    # Calculate difference where both are available
    df['temp_difference'] = df[vehicle_temp_col] - df[weather_temp_col]

    # Create flag: True if difference is valid and exceeds threshold
    df[flag_col_name] = (df['temp_difference'].abs() > max_difference_c) & df['temp_difference'].notna()

    num_flagged = df[flag_col_name].sum()
    num_compared = df['temp_difference'].notna().sum()

    if num_compared > 0:
        print(f" - Compared {num_compared} rows with external temperature data.")
        print(f" - Flagged {num_flagged} rows as suspicious (difference > {max_difference_c}°C).")
        if num_flagged > 0:
            # Prepare columns to display in examples
            example_cols = [vehicle_temp_col, weather_temp_col, 'temp_difference']
            if has_station_dist:
                example_cols.append(station_dist_col) # Add distance if available

            print(f"   Example discrepancies ({', '.join(example_cols)}):")
            # Show rows where the flag is True, selecting the example columns
            print(df.loc[df[flag_col_name], example_cols].head(20))
    else:
        print(" - No external temperature data available for comparison.")

    # Optionally drop the temporary difference column
    # df = df.drop(columns=['temp_difference'])

    print(f"Result shape (with flag added): {df.shape}")
    return df

def filter_unrealistic_ranges(df, range_checks, trip_id_col='trip_id', max_details_to_print=10, plot_output_dir=None):
    """
    Filters or corrects rows with out-of-range values.
    - For 'battery_health', it corrects values and provides a detailed summary of the outcome.
    - For 'speed_array', it identifies and returns the trip IDs with violations without dropping rows.
    - For all other columns, it provides detailed reporting and then drops the out-of-range rows.
    """
    print("\n--- Filtering/Correcting Unrealistic Sensor Ranges ---")
    df_filtered = df.copy()
    speed_violation_trip_ids = set()  # Use a set to store unique trip IDs

    for col, (min_val, max_val) in range_checks.items():
        if col not in df_filtered.columns:
            print(f"Warning: Column '{col}' not found for range check. Skipping.")
            continue

        print(f"\n - Checking range for '{col}' ({min_val} to {max_val}).")
        
        sample = df_filtered[col].dropna().head(100)
        is_array_like = sample.apply(lambda x: isinstance(x, (list, np.ndarray))).any()

        out_of_range_mask = pd.Series(False, index=df_filtered.index)
        if is_array_like:
            print(f"   (Applying range check to each raw value in array column '{col}')")
            def is_any_value_out_of_range(arr):
                if not isinstance(arr, (list, np.ndarray)): return False
                return any(isinstance(v, (int, float)) and not (min_val <= v <= max_val) for v in arr)
            out_of_range_mask = df_filtered[col].apply(is_any_value_out_of_range)
        else:
            numeric_col = pd.to_numeric(df_filtered[col], errors='coerce')
            out_of_range_mask = numeric_col.notna() & ((numeric_col < min_val) | (numeric_col > max_val))

        num_out_of_range_rows = out_of_range_mask.sum()

        if num_out_of_range_rows == 0:
            print(f"   No out-of-range values found for '{col}'.")
            continue

        print(f"   Found {num_out_of_range_rows} rows with out-of-range values for '{col}'.")
        
        affected_trips_df = df_filtered.loc[out_of_range_mask, [trip_id_col]]
        affected_trip_ids = affected_trips_df[trip_id_col].unique()
        num_unique_trips = len(affected_trip_ids)
        print(f"     These rows belong to {num_unique_trips} unique trip(s).")

        # --- SPECIAL HANDLING FOR BATTERY_HEALTH ---
        if col == BATT_HEALTH_COL:
            print("     Detailed analysis (correcting values in-place):")
            
            trips_corrected_count = 0
            trips_all_nan_count = 0
            
            trip_groups = df_filtered.groupby(trip_id_col, observed=True)
            
            # --- CORRECTED LOOP STRUCTURE ---
            for i, trip_id in enumerate(affected_trip_ids):
                
                # --- Reporting Logic (Limited by max_details_to_print) ---
                if i < max_details_to_print:
                    trip_group = trip_groups.get_group(trip_id)
                    oor_mask_for_trip_reporting = out_of_range_mask.loc[trip_group.index]
                    
                    num_oor_rows_in_trip = oor_mask_for_trip_reporting.sum()
                    total_rows_in_trip = len(trip_group)
                    pct_oor_rows = (num_oor_rows_in_trip / total_rows_in_trip) * 100
                    oor_values_in_trip = trip_group.loc[oor_mask_for_trip_reporting, col].unique().tolist()
                    print(f"     - Trip ID: {trip_id} | OOR Rows: {num_oor_rows_in_trip}/{total_rows_in_trip} ({pct_oor_rows:.1f}%) | Value(s): {', '.join(map(str, oor_values_in_trip))}")
                    
                    if pct_oor_rows < 100.0:
                        first_violation_iloc = np.where(oor_mask_for_trip_reporting)[0][0]
                        context_window = 5
                        start_iloc = max(0, first_violation_iloc - context_window)
                        end_iloc = min(len(trip_group), first_violation_iloc + 1 + context_window)
                        context_cols = ['timestamp', col] if 'timestamp' in trip_group.columns else [col]
                        context_df = trip_group.iloc[start_iloc:end_iloc][context_cols].copy()
                        context_df['marker'] = ''
                        violation_original_index = trip_group.index[first_violation_iloc]
                        context_df.loc[violation_original_index, 'marker'] = '<-- OOR'
                        if start_iloc == 0:
                            trip_start_index = trip_group.index[0]
                            if context_df.loc[trip_start_index, 'marker'] == '':
                                context_df.loc[trip_start_index, 'marker'] = '<-- TRIP START'
                            else:
                                context_df.loc[trip_start_index, 'marker'] += ' (TRIP START)'
                        print("       Context for first OOR value:\n" + "\n".join([f"       {line}" for line in context_df.to_string().split('\n')]))
                
                elif i == max_details_to_print:
                    print(f"     ... (details for remaining {num_unique_trips - i} trips omitted)")

                # --- Correction Logic (RUNS FOR EVERY AFFECTED TRIP) ---
                trip_mask = df_filtered[trip_id_col] == trip_id
                oor_mask_for_correction = out_of_range_mask.loc[trip_mask]
                
                df_filtered.loc[trip_mask & oor_mask_for_correction, col] = np.nan
                trip_series = df_filtered.loc[trip_mask, col]
                
                if trip_series.dropna().empty:
                    trips_all_nan_count += 1
                else:
                    trips_corrected_count += 1
                    df_filtered.loc[trip_mask, col] = trip_series.bfill().ffill()

            # --- DETAILED SUMMARY (Now accurate) ---
            print(f"\n   Correction Summary for '{col}':")
            rows_still_nan = df_filtered.loc[out_of_range_mask, col].isnull().sum()
            rows_successfully_filled = num_out_of_range_rows - rows_still_nan
            
            print(f"   - Trips successfully corrected: {trips_corrected_count} (had at least one valid value to fill from)")
            print(f"   - Trips now with all NaN values: {trips_all_nan_count} (had no valid values)")
            print(f"   - Total rows successfully filled: {rows_successfully_filled}")
            print(f"   - Total rows that remain NaN: {rows_still_nan}")

        # --- NEW HANDLING FOR SPEED_ARRAY ---
        elif col == SPEED_ARRAY_COL:
            print("     Detailed analysis (reporting violations, not dropping rows):")
            speed_violation_trip_ids.update(affected_trip_ids)  # Collect the trip IDs
            
            trip_groups = df_filtered.groupby(trip_id_col, observed=True)
            for i, trip_id in enumerate(affected_trip_ids):
                if i >= max_details_to_print:
                    print(f"     ... (details for remaining {num_unique_trips - i} trips omitted)")
                    break
                
                trip_group = trip_groups.get_group(trip_id)
                oor_mask_for_trip = out_of_range_mask.loc[trip_group.index]
                
                num_oor_rows_in_trip = oor_mask_for_trip.sum()
                total_rows_in_trip = len(trip_group)
                pct_oor_rows = (num_oor_rows_in_trip / total_rows_in_trip) * 100
                
                oor_indices = np.where(oor_mask_for_trip)[0]
                num_blocks = np.sum(np.diff(oor_indices) > 1) + 1 if len(oor_indices) > 0 else 0
                
                total_samples_in_trip = trip_group[col].dropna().apply(len).sum()
                def count_outliers(arr):
                    if not isinstance(arr, (list, np.ndarray)): return 0
                    return sum(1 for v in arr if isinstance(v, (int, float)) and not (min_val <= v <= max_val))
                num_oor_samples_in_trip = trip_group.loc[oor_mask_for_trip, col].apply(count_outliers).sum()
                pct_oor_samples = (num_oor_samples_in_trip / total_samples_in_trip) * 100 if total_samples_in_trip > 0 else 0
                print(f"     - Trip ID: {trip_id} | "
                      f"Rows: {num_oor_rows_in_trip}/{total_rows_in_trip} ({pct_oor_rows:.1f}%) | "
                      f"Samples: {num_oor_samples_in_trip}/{total_samples_in_trip} ({pct_oor_samples:.1f}%) | "
                      f"Blocks: {num_blocks}")
            print(f"\n   Identified {len(affected_trip_ids)} trips with speed violations. Rows will NOT be dropped here.")

        # --- DEFAULT HANDLING FOR ALL OTHER COLUMNS ---
        else:
            print("     Detailed analysis (dropping these rows):")
            trip_groups = df_filtered.groupby(trip_id_col, observed=True)

            for i, trip_id in enumerate(affected_trip_ids):
                if i >= max_details_to_print:
                    print(f"     ... (details for remaining {num_unique_trips - i} trips omitted)")
                    break
                
                trip_group = trip_groups.get_group(trip_id)
                oor_mask_for_trip = out_of_range_mask.loc[trip_group.index]
                
                num_oor_rows_in_trip = oor_mask_for_trip.sum()
                total_rows_in_trip = len(trip_group)
                pct_oor_rows = (num_oor_rows_in_trip / total_rows_in_trip) * 100
                
                oor_indices = np.where(oor_mask_for_trip)[0]
                num_blocks = np.sum(np.diff(oor_indices) > 1) + 1 if len(oor_indices) > 0 else 0
                
                if is_array_like:
                    total_samples_in_trip = trip_group[col].dropna().apply(len).sum()
                    def count_outliers(arr):
                        if not isinstance(arr, (list, np.ndarray)): return 0
                        return sum(1 for v in arr if isinstance(v, (int, float)) and not (min_val <= v <= max_val))
                    num_oor_samples_in_trip = trip_group.loc[oor_mask_for_trip, col].apply(count_outliers).sum()
                    pct_oor_samples = (num_oor_samples_in_trip / total_samples_in_trip) * 100 if total_samples_in_trip > 0 else 0
                    print(f"     - Trip ID: {trip_id} | "
                          f"Rows: {num_oor_rows_in_trip}/{total_rows_in_trip} ({pct_oor_rows:.1f}%) | "
                          f"Samples: {num_oor_samples_in_trip}/{total_samples_in_trip} ({pct_oor_samples:.1f}%) | "
                          f"Blocks: {num_blocks}")
                else:
                    oor_values_in_trip = trip_group.loc[oor_mask_for_trip, col].unique().tolist()
                    oor_values_str = ", ".join(map(str, oor_values_in_trip))
                    print(f"     - Trip ID: {trip_id} | "
                          f"OOR Rows: {num_oor_rows_in_trip}/{total_rows_in_trip} ({pct_oor_rows:.1f}%) | "
                          f"Value(s): {oor_values_str} | "
                          f"Blocks: {num_blocks}")

                    if pct_oor_rows < 100.0:
                        first_violation_iloc = oor_indices[0]
                        context_window = 5
                        start_iloc = max(0, first_violation_iloc - context_window)
                        end_iloc = min(len(trip_group), first_violation_iloc + 1 + context_window)
                        
                        context_cols = [col]
                        if 'timestamp' in trip_group.columns:
                            context_cols.insert(0, 'timestamp')

                        context_df = trip_group.iloc[start_iloc:end_iloc][context_cols].copy()
                        context_df['marker'] = ''
                        
                        violation_original_index = trip_group.index[first_violation_iloc]
                        context_df.loc[violation_original_index, 'marker'] = '<-- OOR'
                        
                        if start_iloc == 0:
                            trip_start_index = trip_group.index[0]
                            if context_df.loc[trip_start_index, 'marker'] == '':
                                context_df.loc[trip_start_index, 'marker'] = '<-- TRIP START'
                            else:
                                context_df.loc[trip_start_index, 'marker'] += ' (TRIP START)'
                        
                        print("       Context for first OOR value:")
                        indented_context = "\n".join([f"       {line}" for line in context_df.to_string().split('\n')])
                        print(indented_context)

            initial_row_count = len(df_filtered)
            df_filtered = df_filtered[~out_of_range_mask]
            print(f"\n   Dropped {initial_row_count - len(df_filtered)} rows for '{col}'.")

    print(f"\nResult shape after range checks: {df_filtered.shape}")
    return df_filtered, list(speed_violation_trip_ids)

def find_inconsistent_sequences(df, trip_id_col, time_col, sequence_checks, max_details_to_print=10, context_window=5):
    """
    Finds and reports trips with inconsistent sequential data, returning detailed
    information about each violation for programmatic correction.

    Args:
        df (pd.DataFrame): The input DataFrame. MUST be sorted by trip_id and timestamp.
        trip_id_col (str): The name of the trip identifier column.
        time_col (str): The name of the timestamp column.
        sequence_checks (dict): Configuration for the checks.
        max_details_to_print (int): Max number of trip IDs with issues to print.
        context_window (int): Number of rows to show before and after the inconsistency.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The original, unfiltered DataFrame.
            - dict: A dictionary of trips with errors, where keys are trip IDs and
                    values are lists of detailed error dictionaries.
    """
    print("\n--- Checking for Inconsistent Sequential Data ---")
    if df.empty:
        print("DataFrame is empty. Skipping check.")
        return df, {}

    trips_with_errors = {}
    grouped = df.groupby(trip_id_col, observed=True)

    for trip_id, trip_df in grouped:
        if len(trip_df) < 2:
            continue

        errors_found_in_trip = []

        for col, config in sequence_checks.items():
            if col not in trip_df.columns:
                continue

            series = trip_df[col]
            
            # Prepare the series for checking.
            if col == time_col:
                numeric_series = pd.to_datetime(series, errors='coerce').astype(np.int64)
                if numeric_series.isna().any():
                    # This case is critical and should be handled, but for now, we'll just report.
                    errors_found_in_trip.append({
                        'message': f"'{col}' contains invalid timestamp values.",
                        'context': '', 'column': col, 'violation_index': None,
                        'pre_violation_index': None, 'pre_violation_value': None, 'violation_value': None
                    })
                    continue
                series_to_check = numeric_series
            else:
                numeric_series = pd.to_numeric(series, errors='coerce')
                series_to_check = numeric_series.dropna()

            if len(series_to_check) < 2:
                continue
            
            diffs = series_to_check.diff().dropna()
            if diffs.empty:
                continue

            direction = config.get('direction')
            tolerance = config.get('tolerance', 0)
            strict = config.get('strict', False)

            violations = pd.Series(False, index=diffs.index)
            if direction == 'increasing':
                limit = 0 if strict else -tolerance
                violations = diffs < limit
            elif direction == 'decreasing':
                limit = 0 if strict else tolerance
                violations = diffs > limit

            if violations.any():
                num_violations = violations.sum()
                first_violation_original_idx = violations[violations].index[0]
                
                # Find the index of the last valid point *before* the violation
                prev_valid_series = series_to_check[series_to_check.index < first_violation_original_idx]
                if not prev_valid_series.empty:
                    prev_violation_original_idx = prev_valid_series.index[-1]
                    prev_val = series.loc[prev_violation_original_idx]
                    curr_val = series.loc[first_violation_original_idx]
                    
                    error_msg = (f"'{col}' failed '{direction}' check ({num_violations} times). "
                                 f"First violation at index {first_violation_original_idx}: value changed from {prev_val} to {curr_val}.")
                    
                    # Generate context for printing
                    violation_iloc = trip_df.index.get_loc(first_violation_original_idx)
                    start_iloc = max(0, violation_iloc - context_window)
                    end_iloc = min(len(trip_df), violation_iloc + 1 + context_window)
                    context_df = trip_df.iloc[start_iloc:end_iloc][[time_col, col]].copy()
                    context_df['marker'] = ''
                    context_df.loc[first_violation_original_idx, 'marker'] = '<-- VIOLATION'
                    context_str = context_df.to_string()
                    
                    # Append detailed error information for programmatic use
                    errors_found_in_trip.append({
                        'message': error_msg,
                        'context': context_str,
                        'column': col,
                        'violation_index': first_violation_original_idx,
                        'pre_violation_index': prev_violation_original_idx,
                        'pre_violation_value': prev_val,
                        'violation_value': curr_val
                    })

        if errors_found_in_trip:
            trips_with_errors[trip_id] = errors_found_in_trip

    # --- Reporting ---
    if trips_with_errors:
        num_bad_trips = len(trips_with_errors)
        print(f"Found {num_bad_trips} trips with sequential inconsistencies.")
        print("   Detailed analysis:")
        
        count = 0
        for trip_id, errors in trips_with_errors.items():
            if count >= max_details_to_print:
                print(f"   ... (details for remaining {num_bad_trips - count} trips omitted)")
                break
            print(f"\n   - Trip ID: {trip_id}")
            for error_detail in errors:
                print(f"     - {error_detail['message']}")
                if error_detail['context']:
                    indented_context = "\n".join([f"       {line}" for line in error_detail['context'].split('\n')])
                    print(indented_context)
            count += 1
    else:
        print("No trips with sequential inconsistencies found.")

    return df, trips_with_errors

def report_invalid_array_lengths(df, array_length_checks, trip_id_col='trip_id', time_col='timestamp', max_trips_to_print=10):
    """
    Reports trips and rows where array-like columns do not contain the expected number of samples.
    This function is for reporting only and does not modify the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        array_length_checks (dict): A dictionary where keys are column names and values are the
                                    expected integer length of the array in that column.
                                    Example: {'speed_array': 10, 'altitude_array': 10}
        trip_id_col (str): The name of the trip identifier column for reporting.
        time_col (str): The name of the timestamp column for identifying specific rows.
        max_trips_to_print (int): The maximum number of trip IDs with issues to detail in the report.

    Returns:
        pd.DataFrame: The original, unmodified DataFrame.
    """
    print("\n--- Checking for Invalid Array Sample Counts ---")
    if df.empty:
        print("DataFrame is empty. Skipping check.")
        return df

    # A dictionary to hold detailed error information, structured by trip_id
    trips_with_errors = {}

    for col, expected_length in array_length_checks.items():
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found for array length check. Skipping.")
            continue

        print(f" - Checking array length for '{col}' (expected: {expected_length}).")

        # Apply the helper to get a Series of actual lengths
        actual_lengths = df[col].apply(get_array_length)

        # Create a boolean mask for rows where the length is incorrect.
        # We only care about actual arrays (length != -1) that don't match the expected length.
        invalid_length_mask = (actual_lengths != -1) & (actual_lengths != expected_length)

        if invalid_length_mask.any():
            # Get the subset of the DataFrame with these errors for easier processing
            error_df = df.loc[invalid_length_mask].copy()
            error_df['actual_length'] = actual_lengths[invalid_length_mask]

            print(f"   Found {len(error_df)} rows in '{col}' with incorrect array lengths.")

            # Group by trip_id to collect all errors for each trip
            for trip_id, group in error_df.groupby(trip_id_col, observed=True):
                if trip_id not in trips_with_errors:
                    trips_with_errors[trip_id] = []

                for _, row in group.iterrows():
                    error_info = {
                        'column': col,
                        'timestamp': row[time_col],
                        'actual_length': int(row['actual_length']),
                        'expected_length': expected_length
                    }
                    trips_with_errors[trip_id].append(error_info)
        else:
            print(f"   All arrays in '{col}' have the correct length.")

    # --- Reporting Phase ---
    if trips_with_errors:
        num_bad_trips = len(trips_with_errors)
        print(f"\nFound {num_bad_trips} trips containing rows with invalid array lengths.")
        print("   Detailed analysis:")

        count = 0
        for trip_id, errors in trips_with_errors.items():
            if count >= max_trips_to_print:
                print(f"   ... (details for remaining {num_bad_trips - count} trips omitted)")
                break

            print(f"\n   - Trip ID: {trip_id}")
            # Sort errors by timestamp for chronological reporting
            sorted_errors = sorted(errors, key=lambda x: x['timestamp'])
            for error in sorted_errors:
                print(f"     - Row at {error['timestamp']}: Column '{error['column']}' has "
                      f"{error['actual_length']} samples (expected {error['expected_length']}).")
            count += 1
    else:
        print("No trips with invalid array sample counts found.")

    print(f"Result shape (unchanged): {df.shape}")
    return df

def analyze_consecutive_values(df, trip_id_col, time_col, checks_config,
                               lat_col=None, lon_col=None, h3_col=None,
                               gap_flag_col=None,
                               context_window=3, max_details_to_print=10):
    """
    Analyzes and reports violations in differences between consecutive values.
    If `gap_flag_col` is provided, it will ignore between-row violations that
    occur at a flagged time gap.
    """
    # Determine the title for the print block based on whether this is a filtering run
    run_title = "--- Analyzing Consecutive Value Differences (Filtered for Gaps) ---" if gap_flag_col else "--- Analyzing Consecutive Value Differences (Initial Pass) ---"
    print(f"\n{run_title}")

    if df.empty:
        print("DataFrame is empty. Skipping analysis.")
        return {}

    trips_with_violations = {}
    grouped = df.groupby(trip_id_col, observed=True)

    for trip_id, trip_df in grouped:
        if len(trip_df) < 2:
            continue

        violations_found_in_trip = []

        for col, config in checks_config.items():
            # --- SPECIAL: Location Check using Haversine Distance ---
            if col == 'location':
                coords = []
                if lat_col and lon_col and lat_col in trip_df.columns and lon_col in trip_df.columns:
                    coords = list(zip(trip_df[lat_col], trip_df[lon_col]))
                elif h3_col and h3_col in trip_df.columns:
                    coords = [h3_to_latlon(h3_idx) for h3_idx in trip_df[h3_col]]
                else:
                    continue

                valid_coords_with_indices = [
                    (trip_df.index[i], coord) for i, coord in enumerate(coords)
                    if not (pd.isna(coord[0]) or pd.isna(coord[1]))
                ]

                if len(valid_coords_with_indices) < 2:
                    continue

                indices, points = zip(*valid_coords_with_indices)
                
                distances = [
                    haversine_distance(points[i-1][0], points[i-1][1], points[i][0], points[i][1])
                    for i in range(1, len(points))
                ]
                distance_diffs = pd.Series(distances, index=indices[1:])
                coord_str_series = pd.Series([f"({p[0]:.5f}, {p[1]:.5f})" for p in points], index=indices)

                for check_type, limits in config.items():
                    min_val, max_val = limits.get('min', -np.inf), limits.get('max', np.inf)
                    _collect_violations(
                        distance_diffs, check_type, min_val, max_val, violations_found_in_trip,
                        violation_type='between-row', column_name='location', series=coord_str_series,
                        trip_df=trip_df, gap_flag_col=gap_flag_col
                    )
                continue

            # --- REGULAR: Column-based checks ---
            if col not in trip_df.columns:
                continue

            series = trip_df[col]
            is_array_col = any(isinstance(x, (list, np.ndarray)) for x in series.dropna().head())

            # Part 1: Between-Row Violation Check
            diffs = pd.Series(np.nan, index=trip_df.index)
            if is_array_col:
                get_first = lambda arr: arr[0] if isinstance(arr, (list, np.ndarray)) and len(arr) > 0 else np.nan
                get_last = lambda arr: arr[-1] if isinstance(arr, (list, np.ndarray)) and len(arr) > 0 else np.nan
                diffs = series.apply(get_first) - series.apply(get_last).shift(1)
            elif col == time_col:
                diffs = pd.to_datetime(series, errors='coerce').diff().dt.total_seconds()
            else:
                diffs = pd.to_numeric(series, errors='coerce').diff()

            if not diffs.dropna().empty:
                for check_type, limits in config.items():
                    min_val, max_val = limits.get('min', -np.inf), limits.get('max', np.inf)
                    _collect_violations(
                        diffs, check_type, min_val, max_val, violations_found_in_trip,
                        violation_type='between-row', column_name=col, series=series,
                        trip_df=trip_df, gap_flag_col=gap_flag_col
                    )

            # Part 2: Within-Row Violation Check (unaffected by gap flag)
            if is_array_col:
                for row_idx, arr in series.dropna().items():
                    if isinstance(arr, (list, np.ndarray)) and len(arr) > 1:
                        internal_diffs = pd.Series(np.diff(arr))
                        if not internal_diffs.dropna().empty:
                            for check_type, limits in config.items():
                                min_val, max_val = limits.get('min', -np.inf), limits.get('max', np.inf)
                                _collect_violations(
                                    internal_diffs, check_type, min_val, max_val, violations_found_in_trip,
                                    violation_type='within-row', column_name=col, series=series,
                                    row_index=row_idx, array_data=arr
                                )

        if violations_found_in_trip:
            trips_with_violations[trip_id] = violations_found_in_trip

    # --- Reporting Phase ---
    _report_violations(trips_with_violations, grouped, time_col, context_window, max_details_to_print)
    return trips_with_violations


def _collect_violations(diffs, check_type, min_val, max_val, violation_list, violation_type, column_name, series,
                        row_index=None, array_data=None, trip_df=None, gap_flag_col=None):
    """Helper to find and format violation details, optionally skipping violations at time gaps."""
    target_diffs = pd.Series(dtype=float)
    if check_type == 'pos': target_diffs = diffs[diffs > 0]
    elif check_type == 'neg': target_diffs = diffs[diffs < 0].abs()
    elif check_type == 'both': target_diffs = diffs.abs()

    violating_diffs = target_diffs[(target_diffs < min_val) | (target_diffs > max_val)]

    for viol_idx, diff_val in violating_diffs.items():
        # --- ADDED: Check to skip violations that occur at a known time gap ---
        if violation_type == 'between-row' and gap_flag_col and trip_df is not None and gap_flag_col in trip_df.columns:
            if trip_df.loc[viol_idx, gap_flag_col] == 1:
                continue  # Skip this violation, it's expected due to the gap

        details = {
            'column': column_name, 'type': violation_type, 'check_rule': check_type,
            'limit_broken': 'max' if abs(diff_val) > max_val else 'min',
            'min_limit': min_val, 'max_limit': max_val, 'actual_diff': diffs[viol_idx]
        }
        if violation_type == 'between-row':
            prev_row_loc = series.index.get_loc(viol_idx) - 1
            details['row_index'] = viol_idx
            details['prev_row_index'] = series.index[prev_row_loc]
            details['value1'] = series.loc[details['prev_row_index']]
            details['value2'] = series.loc[details['row_index']]
        elif violation_type == 'within-row':
            details['row_index'] = row_index
            details['array_index'] = viol_idx
            details['value1'] = array_data[viol_idx]
            details['value2'] = array_data[viol_idx + 1]
        violation_list.append(details)

def _report_violations(trips_with_violations, grouped, time_col, context_window, max_details_to_print):
    """Helper to print a formatted report of all found violations."""
    if not trips_with_violations:
        print("No consecutive value violations found in this pass.")
        return

    num_bad_trips = len(trips_with_violations)
    print(f"Found {num_bad_trips} trips with violations in this pass.")
    
    count = 0
    for trip_id, violations in trips_with_violations.items():
        if count >= max_details_to_print:
            print(f"\n   ... (details for remaining {num_bad_trips - count} trips omitted)")
            break
        
        print(f"\n--- Trip ID: {trip_id} ---")
        sorted_violations = sorted(violations, key=lambda x: (x['row_index'], x.get('array_index', -1)))
        
        for v in sorted_violations:
            print(f"  - Type: {v['type']}, Column: '{v['column']}', Rule: '{v['check_rule']}'")
            diff_format = ".4f" if v['column'] == 'location' else ".2f"
            print(f"    - Violation: Change of {v['actual_diff']:{diff_format}} broke limit (Min: {v['min_limit']}, Max: {v['max_limit']})")

            if v['type'] == 'between-row':
                trip_df = grouped.get_group(trip_id)
                if v['column'] == 'location':
                    print("    Context (Location):")
                    prev_time = trip_df.loc[v['prev_row_index'], time_col]
                    curr_time = trip_df.loc[v['row_index'], time_col]
                    print(f"      {prev_time} -> {v['value1']}  <-- PREVIOUS")
                    print(f"      {curr_time} -> {v['value2']}  <-- VIOLATION")
                else:
                    viol_iloc = trip_df.index.get_loc(v['row_index'])
                    start_iloc = max(0, viol_iloc - context_window)
                    end_iloc = min(len(trip_df), viol_iloc + context_window + 1)
                    
                    context_df = trip_df.iloc[start_iloc:end_iloc][[time_col, v['column']]].copy()
                    context_df['marker'] = ''
                    context_df.loc[v['row_index'], 'marker'] = '<-- VIOLATION'
                    context_df.loc[v['prev_row_index'], 'marker'] = '<-- PREVIOUS'
                    print("    Context (Rows):")
                    print("\n".join([f"      {line}" for line in context_df.to_string().split('\n')]))

            elif v['type'] == 'within-row':
                arr = grouped.get_group(trip_id).loc[v['row_index'], v['column']]
                arr_idx = v['array_index']
                start_idx = max(0, arr_idx - context_window)
                end_idx = min(len(arr), arr_idx + context_window + 2)
                
                print(f"    Context (Within array at {time_col} {grouped.get_group(trip_id).loc[v['row_index'], time_col]}):")
                for i in range(start_idx, end_idx):
                    marker = ''
                    if i == arr_idx: marker = '<-- V1'
                    elif i == arr_idx + 1: marker = '<-- V2'
                    print(f"      Index {i}: {arr[i]:<10.2f} {marker}")
        count += 1

def identify_skipped_rows(df: pd.DataFrame, trip_id_col: str, time_col: str, context_window: int = 4) -> list:
    """
    Analyzes a DataFrame to find and return the indices of rows that appear to be
    glitches between two otherwise valid 60-second interval points.

    It detects patterns where the time difference between a row 'x' and row 'x+2'
    or 'x+3' is exactly 60 seconds, and returns the indices of the rows in between.

    Args:
        df (pd.DataFrame): The input DataFrame. MUST be sorted by trip_id and timestamp
                           with a reset index.
        trip_id_col (str): The name of the trip identifier column.
        time_col (str): The name of the timestamp column.
        context_window (int): The number of rows to show in the diagnostic printout.

    Returns:
        list: A sorted list of unique DataFrame indices corresponding to the rows
              that should be removed.
    """
    print("\n--- Identifying Rows from Likely Skipped Timestamps ---")

    if df.empty or time_col not in df.columns or trip_id_col not in df.columns:
        print("DataFrame is empty or missing required columns. Skipping.")
        return []

    df_copy = df.copy()
    df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors='coerce')

    grouped = df_copy.groupby(trip_id_col, observed=True)
    t_plus_2 = grouped[time_col].shift(-2)
    t_plus_3 = grouped[time_col].shift(-3)

    delta_to_t2 = (t_plus_2 - df_copy[time_col]).dt.total_seconds()
    delta_to_t3 = (t_plus_3 - df_copy[time_col]).dt.total_seconds()

    # Find the starting points of the patterns
    pattern_t2_starts = df_copy.index[delta_to_t2 == 60.0]
    pattern_t3_starts = df_copy.index[delta_to_t3 == 60.0]

    if pattern_t2_starts.empty and pattern_t3_starts.empty:
        print("No instances of the specific skipped timestamp pattern were found.")
        return []

    # Use a set to automatically handle overlaps and duplicates
    indices_to_remove = set()

    # Collect indices for the t+2 pattern (remove row x+1)
    for idx in pattern_t2_starts:
        indices_to_remove.add(idx + 1)

    # Collect indices for the t+3 pattern (remove rows x+1 and x+2)
    for idx in pattern_t3_starts:
        indices_to_remove.add(idx + 1)
        indices_to_remove.add(idx + 2)

    print(f"Identified {len(indices_to_remove)} unique rows for removal based on the pattern.")

    # --- Optional: Print context for verification ---
    all_pattern_starts = sorted(list(set(pattern_t2_starts.tolist() + pattern_t3_starts.tolist())))
    flagged_trips = df_copy.loc[all_pattern_starts, trip_id_col].unique()

    for trip_id in flagged_trips[:5]: # Limit printing to first 5 affected trips
        print(f"\n--- Context for Trip ID: {trip_id} ---")
        trip_df = df_copy[df_copy[trip_id_col] == trip_id]
        trip_pattern_starts = [idx for idx in all_pattern_starts if df_copy.loc[idx, trip_id_col] == trip_id]

        for idx in trip_pattern_starts:
            reason = ""
            if idx in pattern_t2_starts: reason = "(t+2 is 60s away)"
            elif idx in pattern_t3_starts: reason = "(t+3 is 60s away)"
            
            print(f"\n  - Pattern starts at index {idx} {reason}")
            
            row_loc = trip_df.index.get_loc(idx)
            start_iloc = max(0, row_loc - 1)
            end_iloc = min(len(trip_df), row_loc + context_window)
            context_df = trip_df.iloc[start_iloc:end_iloc].copy()
            
            context_df['marker'] = ''
            context_df.loc[idx, 'marker'] = '<-- PATTERN START'
            # Mark the rows that would be removed
            for i in range(1, 3):
                if (idx + i) in indices_to_remove and (idx + i) in context_df.index:
                    context_df.loc[idx + i, 'marker'] = '<-- TO BE REMOVED'

            display_cols = [col for col in [time_col, 'current_odo', 'current_soc', 'marker'] if col in context_df.columns]
            print(context_df[display_cols].to_string())

    if len(flagged_trips) > 5:
        print(f"\n... (Context printing omitted for {len(flagged_trips) - 5} more trips) ...")

    return sorted(list(indices_to_remove))

def print_multi_gap_contexts(df, trip_id_col, time_col, gap_flag_col, context_window=5, max_prints=10):
    """
    Finds and prints the context for rows where more than one 'gap' flag
    appears within a specified context window.

    This is useful for diagnosing areas with multiple consecutive data quality issues.

    Args:
        df (pd.DataFrame): The input DataFrame, which should contain the gap flag column.
        trip_id_col (str): The name of the trip identifier column.
        time_col (str): The name of the timestamp column for context.
        gap_flag_col (str): The name of the boolean/integer column indicating a gap (e.g., 'is_start_after_gap').
        context_window (int): The number of rows to show before and after a flagged row.
        max_prints (int): The maximum number of distinct context blocks to print to avoid flooding the output.
    """
    # 1. Basic validation to ensure required columns exist.
    required_cols = [trip_id_col, time_col, gap_flag_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: DataFrame is missing one or more required columns: {required_cols}. Aborting.")
        return

    # 2. Find all rows that are flagged as gaps.
    gap_rows = df[df[gap_flag_col] == 1]
    if gap_rows.empty:
        print("No rows with the gap flag were found.")
        return

    print(f"\n--- Analyzing Contexts for Multiple Gaps (Window: +/- {context_window} rows) ---")

    # 3. Keep track of indices we've already printed to avoid duplicate blocks.
    printed_indices = set()
    prints_count = 0

    # 4. Iterate through each identified gap row.
    for idx, gap_row in gap_rows.iterrows():
        if idx in printed_indices:
            continue

        if prints_count >= max_prints:
            print(f"\n... Reached max prints limit of {max_prints}. Aborting further analysis.")
            break

        # 5. Define the context for the current gap by looking at its trip.
        trip_id = gap_row[trip_id_col]
        trip_df = df[df[trip_id_col] == trip_id]

        try:
            # Get the integer position (iloc) of our current gap row within its trip.
            row_iloc = trip_df.index.get_loc(idx)
        except KeyError:
            # This can happen if the index is not unique within the trip, though it shouldn't
            # with proper preprocessing. We skip if the index is not found.
            continue

        # Define the window slice using integer locations.
        start_iloc = max(0, row_iloc - context_window)
        end_iloc = min(len(trip_df), row_iloc + context_window + 1)
        context_df = trip_df.iloc[start_iloc:end_iloc]

        # 6. Check if this context contains more than one gap.
        gaps_in_context = context_df[gap_flag_col].sum()

        if gaps_in_context > 1:
            prints_count += 1
            print(f"\n--- Multi-Gap Context Found in Trip ID: {trip_id} (around index {idx}) ---")

            # Prepare a list of relevant columns for a clean printout.
            display_cols = [time_col, 'current_odo', 'current_soc', gap_flag_col]
            # Filter to only columns that actually exist in the DataFrame.
            display_cols_exist = [col for col in display_cols if col in context_df.columns]

            print(context_df[display_cols_exist].to_string())

            # 7. Add all indices from this printed context to the set to avoid re-printing this block.
            printed_indices.update(context_df.index)

    if prints_count == 0:
        print("No contexts with multiple gaps were found.")
    else:
        print(f"\n--- Analysis Complete. Found and printed {prints_count} multi-gap contexts. ---")

def flag_erroneous_altitude(
        df, trip_id_col=TRIP_ID_COL, alt_col=ALT_ARRAY_COL, error_alias=-1, 
        max_diff_from_error=4, max_cons_error=6, island_size=2
        ):
    """
    Flags erroneous altitude values (-1) by replacing them with np.nan based on robust heuristics.
    """
    
    df_copy = df.copy()
    samples_per_row = df_copy[alt_col].iat[0].size

    grouped = df_copy.groupby(trip_id_col)

    all_alts = np.concatenate(df_copy[alt_col].values)
    errors_before = np.count_nonzero(all_alts == error_alias)
    print(f"Error alias ratio before cleaning: {errors_before/ len(all_alts)}")
    print(f"Max alt: {np.max(all_alts)}, min alt: {np.min(all_alts)}")
    print(f"Ratio of alt < -1: {np.sum(all_alts < -1) / len(all_alts)}")

    for trip_id, trip_df in grouped:
        if len(trip_df) < 2:
            continue

        trip_alts = np.concatenate(trip_df[alt_col].values).astype(float, copy=False)

        n = len(trip_alts)
        i = 0
        while i < n:
            if trip_alts[i] < MIN_VALID_ALTITUDE:
                trip_alts[i] = np.nan
                i += 1
                continue

            if trip_alts[i] == error_alias:
                j = i + 1
                while j < n and trip_alts[j] == error_alias:
                    j += 1
                
                if len(trip_alts[i:j]) > max_cons_error:
                    trip_alts[i:j] = np.nan
                else:    
                    left_neigh  = i - 1 if i > 0 else None
                    right_neigh = j     if j < n else None
                   
                    if (
                        left_neigh is not None and abs(trip_alts[left_neigh] - error_alias) > max_diff_from_error
                        or right_neigh is not None and abs(trip_alts[right_neigh] - error_alias) > max_diff_from_error
                    ):
                        trip_alts[i:j] = np.nan
                    
                    # else:
                    #     start = left_neigh if left_neigh  is not None else i        # inclusive
                    #     stop  = (right_neigh + 1) if right_neigh is not None else j # exclusive
                    #     #print(trip_alts[start:stop])           
                i = j                  
            
            else:
                i += 1                 
        
        # removing "islands"
        if island_size != 2:
            print("island_size != 2 not implemented yet, setting it to 2.")

        for k in range(1,len(trip_alts)-1):
            if np.isnan(trip_alts[k-1]) and ~np.isnan(trip_alts[k]) and np.isnan(trip_alts[k+1]):
                trip_alts[k] = np.nan
            if np.isnan(trip_alts[k-1]) and not np.isnan(trip_alts[k]) and not np.isnan(trip_alts[k+1]) and np.isnan(trip_alts[k+2]):
                trip_alts[k:k+2] = np.nan              

        cleaned = trip_alts.reshape(-1, samples_per_row)        
        mask = df_copy[trip_id_col] == trip_id
        rows = mask.sum()
        obj_vals = np.empty(rows, dtype=object)
        obj_vals[:] = list(cleaned)
        df_copy.loc[mask, alt_col] = obj_vals
        assert df_copy.loc[mask, alt_col].apply(type).eq(np.ndarray).all()
    
    all_alts = np.concatenate(df_copy[alt_col].values)
    num_of_cleaned = np.sum(np.isnan(all_alts))
    print(f"Ratio of cleaned samples: {num_of_cleaned/ len(all_alts)}")

    return df_copy

def drop_infreq_models(df, min_prop):
    """
    Filters out car models based on a minimum proportion threshold and prints
    a detailed summary of the models that were dropped.

    Args:
        df (pd.DataFrame): The input DataFrame with trip data.
        min_prop (float): The minimum proportion (e.g., 0.01 for 1%) a car
                          model must represent to be kept.

    Returns:
        pd.DataFrame: A DataFrame with the infrequent models removed.
    """
    df_copy = df.copy()

    first_models = df_copy.groupby(TRIP_ID_COL)[CAR_MODEL_COL].first()

    start_n_trips = len(first_models)
    start_n_models = df_copy[CAR_MODEL_COL].nunique() 
    print(f"- Starting with {start_n_trips} trips and {start_n_models} models.\n")

    model_counts  = first_models.value_counts(normalize=True)
    model_proportions = first_models.value_counts(normalize=True)

    model_stats = pd.DataFrame({
        'count': model_counts,
        'proportion': model_proportions
    })

    models_to_drop_mask = model_stats['proportion'] < min_prop
    dropped_summary = model_stats[models_to_drop_mask].copy()

    print("-" * 50)
    if not dropped_summary.empty:
        print(f"Dropping {len(dropped_summary)} models that are each less than {min_prop:.2%} of total trips:")
        dropped_summary['proportion'] = dropped_summary['proportion'].map('{:.4%}'.format)
        print(dropped_summary)
    else:
        print("No models met the criteria to be dropped.")
    print("-" * 50)

    models_to_keep = model_stats[model_stats['proportion'] >= min_prop].index

    df_copy = df_copy[df_copy[CAR_MODEL_COL].isin(models_to_keep)]

    final_n_models = df_copy[CAR_MODEL_COL].nunique() 
    final_n_trips = df_copy[TRIP_ID_COL].nunique() 
    print(f"\n- Removed {start_n_trips - final_n_trips} trips and {start_n_models - final_n_models} models.")

    return df_copy

# --- Main Execution Logic ---
def main():
    """Runs the preprocessing pipeline using parameters defined in the configuration block."""
    print(f"--- Starting Preprocessing Pipeline ---")
    print(f"\nInput file: {INPUT_FILE_PATH}")
    print(f"Output file: {OUTPUT_FILE_PATH}")

    try:
        # 1. Load Data
        print("\n--- Loading Data ---")
        df = load_file(INPUT_FILE_PATH)
        if not isinstance(df, pd.DataFrame): raise TypeError("Loaded data is not a DataFrame.")
        print(f"Initial shape: {df.shape}")

        # --- Subset for Testing (Optional) ---
        if PROCESS_SUBSET is not None and PROCESS_SUBSET < len(df):
             print(f"\n--- Processing Subset: First {PROCESS_SUBSET} rows ---")
             df = df.head(PROCESS_SUBSET).copy()
        # ------------------------------------

        # 2. Rename Columns
        print("\n--- Renaming Columns ---")
        rename_map_existing = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
        df = df.rename(columns=rename_map_existing)
        cols_to_keep = list(rename_map_existing.values())
        if LAT_COL not in cols_to_keep: cols_to_keep.append(LAT_COL)
        if LON_COL not in cols_to_keep: cols_to_keep.append(LON_COL)
        #df = df[cols_to_keep] # Keep only mapped columns
        df = df[[col for col in cols_to_keep if col in df.columns]]
        print(f"Columns after renaming: {df.columns.tolist()}")
        # --- Apply Preprocessing Steps Sequentially ---

        # Convert timestamp to datetime and sort the data
        print("\n--- Converting Timestamp and Sorting Data ---")
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
        initial_rows = len(df)
        df.dropna(subset=[TIME_COL], inplace=True)
        if len(df) < initial_rows:
            print(f"Dropped {initial_rows - len(df)} rows with invalid timestamps.")

        
        df = df.sort_values(by=[TRIP_ID_COL, TIME_COL], ascending=True).reset_index(drop=True)
        print(f"Data sorted by trip and time. Shape: {df.shape}")

        # Report Invalid Array Lengths (Row Level)
        array_length_checks = {
            SPEED_ARRAY_COL: EXPECTED_SPEED_ARRAY_SAMPLES,
            ALT_ARRAY_COL: EXPECTED_ALT_ARRAY_SAMPLES
        }
        # This function only reports and does not filter, so we don't reassign df
        report_invalid_array_lengths(df, array_length_checks, trip_id_col=TRIP_ID_COL, time_col=TIME_COL)

        # 3. Convert H3 Index to Lat/Lon (Add this step early)
        print(f"\n--- Converting H3 Index ('{H3_COL}') to Lat/Lon ---")
        if H3_COL in df.columns:
            # Apply the conversion function; it returns tuples (lat, lon)
            lat_lon_tuples = df[H3_COL].apply(h3_to_latlon)

            # Create the new columns from the tuples
            df[LAT_COL] = lat_lon_tuples.apply(lambda x: x[0])
            df[LON_COL] = lat_lon_tuples.apply(lambda x: x[1])

            # Report potential failures
            failed_conversions = df[LAT_COL].isna().sum()
            if failed_conversions > 0:
                print(f"Warning: {failed_conversions} rows failed H3 to Lat/Lon conversion (resulted in NaN).")
            print(f"Added '{LAT_COL}' and '{LON_COL}' columns.")
        else:
            print(f"Warning: H3 index column '{H3_COL}' not found. Skipping conversion.")
        print(f"Shape after H3 conversion: {df.shape}")
        # ----------------------------------------------------

        # 4. Validate and Filter Lat/Lon (includes trip filter) *** 
        df = validate_and_filter_latlon(
            df,
            lat_col=LAT_COL,
            lon_col=LON_COL,
            trip_id_col=TRIP_ID_COL,
            filter_zeros=FILTER_ZERO_COORDINATES,
            zero_tolerance=1e-3, # Use appropriate tolerance
            filter_mostly_zero_trips=FILTER_MOSTLY_ZERO_TRIPS,
            zero_trip_threshold=ZERO_TRIP_THRESHOLD
        )

        if VALIDATE_POINTS_IN_AREA:
            # 5. Validate and Filter Lat/Lon in target countries.
            # Define the countries of interest (Corsica is part of France).
            TARGET_COUNTRY_CODES = ['FRA', 'LUX', 'BEL', 'DNK', 'CHE', 'NLD', 'PRT', 'ITA', 'UKR', 'DEU', 'GRC', 'NOR', 'HUN']

            # Get the geometry using the new GADM function
            # Note: A smaller buffer is needed because the data is more accurate
            target_area_geometry = get_gadm_countries_geometry(
                gadm_directory=GADM_FOLDER_PATH,
                country_iso_codes=TARGET_COUNTRY_CODES,
                buffer_meters=500 # Start with a smaller buffer (e.g., 500m)
            )

            # Step 2: Run the check if the geometry was loaded successfully
            if target_area_geometry:
                print("\n--- Identifying points outside the target area ---")
                outside_points_df = find_points_outside_target_area(
                    df=df,
                    target_geometry=target_area_geometry,
                    trip_id_col=TRIP_ID_COL, lat_col=LAT_COL, lon_col=LON_COL
                )

                print("\n--- Results: Trip ID and Coordinates of Points OUTSIDE Target Area ---")
                if not outside_points_df.empty:
                    print(outside_points_df)
                else:
                    print("No points were found outside the target area.")
            else:
                print("\nSkipping location checks because the target area geometry could not be created.")

        # 2. Define and apply manual trip removals
        # This dictionary centralizes all manual filtering decisions.
        MANUAL_TRIPS_TO_REMOVE = {
            # --- Geographic Outliers ---
            # Reason: Not representative of mainland Europe, creates massive/impractical map bounding boxes.
            'eb1f4e33-a91a-5556-abf6-59a5975e204e': "Guadeloupe trip (overseas territory)",
            '1f17cd75-fb38-5746-8b50-c95f5dc21187': "Madeira trip (island outlier)",

            # --- Data Integrity Issues ---
            # Reason: Contains a large, unrecoverable GPS gap, likely a non-driving event (e.g., ferry).
            '8de496a7-fdb8-56e3-8b6d-630636d1fae8': "Marseille trip (large GPS gap)"
        }
        
        # Call the function to perform the removal
        df = remove_manual_trips(df, MANUAL_TRIPS_TO_REMOVE, trip_id_col=TRIP_ID_COL)

        # 3. Check Trip Static Data Consistency (Trip Level)
        df = filter_inconsistent_trip_data(df, trip_id_col=TRIP_ID_COL, cols_to_check=STATIC_TRIP_COLS)

        # 4. Check/Drop Redundant ID (Trip Level)
        df = check_and_drop_correlated_id(df, primary_id_col=TRIP_ID_COL, secondary_id_col=MSG_SESSION_ID_COL)

        # 5. Basic Duplicates (Row Level)
        df = remove_duplicates(df, subset_cols=DUPLICATE_CHECK_COLS_STD)

        # 6. Remove Constant/All-NaN Columns
        df = remove_constant_cols(df)

        # Identify and remove rows that match the "skipped sample" pattern.
        indices_to_drop = identify_skipped_rows(df, trip_id_col=TRIP_ID_COL, time_col=TIME_COL)

        if indices_to_drop:
            print(f"\n--- Removing {len(indices_to_drop)} identified glitch rows ---")
            df.drop(index=indices_to_drop, inplace=True)
            # CRITICAL: Reset the index after dropping rows to ensure continuity for subsequent steps.
            df.reset_index(drop=True, inplace=True)
            print(f"Shape after removing glitch rows: {df.shape}")

        _ = identify_skipped_rows(df, trip_id_col=TRIP_ID_COL, time_col=TIME_COL)    
        # --- Fetch Weather Data ---
        if FETCH_WEATHER_DATA:
            print("\n--- Fetching External Weather Data (Row-by-Row) ---")
            print("Applying weather function... This may take a significant amount of time.")
            # Apply the function - result is a DataFrame with new weather columns
            # Make sure standard column names are passed if needed by the function
            weather_cols_df = df.apply(
                get_weather_for_timestamp, # Use the function defined above
                axis=1,
                result_type='expand', # Creates new columns directly
                lat_col=LAT_COL,
                lon_col=LON_COL,
                time_col=TIME_COL,
                h3_col=H3_COL
            )
            # Concatenate the new weather columns to the original DataFrame
            df = pd.concat([df, weather_cols_df], axis=1)
            print(f"Shape after adding weather columns: {df.shape}")
            print(f"Weather columns added: {weather_cols_df.columns.tolist()}")
        else:
            print("\n--- Skipping External Weather Data Fetching ---")

        # --- Flag Suspicious Temperatures ---
        if FETCH_WEATHER_DATA and 'weather_temp' in df.columns:
             df = flag_suspicious_temperatures(
                 df,
                 vehicle_temp_col=VEHICLE_TEMP_COL, # Standard name for vehicle temp
                 weather_temp_col='weather_temp', # Name created by get_weather...
                 station_dist_col='station_dist_km', # Explicitly pass or rely on default
                 max_difference_c=MAX_TEMP_DIFFERENCE_C,
                 flag_col_name='temp_suspicious_flag' # Optional: change flag name if desired
             )
        else:
             print("\n--- Skipping Temperature Comparison (No weather data fetched or temp column missing) ---")

        # 7. Filter Unrealistic Ranges (Row Level)
        range_checks = {
            SOC_COL: (MIN_VALID_SOC, MAX_VALID_SOC),
            VEHICLE_TEMP_COL: (MIN_VALID_TEMP, MAX_VALID_TEMP),
            ALT_ARRAY_COL: (MIN_VALID_ALTITUDE, MAX_VALID_ALTITUDE),
            BATT_HEALTH_COL: (MIN_VALID_BATT_HEALTH, MAX_VALID_BATT_HEALTH),
            ODO_COL: (0, MAX_VALID_ODO), # Odo should start >= 0
            WEIGHT_COL: (MIN_VALID_WEIGHT, MAX_VALID_WEIGHT),
            SPEED_ARRAY_COL: (MIN_VALID_SPEED, MAX_VALID_SPEED)
        }

        df, speed_outlier_trip_ids = filter_unrealistic_ranges(
            df, 
            range_checks, 
            trip_id_col=TRIP_ID_COL, 
            max_details_to_print=5,
            plot_output_dir=SPEED_OUTLIERS_PLOT_DIR # Pass the directory path here
        )

        # The list of speed outlier trip IDs is now returned by the function above.
        # Manually dropping the identified trips.
        if speed_outlier_trip_ids:
            initial_trip_count = df[TRIP_ID_COL].nunique()
            print(f"\n--- Removing {len(speed_outlier_trip_ids)} trips with speed sensor errors ---")
            print(f"Initial number of unique trips before removal: {initial_trip_count}")

            df = df[~df[TRIP_ID_COL].isin(speed_outlier_trip_ids)].copy()

            final_trip_count = df[TRIP_ID_COL].nunique()
            print(f"Dropped {initial_trip_count - final_trip_count} trips.")
            print(f"Final number of unique trips: {final_trip_count}")
        else:
            print("\n--- No trips identified for removal due to speed sensor errors ---")

        df, speed_outlier_trip_ids = filter_unrealistic_ranges(
            df, 
            range_checks, 
            trip_id_col=TRIP_ID_COL, 
            max_details_to_print=5,
            plot_output_dir=SPEED_OUTLIERS_PLOT_DIR # Pass the directory path here
        )

        # 9. Filter Inconsistent Sequences (Trip Level)
        sequence_checks = {
            ODO_COL: {'direction': 'increasing', 'strict': True}, # Allow for minor float inaccuracies
            # SOC_COL: {'direction': 'decreasing', 'strict': True}, can increase due to regenerative braking
            BATT_HEALTH_COL: {'direction': 'decreasing', 'strict': True},
            TIME_COL: {'direction': 'increasing', 'strict': True} # Ensure time is strictly increasing
        }
        df, trips_with_errors = find_inconsistent_sequences(df, TRIP_ID_COL, TIME_COL, sequence_checks)

        # --- DYNAMIC CORRECTION BLOCK ---
        if trips_with_errors:
            print("\n--- Applying Dynamic Corrections for Sequential Errors ---")
            
            indices_to_drop = []
            # Flatten the list of all errors from all trips
            all_errors = [(trip_id, error) for trip_id, trip_errors in trips_with_errors.items() for error in trip_errors]

            for trip_id, error_details in all_errors:
                col_name = error_details['column']
                violation_idx = error_details['violation_index']

                # Handle Odometer Glitch (Drop the single erroneous row)
                if col_name == ODO_COL:
                    print(f"\n - Found Odometer glitch in trip '{trip_id}'.")
                    print(f"   - Marking row with index {violation_idx} for removal.")
                    if violation_idx is not None:
                        indices_to_drop.append(violation_idx)

                # Handle Battery Health Jump (Forward-fill from the last good value)
                elif col_name == BATT_HEALTH_COL:
                    pre_violation_idx = error_details['pre_violation_index']
                    if pre_violation_idx is not None and violation_idx is not None:
                        last_good_val = df.loc[pre_violation_idx, BATT_HEALTH_COL]
                        
                        print(f"\n - Found Battery Health jump in trip '{trip_id}'.")
                        print(f"   - Correcting '{BATT_HEALTH_COL}' from index {violation_idx} onwards to value {last_good_val}.")
                        
                        trip_mask = df[TRIP_ID_COL] == trip_id
                        onwards_mask = df.index >= violation_idx
                        df.loc[trip_mask & onwards_mask, BATT_HEALTH_COL] = last_good_val
                        
                        print("   Context after fix:")
                        trip_df = df[df[TRIP_ID_COL] == trip_id]
                        try:
                            iloc = trip_df.index.get_loc(violation_idx)
                            print(trip_df.iloc[max(0, iloc-5):min(len(trip_df), iloc+6)][[TIME_COL, BATT_HEALTH_COL]].to_string())
                        except KeyError:
                            print(f"     Index {violation_idx} no longer exists for context view.")

            # Drop all collected indices at once for safety
            if indices_to_drop:
                print(f"\n--- Dropping {len(indices_to_drop)} rows due to Odometer glitches ---")
                # Ensure indices are unique and exist before dropping
                valid_indices_to_drop = [idx for idx in set(indices_to_drop) if idx in df.index]
                print(f"Dropping valid indices: {valid_indices_to_drop}")
                df = df.drop(index=valid_indices_to_drop).reset_index(drop=True)
                print(f"Shape after dropping rows: {df.shape}")

            print("\n--- Re-running Sequence Check After Dynamic Fixes ---")
            df, _ = find_inconsistent_sequences(df, TRIP_ID_COL, TIME_COL, sequence_checks)
        # --- END DYNAMIC CORRECTION BLOCK ---

        ########################### Misalignment Report Helpers #########################################

        # first creat a lookup table for battery_capacity_kWh
        BATTERY_LOOKUP = [
            # ─── small- and mid-size e-CMP cars ────────────────────────────
            {"car_model": "208 V2 (P21)",                 "manufacturer": "PEUGEOT", "battery_type": "L2 640 60AH EF 640 EN",         "battery_capacity_kWh": 50.0},
            {"car_model": "2008 V2 (P24)",                "manufacturer": "PEUGEOT", "battery_type": "L2 640 60AH EF 640 EN",         "battery_capacity_kWh": 50.0},
            {"car_model": "C4 V3 (C41E)",                 "manufacturer": "CITROEN", "battery_type": "L2 640 60AH EF 640 EN",         "battery_capacity_kWh": 50.0},
            {"car_model": "DS3 CROSSBACK (D34)",          "manufacturer": "DS",      "battery_type": "L2 640 60AH EF 640 EN",         "battery_capacity_kWh": 50.0},
            {"car_model": "C4X (C43E)",                   "manufacturer": "CITROEN", "battery_type": "L2 640 60AH EF 640 EN",         "battery_capacity_kWh": 50.0},
            {"car_model": "308 V3 (P512)",                "manufacturer": "PEUGEOT", "battery_type": "L3 720 70AH EF 720 EN",         "battery_capacity_kWh": 54.0},

            # ─── e-K9 vans (Berlingo / Rifter / Partner) ───────────────────
            {"car_model": "BERLINGO (K9 EUROPE)",         "manufacturer": "CITROEN", "battery_type": "L2 640 60AH EF 640 EN",         "battery_capacity_kWh": 52.0},
            {"car_model": "RIFTER / PARTNER (K9 EUROPE)", "manufacturer": "PEUGEOT", "battery_type": "L2 640 60AH EF 640 EN",         "battery_capacity_kWh": 52.0},
            {"car_model": "RIFTER / PARTNER (K9 EUROPE)", "manufacturer": "PEUGEOT", "battery_type": "L3 760 70AH EFBC 760 EN EFBC",  "battery_capacity_kWh": 52.0},  # ← NEW

            # ─── larger K0 vans (Spacetourer / Traveller / Expert / Jumpy) ─
            {"car_model": "TRAVELLER / EXPERT 4 (K0)",    "manufacturer": "PEUGEOT", "battery_type": "L2 640 640 EN",                 "battery_capacity_kWh": 50.0},
            {"car_model": "SPACETOURER / JUMPY 4 (K0)",   "manufacturer": "CITROEN", "battery_type": "L2 640 640 EN",                 "battery_capacity_kWh": 50.0},
            {"car_model": "SPACETOURER / JUMPY 4 (K0)",   "manufacturer": "CITROEN", "battery_type": "L3 760 70AH EFBC 760 EN EFBC",  "battery_capacity_kWh": 75.0},
            {"car_model": "TRAVELLER / EXPERT 4 (K0)",    "manufacturer": "PEUGEOT", "battery_type": "L3 760 70AH EFBC 760 EN EFBC",  "battery_capacity_kWh": 75.0},
        ]

        lookup_df = pd.DataFrame(BATTERY_LOOKUP)

        def _clean_series(s: pd.Series) -> pd.Series:
            return (
                s.astype(str)
                .str.replace(r"\s+", " ", regex=True)   # collapse multiple spaces
                .str.replace(r"[^\x00-\x7F]+", " ", regex=True)  # strip non-ASCII artifacts
                .str.strip()
            )

        df["car_model"]      = _clean_series(df["car_model"])
        df["manufacturer"]   = _clean_series(df["manufacturer"])
        df["battery_type"]   = _clean_series(df["battery_type"])
        lookup_df["car_model"]    = _clean_series(lookup_df["car_model"])
        lookup_df["manufacturer"] = _clean_series(lookup_df["manufacturer"])
        lookup_df["battery_type"] = _clean_series(lookup_df["battery_type"])

        df = df.merge(
            lookup_df,
            on=["car_model", "manufacturer", "battery_type"],
            how="left"
        )

        # reset the index to resolve any duplicates created by the merge.
        df = df.reset_index(drop=True)

        missing = df["battery_capacity_kWh"].isna().sum()
        print(f"\nRows still missing battery_capacity_kWh after merge: {missing}")
        
        # # --- 2. Define Configuration and Rules ---
        # CONFIG = MisalignmentConfig(
        #     trip_id=TRIP_ID_COL, time=TIME_COL, lat=LAT_COL, lon=LON_COL, odo=ODO_COL,
        #     speed_array=SPEED_ARRAY_COL, soc=SOC_COL, alt_array=ALT_ARRAY_COL,
        #     battery_capacity_kWh_col="battery_capacity_kWh"
        # )

        # # Define the list of rules to apply.
        # all_rules = [
        #     # A. Foundational Data Integrity
        #     time_gap_rule(min_s=59.0, max_s=61.0),
        #     simple_threshold_rule(name="Odo_Regress", col="Δ_odo_km", threshold=-0.05, op='lt'),
        #     simple_threshold_rule(name="GPS_Teleport", col="Δ_gps_km", threshold=2.0, op='gt'),

        #     # B. High-Impact Sensor Glitch Checks
        #     #    - Within a single 60-second array
        #     speed_spike_rule(limit=25.0), # Checks 'speed_max_internal_delta_kph'
        #     simple_threshold_rule(name="Altitude_Spike_Internal", col="alt_max_internal_delta_m", threshold=25.0, op='gt'),
        #     #    - Between consecutive rows (the physical jump)
        #     simple_threshold_rule(name="Speed_Edge_Jump", col="speed_edge_delta_kph", threshold=25, op='abs_gt'),
        #     # CORRECTED RULE: Now points to the physical jump between arrays, not the change in medians.
        #     simple_threshold_rule(name="Altitude_Edge_Jump", col="alt_edge_delta_m", threshold=30, op='abs_gt'),

        #     # C. Multi-variate Plausibility Checks
        #     speed_spike_while_stationary_rule(speed_spike_thr=20.0, max_dist_km=0.02),
        #     simple_threshold_rule(name="SoC_Jump", col="Δ_soc_pct", threshold=2.0, op='abs_gt'),
            
        #     # D. Consistency Between Different Distance Measures
        #     ratio_rule(lhs='Δ_odo_km', rhs='Δ_gps_km', max_ratio=3.0, min_move=0.1),
        #     ratio_rule(lhs='Δ_odo_km', rhs='Δ_dist_from_speed_km', max_ratio=3.0, min_move=0.1),

        #     # E. Advanced Physics and Cross-Family Plausibility Checks
        #     grade_rule(name="Impossible_Grade_vs_GPS", dist_col='Δ_gps_km', max_grade=0.30),
        #     grade_rule(name="Impossible_Grade_vs_Odo", dist_col='Δ_odo_km', max_grade=0.30),
        #     energy_per_100km_rule("Energy_Consumption_Outlier", min_kWh_100km=3, max_kWh_100km=50, min_move_km=1.0),
        #     charging_while_moving_rule(min_regen_pct=0.5, min_dist_km=0.1),
        #     uphill_no_energy_rule(min_alt_gain_m=50, max_soc_gain_pct=0.2),
        #     descent_no_speed_change_rule(min_alt_loss_m=50.0, max_speed_delta_kph=5.0),
        #     odo_jump_short_time_rule(odo_jump_km=1.0, max_time_s=10.0),
        #     teleport_with_no_energy_change_rule(teleport_dist_km=2.0, max_soc_change_pct=0.5),
        #     unrealistic_regen_on_descent_rule(min_alt_loss_m=100.0, max_regen_pct=2.0),
            
        #     # (Optional) A rule to monitor the change in median altitude if you still want to track it.
        #     # This is now clearly named to reflect what it does. Threshold might need tuning.
        #     simple_threshold_rule(name="Altitude_Trend_Change", col="Δ_alt_m", threshold=55, op='abs_gt'),
        # ]

        # # --- 3. Run the Analysis ---
        # deltas_df, flags_df = analyze_feature_misalignment(df, CONFIG, all_rules)

        # # --- 4. Review the Results ---
        # rule_hit_counts = flags_df.sum().sort_values(ascending=False)
        # trip_hit_counts = flags_df.groupby(df[CONFIG.trip_id]).any().sum().sort_values(ascending=False)

        # print("\n--- Misalignment Results (V3 - Hardened) ---")
        # print("\nTop-15 rules by number of rows flagged:")
        # print(rule_hit_counts.head(15))

        # print("\nTop-15 rules by number of trips flagged:")
        # print(trip_hit_counts.head(15))

        # # --- 5. Build and Print the Detailed Report (for manual inspection) ---
        # final_report = build_report(df, deltas_df, flags_df)
        # print_misalignment_report(
        #     report_df=final_report,
        #     raw_df=df,
        #     deltas=deltas_df,
        #     all_rules=all_rules,
        #     cfg=CONFIG,
        #     context_window=3,
        #     max_trips=5
        # )

        ############################################################################################





    #     # --- Two-Pass Analysis for Consecutive Value Differences ---
    #     print("\n--- Running Detailed Analysis of Consecutive Value Differences ---")

        # Define the configuration for the checks using constants from the top of the script.
        consecutive_checks_config = {
            # 'location': {
            #     'both': {'min': 0.0, 'max': MAX_LOCATION_DIFF}
            # },
            # SOC_COL: {
            #     'pos': {'min': 0.0, 'max': POS_MAX_SOC_DIFF},
            #     'neg': {'min': 0.0, 'max': NEG_MAX_SOC_DIFF}
            # },
            # SPEED_ARRAY_COL: {
            #     'pos': {'min': 0.0, 'max': POS_MAX_SPEED_DIFF},
            #     'neg': {'min': 0.0, 'max': NEG_MAX_SPEED_DIFF}
            # },
            # ALT_ARRAY_COL: {
            #     'pos': {'min': 0.0, 'max': POS_MAX_ALT_DIFF},
            #     'neg': {'min': 0.0, 'max': NEG_MAX_ALT_DIFF}
            # },
            # VEHICLE_TEMP_COL: {
            #     'pos': {'min': 0.0, 'max': POS_MAX_TEMP_DIFF},
            #     'neg': {'min': 0.0, 'max': NEG_MAX_TEMP_DIFF}
            # },
            TIME_COL: {
                'pos': {'min': MIN_TIME_GAP_SECONDS, 'max': MAX_TIME_GAP_SECONDS}
            }#,
            # ODO_COL: {
            #     'pos': {'min': 0.0, 'max': POS_MAX_ODO_DIFF},
            #     'neg': {'min': 0.0, 'max': NEG_MAX_ODO_DIFF}
            # },
            # BATT_HEALTH_COL: {
            #     'neg': {'min': 0.0, 'max': NEG_MAX_SOH_DIFF}
            # }
        }

        # --- Pass 1: Run analysis to find all violations, including time gaps ---
        initial_violations_report = analyze_consecutive_values(
            df,
            trip_id_col=TRIP_ID_COL,
            time_col=TIME_COL,
            checks_config=consecutive_checks_config,
            lat_col=LAT_COL, lon_col=LON_COL, h3_col=H3_COL,
            max_details_to_print=0 # Suppress detailed printing for the first pass
        )

        # --- Flagging Step: Create the 'is_start_after_gap' column based on the initial report ---
        print("\n--- Flagging Rows with Irregular Time Gaps ---")
        df['is_start_after_gap'] = 0
        if initial_violations_report:
            gap_indices = []
            # Iterate through all violations in the generated report
            for violations in initial_violations_report.values():
                for v in violations:
                    # Check if the violation is for the timestamp column (either too small or too large)
                    if v['column'] == TIME_COL:
                        gap_indices.append(v['row_index'])
            
            if gap_indices:
                unique_gap_indices = sorted(list(set(gap_indices)))
                df.loc[unique_gap_indices, 'is_start_after_gap'] = 1
                print(f"Flagged {len(unique_gap_indices)} rows with irregular time gaps (outside range [{MIN_TIME_GAP_SECONDS}s, {MAX_TIME_GAP_SECONDS}s]).")
            else:
                print("No irregular time gaps were found to flag.")
        else:
            print("No violations found in initial pass, skipping time gap flagging.")

        # if 'is_start_after_gap' in df.columns:
        #     print_multi_gap_contexts(
        #         df=df,
        #         trip_id_col=TRIP_ID_COL,
        #         time_col=TIME_COL,
        #         gap_flag_col='is_start_after_gap',
        #         context_window=5,  # Look 4 rows before and 4 after
        #         max_prints=1000      # Stop after printing 10 examples
        #     )
        
    #     # --- Pass 2: Re-run analysis, this time ignoring violations at the flagged gaps ---
    #     final_violations_report = analyze_consecutive_values(
    #         df,
    #         trip_id_col=TRIP_ID_COL,
    #         time_col=TIME_COL,
    #         checks_config=consecutive_checks_config,
    #         lat_col=LAT_COL, lon_col=LON_COL, h3_col=H3_COL,
    #         gap_flag_col='is_start_after_gap', # Activate the filtering logic
    #         max_details_to_print=1000,
    #         context_window=18
    #     )
        
    #     if final_violations_report:
    #         print(f"\nFinal analysis complete. Found {len(final_violations_report)} trips with non-gap-related violations.")
    #     else:
    #         print("\nFinal analysis complete. No non-gap-related violations found.")
    #     # --- End of Two-Pass Analysis ---

        # replacing any altitude == -1, or "island" between -1s with np.nan.
        df = flag_erroneous_altitude(
            df, 
            trip_id_col=TRIP_ID_COL, 
            alt_col=ALT_ARRAY_COL, 
            error_alias=-1, 
            max_diff_from_error=4, 
            max_cons_error=6, 
            island_size=2
        )

        if DROP_INFREQ_MODELS:
            # 5. Drop trips from very infrequent car models
            print(f"\n--- Dropping Infrequent Car Models ---")
            df = drop_infreq_models(
                df,
                min_prop = MIN_MODEL_PROP,
            )
        
        # --- Final Check & Save ---
        print("\n--- Final Data Check ---")
        if df.empty: print("Warning: DataFrame is empty after preprocessing.")
        else: 
            print(f"Final shape: {df.shape}")
            print(f"Final columns: {df.columns.tolist()}")

        # --- Save the cleaned data using save_file helper ---
        print(f"\n--- Saving Cleaned Data ---")
        try:
            # Extract directory, base filename, and format from the full path
            output_dir = os.path.dirname(OUTPUT_FILE_PATH)
            base_file_name = os.path.splitext(os.path.basename(OUTPUT_FILE_PATH))[0]
            output_extension = os.path.splitext(OUTPUT_FILE_PATH)[1].lower().lstrip('.')

            # Determine format for the helper function
            if output_extension in ['parquet', 'pickle']:
                save_format = output_extension
            else:
                # Default or fallback if extension is different (e.g., .csv)
                # save_file currently only supports parquet/pickle, so this might error
                # or you might adjust save_file to handle csv too.
                # For now, let's try passing the extension, save_file will validate.
                warnings.warn(f"Output file extension '.{output_extension}' may not be directly supported by save_file helper (expects 'parquet' or 'pickle'). Attempting anyway.")
                save_format = output_extension # Let save_file handle validation

            # Call the helper function
            save_file(
                data=df,
                path=output_dir,
                file_name=base_file_name,
                format=save_format, # Pass the determined format
                index=False         # Don't save index for parquet
            )
            # Success message is printed inside save_file
            print("Preprocessing complete.")

        except (ValueError, TypeError, ImportError, Exception) as e:
            # Catch errors specifically from save_file or path extraction
            print(f"Failed to save file using helper function: {e}")

    except (FileNotFoundError, TypeError, ValueError, IsADirectoryError, ImportError, KeyError, Exception) as e:
        print(f"\n--- Preprocessing Failed ---")
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
