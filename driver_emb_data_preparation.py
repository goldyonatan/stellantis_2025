from __future__ import annotations
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
from driver_emb_feature_extraction import flag_outliers
from typing import Dict, List, Optional, Tuple, Any, Iterable, Union
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# --- Configuration ---
# Path to your existing, partially incorrect checkpoint file
INPUT_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\checkpoints\windows_checkpoint.pickle"

# Path to save the new, corrected file
OUTPUT_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\windows_corrected.pickle"

# NEW: a different file for the artifacts dict (splits, masks, etc.)
ARTIFACTS_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\embedding_artifacts.pickle"

PLOT_PATH= r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\analysis_reports"

TRIP_ID_COL = 'trip_id'
CAR_MODEL_COL = "car_model"

TEST_SIZE = 0.15

MAX_UNIQUE_TO_PRINT = 50

CORRECT_SPEED_REL_LIMIT_AND_PROPULSIVE = True

# This must match the value in your feature extraction script
WIND_DRAG_FACTOR = 0.05

MIN_SPEED_FOR_LATERAL = 0.5  # m/s

OUTLIER_CONFIGS = {
    # Primary motion
    'speed_mps': {
        # Max seen ≈ 41.8 m/s; keep a safe headroom but clip unrealistic values.
        'raw_limits': (-0.5, 60.0),
    },
    'long_accel_mps2': {
        # Seen: -6.95…8.86; keep road-car envelope.
        'raw_limits': (-10.0, 10.0),
    },

    # Lateral dynamics (derived)
    'centripetal_accel_mps2': {
        # Heavy-tailed due to tiny radii; enforce realistic lateral g.
        'raw_limits': (0.0,20.0),       
        'diff_limits': (-18.0, 18.0),
    },
    'lat_accel_signed_mps2': {
        # v * ω; spikes mostly from heading wrap/noise.
        'raw_limits': (-20.0, 20.0),
        'diff_limits': (-22.0, 22.0),
    },
    'centripetal_jerk_mps3': {
        # Stats show rare large spikes; treat as artifacts.
        'raw_limits': (-100.0, 100.0),
        'diff_limits': (-120.0, 120.0),
    },

    # Longitudinal derivatives
    'jerk_mps3': {
        # Typical car jerk rarely > ~10 m/s^3; keep generous to avoid false positives.
        'raw_limits': (-20.0, 20.0),
        'diff_limits': (-30.0, 30.0),
    },
    'grade_corrected_accel': {
        # Slightly wider than long_accel to absorb grade compensation.
        'raw_limits': (-12.0, 12.0),
    },

    # Yaw
    'yaw_rate_dps': {
        # Tight enough to catch wrap spikes, generous for real U-turns.
        'raw_limits': (-180.0, 180.0),
        'diff_limits': (-200.0, 200.0),
    },
    'yaw_accel_dps2': {
        
        'raw_limits': (-400.0, 400.0),
        'diff_limits': (-600.0, 600.0),
    },

    # Speed vs limit (pick one of these based on which column you feed)
    'speed_limit_ratio': {
        # Median ~0.76, max ~5.0 (includes odd cases/inf when limit=0).
        # Clip egregious values; flag inf/NaN upstream is already handled.
        'raw_limits': (0.0, 3.7),
    },
}

FINAL_KIN_CHANNELS = [
    'jerk_mps3',
    'grade_corrected_accel',
    'yaw_rate_dps',
    'yaw_accel_dps2',
    'lat_accel_signed_mps2',
    'centripetal_jerk_mps3',
    'speed_limit_ratio'
]

# Map child channel -> list of parent/base channels it depends on.
# Derivative-like children will get neighbor expansion (prev/next),
# dependent (non-derivative) children get direct (same-index) masking.
DERIV_DEP_MAP: Dict[str, List[str]] = {
    # --- derivatives (time-derivatives) ---
    "centripetal_jerk_mps3": ["centripetal_accel_mps2"],
    "jerk_mps3": ["long_accel_mps2"],
    "yaw_accel_dps2": ["yaw_rate_dps"],
    # --- dependents (non-derivatives) ---
    "lat_accel_signed_mps2": ["centripetal_accel_mps2"],
}

DERIVATIVE_COLS = {"centripetal_jerk_mps3", "jerk_mps3", "yaw_accel_dps2"}

def gate_low_speed_for_lateral(
    df: pd.DataFrame,
    min_speed_for_lateral: float = 0.8,              # m/s
    yaw_rate_threshold: float = 60.0,                 # deg/s (|yaw_rate| must exceed this)
    lateral_channels: tuple = ("centripetal_accel_mps2", "lat_accel_signed_mps2"),
    fill: str = "nan",                                # "zero" or "nan"
    recompute_centripetal_jerk: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Post-hoc guardrail for array-based windows in a DataFrame.
    Gating applies ONLY where BOTH conditions are true:
        speed_mps < min_speed_for_lateral  AND  abs(yaw_rate_dps) > yaw_rate_threshold

    Mutates df in place and returns it.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("gate_low_speed_for_lateral expects a pandas DataFrame.")
    for req in ("speed_mps", "yaw_rate_dps"):
        if req not in df.columns:
            raise KeyError(f"DataFrame is missing required '{req}' column.")

    # Operate only on channels that exist
    active_channels = tuple(ch for ch in lateral_channels if ch in df.columns)
    if not active_channels:
        if verbose:
            print("gate_low_speed_for_lateral: no lateral channels found; nothing to do.")
        return df

    # Stats
    n_rows = len(df)
    rows_touched = 0
    totals = {ch: 0 for ch in active_channels}        # elements considered (min aligned length per row)
    corrected = {ch: 0 for ch in active_channels}     # elements set to 0/NaN
    total_mask_true = 0
    total_considered = 0

    def _fill_value(dtype):
        return 0.0 if fill == "zero" else np.nan

    for idx, row in df.iterrows():
        v = row["speed_mps"]
        yr = row["yaw_rate_dps"]
        if v is None or yr is None:
            continue

        v = np.asarray(v)
        yr = np.asarray(yr)
        L = min(v.size, yr.size)
        if L == 0:
            continue

        # BOTH conditions must hold
        m = (v[:L] < min_speed_for_lateral) & (np.abs(yr[:L]) > yaw_rate_threshold)
        if not np.any(m):
            # still count considered for overall stats
            total_considered += L
            for ch in active_channels:
                if ch in row and row[ch] is not None:
                    totals[ch] += min(np.asarray(row[ch]).size, L)
            continue

        row_touched = False
        total_mask_true += int(m.sum())
        total_considered += L

        for ch in active_channels:
            a = row.get(ch, None)
            if a is None:
                continue
            a = np.asarray(a)
            Lc = min(a.size, L)
            if Lc == 0:
                continue

            mc = m[:Lc]
            if not np.any(mc):
                totals[ch] += Lc
                continue

            a2 = a.copy()
            a2[mc] = _fill_value(a2.dtype)
            df.at[idx, ch] = a2

            totals[ch] += Lc
            corrected[ch] += int(mc.sum())
            row_touched = True

        if row_touched:
            rows_touched += 1
            # keep centripetal jerk consistent with potentially updated centripetal accel
            if recompute_centripetal_jerk and ("centripetal_accel_mps2" in active_channels) and ("centripetal_jerk_mps3" in df.columns):
                ca = np.asarray(df.at[idx, "centripetal_accel_mps2"])
                if ca.size > 0:
                    df.at[idx, "centripetal_jerk_mps3"] = np.gradient(ca).astype(ca.dtype)

    if verbose:
        print(
            f"[gate_low_speed_for_lateral] min_speed={min_speed_for_lateral} m/s | "
            f"yaw_rate>|{yaw_rate_threshold}| deg/s | fill='{fill}' | "
            f"recompute_centripetal_jerk={recompute_centripetal_jerk}"
        )
        print(
            f"Processed rows: {n_rows:,} | rows touched: {rows_touched:,} | "
            f"mask true (both cond.): {total_mask_true:,}/{total_considered:,} "
            f"({(100.0*total_mask_true/total_considered if total_considered else 0):.2f}%)"
        )
        for ch in active_channels:
            if totals[ch] > 0:
                pct = 100.0 * corrected[ch] / totals[ch]
                print(f"  - {ch}: corrected {corrected[ch]:,}/{totals[ch]:,} values ({pct:.2f}%)")
            else:
                print(f"  - {ch}: no elements considered (totals=0)")

    return df

def correct_propulsion_features(df: pd.DataFrame, wind_drag_factor: float) -> pd.DataFrame:
    """
    Corrects the propulsive acceleration and power features in the windows DataFrame.
    This now corrects both 'propulsive_power_kw' and 'propulsive_power_kw_per_kg'.

    Args:
        df (pd.DataFrame): The input DataFrame with windows data.
        wind_drag_factor (float): The factor to apply to headwind for drag calculation.

    Returns:
        pd.DataFrame: A new DataFrame with the corrected features.
    """
    df_corrected = df.copy()
    print("Applying propulsion feature correction...")

    # Loop through each window and apply the correction
    for index, row in tqdm(df_corrected.iterrows(), total=df_corrected.shape[0], desc="Correcting Propulsion"):
        # Retrieve the necessary arrays and static values from the row
        grade_corr_accel = row['grade_corrected_accel']
        headwind = row['headwind_mps']
        speed = row['speed_mps']
        
        # 1. Get the raw value for vehicle mass.
        vehicle_mass_raw = row['empty_weight_kg']
        
        # 2. Convert it to a numeric type. 'coerce' will turn any non-numeric
        #    strings (like 'Unknown' or '1500 kg') into NaN (Not a Number).
        vehicle_mass = pd.to_numeric(vehicle_mass_raw, errors='coerce')
        
        # 3. Add a safety check. If mass could not be converted, skip this row.
        if pd.isna(vehicle_mass):
            continue

        # Sanity check that we are working with numpy arrays
        if not isinstance(grade_corr_accel, np.ndarray) or not isinstance(headwind, np.ndarray):
            continue

        # --- THE CORRECTION ---
        # 1. Correct Propulsive Accel = grade_corrected_accel + wind_component
        wind_component_accel = headwind * wind_drag_factor
        new_propulsive_accel = grade_corr_accel + wind_component_accel

        # 2. Correct Propulsive Power (in kW)
        propulsive_force = vehicle_mass * new_propulsive_accel
        propulsive_power_watts = propulsive_force * speed
        new_propulsive_power_kw = propulsive_power_watts / 1000.0

        # 3. Correct Propulsive Power per kg (in kW/kg)
        new_propulsive_power_per_kg = new_propulsive_power_kw / vehicle_mass

        # Overwrite the old arrays in the DataFrame using .at for efficiency
        df_corrected.at[index, 'propulsive_accel_mps2'] = new_propulsive_accel
        df_corrected.at[index, 'propulsive_power_kw'] = new_propulsive_power_kw
        df_corrected.at[index, 'propulsive_power_kw_per_kg'] = new_propulsive_power_per_kg

    print("Propulsion correction complete.")
    return df_corrected

def add_speed_limit_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms 'speed_rel_limit_mps' from a difference to a ratio.
    It calculates the original speed limit and adds a new 'speed_limit_ratio' column.

    Args:
        df (pd.DataFrame): The input DataFrame, must contain 'speed_mps' and 'speed_rel_limit_mps'.

    Returns:
        pd.DataFrame: The DataFrame with the new 'speed_limit_ratio' column.
    """
    df_transformed = df.copy()
    print("Adding speed-to-limit ratio feature...")

    required_cols = ['speed_rel_limit_mps', 'speed_mps']
    if not all(col in df_transformed.columns for col in required_cols):
        print("WARNING: Missing 'speed_rel_limit_mps' or 'speed_mps'. Skipping ratio calculation.")
        return df_transformed

    new_ratio_col = []
    for index, row in tqdm(df_transformed.iterrows(), total=df_transformed.shape[0], desc="Calculating Speed Ratio"):
        speed_array = row['speed_mps']
        speed_rel_limit_array = row['speed_rel_limit_mps']

        if not isinstance(speed_array, np.ndarray) or not isinstance(speed_rel_limit_array, np.ndarray):
            new_ratio_col.append(np.full_like(speed_array, np.nan) if isinstance(speed_array, np.ndarray) else None)
            continue

        # Calculate the original speed limit: speed_limit = speed - speed_rel_limit
        speed_limit_array = speed_array - speed_rel_limit_array

        # Initialize the output array with NaNs
        ratio_array = np.full_like(speed_array, np.nan, dtype=float)

        # Case 1: speed_limit is not zero (normal case)
        mask_normal = speed_limit_array != 0
        ratio_array[mask_normal] = speed_array[mask_normal] / speed_limit_array[mask_normal]

        # Case 2: speed_limit is zero
        mask_zero_limit = ~mask_normal
        # Sub-case A: speed is also zero -> ratio is 1.0 (driving at the limit)
        mask_zero_speed = speed_array == 0
        ratio_array[mask_zero_limit & mask_zero_speed] = 1.0
        # Sub-case B: speed is non-zero -> ratio is inf (infinitely over the limit)
        ratio_array[mask_zero_limit & ~mask_zero_speed] = np.inf

        new_ratio_col.append(ratio_array)

    df_transformed['speed_limit_ratio'] = new_ratio_col
    print("Speed ratio calculation complete.")
    return df_transformed

def analyze_hierarchical_structure(df: pd.DataFrame):
    """
    Analyzes and prints a high-level report on the hierarchical structure
    of the data (trips, segments, windows).
    """
    print("===================================================")
    print("==========   HIERARCHICAL DATA STRUCTURE   ==========")
    print("===================================================")

    if 'trip_id' not in df.columns or 'seg_id' not in df.columns:
        print("\nERROR: 'trip_id' or 'seg_id' not found. Cannot analyze structure.")
        return

    n_trips = df['trip_id'].nunique()
    n_segs = df.groupby('trip_id')['seg_id'].nunique().sum()
    n_wins = len(df)

    print(f"\n--- Overall Counts ---")
    print(f"  - Total Unique Trips    : {n_trips}")
    print(f"  - Total Unique Segments : {n_segs}")
    print(f"  - Total Windows         : {n_wins}")

    if n_trips > 0:
        segs_per_trip = df.groupby('trip_id')['seg_id'].nunique()
        print("\n--- Segments per Trip ---")
        print(segs_per_trip.describe().to_string())

    if n_segs > 0:
        wins_per_seg = df.groupby(['trip_id', 'seg_id']).size()
        print("\n--- Windows per Segment ---")
        print(wins_per_seg.describe().to_string())
    
    print("\n" + "="*51 + "\n")

def print_missing_values_report(
    windows_df: pd.DataFrame,
    kinematic_channels: list
):
    """
    Analyzes and prints a comprehensive report on missing values (NaNs) within
    the kinematic channel arrays of the windows DataFrame.
    """
    print("===================================================")
    print("=======   MISSING VALUES REPORT (ARRAYS)   ========")
    print("===================================================")

    if windows_df.empty:
        print("\nWarning: The windows DataFrame is empty. No report to generate.")
        return

    channels_to_process = [ch for ch in kinematic_channels if ch in windows_df.columns]
    if not channels_to_process:
        print("\nWarning: No array-based channels were found. Skipping report.")
        return

    try:
        tensor = np.stack([np.vstack(windows_df[c].to_numpy()) for c in channels_to_process], axis=1)
    except ValueError as e:
        print(f"\nCRITICAL ERROR: Could not stack arrays due to inconsistent lengths.")
        print("Please re-run the feature extraction pipeline with validation.")
        return

    n_win, n_chans, seq_len = tensor.shape
    grand_total_samples = tensor.size
    total_samples_per_chan = n_win * seq_len
    nan_mask = ~np.isfinite(tensor)

    print("\n--- 1. Summary by Individual Samples (Total NaNs) ---")
    for i, ch in enumerate(channels_to_process):
        n_missing = np.sum(nan_mask[:, i, :])
        pct_missing = (n_missing / total_samples_per_chan) * 100 if total_samples_per_chan > 0 else 0
        print(f"{ch:<25}: {n_missing:>8,} missing samples ({pct_missing:7.3f}%)")
    overall_missing = np.sum(nan_mask)
    overall_pct = (overall_missing / grand_total_samples) * 100 if grand_total_samples > 0 else 0
    print("-" * 60)
    print(f"{'TOTAL':<25}: {overall_missing:>8,} missing samples ({overall_pct:7.3f}% of all samples)")

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
    
    print("\n" + "="*51 + "\n")

def analyze_scalar_feature(series: pd.Series, col_name: str):
    """Analyzes and prints a report for a column with scalar values."""
    print(f"======== Analyzing Scalar Feature: '{col_name}' ========")
    
    dtype = series.dtype
    missing_count = series.isnull().sum()
    missing_percent = (missing_count / len(series)) * 100 if len(series) > 0 else 0
    
    print(f"  - Data Type         : {dtype}")
    print(f"  - Missing Values    : {missing_count} ({missing_percent:.2f}%)")

    try:
        nunique = series.nunique()
        print(f"  - Unique Values (est): {nunique}")
        unique_vals = series.unique()
        
        print("  - Unique Samples    :", end=" ")
        if nunique <= MAX_UNIQUE_TO_PRINT:
            try:
                print(np.sort(unique_vals[~pd.isna(unique_vals)]))
            except TypeError:
                print(unique_vals[~pd.isna(unique_vals)])
        else:
            print(f"First {MAX_UNIQUE_TO_PRINT} unique samples shown.")
            
    except TypeError:
        print("  - Unique Values (est): N/A (unhashable type in column)")

    if pd.api.types.is_numeric_dtype(series):
        print("\n  --- Numeric Stats ---")
        stats = series.describe()
        print(stats.to_string())
    
    print("=" * (len(col_name) + 35))
    print("\n")

def analyze_array_feature(series: pd.Series, col_name: str):
    """Analyzes and prints a report for a column containing numpy arrays."""
    print(f"======== Analyzing Array-Based Feature: '{col_name}' ========")
    
    series_non_null = series.dropna()
    if series_non_null.empty:
        print("  - Column is entirely empty or NaN. No analysis possible.")
        print("=" * (len(col_name) + 40))
        print("\n")
        return
        
    dtype = series_non_null.iloc[0].dtype
    missing_count = series.isnull().sum()
    missing_percent = (missing_count / len(series)) * 100 if len(series) > 0 else 0
    
    print(f"  - Data Type (in array): {dtype}")
    print(f"  - Missing Windows     : {missing_count} ({missing_percent:.2f}%)")

    try:
        lengths = series_non_null.apply(len)
        print("\n  --- Array Shape Stats ---")
        print(f"  - Min Length          : {lengths.min()}")
        print(f"  - Max Length          : {lengths.max()}")
        print(f"  - Mean Length         : {lengths.mean():.2f}")
        if lengths.min() != lengths.max():
            print("  - WARNING: Inconsistent array lengths found!")
    except Exception as e:
        print(f"  - Could not compute array shape stats. Error: {e}")

    try:
        flattened_data = np.concatenate(series_non_null.to_list())
        
        total_samples = len(flattened_data)
        nan_samples = np.isnan(flattened_data).sum()
        nan_percent = (nan_samples / total_samples) * 100 if total_samples > 0 else 0
        
        print("\n  --- Flattened Data Stats (all samples from all windows combined) ---")
        print(f"  - Total Samples     : {total_samples:,}")
        print(f"  - NaN Samples       : {nan_samples:,} ({nan_percent:.2f}%)")

        numeric_data = flattened_data[~np.isnan(flattened_data)]
        if numeric_data.size > 0:
            print(f"  - Min               : {numeric_data.min():.4f}")
            print(f"  - Max               : {numeric_data.max():.4f}")
            print(f"  - Mean              : {numeric_data.mean():.4f}")
            print(f"  - Std Dev           : {numeric_data.std():.4f}")
            print(f"  - Median            : {np.median(numeric_data):.4f}")
        else:
            print("  - No valid numeric data to calculate statistics.")

    except Exception as e:
        print(f"\n  - Could not compute flattened data stats. Error: {e}")

    print("=" * (len(col_name) + 40))
    print("\n")

def analyze_array_correlations(df: pd.DataFrame, array_features: list, method: str = 'spearman'):
    """
    Calculates and visualizes the correlation matrix for array-based features.
    Defaults to Spearman for robustness to outliers and non-linear relationships.
    """
    print("===================================================")
    print(f"=======   ARRAY FEATURE CORRELATION (SPEARMAN)   ========")
    print("===================================================")

    if df.empty:
        print("\nInput DataFrame is empty. Skipping correlation analysis.")
        return

    print(f"\nAnalyzing {method.capitalize()} correlations for {len(array_features)} array-based features.")

    flattened_data = {}
    base_length = -1

    for feature in array_features:
        series_non_null = df[feature].dropna()
        if series_non_null.empty:
            print(f"  - Skipping '{feature}': column is entirely NaN.")
            continue
        
        flat_vector = np.concatenate(series_non_null.to_list())
        flattened_data[feature] = flat_vector
        
        if base_length == -1:
            base_length = len(flat_vector)
        elif len(flat_vector) != base_length:
            print("\nCRITICAL ERROR: Mismatched flattened lengths between features.")
            print("This is likely due to inconsistent array lengths in the source DataFrame.")
            print("Aborting correlation analysis.")
            return

    if not flattened_data:
        print("\nNo valid data to form correlation matrix. Aborting.")
        return

    corr_df = pd.DataFrame(flattened_data)
    
    print(f"\nCalculating {method.capitalize()} correlation matrix...")
    correlation_matrix = corr_df.corr(method=method)

    print(f"\n--- {method.capitalize()} Correlation Matrix ---")
    print(correlation_matrix.to_string())

    print("\nGenerating correlation heatmap...")
    plt.figure(figsize=(16, 14))
    should_annotate = len(correlation_matrix.columns) < 20
    
    sns.heatmap(
        correlation_matrix, 
        annot=should_annotate, 
        cmap='coolwarm', 
        fmt='.2f', 
        linewidths=.5,
        vmin=-1, 
        vmax=1
    )
    plt.title(f'{method.capitalize()} Correlation Matrix of Array-Based Features', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    try:
        os.makedirs(PLOT_PATH, exist_ok=True)
        heatmap_path = os.path.join(PLOT_PATH, f"array_feature_{method}_correlation_heatmap.png")
        plt.savefig(heatmap_path, dpi=150)
        print(f"\nSuccessfully saved heatmap to: {heatmap_path}")
    except Exception as e:
        print(f"\nERROR: Could not save heatmap. {e}")
    
    plt.close()
    print("\n" + "="*51 + "\n")


def analyze_all_features(df: pd.DataFrame):
    """
    Orchestrates the analysis of all columns in the DataFrame.
    """
    print("===================================================")
    print("==========   INDIVIDUAL FEATURE ANALYSIS   ==========")
    print("===================================================")

    for col_name in sorted(df.columns):
        if df[col_name].dropna().empty:
            print(f"======== Analyzing Feature: '{col_name}' ========")
            print("  - Column is entirely empty or NaN. Skipping detailed analysis.")
            print("=" * (len(col_name) + 30))
            print("\n")
            continue

        first_item = df[col_name].dropna().iloc[0]

        if isinstance(first_item, (np.ndarray, list)):
            analyze_array_feature(df[col_name], col_name)
        else:
            analyze_scalar_feature(df[col_name], col_name)

def drop_windows_with_high_nonfinite(
    df: pd.DataFrame,
    array_cols: Optional[List[str]] = None,
    per_channel_max_nonfinite_frac: float = 0.30,  # i.e., require >= 70% finite per channel
    overall_max_nonfinite_frac: float = 0.20,      # i.e., require >= 80% finite overall
    min_required_channels: Optional[int] = None,   # require at least this many channels to pass per-channel threshold
    keep_reason_column: str = "drop_reason",       # name for reason column (added to a side report, not to returned df)
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop windows whose arrays contain too high a percentage of non-finite values.

    Parameters
    ----------
    df : DataFrame
        One row per window. Each channel column contains a 1-D ndarray (length ~window_len).
    array_cols : list of str, optional
        Which columns to evaluate. If None, auto-detect columns that look like 1-D numeric arrays.
    per_channel_max_nonfinite_frac : float
        Max allowed non-finite fraction PER CHANNEL (e.g., 0.30 means at least 70% finite required per channel).
    overall_max_nonfinite_frac : float
        Max allowed non-finite fraction across ALL selected channels combined (time x channels).
    min_required_channels : int, optional
        If set, require at least this many channels to pass the per-channel threshold (default: all selected channels).
    keep_reason_column : str
        Column name in the returned *report* (not the filtered df) describing why a row was dropped.
    verbose : bool
        Print a summary.

    Returns
    -------
    filtered_df : DataFrame
        df with failing windows dropped (index preserved).
    report_df : DataFrame
        Per-window diagnostics: per-channel nonfinite fractions, overall nonfinite fraction, and drop reason (if any).

    Notes
    -----
    - Non-finite = {NaN, +Inf, -Inf}. Uses np.isfinite().
    - Arrays with length 0 or None are treated as fully non-finite for that channel.
    - If arrays have mixed lengths across channels, overall fraction is computed over the sum of lengths actually present.
    - This function does not mutate `df`.
    """
    # --- basic validation ---
    if not (0.0 <= per_channel_max_nonfinite_frac < 1.0):
        raise ValueError("per_channel_max_nonfinite_frac must be in [0, 1).")
    if not (0.0 <= overall_max_nonfinite_frac < 1.0):
        raise ValueError("overall_max_nonfinite_frac must be in [0, 1).")

    # --- infer array columns if needed ---
    def _looks_like_numeric_1d_array(x: Any) -> bool:
        if isinstance(x, np.ndarray):
            return x.ndim == 1 and np.issubdtype(x.dtype, np.number)
        # support list/tuple of numbers (rare)
        if isinstance(x, (list, tuple)) and len(x) > 0:
            try:
                arr = np.asarray(x)
                return arr.ndim == 1 and np.issubdtype(arr.dtype, np.number)
            except Exception:
                return False
        return False

    if array_cols is None:
        candidate_cols = []
        # sample a few rows for detection to avoid scanning everything
        sample_idx = df.index[: min(100, len(df))]
        for col in df.columns:
            try:
                sample_vals = df.loc[sample_idx, col].tolist()
            except Exception:
                continue
            if sample_vals and all(_looks_like_numeric_1d_array(v) for v in sample_vals if v is not None):
                candidate_cols.append(col)
        array_cols = candidate_cols

    if len(array_cols) == 0:
        raise ValueError("No array-like channel columns detected. Pass `array_cols` explicitly.")

    # --- compute per-row stats ---
    # We’ll store fractions in a dict-of-lists for vectorized DataFrame construction at the end
    per_chan_nonfinite_frac: Dict[str, List[float]] = {c: [] for c in array_cols}
    per_chan_lengths: Dict[str, List[int]] = {c: [] for c in array_cols}
    overall_nonfinite_frac: List[float] = []
    drop_reason: List[Optional[str]] = []

    # To avoid costly conversions on big frames, iterate rows
    it = df[array_cols].itertuples(index=True, name=None)  # yields (index, col1_val, col2_val, ...)
    for row in it:
        idx = row[0]
        chan_values = row[1:]

        # per-channel
        chan_ok_count = 0
        chan_fail = False
        total_len = 0
        total_nonfinite = 0
        row_reasons = []

        for col_name, values in zip(array_cols, chan_values):
            if values is None:
                length = 0
                nonfinite = 0
                frac = 1.0  # treat as fully non-finite
            else:
                arr = values if isinstance(values, np.ndarray) else np.asarray(values)
                if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.number):
                    # treat invalid structure as fully non-finite
                    length = len(arr) if hasattr(arr, "__len__") else 0
                    nonfinite = length
                    frac = 1.0
                else:
                    finite_mask = np.isfinite(arr)
                    length = arr.size
                    nonfinite = int(length - int(finite_mask.sum()))
                    frac = (nonfinite / length) if length > 0 else 1.0

            per_chan_nonfinite_frac[col_name].append(frac)
            per_chan_lengths[col_name].append(length)

            total_len += length
            total_nonfinite += nonfinite

            if frac > per_channel_max_nonfinite_frac:
                chan_fail = True
                row_reasons.append(f"{col_name}:nonfinite>{per_channel_max_nonfinite_frac:.2f}")

        # overall fraction across channels (weighted by actual lengths)
        overall_frac = (total_nonfinite / total_len) if total_len > 0 else 1.0
        overall_nonfinite_frac.append(overall_frac)

        # per-channel pass count (for min_required_channels)
        chan_pass_count = sum(
            1 for c in array_cols if per_chan_nonfinite_frac[c][-1] <= per_channel_max_nonfinite_frac
        )

        # decide keep/drop
        reason = None
        if overall_frac > overall_max_nonfinite_frac:
            reason = f"overall_nonfinite>{overall_max_nonfinite_frac:.2f}"
        if chan_fail:
            reason = (reason + "; " if reason else "") + "per_channel_threshold_exceeded"
        if min_required_channels is not None and chan_pass_count < min_required_channels:
            reason = (reason + "; " if reason else "") + f"passed_channels<{min_required_channels}"

        drop_reason.append(reason)

    # --- build report df ---
    report_cols = {
        f"{c}_nonfinite_frac": per_chan_nonfinite_frac[c] for c in array_cols
    } | {
        f"{c}_length": per_chan_lengths[c] for c in array_cols
    }
    report_df = pd.DataFrame(report_cols, index=df.index)
    report_df["overall_nonfinite_frac"] = overall_nonfinite_frac
    report_df[keep_reason_column] = drop_reason

    # --- build keep mask and filter ---
    keep_mask = report_df[keep_reason_column].isna()
    filtered_df = df.loc[keep_mask].copy()

    if verbose:
        total = len(df)
        kept = int(keep_mask.sum())
        dropped = total - kept
        print("=== drop_windows_with_high_nonfinite ===")
        print(f"Checked channels: {array_cols}")
        print(f"Per-channel max non-finite: {per_channel_max_nonfinite_frac:.2f}")
        print(f"Overall    max non-finite: {overall_max_nonfinite_frac:.2f}")
        if min_required_channels is not None:
            print(f"Min required channels passing per-channel threshold: {min_required_channels}")
        print(f"Windows total: {total:,} | kept: {kept:,} ({kept/total*100:.2f}%) | dropped: {dropped:,} ({dropped/total*100:.2f}%)")

        # quick breakdown of top drop reasons
        if dropped > 0:
            reason_counts = report_df.loc[~keep_mask, keep_reason_column].value_counts().head(10)
            print("\nTop drop reasons:")
            for r, n in reason_counts.items():
                print(f"  - {r}: {n:,}")

    return filtered_df, report_df

ArrayLike = Union[np.ndarray, List[np.ndarray]]
WindowsLike = Union[pd.DataFrame, Dict[str, ArrayLike]]

def build_nonfinite_masks(
    windows: WindowsLike,
    channels: Iterable[str],
    *,
    packbits: bool = False,
    return_stacked: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], np.ndarray | None]:
    """
    Build per-window NON-FINITE masks for each array channel.

    Args:
        windows:
            - Either a pandas DataFrame where each `channels` column holds a 1D numpy array
              (length = window_len) for each window/row, OR
            - A dict mapping channel -> (N x L) ndarray, or list of (L,) arrays per window.
        channels: iterable of array-column names to process (order defines channel order).
        packbits: if True, pack the time-axis mask into bits to reduce memory (8x smaller).
                  Axis layout is preserved for channel/row dimension; only time is packed.
        return_stacked: if True, also return a single stacked mask of shape:
            - (N, C, L) when packbits=False
            - (N, C, L_packed) when packbits=True
          If False, returns None in that slot.

    Returns:
        masks_by_channel: dict[channel] -> mask ndarray, shape (N, L) or (N, L_packed)
                          dtype=uint8, where 1 = NON-FINITE, 0 = finite.
        meta: {'n_windows': int, 'window_len': int, 'n_channels': int}
        stacked: np.ndarray or None
                 If return_stacked=True:
                   - shape (N, C, L) or (N, C, L_packed), dtype=uint8
                 Else: None

    Notes:
        - NON-FINITE is defined via ~np.isfinite(x) (i.e., NaN, +inf, -inf).
        - The function does NOT modify your data, normalize, or fill values—masking only.
        - Row order is preserved; you can shuffle indices and index both data and masks
          with the same permutation in your DataLoader.
    """
    # Helper: get a dense (N, L) float array for a given channel without copying unnecessarily
    def _as_2d(arr_like: ArrayLike) -> np.ndarray:
        arr = np.asarray(arr_like, dtype=object) if not isinstance(arr_like, np.ndarray) else arr_like
        # Case 1: already numeric 2D (N, L)
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.dtype != object:
            return arr
        # Case 2: list/1D object array of (L,) arrays -> stack
        if isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.dtype == object:
            return np.stack(arr.tolist(), axis=0)
        if isinstance(arr, list):
            return np.stack(arr, axis=0)
        raise ValueError("Unsupported array layout for a channel; expected (N,L) numeric or 1D object/list of (L,) arrays.")

    # Normalize input access
    get_col = None
    if isinstance(windows, pd.DataFrame):
        def get_col(ch: str) -> ArrayLike:
            if ch not in windows.columns:
                raise KeyError(f"Channel '{ch}' not found in DataFrame.")
            return windows[ch].to_numpy()
    elif isinstance(windows, dict):
        def get_col(ch: str) -> ArrayLike:
            if ch not in windows:
                raise KeyError(f"Channel '{ch}' not found in dict.")
            return windows[ch]
    else:
        raise TypeError("`windows` must be a pandas.DataFrame or a dict[channel -> arrays].")

    channels = list(channels)
    if len(channels) == 0:
        raise ValueError("`channels` must contain at least one channel name.")

    # Build masks per channel
    masks_by_channel: Dict[str, np.ndarray] = {}
    N = None
    L = None

    for ch in channels:
        data_2d = _as_2d(get_col(ch))  # (N, L), numeric
        if data_2d.ndim != 2:
            raise ValueError(f"Channel '{ch}' is not 2D after stacking; got shape {data_2d.shape}.")
        if N is None:
            N, L = data_2d.shape
        else:
            if data_2d.shape[0] != N:
                raise ValueError(f"Channel '{ch}' has a different number of windows: {data_2d.shape[0]} vs expected {N}.")
            if data_2d.shape[1] != L:
                raise ValueError(f"Channel '{ch}' has a different window length: {data_2d.shape[1]} vs expected {L}.")

        # NON-FINITE mask: True where value is NaN/Inf
        mask_bool = ~np.isfinite(data_2d)
        mask_u8 = mask_bool.astype(np.uint8)  # 1 = NON-FINITE, 0 = finite

        if packbits:
            # Pack along time axis for memory efficiency (keeps row & channel dims intact)
            # np.packbits expects last axis boolean/uint8; we keep dtype=uint8.
            mask_u8 = np.packbits(mask_u8, axis=1)  # shape: (N, ceil(L/8))

        masks_by_channel[ch] = mask_u8

    meta = {
        "n_windows": int(N if N is not None else 0),
        "window_len": int(L if L is not None else 0),
        "n_channels": int(len(channels)),
    }

    stacked = None
    if return_stacked:
        # Stack in channel order -> (N, C, L_or_packed)
        stacked = np.stack([masks_by_channel[ch] for ch in channels], axis=1).astype(np.uint8)

    return masks_by_channel, meta, stacked

# ---------- helpers ----------

def _as_2d(arr_like: ArrayLike) -> np.ndarray:
    """Return a numeric (N,L) ndarray from various column formats."""
    if isinstance(arr_like, np.ndarray):
        if arr_like.ndim == 2 and arr_like.dtype != object:
            return arr_like
        if arr_like.ndim == 1 and arr_like.dtype == object:
            return np.stack(arr_like.tolist(), axis=0)
    if isinstance(arr_like, list):
        return np.stack(arr_like, axis=0)
    raise ValueError("Unsupported array layout; expected (N,L) numeric or 1D object/list of (L,) arrays.")

def _get_col(windows: WindowsLike, ch: str) -> ArrayLike:
    if isinstance(windows, pd.DataFrame):
        if ch not in windows.columns:
            raise KeyError(f"Channel '{ch}' not found in DataFrame.")
        return windows[ch].to_numpy()
    elif isinstance(windows, dict):
        if ch not in windows:
            raise KeyError(f"Channel '{ch}' not found in dict.")
        return windows[ch]
    else:
        raise TypeError("`windows` must be a pandas.DataFrame or a dict[channel -> arrays].")

def _set_col(windows: WindowsLike, ch: str, data2d: np.ndarray, copy: bool) -> WindowsLike:
    """Write back (N,L) numeric array to match the input structure."""
    if isinstance(windows, pd.DataFrame):
        out = windows.copy(deep=True) if copy else windows
        # store as object array of (L,) rows to keep row-wise array semantics
        out[ch] = [data2d[i].copy() for i in range(data2d.shape[0])]
        return out
    elif isinstance(windows, dict):
        out = dict(windows) if copy else windows
        out[ch] = data2d
        return out
    else:
        raise TypeError("`windows` must be a pandas.DataFrame or a dict[channel -> arrays].")

# ---------- build stats on TRAIN ONLY (finite-aware) ----------

def build_channel_normalizer(
    train_windows: WindowsLike,
    channels: Iterable[str],
    *,
    method: Literal["zscore", "minmax", "robust"] = "zscore",
    clip_percentiles: Tuple[float, float] | None = None,  # e.g., (0.5, 99.5)
    eps: float = 1e-6,
) -> Dict[str, dict]:
    """
    Compute per-channel normalization stats using only FINITE values from TRAIN windows.

    Returns:
        normalizer: dict with:
          {
            'method': 'zscore' | 'minmax' | 'robust',
            'channels': [..],
            'stats': {
               ch: {'mean':..., 'std':...} | {'min':..., 'max':...} | {'median':..., 'iqr':...}
            },
            'meta': {'n_windows': N, 'window_len': L}
          }
    """
    channels = list(channels)
    if not channels:
        raise ValueError("`channels` must contain at least one channel name.")

    stats: Dict[str, dict] = {}
    N = L = None

    for ch in channels:
        x2d = _as_2d(_get_col(train_windows, ch))  # (N,L)
        if N is None:
            N, L = x2d.shape
        else:
            if x2d.shape != (N, L):
                raise ValueError(f"Channel '{ch}' shape mismatch: {x2d.shape} vs expected {(N,L)}.")

        # Flatten finite values only
        finite_mask = np.isfinite(x2d)
        vals = x2d[finite_mask]
        if vals.size == 0:
            # Fall back to neutral stats to avoid NaNs
            if method == "zscore":
                stats[ch] = {"mean": 0.0, "std": 1.0}
            elif method == "minmax":
                stats[ch] = {"min": 0.0, "max": 1.0}
            else:  # robust
                stats[ch] = {"median": 0.0, "iqr": 1.0}
            continue

        # Optional outlier clipping for stats computation
        if clip_percentiles is not None:
            lo, hi = np.percentile(vals, clip_percentiles)
            vals = vals[(vals >= lo) & (vals <= hi)]

        if method == "zscore":
            m = float(vals.mean())
            s = float(vals.std(ddof=0))
            if not np.isfinite(s) or s < eps:
                s = 1.0
            stats[ch] = {"mean": m, "std": s}
        elif method == "minmax":
            vmin = float(vals.min())
            vmax = float(vals.max())
            if not np.isfinite(vmax - vmin) or (vmax - vmin) < eps:
                vmax = vmin + 1.0
            stats[ch] = {"min": vmin, "max": vmax}
        elif method == "robust":
            med = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25.0, 75.0])
            iqr = float(q3 - q1)
            if not np.isfinite(iqr) or iqr < eps:
                iqr = 1.0
            stats[ch] = {"median": med, "iqr": iqr}
        else:
            raise ValueError(f"Unknown method '{method}'.")

    normalizer = {
        "method": method,
        "channels": channels,
        "stats": stats,
        "meta": {"n_windows": int(N or 0), "window_len": int(L or 0)},
        "eps": float(eps),
        "clip_percentiles": clip_percentiles,
    }
    return normalizer

# ---------- apply to ANY split (finite-aware; non-finite left as-is) ----------

def apply_channel_normalization(
    windows: WindowsLike,
    normalizer: Dict[str, dict],
    *,
    copy: bool = True,
    dtype: np.dtype = np.float32,
) -> WindowsLike:
    """
    Apply per-channel normalization IN-PLACE or on a copy.
    Only FINITE entries are transformed; non-finite remain unchanged (you can fill later).

    Args:
        windows: DataFrame of array columns or dict[channel -> arrays]
        normalizer: object returned by `build_channel_normalizer(...)`
        copy: if True, return a deep copy (DataFrame) / shallow copy (dict) with new arrays
        dtype: cast arrays to this dtype after normalization
    """
    method: str = normalizer["method"]
    channels: List[str] = list(normalizer["channels"])
    stats: Dict[str, dict] = normalizer["stats"]
    eps: float = float(normalizer.get("eps", 1e-6))

    out = windows.copy(deep=True) if (copy and isinstance(windows, pd.DataFrame)) else (dict(windows) if (copy and isinstance(windows, dict)) else windows)

    for ch in channels:
        x2d = _as_2d(_get_col(out, ch)).astype(dtype, copy=True)  # (N,L)
        finite = np.isfinite(x2d)

        if method == "zscore":
            m = stats[ch]["mean"]
            s = stats[ch]["std"]
            s = s if (np.isfinite(s) and s >= eps) else 1.0
            # normalize only finite entries
            x2d[finite] = (x2d[finite] - m) / s
        elif method == "minmax":
            vmin = stats[ch]["min"]
            vmax = stats[ch]["max"]
            rng = vmax - vmin
            rng = rng if (np.isfinite(rng) and rng >= eps) else 1.0
            x2d[finite] = (x2d[finite] - vmin) / rng
        else:  # robust
            med = stats[ch]["median"]
            iqr = stats[ch]["iqr"]
            iqr = iqr if (np.isfinite(iqr) and iqr >= eps) else 1.0
            x2d[finite] = (x2d[finite] - med) / iqr

        out = _set_col(out, ch, x2d, copy=False)

    return out

def fill_nonfinite_with_zero(
    windows: WindowsLike,
    channels: Iterable[str],
    masks: Union[np.ndarray, Dict[str, np.ndarray]],
    *,
    packbits: bool = False,
    copy: bool = True,
    dtype: np.dtype = np.float32,
) -> Tuple[WindowsLike, Dict[str, int], int]:
    """
    Fill non-finite entries (NaN, +Inf, -Inf) with zero using precomputed NON-FINITE masks.

    Args:
        windows: DataFrame of array columns OR dict[channel -> (N,L) arrays].
        channels: channel names to process (order matters ONLY if masks is stacked ndarray).
        masks:
          - If dict: {channel -> (N,L) uint8/bool mask} where 1/True == NON-FINITE
            (or packed along time if packbits=True).
          - If ndarray: shape (N, C, L) (or packed (N, C, ceil(L/8)) if packbits=True).
            The channel order must match `channels`.
        packbits: True if masks are packed along the time axis (from np.packbits).
        copy: return a copy (DataFrame deep copy / dict shallow copy) instead of in-place.
        dtype: cast arrays to this dtype after filling.

    Returns:
        windows_filled: same type as `windows`, with non-finite entries set to 0.0 for `channels`.
        filled_counts: dict[channel] -> number of values set to zero
        filled_total: total number filled across all channels

    Notes:
        - This function does NOT recompute non-finiteness; it trusts the mask.
        - Row order is preserved; shuffling is safe as long as you shuffle data and masks together.
    """
    channels = list(channels)
    if not channels:
        raise ValueError("`channels` must contain at least one channel name.")

    # Prepare output container
    out = windows.copy(deep=True) if (copy and isinstance(windows, pd.DataFrame)) else (dict(windows) if (copy and isinstance(windows, dict)) else windows)

    # Helper: get (N, L) boolean mask for a given channel and target length L
    def _get_mask_for_channel(ch_idx: int, ch_name: str, target_L: int) -> np.ndarray:
        if isinstance(masks, dict):
            if ch_name not in masks:
                raise KeyError(f"Mask for channel '{ch_name}' not found in masks dict.")
            m = masks[ch_name]
            if not isinstance(m, np.ndarray):
                m = np.asarray(m)
            if m.ndim != 2:
                raise ValueError(f"Mask for channel '{ch_name}' must be 2D; got shape {m.shape}.")
            if packbits:
                # m shape: (N, ceil(L/8)) -> unpack to (N, L_bits) then trim/pad to target_L
                mu = np.unpackbits(m.astype(np.uint8), axis=1)
                if mu.shape[1] < target_L:
                    pad = np.zeros((mu.shape[0], target_L - mu.shape[1]), dtype=mu.dtype)
                    mu = np.concatenate([mu, pad], axis=1)
                mask_bool = (mu[:, :target_L] != 0)
            else:
                if m.shape[1] != target_L:
                    raise ValueError(f"Mask for channel '{ch_name}' has time length {m.shape[1]} != expected {target_L}.")
                mask_bool = (m != 0)
            return mask_bool
        else:
            # stacked ndarray: (N, C, L) or packed (N, C, Lp)
            m = masks
            if not isinstance(m, np.ndarray) or m.ndim != 3:
                raise ValueError("Stacked masks must be a 3D ndarray of shape (N, C, L) or packed (N, C, ceil(L/8)).")
            if ch_idx >= m.shape[1]:
                raise IndexError(f"Channel index {ch_idx} out of bounds for masks with C={m.shape[1]}.")
            if packbits:
                mu = np.unpackbits(m[:, ch_idx, :].astype(np.uint8), axis=1)
                if mu.shape[1] < target_L:
                    pad = np.zeros((mu.shape[0], target_L - mu.shape[1]), dtype=mu.dtype)
                    mu = np.concatenate([mu, pad], axis=1)
                mask_bool = (mu[:, :target_L] != 0)
            else:
                if m.shape[2] != target_L:
                    raise ValueError(f"Stacked mask time length {m.shape[2]} != expected {target_L}.")
                mask_bool = (m[:, ch_idx, :] != 0)
            return mask_bool

    filled_counts: Dict[str, int] = {}
    filled_total = 0

    # Process each channel independently
    for ci, ch in enumerate(channels):
        x2d = _as_2d(_get_col(out, ch)).astype(dtype, copy=True)  # (N, L)
        N, L = x2d.shape

        mask_bool = _get_mask_for_channel(ci, ch, L)  # (N, L) bool where True==NON-FINITE
        if mask_bool.shape != (N, L):
            raise ValueError(f"Mask for channel '{ch}' has shape {mask_bool.shape}, expected {(N, L)}.")

        # Count fills, then fill
        n_fill = int(mask_bool.sum())
        if n_fill > 0:
            x2d[mask_bool] = 0.0

        filled_counts[ch] = n_fill
        filled_total += n_fill

        out = _set_col(out, ch, x2d, copy=False)

    return out, filled_counts, filled_total

def _to_np_1d(a):
    """Coerce to 1D np.ndarray if possible; otherwise return as-is."""
    if isinstance(a, np.ndarray):
        return a
    try:
        arr = np.asarray(a)
        if arr.ndim == 1:
            return arr
    except Exception:
        pass
    return a  # leave untouched if not a 1D array


def _expand_neighbors(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Expand True values to immediate neighbors (prev/next)."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 1:
        return mask
    out = mask.copy()
    for r in range(1, radius + 1):
        out[:-r] |= mask[r:]
        out[r:] |= mask[:-r]
    return out


def harmonize_derivatives_and_dependents(
    df: pd.DataFrame,
    mapping: Dict[str, List[str]] = None,
    neighbor_radius_for_derivatives: int = 1,
    inplace: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Make derivative/dependent channels NaN exactly where their base inputs are non-finite,
    without recomputing any values.

    - For derivative channels, mask indices that touch a non-finite parent at i-1, i, or i+1
      (mirrors central-difference support of np.gradient; edges naturally use one neighbor).
    - For dependent channels, mask only at the same indices where parents are non-finite.

    Parameters
    ----------
    df : DataFrame with array-valued columns (each cell is a 1D np.ndarray)
    mapping : dict child -> list of parent columns
    neighbor_radius_for_derivatives : typically 1 (prev/next). Use 2 if your derivative stencil is wider.
    inplace : modify df in place if True
    verbose : print a short summary

    Returns
    -------
    DataFrame (same shape), with child arrays patched by setting affected indices to NaN.
    """
    if mapping is None:
        mapping = DERIV_DEP_MAP

    target = df if inplace else df.copy()
    changes_summary = {}

    for child_col, parents in mapping.items():
        if child_col not in target.columns:
            continue  # skip silently if not present

        total_masked = 0
        total_elems = 0

        def _patch_row(row):
            nonlocal total_masked, total_elems

            child = _to_np_1d(row[child_col])
            if not isinstance(child, np.ndarray) or child.ndim != 1:
                return row[child_col]  # skip non-array rows

            n = child.shape[0]
            # build invalid mask from parents
            invalid = np.zeros(n, dtype=bool)
            for p in parents:
                if p not in row:
                    continue
                base = _to_np_1d(row[p])
                if not isinstance(base, np.ndarray) or base.ndim != 1:
                    continue
                # align lengths if needed (defensive)
                m = min(n, base.shape[0])
                base_valid = np.isfinite(base[:m])
                inv = ~base_valid
                if child_col in DERIVATIVE_COLS:
                    inv = _expand_neighbors(inv, radius=neighbor_radius_for_derivatives)
                # pad/truncate to child length
                if m < n:
                    pad = np.zeros(n - m, dtype=bool)
                    inv = np.concatenate([inv, pad])
                invalid |= inv[:n]

            if invalid.any():
                child_fixed = child.copy()
                # Keep dtype if float; if dtype is integer, conversion will happen anyway
                child_fixed[invalid] = np.nan
                total_masked += int(invalid.sum())
                total_elems += n
                return child_fixed
            else:
                total_elems += n
                return row[child_col]

        target[child_col] = target.apply(_patch_row, axis=1)
        changes_summary[child_col] = (total_masked, total_elems)

    if verbose and changes_summary:
        lines = []
        for k, (masked, tot) in changes_summary.items():
            if tot > 0:
                pct = 100.0 * masked / tot
                lines.append(f"- {k}: masked {masked}/{tot} samples ({pct:.3f}%)")
        if lines:
            print("[harmonize] Masked derivative/dependent indices where parents were non-finite:\n" +
                  "\n".join(lines))

    return target

def main():
    """
    Loads the windows DataFrame, applies corrections and transformations,
    and saves the result to a new file.
    """
    print(f"--- Starting Feature Correction and Transformation Script ---")

    # --- 1. Load the data ---
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input file not found at: {INPUT_PATH}")
        return

    print(f"Loading data from: {INPUT_PATH}")
    try:
        with open(INPUT_PATH, 'rb') as f:
            df_list = pickle.load(f)
        # The checkpoint is a list of dicts, convert to DataFrame
        if isinstance(df_list, list):
            df = pd.DataFrame(df_list)
        else: # If it's already a DataFrame
            df = df_list
        print(f"Successfully loaded {len(df)} windows.")
    except Exception as e:
        print(f"ERROR: Failed to load or process the pickle file. {e}")
        return

    if CORRECT_SPEED_REL_LIMIT_AND_PROPULSIVE:

        # # --- 2. Verify required columns exist for propulsion correction ---
        # propulsion_cols = [
        #     'grade_corrected_accel', 'headwind_mps', 'propulsive_accel_mps2',
        #     'speed_mps', 'propulsive_power_kw', 'propulsive_power_kw_per_kg',
        #     'empty_weight_kg' # Added vehicle mass
        # ]
        # if not all(col in df.columns for col in propulsion_cols):
        #     missing = [c for c in propulsion_cols if c not in df.columns]
        #     print(f"ERROR: DataFrame is missing columns for propulsion correction: {missing}")
        #     return

        # --- 3. Apply corrections and transformations by calling the functions ---
        #df = correct_propulsion_features(df, wind_drag_factor=WIND_DRAG_FACTOR)
        df = add_speed_limit_ratio(df)

        # --- 4. Sanity Check (Optional but recommended) ---
        # print("\n--- Sanity Check ---")
        # try:
        #     # Find a window where the correction would have made a difference
        #     check_idx = df['grade_corrected_accel'].apply(lambda x: np.nanmean(np.abs(x)) > 0.1).idxmax()

        #     # Load original row for comparison
        #     with open(INPUT_PATH, 'rb') as f:
        #         original_list = pickle.load(f)
        #     original_row = pd.DataFrame(original_list).iloc[check_idx]

        #     print(f"Checking window at index: {check_idx}")
        #     print(f"Mean Grade-Corrected Accel: {np.nanmean(df.at[check_idx, 'grade_corrected_accel']):.4f}")
        #     print(f"Mean Headwind:              {np.nanmean(df.at[check_idx, 'headwind_mps']):.4f}")
        #     print("-" * 30)
        #     print(f"Old Propulsive Accel (mean): {np.nanmean(original_row['propulsive_accel_mps2']):.4f}")
        #     print(f"New Propulsive Accel (mean): {np.nanmean(df.at[check_idx, 'propulsive_accel_mps2']):.4f}")
        #     print("-" * 30)
        #     print(f"Old Power (mean): {np.nanmean(original_row['propulsive_power_kw']):.4f}")
        #     print(f"New Power (mean): {np.nanmean(df.at[check_idx, 'propulsive_power_kw']):.4f}")
        #     print("-" * 30)
        #     print(f"Old Power/kg (mean): {np.nanmean(original_row['propulsive_power_kw_per_kg']):.6f}")
        #     print(f"New Power/kg (mean): {np.nanmean(df.at[check_idx, 'propulsive_power_kw_per_kg']):.6f}")
        #     print("-" * 30)
        #     if 'speed_limit_ratio' in df.columns:
        #         print(f"New Speed/Limit Ratio (mean): {np.nanmean(df.at[check_idx, 'speed_limit_ratio']):.4f}")

        # except Exception as e:
        #     print(f"Could not perform sanity check. Error: {e}")

    print("\n--- Gate low speed for lateral ---")
    df = gate_low_speed_for_lateral(
        df,
        min_speed_for_lateral=MIN_SPEED_FOR_LATERAL,     # 0.8 m/s ≈ 2.9 km/h
        yaw_rate_threshold=65.0,       # gate only when yaw is “busy”
        lateral_channels=("centripetal_accel_mps2", "lat_accel_signed_mps2"),
        fill="nan",
        recompute_centripetal_jerk=True,
        verbose=True,
    )

    # 1. High-level structure
    analyze_hierarchical_structure(df)
    
    # Dynamically find all array-based columns
    array_features = [
        col for col in df.columns 
        if not df[col].dropna().empty and isinstance(df[col].dropna().iloc[0], np.ndarray)
    ]
    
    # 2. Consolidated missing values report for arrays
    print_missing_values_report(df, array_features)
    
    # 3. Correlation analysis for array features
    #analyze_array_correlations(df, array_features, method='spearman')
    
    # 4. Detailed breakdown of every single feature
    analyze_all_features(df)

    # 9. Flag outliers
    print(f"\n--- Flagging outliers ---")
    df = flag_outliers(
        df,
        OUTLIER_CONFIGS,
        n_preview= 50
    )

    df = harmonize_derivatives_and_dependents(
        df,
        neighbor_radius_for_derivatives=1,  # matches np.gradient support
        inplace=False,
        verbose=True,
    )

    print_missing_values_report(df, array_features)

    qc_cols = [c for c in FINAL_KIN_CHANNELS if c != "speed_limit_ratio"]

    df, qc = drop_windows_with_high_nonfinite(
        df,
        array_cols=qc_cols,
        per_channel_max_nonfinite_frac=0.30,  # require >=70% finite within each channel
        overall_max_nonfinite_frac=0.20,      # require >=80% finite overall across channels
        min_required_channels=None,           # or set e.g. 7 to allow a couple channels to be messy
        verbose=True
    )

    # 3. Stratified split by car model (trip level)
    print("\n--- Train, validation, test split stratified for car model ---")
    trip_car = df.groupby(TRIP_ID_COL)[CAR_MODEL_COL].first().reset_index()

    train_val_ids, test_ids = train_test_split(
    trip_car[TRIP_ID_COL],
    test_size=TEST_SIZE,
    stratify=trip_car[CAR_MODEL_COL],
    random_state=RANDOM_STATE
    )

    train_val = trip_car[trip_car[TRIP_ID_COL].isin(train_val_ids)] 

    train_ids, val_ids = train_test_split(
    train_val[TRIP_ID_COL],
    test_size=len(test_ids),
    stratify=train_val[CAR_MODEL_COL],
    random_state=RANDOM_STATE
    )

    print(f"Trip split -> train:{len(train_ids)} val:{len(val_ids)} test:{len(test_ids)}")

    train_windows = df[df[TRIP_ID_COL].isin(train_ids)]
    val_windows = df[df[TRIP_ID_COL].isin(val_ids)]
    test_windows  = df[df[TRIP_ID_COL].isin(test_ids)]

    train_masks_by_ch, train_meta, train_masks_stacked = build_nonfinite_masks(
    train_windows, FINAL_KIN_CHANNELS, packbits=False, return_stacked=True
    )

    val_masks_by_ch, val_meta, val_masks_stacked = build_nonfinite_masks(
    val_windows, FINAL_KIN_CHANNELS, packbits=False, return_stacked=True
    )

    test_masks_by_ch, test_meta, test_masks_stacked = build_nonfinite_masks(
    test_windows, FINAL_KIN_CHANNELS, packbits=False, return_stacked=True
    )

    # Fit on TRAIN windows only (finite-aware)
    normalizer = build_channel_normalizer(
        train_windows, FINAL_KIN_CHANNELS,
        method="zscore",            # or "minmax" or "robust"
        clip_percentiles=None,      # or (0.5, 99.5) if you want mild clipping for stats
        eps=1e-6
    )

    # 2) Apply to any split; non-finite values remain as-is (you'll fill with zero later)
    train_norm = apply_channel_normalization(train_windows, normalizer, copy=True)
    val_norm   = apply_channel_normalization(val_windows,   normalizer, copy=True)
    test_norm  = apply_channel_normalization(test_windows,  normalizer, copy=True)

    train_windows, train_per_ch, train_total = fill_nonfinite_with_zero(
    train_norm, FINAL_KIN_CHANNELS, masks=train_masks_stacked, packbits=False, copy=True, dtype=np.float32
    )

    val_windows, val_per_ch, val_total = fill_nonfinite_with_zero(
    val_norm, FINAL_KIN_CHANNELS, masks=val_masks_stacked, packbits=False, copy=True, dtype=np.float32
    )

    test_windows, test_per_ch, test_total = fill_nonfinite_with_zero(
    test_norm, FINAL_KIN_CHANNELS, masks=test_masks_stacked, packbits=False, copy=True, dtype=np.float32
    )

    # --- 5. Save the corrected DataFrame ---

    # --- 5. Save minimal artifacts for embedding training ---
    print(f"\nSaving artifacts for embedding training to: {OUTPUT_PATH}")
    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        REQUIRED_ID_COLS = [TRIP_ID_COL, "seg_id", "win_id", "start_sample"]
        CHANNELS = list(FINAL_KIN_CHANNELS)  # ensure a concrete list & order

        def _prune(df_split: pd.DataFrame) -> pd.DataFrame:
            # Keep only ids (if present) + ordered channels; drop everything else
            present_ids = [c for c in REQUIRED_ID_COLS if c in df_split.columns]
            missing_ids = [c for c in REQUIRED_ID_COLS if c not in df_split.columns]
            if missing_ids:
                # Warn, but still proceed with the present ids (you said win_id exists)
                print(f"[save] Warning: missing id columns in split: {missing_ids}")

            # Validate channels exist
            missing_ch = [c for c in CHANNELS if c not in df_split.columns]
            if missing_ch:
                raise ValueError(f"[save] Missing required channel columns: {missing_ch}")

            keep_cols = present_ids + CHANNELS
            pruned = df_split[keep_cols].reset_index(drop=True)

            # Infer window length from the first available channel
            first_ch = next((c for c in CHANNELS if len(pruned[c]) > 0), None)
            if first_ch is None:
                raise ValueError("[save] Split has no rows; cannot infer window length.")
            # Find first non-empty array
            first_idx = pruned[first_ch].first_valid_index()
            if first_idx is None:
                raise ValueError("[save] Cannot infer window length: no valid arrays found.")
            win_len = len(pruned.at[first_idx, first_ch])

            return pruned, win_len

        def _make_payload(df_split: pd.DataFrame, masks_stacked: np.ndarray, split_name: str):
            pruned, win_len = _prune(df_split)

            # Validate mask shape: (N, C, L)
            N = len(pruned)
            C = len(CHANNELS)
            if masks_stacked.shape != (N, C, win_len):
                raise ValueError(
                    f"[save] Mask shape mismatch for {split_name}: "
                    f"expected {(N, C, win_len)}, got {masks_stacked.shape}"
                )

            return {
                "windows": pruned,                                  # DataFrame with ids + arrays
                "masks": np.ascontiguousarray(masks_stacked).astype(np.uint8, copy=False),  # (N, C, L)
            }

        artifact = {
            "channels": CHANNELS,  # order matters for the trainer
            "train": _make_payload(train_windows, train_masks_stacked, "train"),
            "val":   _make_payload(val_windows,   val_masks_stacked,   "val"),
            "test":  _make_payload(test_windows,  test_masks_stacked,  "test"),
        }

        with open(ARTIFACTS_PATH, "wb") as f:
            pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[save] artifacts → {ARTIFACTS_PATH}")

        print("Saved splits + masks + channels. Shapes:")
        for split in ("train", "val", "test"):
            win_df = artifact[split]["windows"]
            msk = artifact[split]["masks"]
            print(
                f"  {split:<5}: windows={len(win_df):>6}  "
                f"masks={tuple(msk.shape)}  ids={[c for c in REQUIRED_ID_COLS if c in win_df.columns]}"
            )

    except Exception as e:
        print(f"ERROR: Failed to save training artifacts. {e}")

        
    print(f"\nSaving final data to: {OUTPUT_PATH}")
    try:
        output_dir = os.path.dirname(OUTPUT_PATH)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_pickle(OUTPUT_PATH)
        print("Successfully saved the final file.")
    except Exception as e:
        print(f"ERROR: Failed to save the final file. {e}")

if __name__ == '__main__':
    main()