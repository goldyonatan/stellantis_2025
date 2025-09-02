# extract_seg_start_nonbehavioral_features.py
# ------------------------------------------------------------
# Build **non-behavioral** features that are strictly known at
# the **start of each segment (t0)**, with destination assumed
# known at t0 as well (end of the segment). No behavior/time
# leakage: we never aggregate over the segment and never peek at
# post‑t0 sensor values. We only use:
#   • static metadata (vehicle/battery)
#   • t0 timestamp/location readings
#   • destination (segment end) static info (coords, DEM elev,
#     planned OSRM route metrics) — acceptable if destination is
#     known before departure
#   • static OSM context around start and destination
#
# Also includes tqdm-style progress (via HelperFuncs.get_tqdm),
# and resilient checkpointing similar to the feature extraction
# script. We also reuse the OSM graph **once per trip** to avoid
# repeated Overpass requests (big speedup).
# ------------------------------------------------------------

import os
import math
import warnings
import traceback
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import osmnx as ox

# ---- Helper imports (provided by your repo) ----
try:
    from HelperFuncs import (
        load_file, save_file, sort_df_by_trip_time, find_seg_start_idx,
        get_tqdm, haversine_np, query_dem_opentopo, get_osrm_route,
        get_path_context_from_osm, get_graph_for_trip_corridor, save_checkpoint
    )
except ImportError as e:
    raise ImportError("Could not import one or more helpers from HelperFuncs.py") from e

# Progress bar (aka "taqdam" in your note)
tqdm = get_tqdm()

# ---------------- Configuration ----------------
# Adjust these paths to your environment (kept consistent with your other scripts)
CLEANED_DATA_PATH = r"C:/Users/goldy/OneDrive/Documents/Y-DATA 24-25/Stellantis Project/End-to-End/clean_df/clean_df.parquet"
SEGMENTATION_DICT_PATH = r"C:/Users/goldy/OneDrive/Documents/Y-DATA 24-25/Stellantis Project/End-to-End/segmentation_dict/trips_seg_dict_v7.pickle"

OUTPUT_DIR = r"C:/Users/goldy/OneDrive/Documents/Y-DATA 24-25/Stellantis Project/End-to-End/nonbehavioral_t0_features"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "t0_nonbehavioral_features.parquet")
OUTPUT_FORMAT = "parquet"  # save_file infers from suffix

# Checkpointing (mirrors your feature extraction script)
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
CHECKPOINT_INTERVAL = 200  # save after every N **trips** fully processed

# OSMnx settings (same spirit as your feature extraction script)
OSMNX_CACHE_FOLDER = r"C:/Users/goldy/OneDrive/Documents/Y-DATA 24-25/Stellantis Project/End-to-End/osmnx_cache"
ox.settings.use_cache = True
ox.settings.cache_folder = OSMNX_CACHE_FOLDER
ox.settings.log_console = False

# OSRM public router (read‑only)
OSRM_BASE_URL = "http://router.project-osrm.org"

# Segmentation rule param used across your code
FULL_STOP = 3  # number of consecutive stop flags that constitute an anchor

# ---------- Column name constants (aligned with your data) ----------
TRIP_ID_COL = "trip_id"
TIME_COL = "timestamp"
LAT_COL = "latitude"
LON_COL = "longitude"
ODO_COL = "current_odo"
SOC_COL = "current_soc"
TEMP_COL = "outside_temp"

CAR_MODEL_COL = "car_model"
MANUFACTURER_COL = "manufacturer"
BATTERY_TYPE_COL = "battery_type"
BATTERY_HEALTH_COL = "battery_health"  # t0 value only (no segment mean!)
BATTERY_CAP_COL = "battery_capacity_kWh"
WEIGHT_COL = "empty_weight_kg"

# DO NOT USE: behavior/time-leaky arrays or flags
ALT_ARRAY_COL = "altitude_array"      # we will NOT read from arrays
SPEED_ARRAY_COL = "speed_array"       # we will NOT read from arrays
GAP_FLAG_COL = "is_start_after_gap"   # explicitly excluded

# ---------- Utilities (no leakage) ----------

def encode_time_features(t: pd.Timestamp) -> Dict[str, float]:
    """Derive time-of-day, day-of-week, month — all from t0 only.
    Uses the provided timestamp as-is (no peeking beyond t0)."""
    t = pd.to_datetime(t)
    hour = int(getattr(t, "hour", pd.to_datetime(t, utc=True).hour))
    dow = int(getattr(t, "dayofweek", pd.to_datetime(t, utc=True).dayofweek))
    month = int(getattr(t, "month", pd.to_datetime(t, utc=True).month))
    return {
        "start_hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "start_hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "day_of_week": dow,  # 0=Mon
        "is_weekend": int(dow >= 5),
        "month": month,
    }


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward azimuth from start to destination (0..360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    brg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return brg


def safe_numeric(val):
    try:
        return pd.to_numeric(val)
    except Exception:
        return np.nan


# ---------- Core builder ----------

def build_t0_features_for_segment(
    trip_df: pd.DataFrame,
    trip_labels: np.ndarray,
    seg_id: int,
    G_trip: Optional[object] = None,
) -> Dict:
    """
    Returns a dictionary of **non-behavioral** features known at t0 for the given segment.

    Non-leaky choices:
      • use the row at segment_start_idx for all sensor/context fields
      • use the last labeled row ONLY for destination coordinates (treated as known at t0)
      • DEM elevations for start and dest (static)
      • OSRM route distance/duration (planned route based on start/dest)
      • OSM static attributes around start & dest (from a trip-level graph)
    """
    # Indices for this seg (no use of arrays/aggregations)
    labeled_idxs = np.where(trip_labels == seg_id)[0]
    if len(labeled_idxs) == 0:
        return {}

    first_labeled = labeled_idxs[0]
    last_labeled = labeled_idxs[-1]
    seg_start_idx = find_seg_start_idx(trip_labels, first_labeled, full_stop=FULL_STOP)

    # t0 row (all features read from here)
    row0 = trip_df.iloc[seg_start_idx]
    t0 = pd.to_datetime(row0[TIME_COL])
    lat0 = float(row0[LAT_COL])
    lon0 = float(row0[LON_COL])

    # Destination (assumed known at t0): last labeled point of this segment
    row_dest = trip_df.iloc[last_labeled]
    lat1 = float(row_dest[LAT_COL])
    lon1 = float(row_dest[LON_COL])

    # DEM elevations (static, no leakage)
    try:
        elevs = query_dem_opentopo([float(lat0), float(lat1)], [float(lon0), float(lon1)])
        start_elev_m = float(elevs[0]) if elevs is not None and len(elevs) == 2 else np.nan
        dest_elev_m = float(elevs[1]) if elevs is not None and len(elevs) == 2 else np.nan
    except Exception:
        start_elev_m = np.nan
        dest_elev_m = np.nan

    # Straight-line metrics
    try:
        # haversine_np expects arrays; returns kilometers in your helpers
        crow_km = float(haversine_np(np.array([lon0]), np.array([lat0]), np.array([lon1]), np.array([lat1]))[0])
    except Exception:
        crow_km = np.nan

    try:
        brg_deg = initial_bearing_deg(lat0, lon0, lat1, lon1)
    except Exception:
        brg_deg = np.nan

    # OSRM planned route (distance/duration only – static at t0)
    try:
        route = get_osrm_route([(lat0, lon0), (lat1, lon1)], base_url=OSRM_BASE_URL)
        osrm_dist_km = float(route.get("distance", np.nan)) / 1000.0 if route else np.nan
        osrm_dur_min = float(route.get("duration", np.nan)) / 60.0 if route else np.nan
    except Exception:
        osrm_dist_km = np.nan
        osrm_dur_min = np.nan

    # OSM static context at start & dest from trip-level graph (if available)
    start_osm = {"start_road_type": np.nan, "start_lanes": np.nan, "start_oneway": np.nan, "start_surface": np.nan}
    dest_osm  = {"dest_road_type": np.nan,  "dest_lanes": np.nan,  "dest_oneway": np.nan,  "dest_surface": np.nan}

    try:
        if G_trip is not None:
            df_points = pd.DataFrame({
                TIME_COL: [t0, t0],  # timestamps irrelevant for static OSM attrs
                LAT_COL:  [lat0, lat1],
                LON_COL:  [lon0, lon1],
            })
            df_points = get_path_context_from_osm(df_points, G_trip, trip_df[TRIP_ID_COL].iloc[0], int(seg_id))
            # Expected columns from helper: 'road_type','lanes_count','is_oneway','surface_type'
            if not df_points.empty:
                s = df_points.iloc[0]
                start_osm = {
                    "start_road_type": s.get("road_type", np.nan),
                    "start_lanes": safe_numeric(s.get("lanes_count", np.nan)),
                    "start_oneway": int(s.get("is_oneway", 0)) if not pd.isna(s.get("is_oneway", np.nan)) else np.nan,
                    "start_surface": s.get("surface_type", np.nan),
                }
                d = df_points.iloc[1]
                dest_osm = {
                    "dest_road_type": d.get("road_type", np.nan),
                    "dest_lanes": safe_numeric(d.get("lanes_count", np.nan)),
                    "dest_oneway": int(d.get("is_oneway", 0)) if not pd.isna(d.get("is_oneway", np.nan)) else np.nan,
                    "dest_surface": d.get("surface_type", np.nan),
                }
    except Exception:
        pass

    # Assemble feature row (ONLY t0 + dest-known-at-t0 + static)
    out = {
        # IDs
        TRIP_ID_COL: trip_df[TRIP_ID_COL].iloc[0],
        "seg_id": int(seg_id),

        # --- Static vehicle/battery ---
        CAR_MODEL_COL: row0.get(CAR_MODEL_COL, "unknown"),
        MANUFACTURER_COL: row0.get(MANUFACTURER_COL, np.nan),
        BATTERY_TYPE_COL: row0.get(BATTERY_TYPE_COL, np.nan),
        BATTERY_CAP_COL: safe_numeric(row0.get(BATTERY_CAP_COL, np.nan)),
        WEIGHT_COL: safe_numeric(row0.get(WEIGHT_COL, np.nan)),
        BATTERY_HEALTH_COL: safe_numeric(row0.get(BATTERY_HEALTH_COL, np.nan)),  # t0 only

        # --- t0 sensor/context ---
        "t0_timestamp": t0,
        "t0_lat": lat0,
        "t0_lon": lon0,
        "t0_soc": safe_numeric(row0.get(SOC_COL, np.nan)),
        "t0_outside_temp_c": safe_numeric(row0.get(TEMP_COL, np.nan)),
        "t0_odo": safe_numeric(row0.get(ODO_COL, np.nan)),
        "t0_elev_m": start_elev_m,
        **encode_time_features(t0),

        # --- Destination (assumed known at t0) ---
        "dest_lat": lat1,
        "dest_lon": lon1,
        "dest_elev_m": dest_elev_m,
        "dest_bearing_deg": brg_deg,

        # --- Start↔Dest geometric/planned ---
        "crow_distance_km": crow_km,
        "osrm_distance_km": osrm_dist_km,
        "osrm_duration_min": osrm_dur_min,

        # --- OSM static attrs ---
        **start_osm,
        **dest_osm,
    }

    # Explicitly DO NOT include behavioral arrays or gap flag
    # (e.g., ALT_ARRAY_COL, SPEED_ARRAY_COL, GAP_FLAG_COL)

    # --- Targets (labels; safe & non-leaky) ---
    # We read the SoC at t0 from the segment START index (already placed in out["t0_soc"])
    # and the SoC at the END of the segment from the *last labeled* row (row_dest).
    # This respects the read indices and never peeks beyond segment end.
    end_soc_pct = safe_numeric(row_dest.get(SOC_COL, np.nan))
    out["soc_end_pct"] = end_soc_pct

    return out

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("--- Loading cleaned data & segmentation dict ---")
    df = load_file(CLEANED_DATA_PATH)
    trips_seg_dict = load_file(SEGMENTATION_DICT_PATH)

    if not isinstance(df, pd.DataFrame):
        raise TypeError("cleaned data is not a DataFrame")
    if not isinstance(trips_seg_dict, dict):
        raise TypeError("segmentation dict is not a dict")

    # Ensure sorting once globally
    df = sort_df_by_trip_time(df, trip_id_col=TRIP_ID_COL, time_col=TIME_COL)

    # --- Resume from checkpoints (same pattern as feature extraction) ---
    print("--- Checking for existing checkpoints to resume ---")
    checkpoint_windows_path = os.path.join(CHECKPOINT_DIR, "windows_checkpoint.pickle")
    checkpoint_trips_path = os.path.join(CHECKPOINT_DIR, "processed_trips_checkpoint.pickle")

    rows: List[Dict] = []            # list of per-segment dicts
    processed_trip_ids: set = set()  # trips fully processed in this script

    if os.path.exists(checkpoint_windows_path) and os.path.exists(checkpoint_trips_path):
        print("Found checkpoint files. Loading...")
        try:
            loaded_rows = load_file(checkpoint_windows_path)
            loaded_trips = load_file(checkpoint_trips_path)
            if isinstance(loaded_rows, list):
                rows = loaded_rows
            if isinstance(loaded_trips, set):
                processed_trip_ids = loaded_trips
            print(f"Resuming. Loaded {len(rows)} segment rows from {len(processed_trip_ids)} processed trips.")
        except Exception as e:
            print(f"Warning: Could not load checkpoint files. Error: {e}. Starting from scratch.")
            rows, processed_trip_ids = [], set()
    else:
        print("No complete checkpoint found. Starting from scratch.")

    # For fast membership checks when skipping already-processed trips
    processed_trip_ids = set(processed_trip_ids) if processed_trip_ids else set()

    # --- Main loop ---
    trip_ids = df[TRIP_ID_COL].unique().tolist()
    try:
        for i, trip_id in enumerate(tqdm(trip_ids, desc="Trips"), 1):
            if trip_id in processed_trip_ids:
                continue

            trip_df = df[df[TRIP_ID_COL] == trip_id].copy()
            labels = trips_seg_dict.get(trip_id, None)
            if labels is None:
                warnings.warn(f"No labels for trip_id {trip_id}; skipping trip")
                processed_trip_ids.add(trip_id)
                continue
            if len(trip_df) != len(labels):
                warnings.warn(f"Length mismatch for trip_id {trip_id}; skipping trip")
                processed_trip_ids.add(trip_id)
                continue

            # Prepare a **trip-level** OSM graph once (speedup vs per-segment)
            try:
                coords_trip = list(zip(trip_df[LAT_COL].astype(float).tolist(), trip_df[LON_COL].astype(float).tolist()))
                G_trip = get_graph_for_trip_corridor(coords_trip, buffer_m=800)
            except Exception:
                G_trip = None

            # Positive segment IDs only
            seg_ids = np.unique(labels[labels > 0]).astype(int)
            if len(seg_ids) == 0:
                processed_trip_ids.add(trip_id)
                continue

            for seg_id in seg_ids:
                try:
                    feat = build_t0_features_for_segment(trip_df, labels, seg_id, G_trip)
                    if feat:
                        rows.append(feat)
                except Exception as e:
                    warnings.warn(f"[{trip_id}-{seg_id}] Failed t0 feature build: {e}")
                    traceback.print_exc()

            # mark trip as processed and maybe checkpoint
            processed_trip_ids.add(trip_id)
            if (len(processed_trip_ids) % CHECKPOINT_INTERVAL) == 0:
                print("--- Periodic checkpoint save ---")
                try:
                    save_checkpoint(rows, processed_trip_ids, CHECKPOINT_DIR)
                except Exception as e:
                    warnings.warn(f"Checkpoint save failed: {e}")

        # done all trips
        out_df = pd.DataFrame(rows)

        # Final sanity: drop any accidental columns that could be leaky
        leaky_cols = [c for c in out_df.columns if c in (ALT_ARRAY_COL, SPEED_ARRAY_COL, GAP_FLAG_COL)]
        if leaky_cols:
            out_df.drop(columns=leaky_cols, inplace=True, errors='ignore')

        # De-duplicate in case of resume after partial writes (keep last)
        if {TRIP_ID_COL, 'seg_id'}.issubset(out_df.columns):
            before = len(out_df)
            out_df = out_df.drop_duplicates(subset=[TRIP_ID_COL, 'seg_id'], keep='last').reset_index(drop=True)
            after = len(out_df)
            if after < before:
                print(f"Removed {before - after} duplicate (trip_id, seg_id) rows after resume.")

        print(f"Built non-behavioral t0 feature table with shape: {out_df.shape}")
        # Persist (parquet recommended). save_file infers from suffix.
        output_dir = os.path.dirname(OUTPUT_PATH)
        base_file_name = os.path.splitext(os.path.basename(OUTPUT_PATH))[0]
        save_file(data=out_df, path=output_dir, file_name=base_file_name, format=OUTPUT_FORMAT)
        print(f"Saved: {OUTPUT_PATH}")

    except (KeyboardInterrupt, Exception) as e:
        print(f"--- Interrupted/Failed: {e}. Will perform final checkpoint save. ---")
        traceback.print_exc()
        raise
    finally:
        # Always checkpoint current progress (rows + processed_trip_ids)
        try:
            save_checkpoint(rows, processed_trip_ids, CHECKPOINT_DIR)
        except Exception as e:
            warnings.warn(f"Final checkpoint save failed: {e}")


if __name__ == "__main__":
    main()
