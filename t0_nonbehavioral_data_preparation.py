from pathlib import Path
import os
import math
import warnings
from typing import List

import numpy as np
import pandas as pd
import time, random

# ---- Helper imports (from your repo) ----
try:
    from HelperFuncs import query_dem_opentopo, get_tqdm
except Exception as e:
    raise ImportError("Could not import required helpers from HelperFuncs.py") from e

# Progress bar
_tqdm = get_tqdm()

# ---------------- Path-style Configuration ----------------
BASE_DIR = Path(r"C:/Users/goldy/OneDrive/Documents/Y-DATA 24-25/Stellantis Project/End-to-End")
INPUT_PATH = BASE_DIR / "nonbehavioral_t0_features" / "t0_nonbehavioral_features.parquet"
OUTPUT_PATH = BASE_DIR / "nonbehavioral_t0_features" / "t0_nonbehavioral_features_curated.parquet"

# ---------------- Cleaning Config (scalars) ----------------
# Apply conservative limits; adjust if domain feedback suggests otherwise
MAX_OSRM_DISTANCE_KM: float = 60.0   # segments beyond this are implausible
MAX_OSRM_DURATION_MIN: float = 45.0  # very long for a segment
MIN_SOC_PCT: float = 1.0             # allow very low, but drop <1%
OSRM_LT_CROW_TOLERANCE_KM: float = 0.30  # if OSRM is smaller than crow by > tol -> NaN


DEM_CHUNK_SIZE = 50            # smaller batches to be nice to the API
DEM_COORD_DECIMALS = 5         # round coords to dedupe (~1–10 m precision)
DEM_SLEEP_BASE_SEC = 1.0       # base delay between requests
DEM_SLEEP_JITTER_SEC = 0.5     # plus some jitter to avoid thundering herd
DEM_MAX_RETRIES = 5            # extra outer retries on our side
DEM_CACHE_PATH = BASE_DIR / "nonbehavioral_t0_features" / "dem_cache.parquet"

# ---------------- Core helpers ----------------
def _load_dem_cache():
    if DEM_CACHE_PATH.exists():
        try:
            dfc = pd.read_parquet(DEM_CACHE_PATH)
            # expect columns: lat_r, lon_r, elev_m
            return {(float(r.lat_r), float(r.lon_r)): float(r.elev_m) 
                    for r in dfc.itertuples(index=False)}
        except Exception:
            return {}
    return {}

def _save_dem_cache(cache_dict):
    try:
        if not cache_dict:
            return
        dfc = pd.DataFrame(
            [(k[0], k[1], v) for k, v in cache_dict.items()],
            columns=["lat_r", "lon_r", "elev_m"]
        )
        dfc.drop_duplicates(subset=["lat_r","lon_r"], inplace=True)
        dfc.to_parquet(DEM_CACHE_PATH, index=False)
    except Exception as e:
        warnings.warn(f"Could not save DEM cache: {e}")

# --- ADD this wrapper with backoff/jitter ---
def _query_dem_with_backoff(lat_list, lon_list):
    """Call query_dem_opentopo with extra backoff & jitter on top of helper's retries."""
    for attempt in range(DEM_MAX_RETRIES):
        try:
            vals = query_dem_opentopo(lat_list, lon_list)  # LISTS (helper already retries)
            # polite pause even on success
            time.sleep(DEM_SLEEP_BASE_SEC + random.uniform(0, DEM_SLEEP_JITTER_SEC))
            return vals
        except Exception as e:
            # If 429 (rate-limited), back off more; otherwise still back off a bit
            wait = (DEM_SLEEP_BASE_SEC * (2 ** attempt)) + random.uniform(0, DEM_SLEEP_JITTER_SEC)
            time.sleep(wait)
            if attempt == DEM_MAX_RETRIES - 1:
                raise
    return [np.nan] * len(lat_list)

def _batch_fill_dem(df_in: pd.DataFrame, lat_col: str, lon_col: str, out_col: str) -> int:
    """Fill DEM for missing rows with rate-limiting, caching, and unique-point batching."""
    mask = df_in[out_col].isna() & df_in[lat_col].notna() & df_in[lon_col].notna()
    if not mask.any():
        return 0

    # Round coords to dedupe near-identical points
    lat_r = df_in.loc[mask, lat_col].astype(float).round(DEM_COORD_DECIMALS).tolist()
    lon_r = df_in.loc[mask, lon_col].astype(float).round(DEM_COORD_DECIMALS).tolist()
    idxs  = df_in.loc[mask].index.tolist()

    # Build key -> list[row_indices] and a unique key list
    key_to_rows = {}
    for rix, la, lo in zip(idxs, lat_r, lon_r):
        key_to_rows.setdefault((la, lo), []).append(rix)
    uniq_keys = list(key_to_rows.keys())

    # Load cache
    dem_cache = _load_dem_cache()

    # Determine which unique keys we still need to request
    need_keys = [k for k in uniq_keys if k not in dem_cache]

    # Query in small chunks with backoff + jitter
    filled_unique = 0
    for s in _tqdm(range(0, len(need_keys), DEM_CHUNK_SIZE), desc=f"DEM {out_col}"):
        chunk_keys = need_keys[s:s+DEM_CHUNK_SIZE]
        lats = [k[0] for k in chunk_keys]
        lons = [k[1] for k in chunk_keys]
        try:
            vals = _query_dem_with_backoff(lats, lons)
        except Exception as e:
            warnings.warn(f"DEM batch failed on {s}:{s+DEM_CHUNK_SIZE}: {e}")
            vals = [np.nan] * len(chunk_keys)

        # Update cache
        for k, v in zip(chunk_keys, vals):
            vv = float(v) if v is not None and np.isfinite(v) else np.nan
            dem_cache[k] = vv
            if np.isfinite(vv):
                filled_unique += 1

    # Write cache to disk (best effort)
    _save_dem_cache(dem_cache)

    # Fill DataFrame rows from cache
    filled_rows = 0
    for k, rows in key_to_rows.items():
        val = dem_cache.get(k, np.nan)
        for rix in rows:
            if pd.isna(df_in.at[rix, out_col]) and np.isfinite(val):
                df_in.at[rix, out_col] = val
                filled_rows += 1

    print(f"DEM unique points resolved: {filled_unique} | rows filled: {filled_rows}")
    return filled_rows


def _add_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    # month / day_of_week already exist as ints
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2*np.pi*(df['month'].astype(float)/12.0))
        df['month_cos'] = np.cos(2*np.pi*(df['month'].astype(float)/12.0))
    if 'day_of_week' in df.columns:
        df['dow_sin'] = np.sin(2*np.pi*(df['day_of_week'].astype(float)/7.0))
        df['dow_cos'] = np.cos(2*np.pi*(df['day_of_week'].astype(float)/7.0))
    return df


def _apply_raw_outlier_rules(df: pd.DataFrame) -> pd.DataFrame:
    # Distances/durations
    if 'osrm_distance_km' in df.columns:
        too_far = df['osrm_distance_km'] > MAX_OSRM_DISTANCE_KM
        n = int(too_far.sum())
        if n:
            print(f"[RawOutliers] osrm_distance_km > {MAX_OSRM_DISTANCE_KM}: {n} rows -> NaN")
            df.loc[too_far, 'osrm_distance_km'] = np.nan
    if 'osrm_duration_min' in df.columns:
        too_long = df['osrm_duration_min'] > MAX_OSRM_DURATION_MIN
        n = int(too_long.sum())
        if n:
            print(f"[RawOutliers] osrm_duration_min > {MAX_OSRM_DURATION_MIN}: {n} rows -> NaN")
            df.loc[too_long, 'osrm_duration_min'] = np.nan
    # SoC
    if 't0_soc' in df.columns:
        bad_soc = (df['t0_soc'] < MIN_SOC_PCT) | (df['t0_soc'] > 100)
        n = int(bad_soc.sum())
        if n:
            print(f"[RawOutliers] t0_soc < {MIN_SOC_PCT} or > 100: {n} rows -> NaN")
            df.loc[bad_soc, 't0_soc'] = np.nan
    return df


def _apply_sanity_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # OSRM < crow by more than tolerance -> null OSRM fields (keep crow)
    if {'osrm_distance_km','crow_distance_km'} <= set(df.columns):
        bad = df['osrm_distance_km'].notna() & df['crow_distance_km'].notna() & \
              ((df['osrm_distance_km'] + OSRM_LT_CROW_TOLERANCE_KM) < df['crow_distance_km'])
        n = int(bad.sum())
        print(f"[CHECK->NaN] OSRM distance < crow - {OSRM_LT_CROW_TOLERANCE_KM}km: {n} rows")
        if n:
            print(df.loc[bad, ['trip_id','seg_id','crow_distance_km','osrm_distance_km','osrm_duration_min']].head(10))
            df.loc[bad, ['osrm_distance_km','osrm_duration_min']] = np.nan

    # Crow==0 but coords differ -> crow NaN
    if {'crow_distance_km','t0_lat','t0_lon','dest_lat','dest_lon'} <= set(df.columns):
        diff_coords = (df['t0_lat'] != df['dest_lat']) | (df['t0_lon'] != df['dest_lon'])
        bad_crow = df['crow_distance_km'].fillna(0).eq(0) & diff_coords
        n = int(bad_crow.sum())
        print(f"[CHECK->NaN] crow_distance_km == 0 with differing coords: {n} rows")
        if n:
            print(df.loc[bad_crow, ['trip_id','seg_id','t0_lat','t0_lon','dest_lat','dest_lon','crow_distance_km']].head(10))
            df.loc[bad_crow, 'crow_distance_km'] = np.nan

    # Bearing normalization & invalid -> NaN
    if 'dest_bearing_deg' in df.columns:
        # Normalize to [0,360)
        df['dest_bearing_deg'] = np.mod(df['dest_bearing_deg'], 360.0)
        df.loc[df['dest_bearing_deg'] >= 360.0, 'dest_bearing_deg'] = 0.0
        bad_brg = ~df['dest_bearing_deg'].between(0, 360-1e-9)
        n = int(bad_brg.sum())
        print(f"[CHECK->NaN] bearing outside [0,360): {n} rows")
        if n:
            print(df.loc[bad_brg, ['trip_id','seg_id','dest_bearing_deg']].head(10))
            df.loc[bad_brg, 'dest_bearing_deg'] = np.nan

    # Lat/Lon range checks -> NaN
    for lat, lon, which in [('t0_lat','t0_lon','t0'), ('dest_lat','dest_lon','dest')]:
        if {lat, lon} <= set(df.columns):
            bad = ~df[lat].between(-90,90) | ~df[lon].between(-180,180)
            n = int(bad.sum())
            print(f"[CHECK->NaN] {which} lat/lon out of range: {n} rows")
            if n:
                print(df.loc[bad, ['trip_id','seg_id', lat, lon]].head(10))
                df.loc[bad, [lat, lon]] = np.nan

    return df


def _drop_full_nan_columns(df: pd.DataFrame) -> pd.DataFrame:
    full_nan_cols: List[str] = [c for c in df.columns if df[c].isna().all()]
    if full_nan_cols:
        print(f"Dropping fully-NaN columns: {full_nan_cols}")
        df = df.drop(columns=full_nan_cols)
    return df


def _print_overview(df: pd.DataFrame, title: str):
    print(f"\n=== {title} ===")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} cols")
    print("\nColumns:")
    print(", ".join(df.columns))
    print("\nDTypes:")
    print(df.dtypes.sort_index())
    print("\nHead:")
    print(df.head(5))
    print("\nTop-20 columns by missingness:")
    miss = df.isna().mean().sort_values(ascending=False).head(20)
    print((miss * 100).round(2).astype(str) + "%")


def main():
    # ---------- Load ----------
    df = pd.read_parquet(INPUT_PATH)
    _print_overview(df, "Loaded t0 nonbehavioral features (raw)")

    # ---------- DEM fill (t0 & dest) ----------
    if 't0_elev_m' in df.columns and 't0_lat' in df.columns and 't0_lon' in df.columns:
        filled_start = _batch_fill_dem(df, 't0_lat', 't0_lon', 't0_elev_m')
    else:
        filled_start = 0
    if 'dest_elev_m' in df.columns and 'dest_lat' in df.columns and 'dest_lon' in df.columns:
        filled_dest = _batch_fill_dem(df, 'dest_lat', 'dest_lon', 'dest_elev_m')
    else:
        filled_dest = 0
    print(f"DEM filled -> start: +{filled_start}, dest: +{filled_dest}")

    # Elevation delta
    if {'t0_elev_m','dest_elev_m'} <= set(df.columns):
        df['elev_delta_m'] = df['dest_elev_m'] - df['t0_elev_m']

    # ---------- Raw outlier cleaning ----------
    df = _apply_raw_outlier_rules(df)

    # ---------- Sanity checks -> NaN offenders ----------
    df = _apply_sanity_cleaning(df)

    # ---------- Cyclic encodings (keep ints too) ----------
    df = _add_cyclic(df)

    # ---------- Drop fully-NaN columns ----------
    df = _drop_full_nan_columns(df)

    _print_overview(df, "After curation")

    # ---------- Save ----------
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved curated parquet -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

