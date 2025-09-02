# build_context_vectors_from_pickle.py
# Loads windows_corrected.pickle (list-of-dicts or DataFrame),
# builds EXOGENOUS context vectors for hard-negative mining, and
# saves artifacts under OUTPUT_DIR\context_index

import os, json, math, pickle
import numpy as np
import pandas as pd

# ---------- YOUR PATHS ----------
INPUT_PATH  = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\windows_corrected.pickle"
OUTPUT_DIR  = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data"
OUT_SUBFOLDER = "context_index"  # change if you want a different folder name
ARTIFACTS_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\embedding_artifacts.pickle"
SPLIT = "train" 

# ---------- FEATURE CONFIG ----------
CAT_FEATURES = [
    "win_road_type",       # one-hot
    "win_junction_type",   # one-hot
]

NUM_FEATURES = [
    # Road / topology (high weight)
    "win_avg_speed_limit_kph",
    "win_oneway_pct",
    "win_median_lanes",
    "win_cum_bearing_change_deg",
    "win_elevation_change_m",
    "win_avg_grade",
    "win_bridge_pct",
    "win_tunnel_pct",
    "win_stop_sign_pct",
    "win_traffic_signal_pct",

    # Weather & light (medium weight)
    "win_weather_temp_c",
    "win_weather_humidity_pct",
    "win_weather_pressure_hpa",
    "win_weather_precip_mm",
    "win_weather_snowfall_mm",
    "win_weather_windspeed_kph",
    "win_weather_windgust_kph",
    "win_avg_crosswind_mps",   # will abs()
    "win_max_crosswind_mps",   # will abs()
    "win_avg_headwind_mps",    # will abs()
    "win_max_headwind_mps",    # will abs()
    "win_weather_cloudcover_pct",
    "win_solar_radiation",

    # Time / seasonality (low–medium)
    "start_hour_sin", "start_hour_cos",
    "win_is_day",
]

ONEHOT_FEATURES = [
    "day_of_week",   # one-hot
    # note: month is encoded cyclic (sin/cos) below
]

# Prefilter/bucketing only (NOT used in distance metric) – saved in META
META_COLS = [
    "trip_id", "seg_id", "win_id",
    "car_model", "battery_capacity_kWh", "empty_weight_kg",
    "battery_type", "seg_mean_batt_health"
]

# Weights
WEIGHTS = {
    # Road / topology (high)
    "win_avg_speed_limit_kph": 2.0,
    "win_oneway_pct": 1.5,
    "win_median_lanes": 1.5,
    "win_cum_bearing_change_deg": 1.5,
    "win_elevation_change_m": 1.5,
    "win_avg_grade": 1.5,
    "win_bridge_pct": 0.75,
    "win_tunnel_pct": 0.75,
    "win_stop_sign_pct": 0.75,
    "win_traffic_signal_pct": 0.75,
    "__cat__win_road_type": 2.0,
    "__cat__win_junction_type": 1.0,

    # Weather & light (medium)
    "win_weather_temp_c": 1.0,
    "win_weather_humidity_pct": 1.0,
    "win_weather_pressure_hpa": 0.75,
    "win_weather_precip_mm": 1.0,
    "win_weather_snowfall_mm": 1.0,
    "win_weather_windspeed_kph": 1.0,
    "win_weather_windgust_kph": 1.0,
    "win_avg_crosswind_mps": 1.0,
    "win_max_crosswind_mps": 1.0,
    "win_avg_headwind_mps": 1.0,
    "win_max_headwind_mps": 1.0,
    "win_weather_cloudcover_pct": 0.75,
    "win_solar_radiation": 0.75,

    # Time / seasonality (low–medium)
    "start_hour_sin": 0.5,
    "start_hour_cos": 0.5,
    "win_is_day": 0.5,
    "__cat__day_of_week": 0.5,
    "month_sin": 0.5,
    "month_cos": 0.5,
}

# ---------- HELPERS ----------
def safe_read_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj
    else:
        # Some checkpoints store a dict with 'rows' or similar – handle generically
        try:
            return pd.DataFrame(obj)
        except Exception as e:
            raise ValueError(f"Unsupported pickle content type: {type(obj)}") from e

def robust_scale(col: pd.Series):
    # winsorize to curb tails
    lo, hi = col.quantile([0.01, 0.99])
    clipped = col.clip(lo, hi)

    q1, med, q3 = clipped.quantile([0.25, 0.5, 0.75])
    iqr = float(q3 - q1)

    if iqr >= 1e-3:
        scaled = (clipped - med) / iqr
        stats = {"center": float(med), "scale": iqr, "method": "iqr", "lo": float(lo), "hi": float(hi)}
    else:
        std = float(clipped.std(ddof=0))
        if std >= 1e-6:
            mean = float(clipped.mean())
            scaled = (clipped - mean) / std
            stats = {"center": mean, "scale": std, "method": "std", "lo": float(lo), "hi": float(hi)}
        else:
            scaled = pd.Series(0.0, index=col.index)
            stats = {"center": 0.0, "scale": 1.0, "method": "const", "lo": float(lo), "hi": float(hi)}
    return scaled.astype("float32"), stats

def month_to_cyc(m: pd.Series):
    m = m.astype("Int64").clip(1, 12).fillna(1)
    angle = 2 * math.pi * (m - 1) / 12.0
    return np.sin(angle).astype("float32"), np.cos(angle).astype("float32")

# ---------- MAIN ----------
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input file not found at: {INPUT_PATH}")
        return

    # Load pickle (list-of-dicts or DataFrame)
    print(f"Loading data from: {INPUT_PATH}")
    try:
        df = safe_read_pickle(INPUT_PATH)
        print(f"Successfully loaded {len(df):,} windows. Columns: {len(df.columns)}")
    except Exception as e:
        print(f"ERROR: Failed to load or process the pickle file. {e}")
        return

    # ---- filter to the chosen split (train) using trip_ids from artifacts ----
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)

    if SPLIT not in artifacts or "windows" not in artifacts[SPLIT]:
        raise KeyError(f"Split '{SPLIT}' not found in artifacts or missing 'windows'.")

    train_trip_ids = set(artifacts[SPLIT]["windows"]["trip_id"].unique())
    before = len(df)
    df = df[df["trip_id"].isin(train_trip_ids)].reset_index(drop=True)
    after = len(df)
    print(f"[context] restricting to split='{SPLIT}': {after:,}/{before:,} windows kept")

    outdir = os.path.join(OUTPUT_DIR, OUT_SUBFOLDER)
    os.makedirs(outdir, exist_ok=True)

    # Drop explicit exclusions
    drop_if_present = ["avg_outside_temp", "win_weather_rain_mm", "win_sunshine_duration_s",
                       "win_lit_pct", "win_surface_type", "win_smoothness_type", "win_intersection_count"]
    to_drop = [c for c in drop_if_present if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    # Absolute winds (use magnitudes)
    for c in ["win_avg_crosswind_mps","win_max_crosswind_mps","win_avg_headwind_mps","win_max_headwind_mps"]:
        if c in df.columns:
            df[c] = df[c].abs()

    # Month cyclic features
    if "month" in df.columns:
        s, c = month_to_cyc(df["month"])
        df["month_sin"] = s
        df["month_cos"] = c

    # Build feature lists actually present
    cat_feats  = [c for c in CAT_FEATURES if c in df.columns]
    num_feats  = [c for c in NUM_FEATURES if c in df.columns]
    onehot_ts  = [c for c in ONEHOT_FEATURES if c in df.columns]
    if "month_sin" in df.columns and "month_cos" in df.columns:
        num_feats += ["month_sin","month_cos"]
    meta_feats = [c for c in META_COLS if c in df.columns]

    # Missingness flags
    miss_flags = []
    for c in num_feats + onehot_ts + cat_feats:
        if c in df.columns and df[c].isna().any():
            flag = f"{c}__isna"
            df[flag] = df[c].isna().astype("int8")
            miss_flags.append(flag)

    # Impute numerics with median
    num_stats = {}
    for c in num_feats:
        med = float(df[c].median()) if c in df.columns else 0.0
        df[c] = df[c].fillna(med)
        num_stats[c] = {"median_impute": med}

    # One-hot time (day_of_week)
    oh_cols = []
    for c in onehot_ts:
        dummies = pd.get_dummies(df[c].astype("Int64"), prefix=c, dummy_na=False)
        df = pd.concat([df, dummies], axis=1)
        oh_cols.extend(dummies.columns.tolist())

    # One-hot road/junction
    cat_cols = []
    for c in cat_feats:
        dummies = pd.get_dummies(df[c].astype("category"), prefix=c, dummy_na=False)
        df = pd.concat([df, dummies], axis=1)
        cat_cols.extend(dummies.columns.tolist())

    # Robust-scale numerics
    scaled_num_cols, scaling_params = [], {}
    for c in num_feats:
        scaled, stats = robust_scale(df[c].astype(float))
        sc = f"{c}__rs"
        df[sc] = scaled
        scaled_num_cols.append(sc)
        scaling_params[c] = stats

    # Assemble final matrix
    final_cols, final_w = [], {}

    # numerics
    for base in num_feats:
        col = f"{base}__rs"
        if col in df.columns:
            final_cols.append(col)
            final_w[col] = WEIGHTS.get(base, 1.0)

    # time one-hot
    for col in oh_cols:
        final_cols.append(col)
        if col.startswith("day_of_week_"):
            final_w[col] = WEIGHTS.get("__cat__day_of_week", 0.5)
        else:
            final_w[col] = 0.5

    # road/junction one-hot
    for col in cat_cols:
        final_cols.append(col)
        if col.startswith("win_road_type_"):
            final_w[col] = WEIGHTS.get("__cat__win_road_type", 2.0)
        elif col.startswith("win_junction_type_"):
            final_w[col] = WEIGHTS.get("__cat__win_junction_type", 1.0)
        else:
            final_w[col] = 1.0

    # missingness flags (low weight)
    for col in miss_flags:
        final_cols.append(col)
        final_w[col] = 0.25

    # Guardrails: prevent behavior leakage (exact feature names only)
    behavior_forbidden = {
        "speed_mps",
        "speed_rel_limit_mps",
        "speed_limit_ratio",
        "long_accel_mps2",
        "lat_accel_signed_mps2",
        "centripetal_accel_mps2",
        "centripetal_jerk_mps3",
        "jerk_mps3",
        "yaw_rate_dps",
        "yaw_accel_dps2",
        "propulsive_accel_mps2",
        "propulsive_power_kw",
        "propulsive_power_kw_per_kg",
        "vert_v_mps",
    }

    def _base_name(col: str) -> str:
        # strip scaler/missing flags
        for suf in ("__rs", "__isna"):
            if col.endswith(suf):
                col = col[: -len(suf)]
        return col

    leaked = []
    for c in final_cols:
        # skip legit one-hots
        if c.startswith(("win_road_type_", "win_junction_type_", "day_of_week_")):
            continue
        base = _base_name(c)
        if base in behavior_forbidden:
            leaked.append(c)

    assert not leaked, f"Forbidden feature(s) leaked into context: {leaked[:5]} ..."

    # Build matrix and weight it
    X = df[final_cols].to_numpy(dtype=np.float32, copy=True)
    w = np.array([final_w[c] for c in final_cols], dtype=np.float32)
    X *= w

    # Save artifacts
    np.savez_compressed(os.path.join(outdir, "context_vectors.npz"), X=X)

    meta = df[meta_feats].copy() if meta_feats else pd.DataFrame(index=df.index)
    # Clean + bin empty_weight_kg if present
    if "empty_weight_kg" in meta.columns:
        meta["empty_weight_kg_num"] = pd.to_numeric(meta["empty_weight_kg"], errors="coerce")
        bins   = [-np.inf, 1500, 1700, 1900, 2100, np.inf]
        labels = ["<1500","1500-1699","1700-1899","1900-2099","2100+"]
        meta["empty_weight_kg_bin"] = pd.cut(meta["empty_weight_kg_num"], bins=bins, labels=labels)

    # Prefer Parquet; fall back to CSV if pyarrow/fastparquet missing
    meta_path_parq = os.path.join(outdir, "context_meta.parquet")
    try:
        meta.to_parquet(meta_path_parq, index=False)
        meta_saved_as = meta_path_parq
    except Exception:
        meta_path_csv = os.path.join(outdir, "context_meta.csv")
        meta.to_csv(meta_path_csv, index=False)
        meta_saved_as = meta_path_csv

    config = {
        "final_cols": final_cols,
        "weights": {c: float(final_w[c]) for c in final_cols},
        "num_features": num_feats,
        "scaling_params": scaling_params,
        "num_impute": num_stats,
        "cat_features": cat_feats,
        "onehot_bases": onehot_ts,
        "oh_cols_time": oh_cols,
        "oh_cols_cats": cat_cols,
        "missing_flags": miss_flags,
        "version": 3,
        "notes": "Exogenous context only; road/topology high weight, weather medium, time low–medium; month cyclic; absolute winds; vehicle fields in META only."
    }
    with open(os.path.join(outdir, "context_vectorizer.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[context] X shape: {X.shape} | nnz: {np.count_nonzero(X):,}")
    print(f"[context] features: {len(final_cols)} | saved in: {outdir}")
    print(f"[context] meta saved to: {meta_saved_as}")

if __name__ == "__main__":
    main()
