# segment_behavior_prior_aggs.py
# -------------------------------------------------------------
# Purpose
#   Build leakage-safe, segment-level **prior behavioral aggregations**
#   to be used as Model 2 features (behavioral priors + non-behavioral-at-start).
#
#   For a target segment x of a trip, every feature here is computed **only**
#   from segments 1..x-1 **of the same trip** (strictly no future leakage).
#   Aggregations are derived from the *pure behavioral* kinematic channels
#   used for the driver-embedding model.
#
# Usage
#   - Adjust the PATHS block to your environment.
#   - Run:  python segment_behavior_prior_aggs.py
#   - Output: parquet + pickle with [trip_id, seg_id] and prior-agg columns.
#
# Assumptions
#   - The input is the *corrected* windows file (already cleaned + gated).
#   - We do not re-run cleaning/gating here.
# -------------------------------------------------------------

from __future__ import annotations
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Project helpers
from HelperFuncs import load_file, save_file
# Use the project's analyzer for column-by-column review
from driver_emb_data_preparation import analyze_scalar_feature as ANALYZE_FN

# =============================================================
# PATHS / CONFIG
# =============================================================
WINDOWS_CORRECTED_PATH: str = (
    r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\windows_corrected.pickle"
)
OUTPUT_DIR: str = (
    r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segment_behavior_agg"
)
OUTPUT_BASENAME: str = "seg_behavior_prior_aggs"

# Keys
TRIP_ID_COL: str = "trip_id"
SEG_ID_COL: str = "seg_id"

# Numeric config
NONZERO_EPS: float = 1e-6
SEGMENT_QUANTILES: Tuple[float, float, float] = (0.10, 0.50, 0.90)

# Behavioral channels (pure) used for driver embeddings
BEHAVIOR_CHANNELS: List[str] = [
    "jerk_mps3",
    "grade_corrected_accel",
    "yaw_rate_dps",
    "yaw_accel_dps2",
    "lat_accel_signed_mps2",
    "centripetal_jerk_mps3",
    "speed_limit_ratio",
]

# Thresholds for exceedance-based priors. mode="abs" uses |x|>thr; mode="pos" uses x>thr
AGG_THRESHOLD_MAP: Dict[str, Dict[str, Union[str, List[float]]]] = {
    "jerk_mps3": {"mode": "abs", "thr": [0.5, 1.0]},
    "grade_corrected_accel": {"mode": "abs", "thr": [1.0, 1.5]},
    "yaw_rate_dps": {"mode": "abs", "thr": [30.0, 60.0]},
    "yaw_accel_dps2": {"mode": "abs", "thr": [50.0, 100.0]},
    "lat_accel_signed_mps2": {"mode": "abs", "thr": [1.5, 3.0]},
    "centripetal_jerk_mps3": {"mode": "abs", "thr": [0.5, 1.0]},
    "speed_limit_ratio": {"mode": "pos", "thr": [1.05, 1.10]},
}

# =============================================================
# LOAD
# =============================================================

def load_windows() -> pd.DataFrame:
    """Load the **corrected** windows file. Error if missing or malformed."""
    path = WINDOWS_CORRECTED_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Corrected windows file not found. Expected at:"
            f"- {path}"
            "Please generate it first (post-cleaning & gating)."
        )
    print(f"Loading corrected windows from: {path}")
    df = load_file(path)

    # Required keys
    for col in (TRIP_ID_COL, SEG_ID_COL):
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in corrected windows data.")
    return df

# =============================================================
# SEGMENT-LEVEL SUMMARIES (per channel)
# =============================================================

def _flatten_channel_arrays(values: pd.Series) -> np.ndarray:
    """Concatenate array-like entries (skip None), cast to float64."""
    arrays: List[np.ndarray] = []
    for v in values.values:
        if isinstance(v, np.ndarray):
            arrays.append(v)
        elif isinstance(v, (list, tuple)):
            arrays.append(np.asarray(v))
    if not arrays:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(arrays).astype(np.float64, copy=False)


def _segment_channel_summaries(group: pd.DataFrame, channel: str) -> Dict[str, float]:
    """Per-(trip_id, seg_id) stats for one channel.
    Returns: sum, sumsq, count, nonzero_count, min, max, quantiles, exceedances.
    """
    arr = _flatten_channel_arrays(group[channel]) if channel in group.columns else np.empty(0)
    if arr.size == 0:
        res = {
            f"{channel}__sum": 0.0,
            f"{channel}__sumsq": 0.0,
            f"{channel}__count": 0.0,
            f"{channel}__nonzero": 0.0,
            f"{channel}__min": np.nan,
            f"{channel}__max": np.nan,
        }
        for q in SEGMENT_QUANTILES:
            res[f"{channel}__q{int(q*100):02d}"] = np.nan
        thr_cfg = AGG_THRESHOLD_MAP.get(channel)
        if thr_cfg is not None:
            mode = thr_cfg.get("mode", "pos")
            for thr in thr_cfg.get("thr", []):
                res[f"{channel}__exceed_{mode}_{thr}"] = 0.0
        return res

    finite = np.isfinite(arr)
    if not finite.any():
        res = {
            f"{channel}__sum": 0.0,
            f"{channel}__sumsq": 0.0,
            f"{channel}__count": 0.0,
            f"{channel}__nonzero": 0.0,
            f"{channel}__min": np.nan,
            f"{channel}__max": np.nan,
        }
        for q in SEGMENT_QUANTILES:
            res[f"{channel}__q{int(q*100):02d}"] = np.nan
        thr_cfg = AGG_THRESHOLD_MAP.get(channel)
        if thr_cfg is not None:
            mode = thr_cfg.get("mode", "pos")
            for thr in thr_cfg.get("thr", []):
                res[f"{channel}__exceed_{mode}_{thr}"] = 0.0
        return res

    arr = arr[finite]
    count = float(arr.size)
    s = float(np.sum(arr))
    ss = float(np.sum(arr * arr))
    nonzero = float(np.sum(np.abs(arr) > NONZERO_EPS))
    mn = float(np.min(arr))
    mx = float(np.max(arr))

    res = {
        f"{channel}__sum": s,
        f"{channel}__sumsq": ss,
        f"{channel}__count": count,
        f"{channel}__nonzero": nonzero,
        f"{channel}__min": mn,
        f"{channel}__max": mx,
    }
    for q in SEGMENT_QUANTILES:
        res[f"{channel}__q{int(q*100):02d}"] = float(np.nanquantile(arr, q))
    thr_cfg = AGG_THRESHOLD_MAP.get(channel)
    if thr_cfg is not None:
        mode = thr_cfg.get("mode", "pos")
        thr_list = thr_cfg.get("thr", [])
        if mode == "abs":
            base = np.abs(arr)
            for thr in thr_list:
                res[f"{channel}__exceed_abs_{thr}"] = float(np.sum(base > float(thr)))
        else:  # "pos"
            for thr in thr_list:
                res[f"{channel}__exceed_pos_{thr}"] = float(np.sum(arr > float(thr)))
    return res


def build_segment_summaries(windows_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate arrays within each (trip_id, seg_id) to segment-level
    summary accumulators per behavioral channel.
    """
    cols_present = [c for c in BEHAVIOR_CHANNELS if c in windows_df.columns]
    if not cols_present:
        raise KeyError("None of the behavioral channels were found in windows data.")

    g = windows_df.groupby([TRIP_ID_COL, SEG_ID_COL], sort=True, as_index=False)
    rows: List[Dict[str, float]] = []
    for (trip_id, seg_id), group in tqdm(g, desc="Per-segment summaries"):
        row: Dict[str, float] = {TRIP_ID_COL: trip_id, SEG_ID_COL: seg_id}
        for ch in cols_present:
            row.update(_segment_channel_summaries(group, ch))
        rows.append(row)

    seg_df = pd.DataFrame(rows)

    # Cast numeric cols to float32 for compactness
    for c in seg_df.columns:
        if c in (TRIP_ID_COL, SEG_ID_COL):
            continue
        if seg_df[c].dtype.kind in {"i", "u"}:
            seg_df[c] = seg_df[c].astype(np.float32)
        elif seg_df[c].dtype.kind == "f":
            seg_df[c] = seg_df[c].astype(np.float32)

    seg_df.sort_values([TRIP_ID_COL, SEG_ID_COL], inplace=True)
    seg_df.reset_index(drop=True, inplace=True)
    return seg_df

# =============================================================
# PRIOR (cumulative, up-to-previous) AGGREGATIONS
# =============================================================

def _expand_prior_features(seg_df: pd.DataFrame) -> pd.DataFrame:
    """Within each trip, compute cumulative (expanding) *prior* features
    for each channel, excluding the current segment via shift(1).

    Adds, per channel:
      - prior_mean_, prior_std_, prior_rms_, prior_min_, prior_max_
      - prior_nonzero_frac_, prior_sample_count_
      - prior_mean_q10/q50/q90_
      - prior_exceed_*_thr_ and _rate counterparts
    Also adds prior_seg_count (segments completed before current).
    """
    seg_df = seg_df.copy()
    seg_df.sort_values([TRIP_ID_COL, SEG_ID_COL], inplace=True)
    seg_df.reset_index(drop=True, inplace=True)

    base = seg_df[[TRIP_ID_COL, SEG_ID_COL]].copy().reset_index(drop=True)
    prior_df = base.copy()

    # prior_seg_count: number of completed segments before current
    prior_df["prior_seg_count"] = seg_df.groupby(TRIP_ID_COL).cumcount().astype("Int32")

    grp = seg_df.groupby(TRIP_ID_COL, group_keys=False)

    for ch in BEHAVIOR_CHANNELS:
        needed = [f"{ch}__sum", f"{ch}__sumsq", f"{ch}__count", f"{ch}__nonzero", f"{ch}__min", f"{ch}__max"]
        if not all(c in seg_df.columns for c in needed):
            continue

        # shift(1) ensures we never include current segment x in its own priors
        prior_sum = grp[f"{ch}__sum"].apply(lambda s: s.shift(1).expanding().sum())
        prior_sumsq = grp[f"{ch}__sumsq"].apply(lambda s: s.shift(1).expanding().sum())
        prior_count = grp[f"{ch}__count"].apply(lambda s: s.shift(1).expanding().sum())
        prior_nonzero = grp[f"{ch}__nonzero"].apply(lambda s: s.shift(1).expanding().sum())
        prior_min = grp[f"{ch}__min"].apply(lambda s: s.shift(1).expanding().min())
        prior_max = grp[f"{ch}__max"].apply(lambda s: s.shift(1).expanding().max())

        with np.errstate(invalid="ignore", divide="ignore"):
            prior_mean = prior_sum / prior_count
            prior_rms = np.sqrt(prior_sumsq / prior_count)
            prior_var = (prior_sumsq / prior_count) - (prior_mean ** 2)
            prior_var = prior_var.clip(lower=0)
            prior_std = np.sqrt(prior_var)
            prior_nonzero_frac = prior_nonzero / prior_count

        prior_df[f"prior_mean_{ch}"] = prior_mean.values.astype(np.float32)
        prior_df[f"prior_std_{ch}"] = prior_std.values.astype(np.float32)
        prior_df[f"prior_rms_{ch}"] = prior_rms.values.astype(np.float32)
        prior_df[f"prior_min_{ch}"] = prior_min.values.astype(np.float32)
        prior_df[f"prior_max_{ch}"] = prior_max.values.astype(np.float32)
        prior_df[f"prior_nonzero_frac_{ch}"] = prior_nonzero_frac.values.astype(np.float32)
        prior_df[f"prior_sample_count_{ch}"] = prior_count.values.astype(np.float32)

        # Means of prior segment quantiles
        for q in SEGMENT_QUANTILES:
            col = f"{ch}__q{int(q*100):02d}"
            if col in seg_df.columns:
                prior_q_mean = grp[col].apply(lambda s: s.shift(1).expanding().mean())
                label = {0.10: "q10", 0.50: "q50", 0.90: "q90"}[q]
                prior_df[f"prior_mean_{label}_{ch}"] = prior_q_mean.values.astype(np.float32)

        # Exceedance counts & rates
        thr_cfg = AGG_THRESHOLD_MAP.get(ch)
        if thr_cfg is not None:
            mode = thr_cfg.get("mode", "pos")
            for thr in thr_cfg.get("thr", []):
                seg_col = f"{ch}__exceed_{mode}_{thr}"
                if seg_col in seg_df.columns:
                    prior_exceed = grp[seg_col].apply(lambda s: s.shift(1).expanding().sum())
                    prior_df[f"prior_exceed_{mode}_{thr}_{ch}"] = prior_exceed.values.astype(np.float32)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        rate = prior_exceed / prior_count
                    prior_df[f"prior_exceed_rate_{mode}_{thr}_{ch}"] = rate.values.astype(np.float32)

    return prior_df

# =============================================================
# SANITY (no-leakage invariants) + REVIEW (analyze all columns)
# =============================================================

def sanity_and_review(prior_df: pd.DataFrame, seg_df: pd.DataFrame) -> None:
    """Checks strict no-leakage invariants + runs column analyzer on ALL columns."""
    print("=== Sanity: shapes & keys ===")
    n_trips = prior_df[TRIP_ID_COL].nunique()
    n_segs = len(prior_df)
    mem_mb = prior_df.memory_usage(deep=True).sum() / (1024**2)
    print(f"Trips: {n_trips} | Segments: {n_segs} | Memory: {mem_mb:.2f} MB")

    # Unique (trip_id, seg_id)
    dups = prior_df.duplicated([TRIP_ID_COL, SEG_ID_COL]).sum()
    if dups:
        raise AssertionError(f"Found {dups} duplicate (trip_id, seg_id) keys in prior_df")

    # Monotonic seg order per trip
    for tid, s in prior_df.sort_values([TRIP_ID_COL, SEG_ID_COL]).groupby(TRIP_ID_COL)[SEG_ID_COL]:
        if not s.is_monotonic_increasing:
            raise AssertionError(f"Non-monotonic seg_id order in trip {tid}")

    # Leak check: prior_sample_count equals shifted expanding sum of per-segment counts
    for ch in BEHAVIOR_CHANNELS:
        cnt_col = f"{ch}__count"
        prior_cnt_col = f"prior_sample_count_{ch}"
        if cnt_col in seg_df.columns and prior_cnt_col in prior_df.columns:
            expected = (
                seg_df.sort_values([TRIP_ID_COL, SEG_ID_COL])
                      .groupby(TRIP_ID_COL)[cnt_col]
                      .apply(lambda s: s.shift(1).expanding().sum())
            ).values
            have = prior_df[prior_cnt_col].values
            mask = np.isfinite(expected) & np.isfinite(have)
            if mask.any():
                max_abs_err = float(np.max(np.abs(expected[mask] - have[mask])))
                if max_abs_err > 1e-3:
                    raise AssertionError(
                        f"Prior sample-count misalignment for '{ch}': max_abs_err={max_abs_err:.4f}"
                    )

    # Algebraic sanity: RMS, STD, MEAN relation; bounds on rates
    for ch in BEHAVIOR_CHANNELS:
        mean_c = f"prior_mean_{ch}"
        std_c = f"prior_std_{ch}"
        rms_c = f"prior_rms_{ch}"
        min_c = f"prior_min_{ch}"
        max_c = f"prior_max_{ch}"
        cnt_c = f"prior_sample_count_{ch}"
        if all(c in prior_df.columns for c in [mean_c, std_c, rms_c, min_c, max_c, cnt_c]):
            with np.errstate(invalid="ignore"):
                if ((prior_df[rms_c] + 1e-6) < np.abs(prior_df[mean_c])).any():
                    raise AssertionError(f"RMS < |mean| detected for '{ch}'")
                lhs = (prior_df[std_c] ** 2)
                rhs = (prior_df[rms_c] ** 2) - (prior_df[mean_c] ** 2)
                if ((lhs - rhs).abs().dropna() > 1e-2).any():
                    raise AssertionError(f"Std/RMS/Mean identity off for '{ch}' beyond tolerance")
            mm = prior_df[[min_c, max_c]].dropna()
            if (mm[min_c] > mm[max_c]).any():
                raise AssertionError(f"min>max encountered for '{ch}'")
        thr_cfg = AGG_THRESHOLD_MAP.get(ch)
        if thr_cfg is not None:
            mode = thr_cfg.get("mode", "pos")
            for thr in thr_cfg.get("thr", []):
                cnt_name = f"prior_exceed_{mode}_{thr}_{ch}"
                rate_name = f"prior_exceed_rate_{mode}_{thr}_{ch}"
                if cnt_name in prior_df.columns and cnt_c in prior_df.columns:
                    counts = prior_df[cnt_name]
                    samples = prior_df[cnt_c]
                    finite = np.isfinite(counts) & np.isfinite(samples)
                    if (counts[finite] - samples[finite] > 1e-3).any():
                        raise AssertionError(f"Exceedance counts > samples for '{ch}', thr={thr}")
                if rate_name in prior_df.columns:
                    rates = prior_df[rate_name].dropna()
                    if ((rates < -1e-6) | (rates > 1 + 1e-6)).any():
                        raise AssertionError(f"Exceedance rates outside [0,1] for '{ch}', thr={thr}")

    # Analyze ALL columns using your analyzer (more informative than a head())
    print("=== Detailed column review (analyze_scalar_feature on ALL columns) ===")
    for col in prior_df.columns:
        ANALYZE_FN(prior_df[col], col)

# =============================================================
# MAIN
# =============================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    windows_df = load_windows()

    # Keep only needed columns (IDs + behavioral arrays)
    keep_cols: List[str] = [TRIP_ID_COL, SEG_ID_COL] + [c for c in BEHAVIOR_CHANNELS if c in windows_df.columns]
    missing = [c for c in BEHAVIOR_CHANNELS if c not in windows_df.columns]
    if missing:
        print(f"WARNING: missing behavioral channels in corrected windows: {missing}")
    windows_df = windows_df[keep_cols].copy()

    # Build
    print("Building prior (up-to-previous) aggregations …")
    seg_df = build_segment_summaries(windows_df)
    prior_df = _expand_prior_features(seg_df)

    # Sanity + full-column review (no leakage, correctness)
    sanity_and_review(prior_df, seg_df)

    # Save outputs (pickle + parquet, CSV fallback)
    print("Saving outputs …")
    save_file(prior_df, OUTPUT_DIR, OUTPUT_BASENAME, format="pickle")

    parquet_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}.parquet")
    try:
        prior_df.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"Parquet save failed (non-fatal): {e}")
        print("Attempting to save CSV instead …")
        csv_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}.csv")
        try:
            prior_df.to_csv(csv_path, index=False)
        except Exception as e2:
            print(f"CSV save also failed: {e2}")


if __name__ == "__main__":
    main()
