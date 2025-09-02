# driver_emb_build_neighbor_cache.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# --- paths (match your run) ---
BASE_DIR = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data"
CTX_DIR  = os.path.join(BASE_DIR, "context_index")
PICKLE   = os.path.join(BASE_DIR, "windows_corrected.pickle")
ARTIFACTS_PATH = os.path.join(BASE_DIR, "embedding_artifacts.pickle")
SPLIT = "train"  # keep consistent with context build

K = 50          # requested neighbors per point
EXTRA = 5       # small over-fetch for bucket passes
BUCKET_OVERFETCH_CAP = 512  # cap per-bucket kneighbors to keep things fast

# ----------------- utils -----------------
def l2_normalize(X: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return X / nrm

def load_pickle_df(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, list):
        if len(obj) == 0:
            return pd.DataFrame()
        if isinstance(obj[0], dict):
            return pd.DataFrame(obj)
    raise ValueError(f"Unsupported pickle content type: {type(obj)}")

def try_read_meta(ctx_dir: str) -> pd.DataFrame:
    pq = os.path.join(ctx_dir, "context_meta.parquet")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    csv = os.path.join(ctx_dir, "context_meta.csv")
    if os.path.exists(csv):
        return pd.read_csv(csv)
    raise FileNotFoundError("context_meta.parquet/csv not found in context_index")

def try_read_X(ctx_dir: str) -> np.ndarray:
    npz = os.path.join(ctx_dir, "context_vectors.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError("context_vectors.npz not found in context_index")
    data = np.load(npz)
    if "X" not in data:
        raise KeyError("context_vectors.npz missing array 'X'")
    X = data["X"]
    if not isinstance(X, np.ndarray):
        raise TypeError("Loaded X is not a numpy array")
    return X.astype(np.float32, copy=False)

def speed_limit_bin(v: float) -> str:
    try:
        x = float(v)
    except Exception:
        return "unknown"
    if np.isnan(x):
        return "unknown"
    if x < 40:   return "<40"
    if x < 60:   return "40-59"
    if x < 80:   return "60-79"
    if x < 100:  return "80-99"
    if x < 120:  return "100-119"
    return "120+"

def ensure_bins(meta: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()

    if "speed_limit_bin" not in meta.columns:
        src = meta if "win_avg_speed_limit_kph" in meta.columns else df_raw
        if "win_avg_speed_limit_kph" in src.columns and len(src) == len(meta):
            vals = src["win_avg_speed_limit_kph"].values
            meta["speed_limit_bin"] = pd.Series(vals, index=meta.index).apply(speed_limit_bin).astype(str)
        else:
            meta["speed_limit_bin"] = "unknown"

    if "empty_weight_kg_bin" not in meta.columns:
        if "empty_weight_kg" in meta.columns:
            ew = pd.to_numeric(meta["empty_weight_kg"], errors="coerce")
            bins   = [-np.inf, 1500, 1700, 1900, 2100, np.inf]
            labels = ["<1500","1500-1699","1700-1899","1900-2099","2100+"]
            meta["empty_weight_kg_bin"] = pd.cut(ew, bins=bins, labels=labels, include_lowest=True).astype(str)
        else:
            meta["empty_weight_kg_bin"] = "unknown"

    for c in ("trip_id", "car_model", "speed_limit_bin", "empty_weight_kg_bin"):
        if c not in meta.columns:
            meta[c] = "unknown"

    return meta

# ----------------- core filling -----------------
def fit_and_fill_bucket(
    Xn: np.ndarray,
    idxs: np.ndarray,
    trip_ids: np.ndarray,
    K: int,
    all_inds: np.ndarray,
    all_dsts: np.ndarray,
    fill_mask: np.ndarray | None,
    metric: str = "cosine",
):
    """
    Fit a KNN model on rows Xn[idxs] and fill *free slots* in all_inds/all_dsts
    for the subset of rows indicated by fill_mask (global boolean mask).
    - Excludes self and same-trip neighbors.
    - Appends into first available (-1) slots without overwriting existing ones.
    """
    if idxs.size <= 1:
        return

    # which rows in this bucket still need neighbors?
    if fill_mask is None:
        rel_q = np.arange(idxs.size, dtype=np.int64)
    else:
        rel_q = np.flatnonzero(fill_mask[idxs])
        if rel_q.size == 0:
            return

    n_group = idxs.size
    # Adaptive over-fetch within the group to survive exclusions
    kq = min(n_group - 1, max(K + EXTRA, BUCKET_OVERFETCH_CAP))

    nbrs = NearestNeighbors(n_neighbors=kq, metric=metric, algorithm="auto")
    Xb = Xn[idxs]
    nbrs.fit(Xb)

    dists, neigh = nbrs.kneighbors(Xb[rel_q], return_distance=True)
    cand_global = idxs[neigh]              # (nq, kq)
    q_global    = idxs[rel_q]              # (nq,)

    same_self = cand_global == q_global[:, None]
    same_trip = (trip_ids[cand_global] == trip_ids[q_global][:, None])
    mask_keep = ~(same_self | same_trip)

    for row_j in range(q_global.size):
        gi = q_global[row_j]
        keep = mask_keep[row_j]
        if not np.any(keep):
            continue
        sel_g = cand_global[row_j, keep]
        sel_d = dists[row_j, keep]
        free = np.flatnonzero(all_inds[gi] < 0)
        if free.size == 0:
            continue
        take = min(free.size, sel_g.size)
        if take > 0:
            all_inds[gi, free[:take]] = sel_g[:take]
            all_dsts[gi, free[:take]] = sel_d[:take]

def main():
    # 1) Load context matrix + meta
    X = try_read_X(CTX_DIR)
    meta = try_read_meta(CTX_DIR)
    if not isinstance(meta, pd.DataFrame) or len(meta) != X.shape[0]:
        raise ValueError(f"Meta rows ({len(meta)}) must match X rows ({X.shape[0]})")
    
    # ---- guard: restrict X/meta to TRAIN trip_ids ----
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)

    train_trip_ids = set(artifacts[SPLIT]["windows"]["trip_id"].unique())
    mask = meta["trip_id"].isin(train_trip_ids).to_numpy()
    if not mask.all():
        print(f"[neighbors] restricting to split='{SPLIT}': {int(mask.sum()):,}/{len(mask):,} rows kept")
        X = X[mask]
        meta = meta.loc[mask].reset_index(drop=True)

    # 2) Load raw windows (for speed_limit if needed)
    try:
        df_raw = load_pickle_df(PICKLE)
    except Exception:
        df_raw = pd.DataFrame(index=np.arange(X.shape[0]))
    if len(df_raw) != X.shape[0]:
        df_raw = pd.DataFrame(index=np.arange(X.shape[0]))

    # 3) Prepare bucketing meta
    meta = ensure_bins(meta, df_raw)
    N = X.shape[0]
    Xn = l2_normalize(X.astype(np.float32, copy=False))
    trip_ids = meta["trip_id"].to_numpy()

    # Compute per-row maximum neighbors allowed once we exclude same-trip + self
    trip_counts = pd.Series(trip_ids).value_counts()
    same_trip_counts = trip_counts.loc[pd.Index(trip_ids)].to_numpy()
    allowed_k = np.maximum(0, N - same_trip_counts - 1).astype(int)
    target_k = np.minimum(K, allowed_k)

    capped = (target_k < K).sum()
    if capped > 0:
        print(f"[neighbors] note: {capped} rows are capped below K due to same-trip exclusions "
              f"(min target_k={target_k.min()}, max={target_k.max()}).")

    # 4) Allocate neighbor arrays (filled progressively)
    all_inds = np.full((N, K), fill_value=-1, dtype=np.int64)
    all_dsts = np.full((N, K), fill_value=np.inf, dtype=np.float32)

    # ---------- PASS 1: STRICT buckets ----------
    # (car_model, empty_weight_kg_bin, speed_limit_bin)
    strict_groups = meta.groupby(
        ["car_model", "empty_weight_kg_bin", "speed_limit_bin"],
        dropna=False,
        observed=False,  # silence future warning; keep current behavior
    ).indices
    print(f"[neighbors] strict buckets: {len(strict_groups)}")

    needs_more = np.ones(N, dtype=bool)  # initially everyone needs neighbors
    for _, idxs in strict_groups.items():
        idxs = np.asarray(idxs, dtype=np.int64)
        fit_and_fill_bucket(Xn, idxs, trip_ids, K, all_inds, all_dsts, fill_mask=needs_more, metric="cosine")

    # recompute needs_more against per-row target_k
    filled_counts = (all_inds >= 0).sum(axis=1)
    needs_more = filled_counts < target_k
    missing_after_pass1 = int(needs_more.sum())
    print(f"[neighbors] missing after pass1: {missing_after_pass1}")

    # ---------- PASS 2: LOOSE buckets ----------
    # (car_model, speed_limit_bin)
    loose_groups = meta.groupby(
        ["car_model", "speed_limit_bin"],
        dropna=False,
        observed=False,
    ).indices
    print(f"[neighbors] loose buckets: {len(loose_groups)}")
    if missing_after_pass1 > 0:
        for _, idxs in loose_groups.items():
            idxs = np.asarray(idxs, dtype=np.int64)
            fit_and_fill_bucket(Xn, idxs, trip_ids, K, all_inds, all_dsts, fill_mask=needs_more, metric="cosine")
        filled_counts = (all_inds >= 0).sum(axis=1)
        needs_more = filled_counts < target_k

    # ---------- PASS 3: GLOBAL fallback ----------
    still_missing = np.flatnonzero(needs_more)
    fallback_rows = 0
    if still_missing.size > 0:
        # Over-fetch globally enough to cover the *largest* target gap, but cap by N-1
        gaps = (target_k - filled_counts)
        gaps[gaps < 0] = 0
        max_gap = int(gaps[still_missing].max()) if still_missing.size else 0
        # Fetch generously (all remaining candidates if feasible)
        kq = min(N - 1, max(K + EXTRA + max_gap * 3, 1024))

        nbrs = NearestNeighbors(n_neighbors=kq, metric="cosine", algorithm="auto")
        nbrs.fit(Xn)
        dists, neigh = nbrs.kneighbors(Xn[still_missing], return_distance=True)
        cand_global = neigh
        q_global = still_missing

        same_self = cand_global == q_global[:, None]
        same_trip = (trip_ids[cand_global] == trip_ids[q_global][:, None])
        mask_keep = ~(same_self | same_trip)

        for row_j in range(q_global.size):
            gi = q_global[row_j]
            keep = mask_keep[row_j]
            if not np.any(keep):
                continue
            sel_g = cand_global[row_j, keep]
            sel_d = dists[row_j, keep]

            # how many do we still need for this row (respect per-row target_k)?
            need = int(target_k[gi] - (all_inds[gi] >= 0).sum())
            if need <= 0:
                continue

            free = np.flatnonzero(all_inds[gi] < 0)
            if free.size == 0:
                continue

            take = min(need, free.size, sel_g.size)
            if take > 0:
                all_inds[gi, free[:take]] = sel_g[:take]
                all_dsts[gi, free[:take]] = sel_d[:take]
                fallback_rows += 1

    print(f"[neighbors] global fallback for {fallback_rows} rows")

    # 5) Final guarantee: fill any remaining holes up to target_k using full global search if necessary
    filled_counts = (all_inds >= 0).sum(axis=1)
    need_rows = np.flatnonzero(filled_counts < target_k)
    if need_rows.size > 0:
        # Last resort: ask for *all* candidates (N-1) for the few rows still missing
        kq = min(N - 1, max(1024, int((target_k[need_rows] - filled_counts[need_rows]).max()) + K + 64))
        nbrs = NearestNeighbors(n_neighbors=kq, metric="cosine", algorithm="auto")
        nbrs.fit(Xn)
        dists, neigh = nbrs.kneighbors(Xn[need_rows], return_distance=True)
        same_self = neigh == need_rows[:, None]
        same_trip = (trip_ids[neigh] == trip_ids[need_rows][:, None])
        mask_keep = ~(same_self | same_trip)

        for row_j, gi in enumerate(need_rows):
            keep = mask_keep[row_j]
            if not np.any(keep):
                continue
            sel_g = neigh[row_j, keep]
            sel_d = dists[row_j, keep]
            # fill up to target_k[gi]
            need = int(target_k[gi] - (all_inds[gi] >= 0).sum())
            if need <= 0:
                continue
            free = np.flatnonzero(all_inds[gi] < 0)
            take = min(need, free.size, sel_g.size)
            if take > 0:
                all_inds[gi, free[:take]] = sel_g[:take]
                all_dsts[gi, free[:take]] = sel_d[:take]

    # Recompute fill counts
    filled_counts = (all_inds >= 0).sum(axis=1)

    # Assert: all rows meet their target_k (cannot exceed allowed neighbors)
    if not np.all(filled_counts >= target_k):
        min_gap = int((target_k - filled_counts).clip(lower=0).max())
        raise AssertionError(f"Some rows still below target_k (max shortfall={min_gap}).")

    # 6) Save neighbor cache (fixed K columns; unused slots remain -1/inf when target_k<K)
    out_npz = os.path.join(CTX_DIR, "neighbor_cache.npz")
    np.savez_compressed(out_npz, indices=all_inds, distances=all_dsts)

    # 7) Save diagnostics/meta
    top1 = np.where(np.isfinite(all_dsts[:, 0]), all_dsts[:, 0], np.nan).astype(np.float32)
    neighbor_meta = pd.DataFrame({
        "row": np.arange(N, dtype=np.int64),
        "trip_id": meta["trip_id"].values,
        "car_model": meta.get("car_model", pd.Series(["unknown"]*N)).values,
        "speed_limit_bin": meta.get("speed_limit_bin", pd.Series(["unknown"]*N)).values,
        "empty_weight_kg_bin": meta.get("empty_weight_kg_bin", pd.Series(["unknown"]*N)).values,
        "neighbors_filled": filled_counts,
        "target_k": target_k,
        "nn1_distance": top1,
    })

    out_meta_pq = os.path.join(CTX_DIR, "neighbor_meta.parquet")
    try:
        neighbor_meta.to_parquet(out_meta_pq, index=False)
        out_meta_path = out_meta_pq
    except Exception:
        out_meta_csv = os.path.join(CTX_DIR, "neighbor_meta.csv")
        neighbor_meta.to_csv(out_meta_csv, index=False)
        out_meta_path = out_meta_csv

    # 8) Report
    met_target = int((filled_counts >= target_k).sum())
    capped = int((target_k < K).sum())
    print(f"[neighbors] rows meeting target_k: {met_target}/{N} (100.0%)")
    if capped > 0:
        print(f"[neighbors] note: {capped} rows have target_k<K due to same-trip exclusions.")
    print(f"[neighbors] saved: {out_npz}")
    print(f"[neighbors] meta:  {out_meta_path}")

if __name__ == "__main__":
    main()
