#!/usr/bin/env python3
"""
Export causal driver fingerprints with class-mean cold-start by car_model,
and also FLATTEN the vector into scalar columns for easy downstream modeling.

Per split (train/val/test):
  1) Compute per-(trip_id, seg_id) segment embeddings from window embeddings.
  2) Build a CAUSAL fingerprint for segment x using segments 1..(x-1) of the SAME trip.
  3) Cold-start (first segment / single-seg trip):
       - Use TRAIN-only class mean by `car_model` (looked up via trip_id from windows_corrected.pickle),
         else zero vector. Always flag with fp_coldstart=1.
  4) Save:
       - Pickle with ndarray column 'driver_fp' (original format)
       - FLATTENED Parquet (or CSV fallback) with fp_000..fp_{D-1} columns

Optional: write a merged all-splits flattened file.
"""

# =======================
# CONFIG — edit these
# =======================
ARTIFACTS_PATH = r"C:/Users/goldy/OneDrive/Documents/Y-DATA 24-25/Stellantis Project/End-to-End/driver_emb_data/embedding_artifacts.pickle"
WINDOWS_CORRECTED_PATH = r"C:/Users/goldy/OneDrive/Documents/Y-DATA 24-25/Stellantis Project/End-to-End/driver_emb_data/windows_corrected.pickle"
MODEL_PATH     = None   # None → use {dirname(ARTIFACTS_PATH)}/driver_embedder.pt
OUT_DIR        = None   # None → use dirname(ARTIFACTS_PATH)
BATCH_SIZE     = 512
DEVICE         = None   # None → 'cuda' if available else 'cpu'

# Class-mean prior
USE_CLASS_MEAN_COLDSTART = True
CLASS_COL = "car_model"  # fetched via trip_id from WINDOWS_CORRECTED_PATH

# Sanity checks
SANITY_CHECK = True
SANITY_NORM_TOL = 1e-3

# Flatten/export controls
EXPORT_FLATTENED = True
MERGE_FLATTENED_ALL_SPLITS = True  # write one merged file across train/val/test
MERGED_BASENAME = "fingerprints_all"  # extension decided by writer (parquet/csv)

# =======================
# IMPLEMENTATION
# =======================
import os
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_artifacts(path: str) -> Dict:
    art = load_pickle(path)
    if "channels" not in art or not isinstance(art["channels"], list):
        raise KeyError("Artifacts must contain a top-level 'channels' list.")
    return art  # splits should include 'windows' (DataFrame) and 'masks' (np.ndarray)


def load_class_map_from_windows_corrected(path: str, class_col: str) -> Dict[object, object]:
    """
    Load trip_id -> class (e.g., car_model) mapping from windows_corrected.pickle.
    Accepts either a DataFrame or dict with 'windows' DataFrame.
    """
    obj = load_pickle(path)
    if isinstance(obj, pd.DataFrame):
        df_wc = obj
    elif isinstance(obj, dict) and "windows" in obj and isinstance(obj["windows"], pd.DataFrame):
        df_wc = obj["windows"]
    else:
        raise ValueError("windows_corrected.pickle must be a DataFrame or dict containing a 'windows' DataFrame.")

    if "trip_id" not in df_wc.columns or class_col not in df_wc.columns:
        raise KeyError(f"windows_corrected is missing required columns 'trip_id' and '{class_col}'.")

    df_map = df_wc[["trip_id", class_col]].dropna().drop_duplicates("trip_id", keep="last")
    class_map = dict(zip(df_map["trip_id"].tolist(), df_map[class_col].tolist()))
    if len(class_map) == 0:
        raise ValueError(f"No valid (trip_id → {class_col}) mappings found in windows_corrected.")
    print(f"[class-map] Loaded {len(class_map):,} trip_id → {class_col} mappings from windows_corrected.")
    return class_map


def _stack_split(df: pd.DataFrame, masks: np.ndarray, channels: List[str]) -> Tuple[np.ndarray, np.ndarray, List, List]:
    if df is None or len(df) == 0:
        raise ValueError("Split has zero windows.")
    req = {"trip_id", "seg_id"}
    if not req.issubset(df.columns):
        raise KeyError(f"Windows DataFrame missing required columns: {sorted(req - set(df.columns))}")
    if masks.shape[0] != len(df):
        raise ValueError(f"masks length {masks.shape[0]} != windows rows {len(df)}")

    # Build (N, C, L) tensor from channel arrays; enforce equal window length L across channels.
    Xs = [np.stack(df[ch].to_numpy(), axis=0) for ch in channels]
    Ls = [arr.shape[-1] for arr in Xs]
    if len(set(Ls)) != 1:
        raise ValueError(f"Window lengths differ across channels: {Ls}")
    X = np.stack(Xs, axis=1).astype(np.float32, copy=False)  # (N, C, L)

    # Artifacts store masks as 1==non-finite; convert to 1==finite for model/time masking.
    if masks.ndim != 3 or masks.shape[1] != len(channels) or masks.shape[2] != Ls[0]:
        raise ValueError(f"masks shape {masks.shape} incompatible with X {(len(df), len(channels), Ls[0])}")
    M = (1.0 - masks.astype(np.float32, copy=False))  # (N, C, L) with 1 for finite

    trips = df["trip_id"].tolist()
    segs  = df["seg_id"].tolist()  # int64 per your profile (1..34)
    return X, M, trips, segs


class ConvGRUEncoder(nn.Module):
    """
    1D CNN → GRU (bidirectional) → linear projection.
    Masked average pooling over time; output is L2-normalized embeddings.
    """
    def __init__(self, in_ch: int, embed_dim: int = 128, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.gru = nn.GRU(input_size=128, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(hidden * 2, embed_dim))

    def _masked_avg_pool(self, x: torch.Tensor, mask_time: torch.Tensor) -> torch.Tensor:
        x = x * mask_time
        denom = mask_time.sum(dim=2).clamp_min(1e-6)
        return x.sum(dim=2) / denom

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x)
        time_mask = (mask.sum(dim=1, keepdim=True) > 0).float()  # (B,1,L)
        x = self.conv(x)                        # (B, 128, L)
        x_t = x.transpose(1, 2)                 # (B, L, 128)
        out, _ = self.gru(x_t)                  # (B, L, 2*hidden)
        out = out.transpose(1, 2)               # (B, 2*hidden, L)
        pooled = self._masked_avg_pool(out, time_mask)  # (B, 2*hidden)
        z = self.proj(pooled)                   # (B, D)
        z = F.normalize(z, p=2, dim=1)          # unit-norm
        return z


@torch.no_grad()
def compute_segment_embeddings(
    encoder: nn.Module,
    X: np.ndarray,
    M: np.ndarray,
    trips: List,
    segs: List,
    device: str = "cpu",
    batch_size: int = 512,
) -> Tuple[np.ndarray, List, List]:
    """
    Aggregate window embeddings into one embedding per (trip, segment)
    via mask-weighted mean over windows; then L2-normalize.
    """
    encoder.eval()
    N = X.shape[0]
    embeds = []
    valid_frac = []
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device=device, dtype=torch.float32)
        mb = torch.from_numpy(M[i : i + batch_size]).to(device=device, dtype=torch.float32)
        zb = encoder(xb, mb)  # (b, D)
        embeds.append(zb.cpu().numpy())
        valid_frac.append(mb.cpu().numpy().mean(axis=(1, 2)))
    Z = np.concatenate(embeds, axis=0).astype(np.float32)
    W = np.concatenate(valid_frac, axis=0).astype(np.float32)

    # Group windows by (trip_id, seg_id)
    seg_map: Dict[Tuple, List[int]] = {}
    for idx, key in enumerate(zip(trips, segs)):
        seg_map.setdefault(key, []).append(idx)

    seg_embeds, seg_trip_ids, seg_seg_ids = [], [], []
    for (t, s), idxs in seg_map.items():
        z_i = Z[idxs]
        w_i = W[idxs]
        if np.isfinite(w_i).sum() == 0 or w_i.sum() < 1e-8:
            m = np.mean(z_i, axis=0)
        else:
            w = w_i / (w_i.sum() + 1e-8)
            m = np.sum(z_i * w[:, None], axis=0)
        m = m / (np.linalg.norm(m) + 1e-8)
        seg_embeds.append(m.astype(np.float32))
        seg_trip_ids.append(t)
        seg_seg_ids.append(s)

    return np.stack(seg_embeds, axis=0), seg_trip_ids, seg_seg_ids


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Handle checkpoints saved with DataParallel/DistributedDataParallel ('module.' prefix)."""
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _infer_hidden_and_dim_from_state(state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    # GRU hidden size from weight_hh_l0: shape (3*hidden, hidden)
    if "gru.weight_hh_l0" not in state or "proj.0.weight" not in state:
        raise KeyError("Checkpoint missing expected keys ('gru.weight_hh_l0', 'proj.0.weight').")
    hidden = state["gru.weight_hh_l0"].shape[1]
    embed_dim = state["proj.0.weight"].shape[0]
    return hidden, embed_dim


@torch.no_grad()
def compute_class_means_from_train(
    encoder: nn.Module,
    train_df: pd.DataFrame,
    train_masks: np.ndarray,
    channels: List[str],
    class_map: Dict[object, object],
    class_col: str,
    device: str,
    batch_size: int,
) -> Dict[object, np.ndarray]:
    """
    Compute unit-norm class-mean prototypes from TRAIN segments grouped by `class_col`.
    `class_col` values are obtained via trip_id → class_map (from windows_corrected).
    """
    X, M, trips, segs = _stack_split(train_df.copy(), train_masks.copy(), channels)
    Zs, seg_trips, seg_segs = compute_segment_embeddings(
        encoder, X, M, trips, segs, device=device, batch_size=batch_size
    )

    # Build segment-level frame with class from trip_id
    sdf = pd.DataFrame({"trip_id": seg_trips, "seg_id": seg_segs})
    sdf["seg_embed"] = [z.astype(np.float32) for z in Zs]
    sdf[class_col] = [class_map.get(t, None) for t in sdf["trip_id"]]

    means: Dict[object, np.ndarray] = {}
    for cls, g in sdf.groupby(class_col, dropna=False):
        if cls is None or len(g) == 0:
            continue
        Mcls = np.stack(g["seg_embed"].to_list(), axis=0).astype(np.float32)
        m = Mcls.mean(axis=0)
        m = m / (np.linalg.norm(m) + 1e-8)
        means[cls] = m.astype(np.float32)
    print(f"[class-means] built {len(means)} prototypes by '{class_col}'.")
    return means


def build_causal_fingerprints_with_prior(
    seg_Z: np.ndarray,
    seg_trips: List,
    seg_segs: List,
    class_map: Dict[object, object],
    class_means: Optional[Dict[object, np.ndarray]],
    class_col: str,
) -> pd.DataFrame:
    """
    For each trip_id, sort numerically by seg_id (int64) and compute:
      fp_x = L2( sum_{j < x} seg_embed_j )  if x > 1
      fp_1 = class_means[class_of_trip] if available else zeros
    Always set fp_coldstart=1 for x==1, else 0.
    """
    if len(seg_Z) == 0:
        return pd.DataFrame(columns=["trip_id", "seg_id", "driver_fp", "fp_count", "fp_coldstart", class_col])

    df = pd.DataFrame({"trip_id": seg_trips, "seg_id": seg_segs})
    df["seg_embed"] = [z.astype(np.float32) for z in seg_Z]
    df[class_col] = [class_map.get(t, None) for t in df["trip_id"]]  # trip-level class

    # numeric ordering guaranteed (seg_id is int64)
    df = df.sort_values(["trip_id", "seg_id"]).reset_index(drop=True)

    fp_vecs, fp_count, fp_cold = [], [], []
    last_trip = None
    sum_vec = None
    count = 0

    zero_proto = df["seg_embed"].iloc[0] * 0.0  # correct dimension
    for _, row in df.iterrows():
        t = row.trip_id
        v = row.seg_embed
        cls_val = row.get(class_col, None)

        if t != last_trip:
            sum_vec = np.zeros_like(v, dtype=np.float32)
            count = 0
            last_trip = t

        if count == 0:
            # class-mean cold-start if available, else zero
            if class_means and (cls_val in class_means):
                fp = class_means[cls_val]
            else:
                fp = zero_proto.copy()
            fp_cold.append(1)
        else:
            fp = sum_vec / (np.linalg.norm(sum_vec) + 1e-8)
            fp_cold.append(0)

        fp_vecs.append(fp)
        fp_count.append(count)
        sum_vec = sum_vec + v
        count += 1

    out = df[["trip_id", "seg_id", class_col]].copy()
    out["driver_fp"] = fp_vecs
    out["fp_count"] = fp_count
    out["fp_coldstart"] = fp_cold
    return out


def _save_pickle(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_pickle(out_path)
    print(f"[pickle] wrote {len(df):,} rows → {out_path}")


def _save_parquet_or_csv(df: pd.DataFrame, out_base: str) -> str:
    """
    Try to write Parquet; if not available, fall back to CSV.
    Returns the actual path written.
    """
    pq_path = out_base + ".parquet"
    try:
        df.to_parquet(pq_path, index=False)
        print(f"[flat] wrote {len(df):,} rows → {pq_path}")
        return pq_path
    except Exception as e:
        csv_path = out_base + ".csv"
        df.to_csv(csv_path, index=False)
        print(f"[flat] wrote {len(df):,} rows → {csv_path} (Parquet unavailable: {e})")
        return csv_path


def _flatten_and_save(fps: pd.DataFrame, embed_dim: int, out_base: str) -> str:
    """
    Flatten ndarray 'driver_fp' into fp_000..fp_{D-1} float32 columns and save.
    Keeps: trip_id, seg_id, fp_count, fp_coldstart, split, CLASS_COL, fp_*.
    Uses a single concat to avoid pandas fragmentation warnings.
    """
    # Write empty file if needed
    if fps.empty:
        base = fps.drop(columns=["driver_fp"], errors="ignore").reset_index(drop=True)
        return _save_parquet_or_csv(base, out_base)

    # sanity: all vectors same dimension
    dims = [v.shape[0] for v in fps["driver_fp"]]
    if not all(d == dims[0] == embed_dim for d in dims):
        raise ValueError(f"Inconsistent fingerprint dims in dataframe: {set(dims)} vs embed_dim={embed_dim}")

    # Build fp_* block once, then concat (no repeated inserts -> no fragmentation)
    mat = np.stack(fps["driver_fp"].to_list(), axis=0).astype(np.float32)
    fp_cols = [f"fp_{i:03d}" for i in range(embed_dim)]
    fp_df = pd.DataFrame(mat, columns=fp_cols)  # float32 already

    base = fps.drop(columns=["driver_fp"]).reset_index(drop=True)
    out = pd.concat([base, fp_df], axis=1, copy=False)

    # Column order for readability
    front = ["trip_id", "seg_id", "fp_count", "fp_coldstart", "split", CLASS_COL]
    ordered = front + [c for c in out.columns if c not in front]
    out = out[ordered]

    return _save_parquet_or_csv(out, out_base)


def _available_splits(art: Dict) -> List[str]:
    return [s for s in ("train", "val", "test") if s in art and "windows" in art[s] and "masks" in art[s]]


def _sanity_check(
    split: str,
    fps: pd.DataFrame,
    class_means: Dict[object, np.ndarray],
    embed_dim: int,
    class_col: str,
    tol: float = 1e-3,
):
    if fps.empty:
        print(f"[sanity:{split}] empty → OK")
        return
    req = {"trip_id", "seg_id", "driver_fp", "fp_count", "fp_coldstart", class_col}
    assert req.issubset(fps.columns), "[sanity] required columns missing"
    shapes_ok = all(isinstance(v, np.ndarray) and v.shape == (embed_dim,) for v in fps["driver_fp"])
    assert shapes_ok, "[sanity] some driver_fp vectors have incorrect shapes"
    # non-coldstart: norm ~ 1
    mask_nc = fps["fp_coldstart"].values == 0
    if mask_nc.any():
        norms = np.array([np.linalg.norm(v) for v in fps.loc[mask_nc, "driver_fp"]])
        assert np.all(np.isfinite(norms)), "[sanity] non-finite norms"
        assert np.all(np.abs(norms - 1.0) <= tol), "[sanity] non-cold-start fingerprints not unit-norm"
    # coldstart checks
    mask_cs = ~mask_nc
    if mask_cs.any():
        zeros = np.zeros((embed_dim,), dtype=np.float32)
        for _, r in fps.loc[mask_cs].iterrows():
            fp = r["driver_fp"]; cls = r.get(class_col, None)
            if class_means and (cls in class_means):
                assert np.allclose(fp, class_means[cls], atol=1e-6), "[sanity] cold-start not equal to class mean"
            else:
                assert np.allclose(fp, zeros, atol=1e-6), "[sanity] cold-start not zero for unseen/missing class"
    # monotonic seg order & counts per trip
    for _, g in fps.sort_values(["trip_id", "seg_id"]).groupby("trip_id", sort=False):
        seg_ids = g["seg_id"].to_numpy()
        assert np.all(seg_ids[:-1] <= seg_ids[1:]), "[sanity] seg_id not non-decreasing within trip"
        counts = g["fp_count"].to_numpy()
        assert counts[0] == 0 and np.all(counts[1:] == counts[:-1] + 1), "[sanity] fp_count not incremental per trip"
        starts = g["fp_coldstart"].to_numpy()
        assert starts[0] == 1 and np.all(starts[1:] == 0), "[sanity] coldstart flag invalid per trip"
    print(f"[sanity:{split}] OK")


def main():
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUT_DIR or os.path.dirname(os.path.abspath(ARTIFACTS_PATH))
    model_path = MODEL_PATH or os.path.join(out_dir, "driver_embedder.pt")

    # Load artifacts and class map
    art = load_artifacts(ARTIFACTS_PATH)
    channels: List[str] = art["channels"]
    class_map = load_class_map_from_windows_corrected(WINDOWS_CORRECTED_PATH, CLASS_COL)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best checkpoint not found at: {model_path}")

    # Instantiate encoder using shapes inferred from checkpoint
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # if saved with DataParallel
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    hidden, embed_dim = _infer_hidden_and_dim_from_state(state)
    encoder = ConvGRUEncoder(in_ch=len(channels), embed_dim=embed_dim, hidden=hidden).to(device)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()

    present = _available_splits(art)

    # ---- Build class-mean prototypes from TRAIN only (if enabled) ----
    class_means: Dict[object, np.ndarray] = {}
    if USE_CLASS_MEAN_COLDSTART and "train" in present:
        tr_df = art["train"]["windows"]
        tr_masks = art["train"]["masks"]
        if tr_df is not None and tr_masks is not None and len(tr_df) > 0:
            class_means = compute_class_means_from_train(
                encoder, tr_df, tr_masks, channels, class_map, CLASS_COL, device=device, batch_size=BATCH_SIZE
            )
        else:
            print("[warn] TRAIN split empty; class-mean cold-start disabled.")
    elif USE_CLASS_MEAN_COLDSTART:
        print("[warn] TRAIN split not found; class-mean cold-start disabled.")

    flattened_paths = []

    # ---- Export fingerprints for splits ----
    for split in ["train", "val", "test"]:
        if split not in present:
            print(f"[skip] split '{split}' not found in artifacts")
            continue

        df = art[split]["windows"]
        masks = art[split]["masks"]
        if df is None or masks is None or len(df) == 0:
            print(f"[skip] split '{split}' is empty")
            continue

        # Stack windows
        X, M, trips, segs = _stack_split(df.copy(), masks.copy(), channels)

        # Segment embeddings (per (trip_id, seg_id))
        Zs, seg_trips, seg_segs = compute_segment_embeddings(
            encoder, X, M, trips, segs, device=device, batch_size=BATCH_SIZE
        )

        # Causal fingerprints with class-mean prior on cold-start (car_model via class_map)
        fps = build_causal_fingerprints_with_prior(
            Zs, seg_trips, seg_segs,
            class_map=class_map,
            class_means=class_means if USE_CLASS_MEAN_COLDSTART else {},
            class_col=CLASS_COL,
        )
        fps["split"] = split

        # Save pickle (array cell)
        out_pkl = os.path.join(out_dir, f"fingerprints_{split}.pickle")
        _save_pickle(fps[["trip_id", "seg_id", "driver_fp", "fp_count", "fp_coldstart", "split", CLASS_COL]], out_pkl)

        # Sanity checks (optional)
        if SANITY_CHECK:
            _sanity_check(split, fps, class_means if USE_CLASS_MEAN_COLDSTART else {}, embed_dim, CLASS_COL, tol=SANITY_NORM_TOL)

        # Flatten & save for tabular pipelines
        if EXPORT_FLATTENED:
            out_base = os.path.join(out_dir, f"fingerprints_{split}")
            written_path = _flatten_and_save(fps[["trip_id", "seg_id", "driver_fp", "fp_count", "fp_coldstart", "split", CLASS_COL]].copy(),
                                             embed_dim, out_base)
            flattened_paths.append(written_path)

    # Optionally merge all flattened splits into one file
    if EXPORT_FLATTENED and MERGE_FLATTENED_ALL_SPLITS and flattened_paths:
        merged_path_base = os.path.join(out_dir, MERGED_BASENAME)
        # Read back the flattened files (parquet or csv) and concat
        dfs = []
        for p in flattened_paths:
            if p.endswith(".parquet"):
                dfs.append(pd.read_parquet(p))
            else:
                dfs.append(pd.read_csv(p))
        merged = pd.concat(dfs, ignore_index=True)
        _ = _save_parquet_or_csv(merged, merged_path_base)

    print("[done] fingerprints exported (pickle + flattened).")


if __name__ == "__main__":
    main()
