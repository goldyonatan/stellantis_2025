#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver fingerprints (v3): leakage-free & more robust

Improvements over v2:
  • Train-only whitening of segment embeddings (mean removal + PCA/SVD whitening with ε)
  • Hierarchical cold-start: class-mean blended with global mean using empirical-Bayes shrinkage by class size
  • Robust incremental prototype update: weight each previous segment by cosine similarity to the running prototype
      - For seg x, fingerprint uses ONLY segments 1..(x-1)
      - Weight w_j = clip((cos(fp_{j-1}, z_j))^gamma, w_min, 1.0), gamma∈{1,2}, defaults: gamma=1.5, w_min=0.25
  • Stability metric (fp_stability): computed from previous segments only as mean cosine of prior z_j to fp_{x-1} (mapped to [-1,1])
  • Flattened outputs retain: trip_id, seg_id, split, fp_count, fp_coldstart, fp_stability, CLASS_COL, fp_000..fp_{D-1}

No future leakage: every quantity for seg x is computed strictly from 1..(x-1).

Outputs:
  fingerprints_{split}.pickle  (array cell column 'driver_fp')
  fingerprints_{split}.parquet (flattened fp_*)
  fingerprints_all.parquet     (merged flattened across splits)
"""

# =======================
# CONFIG
# =======================
ARTIFACTS_PATH = r"C:/Users/goldy/OneDrive/Documents/Y-DATA 24-25/Stellantis Project/End-to-End/driver_emb_data/embedding_artifacts.pickle"
WINDOWS_CORRECTED_PATH = r"C:/Users/goldy/OneDrive/Documents/Y-DATA 24-25/Stellantis Project/End-to-End/driver_emb_data/windows_corrected.pickle"
MODEL_PATH     = None   # None → {dirname(ARTIFACTS_PATH)}/driver_embedder.pt
OUT_DIR        = None   # None → dirname(ARTIFACTS_PATH)
BATCH_SIZE     = 512
DEVICE         = None   # None → 'cuda' if available else 'cpu'

# Whitening (fit on TRAIN only; applied to all splits)
USE_WHITENING = True
WHITEN_EPS = 1e-3

# Class-mean prior (hierarchical shrinkage)
USE_CLASS_MEAN_COLDSTART = True
CLASS_COL = "car_model"
SHRINK_TAU = 5.0  # larger → stronger blend toward global

# Robust prototype update
GAMMA = 1.5
W_MIN = 0.25

SANITY_CHECK = True
SANITY_NORM_TOL = 1e-3

EXPORT_FLATTENED = True
MERGE_FLATTENED_ALL_SPLITS = True
MERGED_BASENAME = "fingerprints_all"

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
    return art


def load_class_map(path: str, class_col: str) -> Dict[object, object]:
    obj = load_pickle(path)
    if isinstance(obj, pd.DataFrame):
        df_wc = obj
    elif isinstance(obj, dict) and "windows" in obj and isinstance(obj["windows"], pd.DataFrame):
        df_wc = obj["windows"]
    else:
        raise ValueError("windows_corrected.pickle must be a DataFrame or dict containing a 'windows' DataFrame.")
    if "trip_id" not in df_wc.columns or class_col not in df_wc.columns:
        raise KeyError(f"windows_corrected is missing 'trip_id' and '{class_col}'.")
    df_map = df_wc[["trip_id", class_col]].dropna().drop_duplicates("trip_id", keep="last")
    return dict(zip(df_map["trip_id"].tolist(), df_map[class_col].tolist()))


def _stack_split(df: pd.DataFrame, masks: np.ndarray, channels: List[str]) -> Tuple[np.ndarray, np.ndarray, List, List]:
    if df is None or len(df) == 0:
        raise ValueError("Split has zero windows.")
    if not {"trip_id", "seg_id"}.issubset(df.columns):
        raise KeyError("Windows DataFrame missing 'trip_id'/'seg_id'.")
    Xs = [np.stack(df[ch].to_numpy(), axis=0) for ch in channels]
    Ls = [arr.shape[-1] for arr in Xs]
    if len(set(Ls)) != 1:
        raise ValueError(f"Window lengths differ across channels: {Ls}")
    X = np.stack(Xs, axis=1).astype(np.float32, copy=False)  # (N,C,L)
    if masks.ndim != 3 or masks.shape[1] != len(channels) or masks.shape[2] != Ls[0]:
        raise ValueError("masks shape incompatible")
    M = (1.0 - masks.astype(np.float32, copy=False))  # 1 for finite
    trips = df["trip_id"].tolist(); segs = df["seg_id"].tolist()
    return X, M, trips, segs


class ConvGRUEncoder(nn.Module):
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
        if mask is None: mask = torch.ones_like(x)
        time_mask = (mask.sum(dim=1, keepdim=True) > 0).float()
        x = self.conv(x)                 # (B,128,L)
        out, _ = self.gru(x.transpose(1,2))
        out = out.transpose(1,2)         # (B,2*hidden,L)
        pooled = self._masked_avg_pool(out, time_mask)  # (B,2*hidden)
        z = self.proj(pooled)            # (B,D)
        z = F.normalize(z, p=2, dim=1)   # unit-norm
        return z


def _strip_ddp(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("module.", "", 1): v for k, v in state.items()} if any(k.startswith("module.") for k in state) else state


def _infer_hidden_and_dim(state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    if "gru.weight_hh_l0" not in state or "proj.0.weight" not in state:
        raise KeyError("Checkpoint missing 'gru.weight_hh_l0' or 'proj.0.weight'.")
    hidden = state["gru.weight_hh_l0"].shape[1]
    dim = state["proj.0.weight"].shape[0]
    return hidden, dim


@torch.no_grad()
def encode_segment_windows(encoder, X, M, trips, segs, device="cpu", batch_size=512):
    encoder.eval()
    N = X.shape[0]; outs = []; valid = []
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).to(device)
        mb = torch.from_numpy(M[i:i+batch_size]).to(device)
        zb = encoder(xb, mb)
        outs.append(zb.cpu().numpy())
        valid.append(mb.cpu().numpy().mean(axis=(1,2)))
    Z = np.concatenate(outs, axis=0).astype(np.float32)
    V = np.concatenate(valid, axis=0).astype(np.float32)

    seg_map = {}
    for idx, key in enumerate(zip(trips, segs)):
        seg_map.setdefault(key, []).append(idx)

    seg_embeds, seg_trips, seg_segs = [], [], []
    for (t, s), idxs in seg_map.items():
        z_i = Z[idxs]; w_i = V[idxs]
        if np.isfinite(w_i).sum() == 0 or w_i.sum() < 1e-8:
            m = np.mean(z_i, axis=0)
        else:
            w = w_i / (w_i.sum() + 1e-8)
            m = np.sum(z_i * w[:, None], axis=0)
        m = m / (np.linalg.norm(m) + 1e-8)
        seg_embeds.append(m.astype(np.float32))
        seg_trips.append(t); seg_segs.append(s)
    return np.stack(seg_embeds, axis=0), seg_trips, seg_segs


def fit_whitener_from_train(Z_train: np.ndarray, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (mu, W) such that z' = (z - mu) @ W ~ whitened.
    W = U @ diag(1/sqrt(s + eps)) @ U^T from SVD of covariance.
    """
    mu = Z_train.mean(axis=0)
    Xc = Z_train - mu
    # symmetric covariance
    C = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    # eigh for symmetric matrix
    s, U = np.linalg.eigh(C)
    s = np.clip(s, eps, None)
    W = U @ np.diag(1.0 / np.sqrt(s)) @ U.T
    return mu.astype(np.float32), W.astype(np.float32)


def apply_whitening(Z: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    Xc = Z - mu
    Zw = Xc @ W
    # re-normalize to unit sphere
    Zw = Zw / (np.linalg.norm(Zw, axis=1, keepdims=True) + 1e-8)
    return Zw.astype(np.float32)


def build_fp_for_trip(prev_vectors: List[np.ndarray], class_proto: Optional[np.ndarray], gamma: float, w_min: float):
    """
    Given previous segment embeddings, compute the fingerprint vector (unit-norm),
    fp_count, fp_coldstart, and a stability score from previous segments only.

    Fixes:
      • Use positive-part of cosine for non-integer powers: w = max(w_min, max(sim,0)**gamma), then cap to 1.0
      • Stability is the mean cosine similarity ∈ [-1,1] (no remapping)
    """
    D = prev_vectors[0].shape[0] if prev_vectors else None
    zero = np.zeros((D,), dtype=np.float32) if D is not None else None

    # Cold-start: return class prior (or zeros), count=0, coldstart=1, stability=0.0
    if len(prev_vectors) == 0:
        fp = class_proto if class_proto is not None else zero
        fp = (fp / (np.linalg.norm(fp) + 1e-8)).astype(np.float32) if fp is not None else np.zeros((0,), dtype=np.float32)
        return fp, 0, 1, 0.0

    # Initialize prototype with the first previous vector
    proto = prev_vectors[0].astype(np.float32)
    proto = proto / (np.linalg.norm(proto) + 1e-8)

    # Robust incremental update using cosine-based weights
    g = float(max(0.0, gamma))
    w_floor = float(np.clip(w_min, 0.0, 1.0))
    for j in range(1, len(prev_vectors)):
        v = prev_vectors[j].astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        sim = float(np.clip(np.dot(proto, v), -1.0, 1.0))
        pos = max(sim, 0.0)  # avoid complex numbers for non-integer powers
        w = max(w_floor, pos ** g)
        w = float(min(w, 1.0))
        proto = proto + w * v
        proto = proto / (np.linalg.norm(proto) + 1e-8)

    # Stability: mean cosine of previous vectors to final proto ∈ [-1,1]
    sims = []
    for v in prev_vectors:
        vv = v.astype(np.float32)
        vv = vv / (np.linalg.norm(vv) + 1e-8)
        sims.append(float(np.clip(np.dot(proto, vv), -1.0, 1.0)))
    stability = float(np.mean(sims)) if len(sims) > 0 else 0.0

    return proto.astype(np.float32), len(prev_vectors), 0, stability
def build_causal_fingerprints_with_prior_whitened(
    seg_Z: np.ndarray,
    seg_trips: List,
    seg_segs: List,
    class_map: Dict[object, object],
    class_means: Dict[object, np.ndarray],
    class_sizes: Dict[object, int],
    global_mean: Optional[np.ndarray],
    class_col: str,
    gamma: float,
    w_min: float,
) -> pd.DataFrame:
    """
    As in v2, but with robust prototype & stability and hierarchical cold-start:
      fp_1 = a*class_mean + (1-a)*global_mean, a = n_cls / (n_cls + tau)
    """
    df = pd.DataFrame({"trip_id": seg_trips, "seg_id": seg_segs})
    df["seg_embed"] = [z.astype(np.float32) for z in seg_Z]
    df[class_col] = [class_map.get(t, None) for t in df["trip_id"]]
    df = df.sort_values(["trip_id", "seg_id"]).reset_index(drop=True)

    fp_vecs, fp_count, fp_cold, fp_stab = [], [], [], []
    last_trip = None
    prev_list: List[np.ndarray] = []

    zero_proto = df["seg_embed"].iloc[0] * 0.0 if len(df) else None

    for _, row in df.iterrows():
        t = row.trip_id; v = row.seg_embed; cls_val = row.get(class_col, None)
        if t != last_trip:
            prev_list = []
            last_trip = t

        # build class prior
        if USE_CLASS_MEAN_COLDSTART and cls_val in class_means:
            n_cls = class_sizes.get(cls_val, 0)
            a = n_cls / (n_cls + SHRINK_TAU)
            prior = (a * class_means[cls_val] + (1 - a) * (global_mean if global_mean is not None else zero_proto))
            prior = prior / (np.linalg.norm(prior) + 1e-8)
        else:
            prior = zero_proto

        fp, cnt, cold, stab = build_fp_for_trip(prev_list, class_proto=prior, gamma=gamma, w_min=w_min)
        fp_vecs.append(fp); fp_count.append(cnt); fp_cold.append(cold); fp_stab.append(stab)

        # update prev_list after computing fingerprint for current seg (so seg x uses only 1..x-1)
        prev_list.append(v)

    out = df[["trip_id", "seg_id", class_col]].copy()
    out["driver_fp"] = fp_vecs
    out["fp_count"] = fp_count
    out["fp_coldstart"] = fp_cold
    out["fp_stability"] = fp_stab
    return out


def _save_parquet_or_csv(df: pd.DataFrame, out_base: str) -> str:
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
    if fps.empty:
        base = fps.drop(columns=["driver_fp"], errors="ignore").reset_index(drop=True)
        return _save_parquet_or_csv(base, out_base)
    dims = [v.shape[0] for v in fps["driver_fp"]]
    if not all(d == dims[0] == embed_dim for d in dims):
        raise ValueError("Inconsistent fingerprint dims.")
    mat = np.stack(fps["driver_fp"].to_list(), axis=0).astype(np.float32)
    fp_cols = [f"fp_{i:03d}" for i in range(embed_dim)]
    fp_df = pd.DataFrame(mat, columns=fp_cols)
    base = fps.drop(columns=["driver_fp"]).reset_index(drop=True)
    out = pd.concat([base, fp_df], axis=1, copy=False)
    front = ["trip_id", "seg_id", "fp_count", "fp_coldstart", "fp_stability", "split", CLASS_COL]
    ordered = front + [c for c in out.columns if c not in front]
    out = out[ordered]
    return _save_parquet_or_csv(out, out_base)


def _available_splits(art: Dict) -> List[str]:
    return [s for s in ("train", "val", "test") if s in art and "windows" in art[s] and "masks" in art[s]]


def main():
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUT_DIR or os.path.dirname(os.path.abspath(ARTIFACTS_PATH))
    model_path = MODEL_PATH or os.path.join(out_dir, "driver_embedder.pt")

    art = load_artifacts(ARTIFACTS_PATH)
    channels: List[str] = art["channels"]
    class_map = load_class_map(WINDOWS_CORRECTED_PATH, CLASS_COL)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best checkpoint not found at: {model_path}")

    # build encoder
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
    state = _strip_ddp(state)
    hidden, embed_dim = _infer_hidden_and_dim(state)
    encoder = ConvGRUEncoder(in_ch=len(channels), embed_dim=embed_dim, hidden=hidden).to(device)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()

    present = _available_splits(art)

    # ---- TRAIN segment embeddings for whitening + class means ----
    mu, W = None, None
    class_means, class_sizes = {}, {}
    global_mean = None

    if "train" in present:
        df_tr = art["train"]["windows"]; mk_tr = art["train"]["masks"]
        X, M, trips, segs = _stack_split(df_tr.copy(), mk_tr.copy(), channels)
        Z_tr, tr_trips, tr_segs = encode_segment_windows(encoder, X, M, trips, segs, device=device, batch_size=BATCH_SIZE)

        # Whitening
        if USE_WHITENING:
            mu, W = fit_whitener_from_train(Z_tr, eps=WHITEN_EPS)
            Z_tr = apply_whitening(Z_tr, mu, W)

        # Class means & sizes (by CLASS_COL)
        cvals = [class_map.get(t, None) for t in tr_trips]
        sdf = pd.DataFrame({"trip_id": tr_trips, "seg_id": tr_segs, CLASS_COL: cvals})
        sdf["seg_embed"] = [z for z in Z_tr]
        for cls, g in sdf.groupby(CLASS_COL, dropna=False):
            if cls is None or len(g) == 0: continue
            Mcls = np.stack(g["seg_embed"].to_list(), axis=0).astype(np.float32)
            m = Mcls.mean(axis=0); m = m / (np.linalg.norm(m) + 1e-8)
            class_means[cls] = m.astype(np.float32)
            class_sizes[cls] = int(len(g))
        if len(sdf) > 0:
            global_mean = (np.mean(np.stack(sdf["seg_embed"].to_list(), axis=0), axis=0)).astype(np.float32)
            global_mean = global_mean / (np.linalg.norm(global_mean) + 1e-8)
    else:
        print("[warn] TRAIN split not found; whitening and class means disabled.")

    flattened_paths = []

    # ---- Per split export ----
    for split in ["train", "val", "test"]:
        if split not in present:
            print(f"[skip] split '{split}' not found in artifacts")
            continue
        df = art[split]["windows"]; masks = art[split]["masks"]
        if df is None or masks is None or len(df) == 0:
            print(f"[skip] split '{split}' is empty")
            continue
        X, M, trips, segs = _stack_split(df.copy(), masks.copy(), channels)
        Z, seg_trips, seg_segs = encode_segment_windows(encoder, X, M, trips, segs, device=device, batch_size=BATCH_SIZE)
        if USE_WHITENING and (mu is not None) and (W is not None):
            Z = apply_whitening(Z, mu, W)

        fps = build_causal_fingerprints_with_prior_whitened(
            Z, seg_trips, seg_segs,
            class_map=class_map,
            class_means=class_means,
            class_sizes=class_sizes,
            global_mean=global_mean,
            class_col=CLASS_COL,
            gamma=GAMMA,
            w_min=W_MIN,
        )
        fps["split"] = split

        # Save array-cell pickle
        out_pkl = os.path.join(out_dir, f"fingerprints_{split}.pickle")
        os.makedirs(out_dir, exist_ok=True)
        fps[["trip_id","seg_id","driver_fp","fp_count","fp_coldstart","fp_stability","split",CLASS_COL]].to_pickle(out_pkl)
        print(f"[pickle] wrote {len(fps):,} rows → {out_pkl}")

        # Sanity (no leakage; only 1..(x-1) used)
        if SANITY_CHECK and len(fps) > 0:
            # Unit norms for non-coldstart
            mask_nc = fps["fp_coldstart"].values == 0
            if mask_nc.any():
                norms = np.array([np.linalg.norm(v) for v in fps.loc[mask_nc, "driver_fp"]])
                assert np.all(np.isfinite(norms)), "[sanity] non-finite norms"
                assert np.all(np.abs(norms - 1.0) <= SANITY_NORM_TOL), "[sanity] non-cold-start fingerprints not unit-norm"
            # Monotone counts per trip
            for _, g in fps.sort_values(["trip_id","seg_id"]).groupby("trip_id", sort=False):
                seg_ids = g["seg_id"].to_numpy()
                counts  = g["fp_count"].to_numpy()
                cflags  = g["fp_coldstart"].to_numpy()
                assert counts[0] == 0 and np.all(counts[1:] == counts[:-1] + 1), "[sanity] fp_count not incremental"
                assert cflags[0] == 1 and np.all(cflags[1:] == 0), "[sanity] coldstart flags invalid order"

        # Flatten
        if EXPORT_FLATTENED:
            out_base = os.path.join(out_dir, f"fingerprints_{split}")
            # add fp_stability to flattened outputs
            cols = ["trip_id","seg_id","driver_fp","fp_count","fp_coldstart","fp_stability","split",CLASS_COL]
            written = _flatten_and_save(fps[cols].copy(), embed_dim, out_base)
            flattened_paths.append(written)

    if EXPORT_FLATTENED and MERGE_FLATTENED_ALL_SPLITS and flattened_paths:
        merged_base = os.path.join(out_dir, MERGED_BASENAME)
        dfs = [(pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)) for p in flattened_paths]
        merged = pd.concat(dfs, ignore_index=True)
        _save_parquet_or_csv(merged, merged_base)

    print("[done] v3 fingerprints exported.")

if __name__ == "__main__":
    main()