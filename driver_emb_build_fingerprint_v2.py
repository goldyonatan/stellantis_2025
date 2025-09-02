#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver fingerprint builder (compat) — strictly causal, no recency bias,
compatible with your existing artifacts format.

Key improvements (all causal: for seg x use only 1..(x−1)):
  • Robust trimmed-mean of prior segment embeddings (no EMA/time weights).
  • James–Stein style shrinkage toward TRAIN-only class prototype (e.g., car_model)
    when history is short; fades out as history grows.
  • Residualized vector vs. prototype (exported as fp_res_*).
  • Stability scalar: mean pairwise cosine among prior seg embeddings (n<2 ⇒ 0).
Outputs mirror your prior script, plus fp_res_* and fp_stability in the flattened files.
"""

import os, pickle, math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# =======================
# CONFIG — edit as needed
# =======================
ARTIFACTS_PATH = r"C:/Users/goldy/OneDrive/Documents/Y-DATA 24-25/Stellantis Project/End-to-End/driver_emb_data/embedding_artifacts.pickle"
WINDOWS_CORRECTED_PATH = r"C:/Users/goldy/OneDrive/Documents/Y-DATA 24-25/Stellantis Project/End-to-End/driver_emb_data/windows_corrected.pickle"
MODEL_PATH     = None     # None → same directory as artifacts / driver_embedder.pt
OUT_DIR        = None     # None → same directory as artifacts
CLASS_COL      = "car_model"

# Robust + shrink settings
TRIM_FRAC      = 0.20     # drop bottom 20% by cosine when n>=5
SHRINK_LAMBDA  = 6.0      # gamma = n/(n+lambda)

# IO options
EXPORT_FLATTENED = True
MERGE_FLATTENED_ALL_SPLITS = True
MERGED_BASENAME  = "fingerprints_all"  # .parquet

DEVICE = None  # "cuda" | "cpu" | None (auto)
BATCH_SIZE = 512

# =======================
# Helpers
# =======================
def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def _save_pickle(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_pickle(out_path)
    print(f"[pickle] wrote {len(df):,} rows → {out_path}")

def _save_parquet_or_csv(df: pd.DataFrame, out_base: str) -> str:
    pq = out_base + ".parquet"
    try:
        df.to_parquet(pq, index=False)
        print(f"[flat] wrote {len(df):,} rows → {pq}")
        return pq
    except Exception as e:
        csv = out_base + ".csv"
        df.to_csv(csv, index=False)
        print(f"[flat] wrote {len(df):,} rows → {csv} (Parquet unavailable: {e})")
        return csv

def load_artifacts(path: str) -> Dict:
    art = _load_pickle(path)
    # Expect: {'channels': [...], 'train': {'windows': df, 'masks': np}, 'val': {...}, 'test': {...?}}
    if not all(k in art for k in ["channels", "train", "val"]):
        raise KeyError("Artifacts must contain 'channels', 'train', 'val' (and optionally 'test').")
    for split in ["train", "val"]:
        if "windows" not in art[split] or "masks" not in art[split]:
            raise KeyError(f"Artifacts['{split}'] must include 'windows' and 'masks'.")
    return art

def load_class_map_from_windows_corrected(path: str, class_col: str) -> Dict[object, object]:
    obj = _load_pickle(path)
    if isinstance(obj, pd.DataFrame):
        df_wc = obj
    elif isinstance(obj, dict) and "windows" in obj and isinstance(obj["windows"], pd.DataFrame):
        df_wc = obj["windows"]
    else:
        raise ValueError("windows_corrected must be a DataFrame or dict with a 'windows' DataFrame.")
    if "trip_id" not in df_wc.columns or class_col not in df_wc.columns:
        raise KeyError(f"windows_corrected missing required columns: 'trip_id' and '{class_col}'.")
    df_map = df_wc[["trip_id", class_col]].dropna().drop_duplicates("trip_id", keep="last")
    m = dict(zip(df_map["trip_id"].tolist(), df_map[class_col].tolist()))
    if not m:
        raise ValueError("No (trip_id → class) mappings found in windows_corrected.")
    print(f"[class-map] Loaded {len(m):,} trip_id → {class_col} mappings.")
    return m

# =======================
# Model (mirrors your training encoder)
# =======================
class ConvGRUEncoder(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, hidden: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(input_size=128, hidden_size=hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(2*hidden, embed_dim))

    def _masked_avg_pool(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L), time_mask: (B,1,L) in {0,1}
        xm = x * time_mask
        denom = time_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        return xm.sum(dim=2) / denom.squeeze(2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x)
        time_mask = (mask.sum(dim=1, keepdim=True) > 0).float()  # (B,1,L)
        x = self.conv(x)                        # (B,128,L)
        x_t = x.transpose(1, 2)                 # (B,L,128)
        out, _ = self.gru(x_t)                  # (B,L,2*hidden)
        out = out.transpose(1, 2)               # (B,2*hidden,L)
        pooled = self._masked_avg_pool(out, time_mask)  # (B,2*hidden)
        z = self.proj(pooled)                   # (B,D)
        z = F.normalize(z, p=2, dim=1)
        return z

def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state

def _infer_hidden_and_dim_from_state(state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    if "gru.weight_hh_l0" not in state:
        raise KeyError("Checkpoint missing key: 'gru.weight_hh_l0'")
    hidden = state["gru.weight_hh_l0"].shape[1]
    # support either proj.weight or proj.0.weight depending on Sequential
    if "proj.0.weight" in state:
        embed_dim = state["proj.0.weight"].shape[0]
    elif "proj.weight" in state:
        embed_dim = state["proj.weight"].shape[0]
    else:
        raise KeyError("Checkpoint missing projection weight (proj.weight / proj.0.weight).")
    return hidden, embed_dim

# =======================
# Data → segment embeddings
# =======================
def _stack_split(df: pd.DataFrame, masks: np.ndarray, channels: List[str]) -> Tuple[np.ndarray, np.ndarray, List, List]:
    if df is None or len(df) == 0:
        raise ValueError("Split has zero windows.")
    req = {"trip_id", "seg_id"}
    if not req.issubset(df.columns):
        raise KeyError(f"Windows missing required columns: {sorted(req - set(df.columns))}")
    if masks.shape[0] != len(df):
        raise ValueError(f"masks length {masks.shape[0]} != windows rows {len(df)}")

    # Build (N,C,L) tensor; enforce same L
    arrs = [np.stack(df[ch].to_numpy(), axis=0) for ch in channels]  # each (N,L)
    Ls = [a.shape[-1] for a in arrs]
    if len(set(Ls)) != 1:
        raise ValueError(f"Inconsistent window length across channels: {Ls}")
    X = np.stack(arrs, axis=1).astype(np.float32, copy=False)  # (N,C,L)

    if masks.ndim != 3 or masks.shape[1] != len(channels) or masks.shape[2] != Ls[0]:
        raise ValueError(f"masks shape {masks.shape} incompatible with X {(len(df), len(channels), Ls[0])}")
    M = (1.0 - masks.astype(np.float32, copy=False))  # 1 for valid
    return X, M, df["trip_id"].tolist(), df["seg_id"].tolist()

@torch.no_grad()
def compute_segment_embeddings(
    encoder: nn.Module, X: np.ndarray, M: np.ndarray, trips: List, segs: List,
    device: str = "cpu", batch_size: int = 512
) -> Tuple[np.ndarray, List, List]:
    encoder.eval()
    Zw = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).to(device)
        mb = torch.from_numpy(M[i:i+batch_size]).to(device)
        zb = encoder(xb, mb).detach().cpu().numpy()  # (B,D), unit-norm
        Zw.append(zb)
    Zw = np.concatenate(Zw, axis=0)  # (Nw,D)

    # Aggregate to segments by mean (optionally weighted by window mask coverage)
    dfw = pd.DataFrame({"trip_id": trips, "seg_id": segs})
    dfw["w_embed"] = [z.astype(np.float32) for z in Zw]
    seg_map: Dict[Tuple, List[int]] = {}
    for idx, row in dfw.iterrows():
        key = (row.trip_id, row.seg_id)
        seg_map.setdefault(key, []).append(idx)

    seg_embeds, seg_trips, seg_segs = [], [], []
    for (t, s), idxs in seg_map.items():
        Zs = np.stack([dfw["w_embed"].iat[k] for k in idxs], axis=0).astype(np.float32)
        m = Zs.mean(axis=0)
        m = m / (np.linalg.norm(m) + 1e-8)
        seg_embeds.append(m)
        seg_trips.append(t); seg_segs.append(s)
    return np.stack(seg_embeds, axis=0), seg_trips, seg_segs

# =======================
# Robust trimmed-mean + shrinkage + residual + stability
# =======================
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def _stability(prev: List[np.ndarray]) -> float:
    n = len(prev)
    if n < 2: return 0.0
    vals = []
    for i in range(n):
        for j in range(i+1, n):
            vals.append(_cos(prev[i], prev[j]))
    if not vals: return 0.0
    arr = np.clip(np.array(vals, dtype=np.float32), -1.0, 1.0)
    return float(arr.mean())

def _robust_mean(prev_list: List[np.ndarray]) -> np.ndarray:
    n = len(prev_list)
    if n == 0:
        return None
    V = np.stack(prev_list, axis=0).astype(np.float32)
    mu0 = V.mean(axis=0); mu0 = mu0 / (np.linalg.norm(mu0) + 1e-8)
    if n >= 5 and TRIM_FRAC > 0.0:
        cos_to_mu = (V @ mu0) / (np.linalg.norm(V, axis=1) + 1e-8)
        k = max(1, min(n, int(round(n * (1.0 - TRIM_FRAC)))))
        keep = np.argsort(cos_to_mu)[-k:]
        V = V[keep]
    mu = V.mean(axis=0); mu = mu / (np.linalg.norm(mu) + 1e-8)
    return mu.astype(np.float32)

def _shrink(mu: Optional[np.ndarray], proto: Optional[np.ndarray], n: int, dim: int) -> np.ndarray:
    if n <= 0:
        return (proto.copy() if proto is not None else np.zeros((dim,), dtype=np.float32))
    if mu is None:
        return (proto.copy() if proto is not None else np.zeros((dim,), dtype=np.float32))
    if proto is None or np.allclose(proto, 0.0):
        v = mu.copy()
    else:
        gamma = n / (n + SHRINK_LAMBDA)
        v = gamma * mu + (1.0 - gamma) * proto
    v = v / (np.linalg.norm(v) + 1e-8)
    return v.astype(np.float32)

def build_causal_fps_for_split(
    seg_Z: np.ndarray, seg_trips: List, seg_segs: List,
    class_map: Dict[object, object], class_means: Dict[object, np.ndarray],
    class_col: str, split: str
) -> pd.DataFrame:
    df = pd.DataFrame({"trip_id": seg_trips, "seg_id": seg_segs})
    df["seg_embed"] = [z.astype(np.float32) for z in seg_Z]
    df[class_col] = [class_map.get(t, None) for t in df["trip_id"]]
    df = df.sort_values(["trip_id", "seg_id"]).reset_index(drop=True)

    rows = []
    for trip_id, g in df.groupby("trip_id", sort=False):
        g = g.sort_values("seg_id").reset_index(drop=True)
        cls = g[class_col].iloc[0]
        proto = class_means.get(cls, None)
        prev_list: List[np.ndarray] = []

        for idx, r in g.iterrows():
            seg_id = int(r["seg_id"])
            fp_coldstart = 1 if idx == 0 else 0
            n_prior = idx

            if fp_coldstart:
                fp_vec = proto.copy() if proto is not None else np.zeros_like(r["seg_embed"], dtype=np.float32)
                fp_res = np.zeros_like(fp_vec, dtype=np.float32)
                stab = 0.0
            else:
                mu = _robust_mean(prev_list)                  # robust mean of prior segs
                fp_vec = _shrink(mu, proto, n_prior, len(mu)) # shrink for small n
                if proto is not None:
                    res = fp_vec - proto
                    nr = float(np.linalg.norm(res))
                    fp_res = (res / nr).astype(np.float32) if nr > 0 else np.zeros_like(res, dtype=np.float32)
                else:
                    fp_res = np.zeros_like(fp_vec, dtype=np.float32)
                stab = _stability(prev_list)

            rows.append({
                "trip_id": trip_id, "seg_id": seg_id, "split": split, class_col: cls,
                "driver_fp": fp_vec.astype(np.float32),
                "fp_residual_vec": fp_res.astype(np.float32),
                "fp_count": n_prior, "fp_coldstart": fp_coldstart,
                "fp_stability": float(stab),
            })

            # update (only after emitting fp for this seg) → keeps causality
            prev_list.append(r["seg_embed"])

    return pd.DataFrame(rows)

def flatten_and_save(fps: pd.DataFrame, embed_dim: int, out_base: str) -> str:
    if fps.empty:
        return _save_parquet_or_csv(fps.drop(columns=["driver_fp","fp_residual_vec"], errors="ignore"), out_base)

    # sanity: consistent dims
    dims_fp  = [v.shape[0] for v in fps["driver_fp"]]
    dims_res = [v.shape[0] for v in fps["fp_residual_vec"]]
    if not all(d == dims_fp[0] == embed_dim for d in dims_fp) or not all(d == dims_res[0] == embed_dim for d in dims_res):
        raise ValueError("Inconsistent vector dimensions in fps dataframe.")

    mat_fp  = np.stack(fps["driver_fp"].to_list(), axis=0).astype(np.float32)
    mat_res = np.stack(fps["fp_residual_vec"].to_list(), axis=0).astype(np.float32)
    fp_cols   = [f"fp_{i:03d}" for i in range(embed_dim)]
    res_cols  = [f"fp_res_{i:03d}" for i in range(embed_dim)]

    base = fps.drop(columns=["driver_fp","fp_residual_vec"]).reset_index(drop=True)
    df_fp  = pd.DataFrame(mat_fp,  columns=fp_cols)
    df_res = pd.DataFrame(mat_res, columns=res_cols)
    out = pd.concat([base, df_fp, df_res], axis=1, copy=False)
    # keep a norm for sanity (starts with fp_ so pipelines can optionally use/ignore)
    out["fp_norm"] = np.sqrt((mat_fp**2).sum(axis=1)).astype(np.float32)
    # re-order for readability
    front = ["trip_id","seg_id","fp_count","fp_coldstart","split",CLASS_COL,"fp_stability","fp_norm"]
    ordered = front + [c for c in out.columns if c not in front]
    out = out[ordered]

    return _save_parquet_or_csv(out, out_base)

# =======================
# Main
# =======================
def main():
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUT_DIR or os.path.dirname(os.path.abspath(ARTIFACTS_PATH))
    model_path = MODEL_PATH or os.path.join(out_dir, "driver_embedder.pt")

    art = load_artifacts(ARTIFACTS_PATH)
    channels: List[str] = art["channels"]
    class_map = load_class_map_from_windows_corrected(WINDOWS_CORRECTED_PATH, CLASS_COL)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at: {model_path}")

    # Build encoder from checkpoint
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = _strip_module_prefix(state)
    hidden, embed_dim = _infer_hidden_and_dim_from_state(state)
    encoder = ConvGRUEncoder(in_ch=len(channels), embed_dim=embed_dim, hidden=hidden).to(device)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()

    present_splits = [s for s in ("train","val","test") if s in art and "windows" in art[s] and "masks" in art[s]]
    print(f"[splits] available: {present_splits}")

    # TRAIN-only prototypes
    class_means: Dict[object, np.ndarray] = {}
    if "train" in present_splits:
        Xtr, Mtr, trips_tr, segs_tr = _stack_split(art["train"]["windows"], art["train"]["masks"], channels)
        Ztr, t_tr, s_tr = compute_segment_embeddings(encoder, Xtr, Mtr, trips_tr, segs_tr, device=device, batch_size=BATCH_SIZE)
        df_tr = pd.DataFrame({"trip_id": t_tr, "seg_id": s_tr, "seg_embed": [z for z in Ztr]})
        df_tr[CLASS_COL] = [class_map.get(t, None) for t in df_tr["trip_id"]]
        for cls, g in df_tr.groupby(CLASS_COL, dropna=False):
            if cls is None or len(g)==0: continue
            Mcls = np.stack(g["seg_embed"].to_list(), axis=0).astype(np.float32)
            m = Mcls.mean(axis=0); m = m / (np.linalg.norm(m) + 1e-8)
            class_means[cls] = m.astype(np.float32)
        print(f"[class-means] built {len(class_means)} prototypes from TRAIN by '{CLASS_COL}'.")
    else:
        print("[warn] TRAIN split missing; class prototypes unavailable.")

    flattened = []
    for split in ["train","val","test"]:
        if split not in present_splits:
            print(f"[skip] split '{split}' not found in artifacts"); continue
        d = art[split]
        X, M, trips, segs = _stack_split(d["windows"], d["masks"], channels)
        Zs, seg_trips, seg_segs = compute_segment_embeddings(encoder, X, M, trips, segs, device=device, batch_size=BATCH_SIZE)

        fps = build_causal_fps_for_split(Zs, seg_trips, seg_segs, class_map, class_means, CLASS_COL, split)
        # quick sanity: per-trip counters monotonic & coldstart only first
        for _, g in fps.sort_values(["trip_id","seg_id"]).groupby("trip_id", sort=False):
            counts = g["fp_count"].to_numpy()
            starts = g["fp_coldstart"].to_numpy()
            assert counts[0] == 0 and np.all(counts[1:] == counts[:-1] + 1), "[sanity] fp_count not incremental"
            assert starts[0] == 1 and np.all(starts[1:] == 0), "[sanity] coldstart flags broken"

        # Save pickle
        _save_pickle(fps[["trip_id","seg_id","driver_fp","fp_residual_vec","fp_count","fp_coldstart","fp_stability","split",CLASS_COL]],
                     os.path.join(out_dir, f"fingerprints_{split}.pickle"))

        # Flatten & save
        if EXPORT_FLATTENED:
            out_base = os.path.join(out_dir, f"fingerprints_{split}")
            flattened.append(flatten_and_save(fps, embed_dim, out_base))

    # Merge all flattened
    if EXPORT_FLATTENED and MERGE_FLATTENED_ALL_SPLITS and flattened:
        dfs = []
        for p in flattened:
            if p.endswith(".parquet"):
                dfs.append(pd.read_parquet(p))
            else:
                dfs.append(pd.read_csv(p))
        merged = pd.concat(dfs, ignore_index=True)
        _save_parquet_or_csv(merged, os.path.join(out_dir, MERGED_BASENAME))

    print("[done] fingerprints_v3_compat exported.")

if __name__ == "__main__":
    main()
