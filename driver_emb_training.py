import os
import sys
import time
from datetime import datetime
import math
import pickle
import random
import logging
import csv
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----------------------
# User-provided paths
# ----------------------
ARTIFACTS_PATH = "C:/Users/goldy/OneDrive/Documents/Y-DATA 24-25/Stellantis Project/End-to-End/driver_emb_data/embedding_artifacts.pickle"
NEIGHBOR_CACHE_PATH = "C:/Users/goldy/OneDrive/Documents/Y-DATA 24-25/Stellantis Project/End-to-End/driver_emb_data/context_index/neighbor_cache.npz"

# Will auto-resolve alongside NEIGHBOR_CACHE_PATH
CONTEXT_META_PATHS = [
    os.path.join(os.path.dirname(NEIGHBOR_CACHE_PATH), "context_meta.parquet"),
    os.path.join(os.path.dirname(NEIGHBOR_CACHE_PATH), "context_meta.csv"),
]

# ----------------------
# Hyperparameters
# ----------------------
SKIP_TRAIN = True  # True -> skip training loop; just (re)generate plots from latest CSV / checkpoint
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")  # unique id per training run for artifact/CSV isolation
SEED = 42
BATCH_SIZE = 128
EPOCHS = 30
LR = 2e-3
WEIGHT_DECAY = 1e-4
EMBED_DIM = 128
MIN_SEG_GAP = 2            # "at least X segments apart if possible"
HARD_NEG_TOPK = 10         # draw negatives from top-k of neighbor cache
HARD_NEGATIVE_RATIO = 0.8  # probability to use hard negatives; else random

LOG_INTERVAL = 100         # steps between detailed training logs
MARGIN = 0.3               # margin for triplet

# SupCon (InfoNCE) regularization across the batch to prevent collapse
USE_SUPCON = True
# cosine schedule for SupCon weight & temperature
SUPCON_WEIGHT_MAX = 0.5
SUPCON_WEIGHT_MIN = 0.1
SUPCON_TEMP_MAX = 0.2
SUPCON_TEMP_MIN = 0.07

# Batch-hard negatives: for each anchor, embed a small pool of negatives
USE_BATCH_HARD = True
NEG_POOL_SIZE = 4  # candidates sampled from neighbor cache per anchor (>=1)

# Lightweight data augmentation for robustness
AUG_ENABLE = True
AUG_TIME_MASK_P = 0.5       # probability to apply a temporal mask to anchor/positive
AUG_TIME_MASK_FRAC = 0.10   # ~10% of the window length
AUG_NOISE_STD = 0.01        # small Gaussian noise on anchor/positive

# ----------------------
# Utilities
# ----------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_int_or_none(x):
    try:
        return int(x)
    except Exception:
        return None

def find_existing_path(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# ----------------------
# Data loading helpers
# ----------------------

def load_artifacts(path: str):
    with open(path, "rb") as f:
        art = pickle.load(f)
    if not all(k in art for k in ["channels", "train", "val"]):
        raise KeyError("Artifacts must contain 'channels', 'train', 'val' (and optionally 'test').")
    for split in ["train", "val"]:
        if "windows" not in art[split] or "masks" not in art[split]:
            raise KeyError(f"Artifacts['{split}'] must contain 'windows' DataFrame and 'masks' ndarray.")
    return art

def df_id_tuple(row) -> Tuple:
    # Unique key available in both artifacts windows and context_meta
    return (row.get("trip_id"), row.get("seg_id"), row.get("win_id"))

def normalize_ids(x):
    if isinstance(x, (np.integer, np.floating)):
        x = x.item()
    return x

def build_index_mapping(windows_df: pd.DataFrame) -> Dict[Tuple, int]:
    mapping = {}
    for i, row in windows_df.iterrows():
        key = tuple(normalize_ids(v) for v in df_id_tuple(row))
        mapping[key] = i
    return mapping

def load_context_meta(meta_paths: List[str]) -> pd.DataFrame:
    p = find_existing_path(meta_paths)
    if p is None:
        raise FileNotFoundError("context_meta.parquet/csv not found next to neighbor_cache.npz")
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    else:
        return pd.read_csv(p)

def load_neighbor_cache(npz_path: str):
    data = np.load(npz_path)
    if "indices" not in data or "distances" not in data:
        raise KeyError("neighbor_cache.npz must contain 'indices' and 'distances'.")
    return data["indices"], data["distances"]

# ----------------------
# Dataset
# ----------------------

class TripletDataset(Dataset):
    """
    Creates on-the-fly (anchor, positive, negative-pool) tuples.

    Positives: same trip, different segment; try to ensure abs(seg - seg_anchor) >= MIN_SEG_GAP.
    If not possible, choose any other segment from the same trip.

    Negatives: pool (size NEG_POOL_SIZE) per anchor. With probability HARD_NEGATIVE_RATIO we draw
    from hard-neighbor cache; otherwise we draw random negatives from other trips (regularization).

    Returns (xa, xp, xneg_pool, ma, mp, mneg_pool, used_hard_neg_flag, cls_a, cls_p)
    where xneg_pool has shape (K, C, L) and masks same.
    """

    def __init__(
        self,
        windows_df: pd.DataFrame,
        masks: np.ndarray,
        channels: List[str],
        context_meta: pd.DataFrame,
        neighbor_indices: np.ndarray,
        min_seg_gap: int = 3,
        hard_topk: int = 10,
        hard_ratio: float = 0.8,
    ):
        assert len(windows_df) == masks.shape[0], "windows vs masks length mismatch"
        self.df = windows_df.reset_index(drop=True)
        self.masks = masks.astype(np.uint8, copy=False)  # (N, C, L), 1 == non-finite
        self.channels = channels
        self.N, self.C, self.L = self.masks.shape

        # Build mapping from (trip, seg, win) -> dataset idx
        self.idx_by_key = build_index_mapping(self.df)

        # Build structures to sample positives efficiently
        # map trip_id -> dict(segment_id -> list of indices)
        self.trip_to_seg_to_idxs: Dict[object, Dict[object, List[int]]] = {}
        for i, row in self.df.iterrows():
            trip = normalize_ids(row.get("trip_id"))
            seg = normalize_ids(row.get("seg_id"))
            self.trip_to_seg_to_idxs.setdefault(trip, {}).setdefault(seg, []).append(i)

        # class id per trip for SupCon
        unique_trips = list(self.trip_to_seg_to_idxs.keys())
        self.trip_to_class = {t: k for k, t in enumerate(unique_trips)}

        # Map order in context_meta -> dataset idx (for negatives)
        required = ["trip_id", "seg_id", "win_id"]
        if not all(c in context_meta.columns for c in required):
            raise KeyError(f"context_meta must include columns {required}")
        context_keys = [
            (normalize_ids(r["trip_id"]), normalize_ids(r["seg_id"]), normalize_ids(r["win_id"]))
            for _, r in context_meta.iterrows()
        ]

        self.ctx_row_to_dataset_idx: Dict[int, int] = {}
        miss = 0
        for ctx_row, key in enumerate(context_keys):
            ds_idx = self.idx_by_key.get(key, None)
            if ds_idx is not None:
                self.ctx_row_to_dataset_idx[ctx_row] = ds_idx
            else:
                miss += 1
        logging.info(
            f"[mapping] context_meta rows mapped to dataset: {len(self.ctx_row_to_dataset_idx):,}/{len(context_keys):,} "
            f"(miss={miss:,})"
        )

        # Build a per-dataset-index list of negative candidate dataset indices
        self.neg_cands: Dict[int, List[int]] = {i: [] for i in range(self.N)}
        self.all_indices = np.arange(self.N, dtype=np.int64)

        # precompute trip_id per dataset idx and class id
        self.trip_ids = [normalize_ids(r) for r in self.df["trip_id"].tolist()]
        self.class_ids = np.array([self.trip_to_class[t] for t in self.trip_ids], dtype=np.int64)

        K = neighbor_indices.shape[1]
        for ctx_row in range(neighbor_indices.shape[0]):
            anchor_ds = self.ctx_row_to_dataset_idx.get(ctx_row, None)
            if anchor_ds is None:
                continue
            neigh_rows = neighbor_indices[ctx_row, : min(hard_topk, K)]
            cand_ds = []
            for nr in neigh_rows:
                ds_idx = self.ctx_row_to_dataset_idx.get(int(nr), None)
                if ds_idx is None:
                    continue
                if self.trip_ids[ds_idx] == self.trip_ids[anchor_ds]:
                    continue
                cand_ds.append(ds_idx)
            if cand_ds:
                self.neg_cands[anchor_ds] = cand_ds

        with_hard = sum(1 for v in self.neg_cands.values() if len(v) > 0)
        logging.info(
            f"[negatives] hard-neighbor candidates available for {with_hard:,}/{self.N:,} anchors "
            f"({with_hard/self.N*100:.1f}%). Others will use random negatives."
        )

        # Build positive pools per anchor
        self.min_seg_gap = int(min_seg_gap)
        self.hard_topk = int(hard_topk)
        self.hard_ratio = float(hard_ratio)

        self.pos_pool: Dict[int, List[int]] = {}
        pos_counts: List[int] = []
        for i in range(self.N):
            trip = self.trip_ids[i]
            seg_i = normalize_ids(self.df.at[i, "seg_id"])
            seg_to_idxs = self.trip_to_seg_to_idxs.get(trip, {})
            cand = [j for s, idxs in seg_to_idxs.items() if s != seg_i for j in idxs]
            chosen = []
            if cand:
                seg_i_int = to_int_or_none(seg_i)
                if seg_i_int is not None:
                    good = []
                    for j in cand:
                        s = normalize_ids(self.df.at[j, "seg_id"])
                        sj = to_int_or_none(s)
                        if sj is not None and abs(sj - seg_i_int) >= self.min_seg_gap:
                            good.append(j)
                    chosen = good if good else cand
                else:
                    chosen = cand
            self.pos_pool[i] = chosen
            pos_counts.append(len(chosen))

        neg_counts = [len(self.neg_cands[i]) for i in range(self.N)]
        def pctl(a, q):
            a = np.asarray(a)
            return float(np.percentile(a, q)) if len(a) else 0.0

        logging.info(
            "[dataset] N=%d, C=%d, L=%d | anchors with >=1 positive: %d (%.1f%%) | "
            "pos per anchor (min/med/p95/max)= %.0f/%.0f/%.0f/%.0f | "
            "hard neg per anchor (min/med/p95/max)= %.0f/%.0f/%.0f/%.0f" % (
                self.N, self.C, self.L,
                sum(c > 0 for c in pos_counts), 100.0 * sum(c > 0 for c in pos_counts)/self.N,
                min(pos_counts) if pos_counts else 0, pctl(pos_counts, 50), pctl(pos_counts, 95), max(pos_counts) if pos_counts else 0,
                min(neg_counts) if neg_counts else 0, pctl(neg_counts, 50), pctl(neg_counts, 95), max(neg_counts) if neg_counts else 0,
            )
        )

        if sum(c > 0 for c in pos_counts) < self.N:
            logging.warning("[positives] Some anchors have no positives (single-segment trips); they are excluded from sampling.")

        # Precompute valid anchor indices (those with at least one positive)
        self.valid_anchors = np.array([i for i in range(self.N) if len(self.pos_pool[i]) > 0], dtype=np.int64)
        logging.info(f"[dataset] valid anchors with positives={len(self.valid_anchors):,}")

        # Pre-stack X tensors for fast access (C,L)
        Xs = []
        for ch in self.channels:
            arrs = self.df[ch].to_numpy()
            Xs.append(np.stack(arrs, axis=0))
        self.X = np.stack(Xs, axis=1).astype(np.float32, copy=False)  # (N, C, L)

        # Finite mask (1=finite, 0=nonfinite)
        self.finite_mask = 1.0 - self.masks.astype(np.float32, copy=False)

    def __len__(self):
        return len(self.valid_anchors)

    def __getitem__(self, idx: int):
        a_idx = int(self.valid_anchors[idx])

        # sample positive
        pos_list = self.pos_pool[a_idx]
        p_idx = int(random.choice(pos_list))

        # sample negative pool (mix hard/random per HARD_NEGATIVE_RATIO)
        use_hard = (random.random() < self.hard_ratio) and (len(self.neg_cands[a_idx]) > 0)
        used_hard_flag = 1 if use_hard else 0
        neg_pool_idx: List[int] = []
        if use_hard:
            cand_list = self.neg_cands[a_idx]
            if len(cand_list) >= NEG_POOL_SIZE:
                neg_pool_idx = random.sample(cand_list, NEG_POOL_SIZE)
            else:
                neg_pool_idx = [random.choice(cand_list) for _ in range(NEG_POOL_SIZE)]
        else:
            while len(neg_pool_idx) < NEG_POOL_SIZE:
                cand = int(random.randrange(self.N))
                if cand != a_idx and self.trip_ids[cand] != self.trip_ids[a_idx]:
                    neg_pool_idx.append(cand)

        xa = self.X[a_idx]              # (C,L)
        xp = self.X[p_idx]
        ma = self.finite_mask[a_idx]    # (C,L)
        mp = self.finite_mask[p_idx]

        xn_pool = np.stack([self.X[j] for j in neg_pool_idx], axis=0)           # (K,C,L)
        mn_pool = np.stack([self.finite_mask[j] for j in neg_pool_idx], axis=0) # (K,C,L)

        cls_a = self.class_ids[a_idx]
        cls_p = self.class_ids[p_idx]
        return xa, xp, xn_pool, ma, mp, mn_pool, used_hard_flag, cls_a, cls_p

# ----------------------
# Model
# ----------------------

class ConvGRUEncoder(nn.Module):
    """
    Simple 1D CNN -> GRU -> projection encoder.
    Input shape: (B, C, L), mask: (B, C, L) with 1 for finite, 0 for nonfinite.
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
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, embed_dim),
        )

    def masked_avg_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, F, L), mask: (B, 1, L) with 1 for valid
        w = mask
        x = x * w
        denom = w.sum(dim=2).clamp_min(1e-6)  # (B,1)
        return (x.sum(dim=2) / denom)  # (B,F)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x)
        time_mask = (mask.sum(dim=1, keepdim=True) > 0).float()  # (B,1,L)
        x = self.conv(x)  # (B, 128, L)
        x_t = x.transpose(1, 2)  # (B, L, 128)
        out, _ = self.gru(x_t)   # (B, L, 2*hidden)
        out = out.transpose(1, 2)  # (B, 2*hidden, L)
        pooled = self.masked_avg_pool(out, time_mask)  # (B, 2*hidden)
        z = self.proj(pooled)                          # (B, D)
        z = F.normalize(z, p=2, dim=1)                # unit-norm for cosine
        return z

# ----------------------
# Loss & Metrics
# ----------------------

def triplet_cosine_loss(za: torch.Tensor, zp: torch.Tensor, zn: torch.Tensor, margin: float = MARGIN):
    """Triplet loss with cosine distance: d = 1 - cos_sim"""
    sim_ap = F.cosine_similarity(za, zp)
    sim_an = F.cosine_similarity(za, zn)
    losses = F.relu(margin - (sim_ap - sim_an))
    return losses.mean(), sim_ap.detach(), sim_an.detach(), (losses.detach() > 0).float()


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Supervised Contrastive (SupCon) loss over a batch of normalized embeddings.
    z: (N, D) L2-normalized
    labels: (N,) int class ids; positives share the same label.
    """
    z = F.normalize(z, p=2, dim=1)
    sim = torch.matmul(z, z.T) / max(temperature, 1e-6)  # (N,N)
    N = z.size(0)
    logits = sim - torch.eye(N, device=z.device) * 1e9
    labels = labels.view(-1, 1)
    mask_pos = (labels == labels.T).float() - torch.eye(N, device=z.device)
    denom = torch.logsumexp(logits, dim=1)
    logits_pos = logits + torch.log(mask_pos.clamp_min(1e-12))
    numer = torch.logsumexp(logits_pos, dim=1)
    has_pos = (mask_pos.sum(dim=1) > 0).float()
    loss = -(numer - denom) * has_pos
    loss = loss.sum() / has_pos.clamp_min(1e-6).sum()
    return loss

@torch.no_grad()
def compute_val_metrics(
    encoder: nn.Module,
    X: np.ndarray,
    M: np.ndarray,
    device: str = "cpu",
    trip_ids: Optional[List] = None,
    seg_ids: Optional[List] = None,
) -> Dict[str, float]:
    """
    Window-level metrics:
      - recall@1 and recall@5 (same-trip & different-segment success preferred)
      - mean top-1 similarity to a positive (same trip, different segment) when available
      - mean top-1 similarity to a negative (different trip)
    """
    encoder.eval()
    N = X.shape[0]
    B = 512
    embeds = []
    for i in range(0, N, B):
        xb = torch.from_numpy(X[i : i + B]).to(device=device, dtype=torch.float32)
        mb = torch.from_numpy(M[i : i + B]).to(device=device, dtype=torch.float32)
        zb = encoder(xb, mb)
        embeds.append(zb.cpu())
    Z = torch.cat(embeds, dim=0).numpy().astype(np.float32)  # (N, D)

    Zt = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(Zt, Zt.T)
    np.fill_diagonal(sims, -np.inf)

    def recall_at_k(k: int) -> float:
        correct = 0
        for i in range(N):
            nn_idx = np.argpartition(-sims[i], kth=k - 1)[:k]
            ok = False
            if trip_ids is not None and seg_ids is not None:
                for j in nn_idx:
                    if trip_ids[j] == trip_ids[i] and seg_ids[j] != seg_ids[i]:
                        ok = True
                        break
                if not ok:
                    ok = any(trip_ids[j] == trip_ids[i] for j in nn_idx)
            else:
                ok = True
            if ok:
                correct += 1
        return correct / float(N)

    r1 = recall_at_k(1)
    r5 = recall_at_k(5)

    pos_sims = []
    neg_sims = []
    has_ids = trip_ids is not None and seg_ids is not None
    for i in range(N):
        if has_ids:
            same_trip = (np.array(trip_ids) == trip_ids[i])
            diff_trip = ~same_trip
            diff_seg = (np.array(seg_ids) != seg_ids[i])
            pos_mask = same_trip & diff_seg
            if pos_mask.any():
                pos_sims.append(np.max(sims[i][pos_mask]))
            if diff_trip.any():
                neg_sims.append(np.max(sims[i][diff_trip]))
        else:
            neg_sims.append(np.max(sims[i]))

    pos_top1 = float(np.mean(pos_sims)) if len(pos_sims) else float('nan')
    neg_top1 = float(np.mean(neg_sims)) if len(neg_sims) else float('nan')

    return {"recall@1": r1, "recall@5": r5, "pos_top1_sim": pos_top1, "neg_top1_sim": neg_top1}


@torch.no_grad()
def compute_segment_embeddings(
    encoder: nn.Module,
    X: np.ndarray,
    M: np.ndarray,
    trips: List,
    segs: List,
    device: str = "cpu",
) -> Tuple[np.ndarray, List, List]:
    """Aggregate window embeddings into one embedding per (trip, segment) by weighted mean, then L2-normalize."""
    encoder.eval()
    N = X.shape[0]
    B = 512
    embeds = []
    valid_frac = []
    for i in range(0, N, B):
        xb = torch.from_numpy(X[i : i + B]).to(device=device, dtype=torch.float32)
        mb = torch.from_numpy(M[i : i + B]).to(device=device, dtype=torch.float32)
        zb = encoder(xb, mb)
        embeds.append(zb.cpu().numpy())
        mb_np = mb.cpu().numpy()
        valid_frac.append(mb_np.mean(axis=(1, 2)))
    Z = np.concatenate(embeds, axis=0).astype(np.float32)
    W = np.concatenate(valid_frac, axis=0).astype(np.float32)  # (N,)

    keys = list(zip(trips, segs))
    seg_map: Dict[Tuple, List[int]] = {}
    for idx, k in enumerate(keys):
        seg_map.setdefault(k, []).append(idx)

    seg_embeds = []
    seg_trip_ids = []
    seg_seg_ids = []
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


@torch.no_grad()
def compute_seg_val_metrics(
    encoder: nn.Module,
    X: np.ndarray,
    M: np.ndarray,
    trips: List,
    segs: List,
    device: str = "cpu",
) -> Dict[str, float]:
    """Segment-level recall@k and similarity stats. Excludes segments that have no positive (no sibling segment in same trip)."""
    Zs, seg_trips, seg_segs = compute_segment_embeddings(encoder, X, M, trips, segs, device=device)
    S = Zs.shape[0]
    Zt = Zs / (np.linalg.norm(Zs, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(Zt, Zt.T)
    np.fill_diagonal(sims, -np.inf)

    seg_trips_arr = np.array(seg_trips)
    seg_segs_arr = np.array(seg_segs)
    eligible = []
    pos_masks = []
    neg_masks = []
    for i in range(S):
        same_trip = (seg_trips_arr == seg_trips_arr[i])
        diff_seg = (seg_segs_arr != seg_segs_arr[i])
        pos_mask = same_trip & diff_seg
        if pos_mask.any():
            eligible.append(i)
        pos_masks.append(pos_mask)
        neg_masks.append(~same_trip)

    if len(eligible) == 0:
        return {
            "seg_recall@1": float('nan'),
            "seg_recall@5": float('nan'),
            "seg_pos_top1_sim": float('nan'),
            "seg_neg_top1_sim": float('nan'),
            "seg_N": float(S),
            "seg_eval_N": 0.0,
        }

    def seg_recall_at_k(k: int) -> float:
        correct = 0
        for i in eligible:
            nn_idx = np.argpartition(-sims[i], kth=k - 1)[:k]
            if pos_masks[i][nn_idx].any():
                correct += 1
        return correct / float(len(eligible))

    r1 = seg_recall_at_k(1)
    r5 = seg_recall_at_k(5)

    pos_top1 = []
    neg_top1 = []
    for i in eligible:
        pos_vals = sims[i][pos_masks[i]]
        neg_vals = sims[i][neg_masks[i]]
        if pos_vals.size:
            pos_top1.append(pos_vals.max())
        if neg_vals.size:
            neg_top1.append(neg_vals.max())

    return {
        "seg_recall@1": float(np.mean(r1)),
        "seg_recall@5": float(np.mean(r5)),
        "seg_pos_top1_sim": float(np.mean(pos_top1)) if len(pos_top1) else float('nan'),
        "seg_neg_top1_sim": float(np.mean(neg_top1)) if len(neg_top1) else float('nan'),
        "seg_N": float(S),
        "seg_eval_N": float(len(eligible)),
    }


def _pca2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def generate_cluster_viz(
    seg_Z: np.ndarray,
    seg_trips: List,
    seg_segs: List,
    out_dir: str,
    max_trips: int = 6,
    max_points_per_trip: int = 50,
) -> Tuple[str, str]:
    """
    Create two presentation-ready PCA scatter plots of segment embeddings:
      1) Per-trip view: select up to `max_trips` trips with the most segments; color by trip.
      2) Unsupervised view: KMeans clusters on the same subset; color by cluster id.
    Returns paths to the two saved PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)
    trips = np.array(seg_trips)
    segs = np.array(seg_segs)
    unique, counts = np.unique(trips, return_counts=True)
    top_idx = np.argsort(-counts)[:max_trips]
    top_trips = set(unique[top_idx])
    mask = np.array([t in top_trips for t in trips])

    Z = seg_Z[mask]
    trips_sel = trips[mask]
    segs_sel = segs[mask]

    idx_keep = []
    for t in np.unique(trips_sel):
        idx_t = np.where(trips_sel == t)[0]
        if len(idx_t) > max_points_per_trip:
            idx_t = np.random.choice(idx_t, size=max_points_per_trip, replace=False)
        idx_keep.extend(idx_t.tolist())
    idx_keep = np.array(sorted(idx_keep))

    Z = Z[idx_keep]
    trips_sel = trips_sel[idx_keep]
    segs_sel = segs_sel[idx_keep]

    P = _pca2d(Z)

    plt.figure(figsize=(7.5, 5), dpi=200)
    for t in np.unique(trips_sel):
        m = trips_sel == t
        plt.scatter(P[m, 0], P[m, 1], s=24, alpha=0.85, label=f"trip {t}")
    plt.legend(ncol=2, fontsize=8)
    plt.title("Segment embeddings — colored by trip")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "segment_clusters_by_trip.png")
    plt.savefig(p1)
    plt.close()

    k = min(len(np.unique(trips_sel)), 8)
    if k < 2:
        k = min(4, max(2, Z.shape[0] // 10))

    try:
        from sklearn.cluster import KMeans  # type: ignore
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        cl = km.fit_predict(Z)
    except Exception:
        def _kmeans_np(X, k, iters=50):
            rng = np.random.default_rng(SEED)
            centers = X[rng.choice(len(X), size=k, replace=False)]
            for _ in range(iters):
                d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                labels = d.argmin(axis=1)
                new_centers = np.stack([X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j] for j in range(k)])
                if np.allclose(new_centers, centers):
                    break
                centers = new_centers
            return labels
        cl = _kmeans_np(Z, k)

    plt.figure(figsize=(7.5, 5), dpi=200)
    for j in np.unique(cl):
        m = cl == j
        plt.scatter(P[m, 0], P[m, 1], s=24, alpha=0.85, label=f"cluster {int(j)}")
    plt.legend(ncol=2, fontsize=8)
    plt.title("Segment embeddings — KMeans clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "segment_clusters_kmeans.png")
    plt.savefig(p2)
    plt.close()

    return p1, p2


# --------- Plotting utilities ---------

def _save_presentation_plots_from_history(history: List[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(out_dir, f"training_metrics_{RUN_ID}.csv"), index=False)
    df.to_csv(os.path.join(out_dir, "training_metrics_latest.csv"), index=False)

    # Loss & violation rate
    plt.figure(figsize=(8, 4.8), dpi=200)
    plt.plot(df['epoch'], df['train_loss_triplet'], label='Triplet loss')
    if 'train_loss_supcon' in df:
        plt.plot(df['epoch'], df['train_loss_supcon'], label='SupCon loss')
    plt.plot(df['epoch'], df['train_violation_rate'], label='Violation rate')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training losses & violation rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'driver_embedder_training_loss.png'))
    plt.close()

    # Validation recalls (window & segment)
    plt.figure(figsize=(8, 4.8), dpi=200)
    plt.plot(df['epoch'], df['val_recall@1'], label='WIN r@1')
    plt.plot(df['epoch'], df['val_recall@5'], label='WIN r@5')
    plt.plot(df['epoch'], df['seg_recall@1'], label='SEG r@1')
    plt.plot(df['epoch'], df['seg_recall@5'], label='SEG r@5')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Validation recalls (window & segment)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'driver_embedder_validation_recalls.png'))
    plt.close()

    # Save extra single-focus plots
    try:
        _save_combined_loss_only_from_df(df, out_dir)
        _save_seg_recalls_only_from_df(df, out_dir)
    except Exception:
        pass

def _supcon_weight_for_epoch(epoch: int, total_epochs: int = EPOCHS) -> float:
    """Recreate the cosine schedule used during training."""
    phase = (epoch - 1) / max(total_epochs, 1)
    return SUPCON_WEIGHT_MIN + (SUPCON_WEIGHT_MAX - SUPCON_WEIGHT_MIN) * (0.5 * (1 + math.cos(math.pi * phase)))


def _save_combined_loss_only_from_df(df: pd.DataFrame, out_dir: str):
    """Plot only the true training objective: triplet + scheduled SupCon."""
    # if SupCon was disabled or column missing, this gracefully reduces to triplet-only
    if USE_SUPCON and 'train_loss_supcon' in df.columns:
        weights = np.array([_supcon_weight_for_epoch(e) for e in df['epoch']])
        combined = df['train_loss_triplet'].to_numpy() + weights * df['train_loss_supcon'].to_numpy()
    else:
        combined = df['train_loss_triplet'].to_numpy()

    plt.figure(figsize=(8, 4.8), dpi=200)
    plt.plot(df['epoch'], combined, label='Combined loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training — combined loss')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'driver_embedder_combined_loss.png'))
    plt.close()


def _save_seg_recalls_only_from_df(df: pd.DataFrame, out_dir: str):
    """Plot only segment-level retrieval recalls."""
    plt.figure(figsize=(8, 4.8), dpi=200)
    plt.plot(df['epoch'], df['seg_recall@1'], label='SEG r@1')
    plt.plot(df['epoch'], df['seg_recall@5'], label='SEG r@5')
    plt.xlabel('Epoch'); plt.ylabel('Recall'); plt.title('Validation — segment recalls')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'driver_embedder_seg_recalls.png'))
    plt.close()


@torch.no_grad()
def _save_posneg_similarity_plot(
    encoder: nn.Module,
    X: np.ndarray, M: np.ndarray, trips: List, segs: List,
    out_dir: str, device: str = "cpu", max_pairs: int = 200_000
):
    """
    Presentation plot: distributions of cosine similarity for
    POS (same trip, different segment) vs NEG (different trip) segment pairs.
    """
    Zs, seg_trips, seg_segs = compute_segment_embeddings(encoder, X, M, trips, segs, device=device)
    Zs = Zs / (np.linalg.norm(Zs, axis=1, keepdims=True) + 1e-8)
    sims = (Zs @ Zs.T).astype(np.float32)
    S = sims.shape[0]
    # build upper-triangular index pairs
    iu = np.triu_indices(S, k=1)
    t = np.array(seg_trips); s = np.array(seg_segs)
    pos_mask = (t[iu[0]] == t[iu[1]]) & (s[iu[0]] != s[iu[1]])
    neg_mask = (t[iu[0]] != t[iu[1]])

    pos_vals = sims[iu][pos_mask]
    neg_vals = sims[iu][neg_mask]

    # optional subsample for speed/clarity
    rng = np.random.default_rng(42)
    if pos_vals.size > max_pairs:
        pos_vals = rng.choice(pos_vals, size=max_pairs, replace=False)
    if neg_vals.size > max_pairs:
        neg_vals = rng.choice(neg_vals, size=max_pairs, replace=False)

    plt.figure(figsize=(8, 4.8), dpi=200)
    bins = 60
    plt.hist(neg_vals, bins=bins, density=True, alpha=0.6, label='NEG (different trips)')
    plt.hist(pos_vals, bins=bins, density=True, alpha=0.6, label='POS (same trip, diff. segment)')
    plt.xlabel('Cosine similarity'); plt.ylabel('Density')
    plt.title('Embedding separates behavior — POS vs. NEG similarity (segments)')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'driver_embedder_posneg_similarity.png'))
    plt.close()

# --------- Training ---------

def train():
    set_seed(SEED)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    art = load_artifacts(ARTIFACTS_PATH)
    channels: List[str] = art["channels"]
    tr_df: pd.DataFrame = art["train"]["windows"].copy()
    tr_masks: np.ndarray = art["train"]["masks"].copy()
    va_df: pd.DataFrame = art["val"]["windows"].copy()
    va_masks: np.ndarray = art["val"]["masks"].copy()

    logging.info(f"Train windows: {len(tr_df):,}, masks: {tr_masks.shape}, channels: {len(channels)}")
    logging.info(f"Val   windows: {len(va_df):,}, masks: {va_masks.shape}")

    ctx_meta = load_context_meta(CONTEXT_META_PATHS)
    nn_indices, nn_dists = load_neighbor_cache(NEIGHBOR_CACHE_PATH)
    logging.info(f"Neighbor cache loaded: indices={nn_indices.shape}, distances={nn_dists.shape}")

    train_ds = TripletDataset(
        windows_df=tr_df,
        masks=tr_masks,
        channels=channels,
        context_meta=ctx_meta,
        neighbor_indices=nn_indices,
        min_seg_gap=MIN_SEG_GAP,
        hard_topk=HARD_NEG_TOPK,
        hard_ratio=HARD_NEGATIVE_RATIO,
    )

    def stack_split(df: pd.DataFrame, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List, List]:
        Xs = [np.stack(df[ch].to_numpy(), axis=0) for ch in channels]
        X = np.stack(Xs, axis=1).astype(np.float32, copy=False)
        M = (1.0 - masks.astype(np.float32, copy=False))
        trips = df["trip_id"].tolist() if "trip_id" in df.columns else None
        segs = df["seg_id"].tolist() if "seg_id" in df.columns else None
        return X, M, trips, segs

    va_X, va_M, va_trips, va_segs = stack_split(va_df, va_masks)

    encoder = ConvGRUEncoder(in_ch=len(channels), embed_dim=EMBED_DIM, hidden=128).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.1)

    encoder.train()

    def collate(batch):
        xa, xp, xn_pool, ma, mp, mn_pool, used_hard, cls_a, cls_p = zip(*batch)
        return (
            torch.from_numpy(np.stack(xa)).float(),
            torch.from_numpy(np.stack(xp)).float(),
            torch.from_numpy(np.stack(xn_pool)).float(),
            torch.from_numpy(np.stack(ma)).float(),
            torch.from_numpy(np.stack(mp)).float(),
            torch.from_numpy(np.stack(mn_pool)).float(),
            torch.tensor(used_hard, dtype=torch.int32),
            torch.tensor(cls_a, dtype=torch.int64),
            torch.tensor(cls_p, dtype=torch.int64),
        )

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate, drop_last=True)

    save_dir = os.path.dirname(ARTIFACTS_PATH)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "driver_embedder.pt")  # best-by-SEG checkpoint

    # Per-run CSV (avoids header/column mismatches across runs)
    csv_path = os.path.join(save_dir, f"training_metrics_{RUN_ID}.csv")
    history: List[dict] = []

    logging.info(
        f"[config] epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, weight_decay={WEIGHT_DECAY}, embed_dim={EMBED_DIM}, "
        f"min_seg_gap={MIN_SEG_GAP}, hard_topk={HARD_NEG_TOPK}, hard_ratio={HARD_NEGATIVE_RATIO}, use_supcon={USE_SUPCON}, "
        f"supcon_w_max={SUPCON_WEIGHT_MAX}, supcon_w_min={SUPCON_WEIGHT_MIN}, margin={MARGIN}, use_batch_hard={USE_BATCH_HARD}, "
        f"neg_pool_size={NEG_POOL_SIZE}, aug={AUG_ENABLE}"
    )

    best_seg_r1 = -1.0
    best_seg_r5_at_best = -1.0
    best_win_r1_at_best = -1.0
    best_win_r5_at_best = -1.0
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        num_batches = len(loader)
        logging.info(f"[epoch {epoch:02d}] starting — batches this epoch: {num_batches}")
        t0 = time.time()
        encoder.train()

        epoch_triplet_sum = 0.0
        epoch_supcon_sum = 0.0
        epoch_steps = 0
        epoch_violation_sum = 0.0
        epoch_ap_sum = 0.0
        epoch_an_sum = 0.0
        total_hard_used = 0
        total_rand_used = 0

        int_total_sum = 0.0
        int_violation_sum = 0.0
        int_ap_sum = 0.0
        int_an_sum = 0.0
        int_steps = 0

        for step, batch in enumerate(loader, start=1):
            xa, xp, xn_pool, ma, mp, mn_pool, used_hard, cls_a, cls_p = batch
            xa, xp = xa.to(device), xp.to(device)
            ma, mp = ma.to(device), mp.to(device)
            xn_pool = xn_pool.to(device)
            mn_pool = mn_pool.to(device)
            cls_a = cls_a.to(device)
            cls_p = cls_p.to(device)

            if AUG_ENABLE:
                if AUG_NOISE_STD > 0:
                    xa = xa + torch.randn_like(xa) * AUG_NOISE_STD
                    xp = xp + torch.randn_like(xp) * AUG_NOISE_STD
                Bsz, C, L = xa.shape
                if AUG_TIME_MASK_P > 0 and AUG_TIME_MASK_FRAC > 0:
                    span = max(1, int(L * AUG_TIME_MASK_FRAC))
                    for i in range(Bsz):
                        if random.random() < AUG_TIME_MASK_P:
                            s = random.randint(0, L - span)
                            xa[i, :, s:s + span] = 0.0
                            ma[i, :, s:s + span] = 0.0
                        if random.random() < AUG_TIME_MASK_P:
                            s = random.randint(0, L - span)
                            xp[i, :, s:s + span] = 0.0
                            mp[i, :, s:s + span] = 0.0

            xa = xa * ma
            xp = xp * mp
            xn_pool = xn_pool * mn_pool

            za = encoder(xa, ma)
            zp = encoder(xp, mp)

            Bsz, K, C, L = xn_pool.shape
            xn_flat = xn_pool.view(Bsz * K, C, L)
            mn_flat = mn_pool.view(Bsz * K, C, L)
            zn_flat = encoder(xn_flat, mn_flat)
            zn_pool = zn_flat.view(Bsz, K, -1)

            za_n = F.normalize(za, p=2, dim=1)
            zp_n = F.normalize(zp, p=2, dim=1)
            zn_n = F.normalize(zn_pool, p=2, dim=2)

            sim_ap = (za_n * zp_n).sum(dim=1)
            sim_an_pool = torch.einsum('bd,bkd->bk', za_n, zn_n)

            margin_vec = sim_ap.unsqueeze(1) - MARGIN
            violate_mask = sim_an_pool > margin_vec
            max_all, idx_all = sim_an_pool.max(dim=1)
            masked = sim_an_pool.masked_fill(~violate_mask, float('-inf'))
            max_viol, idx_viol = masked.max(dim=1)
            use_viol = torch.isfinite(max_viol)
            chosen_idx = torch.where(use_viol, idx_viol, idx_all)

            idx_exp = chosen_idx.view(-1, 1, 1).expand(-1, 1, zn_pool.size(2))
            zn = zn_pool.gather(1, idx_exp).squeeze(1)

            loss_triplet, sim_ap_det, sim_an_det, violations = triplet_cosine_loss(za, zp, zn, margin=MARGIN)

            if USE_SUPCON:
                phase = (epoch - 1) / max(EPOCHS, 1)
                supcon_w = SUPCON_WEIGHT_MIN + (SUPCON_WEIGHT_MAX - SUPCON_WEIGHT_MIN) * (0.5 * (1 + math.cos(math.pi * phase)))
                supcon_temp = SUPCON_TEMP_MIN + (SUPCON_TEMP_MAX - SUPCON_TEMP_MIN) * (0.5 * (1 + math.cos(math.pi * phase)))
                z_sup = torch.cat([za, zp], dim=0)
                y_sup = torch.cat([cls_a, cls_p], dim=0)
                loss_sup = supcon_loss(z_sup, y_sup, temperature=supcon_temp)
                loss = loss_triplet + supcon_w * loss_sup
            else:
                loss_sup = torch.tensor(0.0, device=device)
                loss = loss_triplet

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
            opt.step()

            bsz = xa.size(0)
            hard_used = int(used_hard.sum().item()) if hasattr(used_hard, 'sum') else 0
            total_hard_used += hard_used
            total_rand_used += (bsz - hard_used)

            epoch_triplet_sum += float(loss_triplet.item())
            epoch_supcon_sum += float(loss_sup.item())
            epoch_steps += 1
            epoch_violation_sum += float(violations.mean().item())
            epoch_ap_sum += float(sim_ap_det.mean().item())
            chosen_sim_an = float(torch.gather(sim_an_pool, 1, chosen_idx.view(-1, 1)).mean().item())
            epoch_an_sum += chosen_sim_an

            int_total_sum += float(loss.item())
            int_violation_sum += float(violations.mean().item())
            int_ap_sum += float(sim_ap_det.mean().item())
            int_an_sum += chosen_sim_an
            int_steps += 1

            if step % LOG_INTERVAL == 0:
                logging.info(
                    f"[epoch {epoch:02d}] step {step:05d} | total_loss={int_total_sum/int_steps:.4f} (triplet≈{epoch_triplet_sum/max(epoch_steps,1):.4f}, supcon≈{epoch_supcon_sum/max(epoch_steps,1):.4f}) | "
                    f"sim_ap={int_ap_sum/int_steps:.3f} sim_an={int_an_sum/int_steps:.3f} | viol_rate={int_violation_sum/int_steps:.3f} | "
                    f"hard:rand={total_hard_used}:{total_rand_used}"
                )
                int_total_sum = int_violation_sum = int_ap_sum = int_an_sum = 0.0
                int_steps = 0

        # ---- Validation (window & segment) ----
        val_win = compute_val_metrics(encoder, va_X, va_M, device=device, trip_ids=va_trips, seg_ids=va_segs)
        val_seg = compute_seg_val_metrics(encoder, va_X, va_M, va_trips, va_segs, device=device)
        dt = time.time() - t0

        epoch_loss_triplet = epoch_triplet_sum / max(epoch_steps, 1)
        epoch_loss_supcon = epoch_supcon_sum / max(epoch_steps, 1)
        epoch_violation = epoch_violation_sum / max(epoch_steps, 1)
        epoch_sim_ap = epoch_ap_sum / max(epoch_steps, 1)
        epoch_sim_an = epoch_an_sum / max(epoch_steps, 1)
        hard_ratio_used = total_hard_used / max(total_hard_used + total_rand_used, 1)

        logging.info(
            f"[epoch {epoch:02d}] train triplet={epoch_loss_triplet:.4f} supcon={epoch_loss_supcon:.4f} | sim_ap={epoch_sim_ap:.3f} sim_an={epoch_sim_an:.3f} | "
            f"viol_rate={epoch_violation:.3f} | hard_used={total_hard_used} ({hard_ratio_used*100:.1f}%) | "
            f"SEG r@1={val_seg['seg_recall@1']:.4f} r@5={val_seg['seg_recall@5']:.4f} (eval_N={int(val_seg['seg_eval_N'])}/{int(val_seg['seg_N'])}) | "
            f"WIN r@1={val_win['recall@1']:.4f} r@5={val_win['recall@5']:.4f} | {dt:.1f}s"
        )

        # Save per-epoch record in-memory (for plotting and CSV at end)
        history.append({
            "epoch": epoch,
            "train_loss_triplet": epoch_loss_triplet,
            "train_loss_supcon": epoch_loss_supcon,
            "train_violation_rate": epoch_violation,
            "train_sim_ap": epoch_sim_ap,
            "train_sim_an": epoch_sim_an,
            "hard_neg_ratio_used": hard_ratio_used,
            "val_recall@1": val_win['recall@1'],
            "val_recall@5": val_win['recall@5'],
            "val_pos_top1_sim": val_win['pos_top1_sim'],
            "val_neg_top1_sim": val_win['neg_top1_sim'],
            "seg_recall@1": val_seg['seg_recall@1'],
            "seg_recall@5": val_seg['seg_recall@5'],
            "seg_pos_top1_sim": val_seg['seg_pos_top1_sim'],
            "seg_neg_top1_sim": val_seg['seg_neg_top1_sim'],
            "seg_eval_N": val_seg['seg_eval_N'],
            "epoch_time_sec": dt,
            "lr": opt.param_groups[0].get('lr', LR),
            "use_supcon": int(USE_SUPCON),
            "supcon_weight_max": SUPCON_WEIGHT_MAX,
            "supcon_weight_min": SUPCON_WEIGHT_MIN,
            "margin": MARGIN,
        })

        # ---- Model selection: leading metric = segment-level r@1 ----
        if val_seg['seg_recall@1'] > best_seg_r1:
            best_seg_r1 = val_seg['seg_recall@1']
            best_seg_r5_at_best = val_seg['seg_recall@5']
            best_win_r1_at_best = val_win['recall@1']
            best_win_r5_at_best = val_win['recall@5']
            best_epoch = epoch
            torch.save(encoder.state_dict(), model_path)
            logging.info(
                f"[checkpoint] new BEST by SEG r@1={best_seg_r1:.4f} (r@5={best_seg_r5_at_best:.4f}) at epoch {epoch} → saved to {model_path}"
            )

        scheduler.step()

    # --------- End of training: report best, save plots, cluster viz ---------
    logging.info(
        f"[best] epoch={best_epoch} | SEG r@1={best_seg_r1:.4f} r@5={best_seg_r5_at_best:.4f} | "
        f"WIN r@1={best_win_r1_at_best:.4f} r@5={best_win_r5_at_best:.4f}"
    )
    print(
        f"BEST MODEL (by SEG) — epoch {best_epoch}: SEG r@1={best_seg_r1:.4f}, r@5={best_seg_r5_at_best:.4f} | "
        f"WIN r@1={best_win_r1_at_best:.4f}, r@5={best_win_r5_at_best:.4f}"
    )

    out_dir = os.path.dirname(ARTIFACTS_PATH)
    try:
        _save_presentation_plots_from_history(history, out_dir)
    except Exception as e:
        logging.warning(f"[plots] failed to save: {e}")

    # Reload best checkpoint and compute segment embeddings for clustering viz
    try:
        best_encoder = ConvGRUEncoder(in_ch=len(channels), embed_dim=EMBED_DIM, hidden=128).to(device)
        best_encoder.load_state_dict(torch.load(model_path, map_location=device))
        best_encoder.eval()
        seg_Z, seg_trips, seg_segs = compute_segment_embeddings(best_encoder, va_X, va_M, va_trips, va_segs, device=device)
        p1, p2 = generate_cluster_viz(seg_Z, seg_trips, seg_segs, out_dir)
        logging.info(f"[viz] saved cluster plots: {p1} | {p2}")
    except Exception as e:
        logging.warning(f"[best] reload/eval failed: {e}")
    
    # Extra presentation plot: POS vs NEG similarity distributions (segment level) 
    try:
        _save_posneg_similarity_plot(best_encoder, va_X, va_M, va_trips, va_segs, out_dir, device=device)
        logging.info("[viz] saved POS/NEG similarity distribution plot")
    except Exception as e:
        logging.warning(f"[viz] pos/neg similarity plot failed: {e}")


if __name__ == "__main__":
    if SKIP_TRAIN:
        # Just regenerate the single-focus plots from the latest CSV and the POS/NEG plot from the best checkpoint
        out_dir = os.path.dirname(ARTIFACTS_PATH)
        csv_latest = os.path.join(out_dir, "training_metrics_latest.csv")
        try:
            df_latest = pd.read_csv(csv_latest)
            _save_combined_loss_only_from_df(df_latest, out_dir)
            _save_seg_recalls_only_from_df(df_latest, out_dir)
            print(f"[plots] regenerated: combined loss + segment recalls → {out_dir}")
        except Exception as e:
            print(f"[plots] could not regenerate from CSV ({e}); run a training pass first.")

        # POS/NEG plot using best checkpoint, if available
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            art = load_artifacts(ARTIFACTS_PATH)
            channels = art["channels"]
            va_df = art["val"]["windows"].copy()
            va_masks = art["val"]["masks"].copy()
            Xs = [np.stack(va_df[ch].to_numpy(), axis=0) for ch in channels]
            va_X = np.stack(Xs, axis=1).astype(np.float32, copy=False)
            va_M = (1.0 - va_masks.astype(np.float32, copy=False))
            va_trips = va_df["trip_id"].tolist() if "trip_id" in va_df.columns else None
            va_segs = va_df["seg_id"].tolist() if "seg_id" in va_df.columns else None

            model_path = os.path.join(out_dir, "driver_embedder.pt")
            enc = ConvGRUEncoder(in_ch=len(channels), embed_dim=EMBED_DIM, hidden=128).to(device)
            enc.load_state_dict(torch.load(model_path, map_location=device))
            enc.eval()
            _save_posneg_similarity_plot(enc, va_X, va_M, va_trips, va_segs, out_dir, device=device)
            print(f"[plots] regenerated: POS/NEG similarity → {out_dir}")
        except Exception as e:
            print(f"[plots] POS/NEG similarity plot skipped ({e}).")
    else:
        train()

