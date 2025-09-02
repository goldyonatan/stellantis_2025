from __future__ import annotations
import os
import pickle
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

# ======= Paths (edit to your project layout) =======
ARTIFACTS_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\embedding_artifacts.pickle"
NEIGHBOR_CACHE_PATH = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\driver_emb_data\context_index\neighbor_cache.npz"
# ====================================================


# -----------------------------
# Artifact / cache I/O
# -----------------------------
def load_embedding_artifacts(path: str) -> Dict:
    """
    Load the pickle produced by driver_emb_data_preperation.py.

    Expected structure:
      {
        "channels": List[str],
        "train": {"windows": pd.DataFrame, "masks": np.ndarray (N,C,L)},
        "val":   {"windows": pd.DataFrame, "masks": np.ndarray (N,C,L)},
        "test":  {"windows": pd.DataFrame, "masks": np.ndarray (N,C,L)},
      }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifacts file not found: {path}")
    with open(path, "rb") as f:
        art = pickle.load(f)

    if "channels" not in art or not isinstance(art["channels"], (list, tuple)) or len(art["channels"]) == 0:
        raise ValueError("Artifacts missing non-empty 'channels' list.")
    ch = list(art["channels"])

    for split in ("train", "val", "test"):
        if split not in art or "windows" not in art[split] or "masks" not in art[split]:
            raise KeyError(f"Artifacts missing '{split}.windows' or '{split}.masks'")
        win_df = art[split]["windows"]
        masks = art[split]["masks"]
        if not isinstance(win_df, pd.DataFrame):
            raise TypeError(f"{split}.windows must be a pandas DataFrame")
        if not (isinstance(masks, np.ndarray) and masks.ndim == 3):
            raise ValueError(f"{split}.masks must be an ndarray of shape (N, C, L)")

        # channels present?
        missing = [c for c in ch if c not in win_df.columns]
        if missing:
            raise KeyError(f"{split}.windows is missing channel columns: {missing}")

        # shape consistency (N, C, L) with the DF and channel count
        if masks.shape[0] != len(win_df) or masks.shape[1] != len(ch):
            raise ValueError(
                f"{split}.masks shape {masks.shape} inconsistent with windows len {len(win_df)} "
                f"and channels {len(ch)}"
            )
    return art


def load_neighbor_cache(path: str, expected_n: int) -> np.ndarray:
    """
    Load neighbor cache built on TRAIN windows only (context-based), shape (N, K) int.
    Must align 1:1 with art['train']['windows'] row order.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Neighbor cache not found: {path}")
    with np.load(path) as z:
        for key in ("indices", "nbr_indices", "neighbors"):
            if key in z:
                idx = z[key]
                break
        else:
            raise KeyError("Neighbor cache npz missing 'indices' (or 'nbr_indices'/'neighbors').")
    if not (isinstance(idx, np.ndarray) and idx.ndim == 2 and np.issubdtype(idx.dtype, np.integer)):
        raise ValueError("Neighbor indices must be a 2D integer ndarray of shape (N, K).")
    if idx.shape[0] != expected_n:
        raise ValueError(f"Neighbor cache N={idx.shape[0]} does not match train windows N={expected_n}.")
    return idx.astype(np.int64, copy=False)


# -----------------------------
# Internal utilities
# -----------------------------
def _stack_channel_col(col: pd.Series, dtype=np.float32) -> np.ndarray:
    """Convert a DataFrame column (each cell is a 1D ndarray of length L) into a (N, L) array."""
    arr = col.to_numpy()
    if arr.ndim != 1:
        raise ValueError("Window column must be a 1D object array of per-row ndarrays.")
    return np.stack(arr.tolist(), axis=0).astype(dtype, copy=False)


def _pack_windows_tensor(windows_df: pd.DataFrame, channels: List[str], dtype=np.float32) -> np.ndarray:
    """Stack all channel columns into a dense (N, C, L) ndarray (float32)."""
    mats = []
    L_ref = None
    for c in channels:
        x2d = _stack_channel_col(windows_df[c], dtype=dtype)  # (N, L)
        if L_ref is None:
            L_ref = x2d.shape[1]
        elif x2d.shape[1] != L_ref:
            raise ValueError(f"Channel '{c}' has window length {x2d.shape[1]}, expected {L_ref}")
        mats.append(x2d[:, None, :])  # (N,1,L)
    X = np.concatenate(mats, axis=1)  # (N,C,L)
    return X


def _infer_seg_order(windows_df: pd.DataFrame) -> np.ndarray:
    """
    Return zero-based segment order per row, per trip.
    If seg_id is already an ordinal that matches temporal order within each trip,
    use seg_id (normalized to start at 0). Otherwise compute order from start_sample.
    """
    required = ("trip_id", "seg_id", "start_sample")
    for r in required:
        if r not in windows_df.columns:
            raise KeyError(f"'{r}' column required to determine segment order.")

    df = windows_df[["trip_id", "seg_id", "start_sample"]].copy()

    seg_min = (
        df.groupby(["trip_id", "seg_id"], sort=False)["start_sample"]
          .min()
          .reset_index()
          .rename(columns={"start_sample": "seg_start"})
    )

    def _trip_is_ordinal(g: pd.DataFrame) -> bool:
        seg_sorted_by_id = g.sort_values("seg_id")["seg_id"].to_list()
        seg_sorted_by_time = g.sort_values("seg_start")["seg_id"].to_list()
        return seg_sorted_by_id == seg_sorted_by_time

    all_ordinal = True
    for _, g in seg_min.groupby("trip_id", sort=False):
        if not _trip_is_ordinal(g):
            all_ordinal = False
            break

    if all_ordinal:
        seg_min["seg_order_within_trip"] = (
            seg_min["seg_id"] - seg_min.groupby("trip_id")["seg_id"].transform("min")
        ).astype(int)
    else:
        seg_min["seg_order_within_trip"] = (
            seg_min.groupby("trip_id")["seg_start"].rank(method="dense").astype(int) - 1
        )

    merged = windows_df.merge(
        seg_min.drop(columns=["seg_start"]),
        on=["trip_id", "seg_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["seg_order_within_trip"].isna().any():
        raise RuntimeError("Failed to assign segment order for some rows.")
    return merged["seg_order_within_trip"].to_numpy(np.int64, copy=False)


# -----------------------------
# Dataset
# -----------------------------
class EmbeddingWindowsDataset(Dataset):
    """
    Yields dicts with:
      - 'x':   FloatTensor (C, L)   — normalized windows with non-finites already filled with 0.0
      - 'mask':BoolTensor (C, L)    — True where original value was NON-FINITE (1 in artifacts)
      - id fields if present: 'trip_id', 'seg_id', 'win_id', 'start_sample'
      - 'index': int                — original row index in this split
    """
    def __init__(
        self,
        windows_df: pd.DataFrame,
        masks: np.ndarray,              # (N, C, L) uint8: 1 = NON-FINITE
        channels: List[str],
        *,
        precompute: bool = True,
        id_cols: Iterable[str] = ("trip_id", "seg_id", "win_id", "start_sample"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.windows_df = windows_df.reset_index(drop=True)
        self.channels = list(channels)
        self.id_cols = [c for c in id_cols if c in self.windows_df.columns]
        self.dtype = dtype

        N = len(self.windows_df)
        if not (isinstance(masks, np.ndarray) and masks.ndim == 3 and masks.shape[0] == N and masks.shape[1] == len(self.channels)):
            raise ValueError(f"masks must be (N,C,L) with N={N}, C={len(self.channels)}; got {masks.shape}")
        self._mask_np = np.ascontiguousarray(masks, dtype=np.uint8)
        self.C = masks.shape[1]
        self.L = masks.shape[2]

        self._x_np: Optional[np.ndarray] = None
        if precompute:
            X = _pack_windows_tensor(self.windows_df, self.channels, dtype=np.float32)  # (N,C,L)
            if X.shape != self._mask_np.shape:
                raise ValueError(f"Data shape {X.shape} != mask shape {self._mask_np.shape}")
            self._x_np = np.ascontiguousarray(X, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.windows_df)

    def _row_to_x(self, idx: int) -> np.ndarray:
        mats = []
        for c in self.channels:
            a = self.windows_df.at[idx, c]
            if not isinstance(a, np.ndarray) or a.ndim != 1:
                raise ValueError(f"Row {idx} channel '{c}' is not a 1D ndarray")
            mats.append(a[None, :])  # (1,L)
        x = np.concatenate(mats, axis=0)  # (C,L)
        return x.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> Dict:
        if self._x_np is not None:
            x = torch.from_numpy(self._x_np[idx])       # (C,L) float32
        else:
            x = torch.from_numpy(self._row_to_x(idx))   # (C,L) float32

        mask = torch.from_numpy(self._mask_np[idx].astype(np.bool_))  # (C,L) bool

        sample = {
            "x": x.to(self.dtype),
            "mask": mask,
            "index": idx,
        }
        for c in self.id_cols:
            sample[c] = self.windows_df.at[idx, c]
        return sample


# -----------------------------
# Triplet sampler with hard negatives
# -----------------------------
class TripletBatchSampler(Sampler[List[int]]):
    """
    Yields batches of indices arranged as [a0, p0, n0, a1, p1, n1, ...].
    - Anchor/Positive: same trip, DIFFERENT segment; prefer >= min_seg_gap apart.
    - Negative: from precomputed hard-neighbor cache (different trip).
    """
    def __init__(
        self,
        windows_df: pd.DataFrame,
        neighbor_idx: np.ndarray,   # (N, K) int64, train-only, excludes same-trip neighbors
        *,
        n_triplets_per_batch: int = 128,
        min_seg_gap: int = 1,
        top_hard_k: int = 20,
        drop_last: bool = True,
        seed: int = 42,
    ):
        super().__init__(None)
        required_cols = ("trip_id", "seg_id", "start_sample")
        for c in required_cols:
            if c not in windows_df.columns:
                raise KeyError(f"{c} column required in windows_df for triplet sampling.")
        self.df = windows_df.reset_index(drop=True)
        self.N = len(self.df)
        if neighbor_idx.shape[0] != self.N:
            raise ValueError("neighbor_idx must match number of TRAIN windows.")
        if top_hard_k < 1 or top_hard_k > neighbor_idx.shape[1]:
            raise ValueError(f"top_hard_k must be in [1, {neighbor_idx.shape[1]}].")

        self.n_triplets_per_batch = int(n_triplets_per_batch)
        self.min_seg_gap = int(min_seg_gap)
        self.top_hard_k = int(top_hard_k)
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        self.trip_ids = self.df["trip_id"].to_numpy()
        self.seg_ids  = self.df["seg_id"].to_numpy()
        self.seg_order = _infer_seg_order(self.df)  # <- robust order
        self.neigh = neighbor_idx  # (N,K)

        # Map trip -> indices, and per trip -> by segment
        self.trip_to_idx: Dict[object, np.ndarray] = {}
        self.trip_to_seg_to_idx: Dict[object, Dict[object, np.ndarray]] = {}
        self.trip_to_seg_orders: Dict[object, Dict[object, int]] = {}

        for t, grp in self.df.groupby("trip_id", sort=False):
            idxs = grp.index.to_numpy(np.int64, copy=False)
            self.trip_to_idx[t] = idxs
            seg_map = {}
            order_map = {}
            for seg, g2 in grp.groupby("seg_id", sort=False):
                seg_idxs = g2.index.to_numpy(np.int64, copy=False)
                seg_map[seg] = seg_idxs
                order_map[seg] = int(self.seg_order[seg_idxs[0]])
            self.trip_to_seg_to_idx[t] = seg_map
            self.trip_to_seg_orders[t] = order_map

        # Precompute positive candidates (far-preferred, then loose fallback)
        self.pos_candidates_far: List[np.ndarray] = [None] * self.N
        self.pos_candidates_loose: List[np.ndarray] = [None] * self.N

        for i in range(self.N):
            t = self.trip_ids[i]
            seg_anchor = self.seg_ids[i]
            order_anchor = int(self.seg_order[i])

            cand_far: List[int] = []
            cand_loose: List[int] = []
            for seg, seg_idxs in self.trip_to_seg_to_idx[t].items():
                if seg == seg_anchor:
                    continue
                if abs(self.trip_to_seg_orders[t][seg] - order_anchor) >= self.min_seg_gap:
                    cand_far.extend(seg_idxs.tolist())
                else:
                    cand_loose.extend(seg_idxs.tolist())
            self.pos_candidates_far[i] = np.asarray(cand_far, dtype=np.int64) if cand_far else np.empty(0, dtype=np.int64)
            self.pos_candidates_loose[i] = np.asarray(cand_loose, dtype=np.int64) if cand_loose else np.empty(0, dtype=np.int64)

        # Eligible anchors: have at least one positive candidate
        self.eligible_anchor_idx = np.where(
            np.array([len(self.pos_candidates_far[i]) + len(self.pos_candidates_loose[i]) > 0 for i in range(self.N)])
        )[0].astype(np.int64)

        total_triplets = len(self.eligible_anchor_idx)
        self._length = total_triplets // self.n_triplets_per_batch if self.drop_last else int(np.ceil(total_triplets / self.n_triplets_per_batch))

    def __len__(self) -> int:
        return max(self._length, 0)

    def __iter__(self):
        anchors = self.rng.permutation(self.eligible_anchor_idx)
        ptr = 0
        for _ in range(len(self)):
            batch_idx: List[int] = []
            while len(batch_idx) < 3 * self.n_triplets_per_batch and ptr < len(anchors):
                a = int(anchors[ptr]); ptr += 1

                # --- positive ---
                pos_pool = self.pos_candidates_far[a]
                if pos_pool.size == 0:
                    pos_pool = self.pos_candidates_loose[a]
                if pos_pool.size == 0:
                    continue
                p = int(self.rng.choice(pos_pool))

                # --- hard negative ---
                neighs = self.neigh[a, :self.top_hard_k]
                good = neighs[(neighs != a) & (self.trip_ids[neighs] != self.trip_ids[a]) & (neighs != p)]
                if good.size == 0:
                    neighs = self.neigh[a]
                    good = neighs[(neighs != a) & (self.trip_ids[neighs] != self.trip_ids[a]) & (neighs != p)]
                if good.size == 0:
                    other_pool = np.where(self.trip_ids != self.trip_ids[a])[0]
                    if other_pool.size == 0:
                        continue
                    n = int(self.rng.choice(other_pool))
                else:
                    n = int(self.rng.choice(good))

                batch_idx.extend([a, p, n])

            if len(batch_idx) == 3 * self.n_triplets_per_batch:
                yield batch_idx
            elif not self.drop_last and len(batch_idx) >= 3:
                trim = (len(batch_idx) // 3) * 3
                yield batch_idx[:trim]


# -----------------------------
# Collate: turn flat [a,p,n,...] into triplet tensors
# -----------------------------
def triplet_collate(samples: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Expects the DataLoader to deliver samples in groups of 3: (anchor, positive, negative).
    Produces dict with x_a/x_p/x_n and mask_a/mask_p/mask_n stacked along batch dim.
    """
    if len(samples) % 3 != 0:
        raise ValueError("Triplet collate expects a multiple of 3 samples (a,p,n groups).")

    xs    = torch.stack([s["x"] for s in samples], dim=0)      # (3B, C, L)
    masks = torch.stack([s["mask"] for s in samples], dim=0)   # (3B, C, L)

    B3, C, L = xs.shape
    B = B3 // 3
    xs    = xs.view(B, 3, C, L)
    masks = masks.view(B, 3, C, L)

    batch = {
        "x_a": xs[:, 0],
        "x_p": xs[:, 1],
        "x_n": xs[:, 2],
        "mask_a": masks[:, 0],
        "mask_p": masks[:, 1],
        "mask_n": masks[:, 2],
    }
    for key in ("trip_id", "seg_id", "index", "win_id", "start_sample"):
        if key in samples[0]:
            batch[key + "_a"] = [s[key] for i, s in enumerate(samples) if (i % 3) == 0]
            batch[key + "_p"] = [s[key] for i, s in enumerate(samples) if (i % 3) == 1]
            batch[key + "_n"] = [s[key] for i, s in enumerate(samples) if (i % 3) == 2]
    return batch


# -----------------------------
# Factories
# -----------------------------
def make_dataset(artifacts: Dict, split: str, *, precompute: bool = True) -> EmbeddingWindowsDataset:
    split = split.lower()
    if split not in ("train", "val", "test"):
        raise ValueError("split must be 'train', 'val', or 'test'")
    return EmbeddingWindowsDataset(
        windows_df=artifacts[split]["windows"],
        masks=artifacts[split]["masks"],
        channels=artifacts["channels"],
        precompute=precompute,
    )


def make_triplet_dataloader(
    artifacts: Dict,
    neighbor_cache_path: str,
    *,
    n_triplets_per_batch: int = 128,
    min_seg_gap: int = 1,
    top_hard_k: int = 20,
    num_workers: int = 0,
    pin_memory: bool = True,
    precompute: bool = True,
) -> Tuple[EmbeddingWindowsDataset, DataLoader]:
    """
    Train-time loader: returns triplets (anchor, positive, hard-negative).
    - Positives: same trip, different segment; prefer >= min_seg_gap apart (fallback: any other segment).
    - Negatives: from neighbor cache (different trip; hard within top_k, with safe fallbacks).
    """
    ds = make_dataset(artifacts, "train", precompute=precompute)
    train_df = artifacts["train"]["windows"]
    neigh_idx = load_neighbor_cache(neighbor_cache_path, expected_n=len(train_df))

    sampler = TripletBatchSampler(
        windows_df=train_df,
        neighbor_idx=neigh_idx,
        n_triplets_per_batch=n_triplets_per_batch,
        min_seg_gap=min_seg_gap,
        top_hard_k=top_hard_k,
        drop_last=True,
        seed=42,
    )
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=triplet_collate,
    )
    return ds, loader


def make_eval_dataloader(
    artifacts: Dict,
    split: str,
    *,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = True,
    precompute: bool = True,
    shuffle: bool = False,
) -> Tuple[EmbeddingWindowsDataset, DataLoader]:
    """
    Eval-time loader: plain batches for embedding extraction / retrieval metrics.
    """
    ds = make_dataset(artifacts, split, precompute=precompute)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle and (split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return ds, loader


# -----------------------------
# Optional quick self-check
# -----------------------------
if __name__ == "__main__":
    art = load_embedding_artifacts(ARTIFACTS_PATH)

    # Triplet loader for training (uses hard negatives)
    train_ds, train_triplet_dl = make_triplet_dataloader(
        art,
        NEIGHBOR_CACHE_PATH,
        n_triplets_per_batch=8,   # small for a quick check
        min_seg_gap=1,            # prefer at least 1 segment apart
        top_hard_k=20,
        num_workers=0,
    )
    batch = next(iter(train_triplet_dl))
    print("Triplet batch shapes:")
    print("x_a:", batch["x_a"].shape, "x_p:", batch["x_p"].shape, "x_n:", batch["x_n"].shape)
    print("mask_a:", batch["mask_a"].shape, "mask_p:", batch["mask_p"].shape, "mask_n:", batch["mask_n"].shape)

    # Optional: plain eval loader for validation
    val_ds, val_dl = make_eval_dataloader(art, "val", batch_size=32)
    val_batch = next(iter(val_dl))
    print("Eval batch shape:", val_batch["x"].shape, val_batch["mask"].shape)
