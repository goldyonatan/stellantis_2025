#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual modeling — SOLO FP focus (v6).

Compares on TEST:
  a) base (t0 only)                          → R3_none
  b) base + fp SOLO residual (gated)         → R1_fp_solo_gated
  c) base + fp cluster-offset (gated)        → R1_fp_cluster_gated
  d) base + fp ensemble (reg + cluster)      → R1_fp_ens_gated
  e) base + prior residual (reference)       → R2_prior

Design to push SOLO fp:
  • Residual targets from OOF base (GroupKFold by trip_id) — no leakage
  • PLS latents V(fp) supervised on residual (Train-only for selection; Train+Val OOF for final)
  • Pure-fp residual learners on V and Poly2(V): Ridge or Huber + StandardScaler
  • Validation-tuned gating on (fp_count, fp_stability) and shrinkage λ in [0.3..1.0]
  • Optional cluster-offset on V (KMeans) with per-cluster residual means; gated & shrunk
  • Final ensemble blends reg + cluster via Val-tuned weight ∈ {0.3, 0.5, 0.7}

Outputs: End-to-End/model_soc_segment_residual_v6_fp_solo/
"""

from __future__ import annotations
import os, math, json, pickle, warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans

# --- Spherical KMeans override: normalize rows on the unit sphere for clustering ---
class KMeans(KMeans):  # type: ignore[no-redef]
    def fit(self, X, y=None, sample_weight=None):
        import numpy as np
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return super().fit(Xn, y=y, sample_weight=sample_weight)
    def predict(self, X):
        import numpy as np
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return super().predict(Xn)


from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, HuberRegressor

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ---------------- Paths ----------------
BASE_DIR = Path(r"C:/Users/goldy/OneDrive/Documents/Y-DATA 24-25/Stellantis Project/End-to-End")

CLEANED_DATA_PATH = BASE_DIR / "clean_df" / "clean_df.parquet"
SEGMENTATION_DICT_PATH = BASE_DIR / "segmentation_dict" / "trips_seg_dict_v7.pickle"

DRIVER_EMB_DIR = BASE_DIR / "driver_emb_data"
FINGERPRINTS_ALL = None
for name in ["fingerprints_all.parquet", "fingerprints_all.csv"]:
    p = DRIVER_EMB_DIR / name
    if p.exists():
        FINGERPRINTS_ALL = p
if FINGERPRINTS_ALL is None:
    FINGERPRINTS_ALL = DRIVER_EMB_DIR / "fingerprints_all.parquet"

NONBEHAV_T0_CURATED_PATH = BASE_DIR / "nonbehavioral_t0_features" / "t0_nonbehavioral_features_curated.parquet"
BEHAVIOR_PRIOR_AGGS_PATH = BASE_DIR / "segment_behavior_agg" / "seg_behavior_prior_aggs.parquet"

OUT_DIR = BASE_DIR / "model_soc_segment_residual_v6_fp_solo"
PLOTS_DIR = OUT_DIR / "plots"
PRED_DIR = OUT_DIR / "predictions"
SELECTION_DIR = OUT_DIR / "selection_reports"

TRIP_ID_COL = "trip_id"
SEG_ID_COL = "seg_id"
SOC_COL = "current_soc"
TARGET_COL = "soc_end_pct"

RANDOM_STATE = 42
N_SPLITS_OOF = 5

# ---------------- Utils ----------------
def _ensure_dirs():
    for p in [OUT_DIR, PLOTS_DIR, PRED_DIR, SELECTION_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _print_shape(df: pd.DataFrame, name: str):
    print(f"{name:>30s}: {df.shape[0]:,} rows x {df.shape[1]:,} cols")

def _rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))

# ---------------- Target construction ----------------
def build_segment_targets(clean_path: Path, segdict_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(clean_path)
    if TRIP_ID_COL not in df.columns or SOC_COL not in df.columns:
        raise KeyError(f"clean_df must contain '{TRIP_ID_COL}' and '{SOC_COL}'")
    time_col = "timestamp" if "timestamp" in df.columns else None
    if time_col:
        df = df.sort_values([TRIP_ID_COL, time_col]).reset_index(drop=True)
    else:
        df = df.sort_values([TRIP_ID_COL]).reset_index(drop=True)

    seg_map = _load_pickle(segdict_path)
    rows = []
    for trip_id, g in df.groupby(TRIP_ID_COL, sort=False):
        labels = seg_map.get(trip_id, None)
        if labels is None:
            continue
        labels = np.asarray(labels)
        if len(labels) != len(g):
            print(f"[warn] seg_dict mismatch for trip {trip_id}: {len(labels)} vs {len(g)} -> skip")
            continue
        if (labels > 0).any():
            seg_ids = np.unique(labels[labels > 0])
        else:
            continue
        soc_vals = g[SOC_COL].to_numpy()
        for sid in seg_ids:
            idxs = np.where(labels == sid)[0]
            if idxs.size == 0:
                continue
            end_idx = int(idxs[-1])
            val = float(soc_vals[end_idx]) if pd.notna(soc_vals[end_idx]) else np.nan
            rows.append({TRIP_ID_COL: trip_id, SEG_ID_COL: int(sid), TARGET_COL: val})
    tgt = pd.DataFrame(rows)
    _print_shape(tgt, "Built targets")
    return tgt

# ---------------- Feature loading ----------------
def load_fingerprints(merged_all: Path) -> pd.DataFrame:
    def _read_any(p: Path) -> pd.DataFrame:
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    if merged_all.exists():
        df = _read_any(merged_all)
    else:
        base = merged_all.parent
        parts = []
        for split in ["train", "val", "test"]:
            for ext in [".parquet", ".csv"]:
                p = base / f"fingerprints_{split}{ext}"
                if p.exists():
                    d = _read_any(p)
                    if "split" not in d.columns: d["split"] = split
                    parts.append(d); break
        if not parts:
            raise FileNotFoundError(f"No fingerprints files found under {merged_all.parent}")
        df = pd.concat(parts, ignore_index=True)

    keep = [TRIP_ID_COL, SEG_ID_COL, "split", "fp_count", "fp_coldstart", "fp_stability"]
    fp_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("fp_")]
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "timedelta64[ns]"]).columns.tolist()
    df = df.drop(columns=dt_cols, errors="ignore")
    df = df[[c for c in keep + fp_cols if c in df.columns]].copy()
    _print_shape(df, "Fingerprints (strict keep)")
    return df

def load_prior_aggs(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    keep = [TRIP_ID_COL, SEG_ID_COL, "prior_seg_count"] + [c for c in df.columns if isinstance(c, str) and c.startswith("prior_")]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    _print_shape(df, "Prior aggs (strict keep)")
    return df

def load_t0_curated(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if TRIP_ID_COL not in df.columns or SEG_ID_COL not in df.columns:
        raise KeyError("t0 curated must include trip_id, seg_id")
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "timedelta64[ns]"]).columns.tolist()
    if dt_cols: df = df.drop(columns=dt_cols)
    df = df.dropna(axis=1, how="all")
    _print_shape(df, "t0 curated (after drop dt/timedelta)")
    return df

# ---------------- Assembly ----------------
def assemble_dataset(
    target_df: pd.DataFrame,
    fp_df: pd.DataFrame,
    t0_df: pd.DataFrame,
    prior_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    base = fp_df[[TRIP_ID_COL, SEG_ID_COL, "split"]].drop_duplicates()
    df = target_df.merge(base, on=[TRIP_ID_COL, SEG_ID_COL], how="inner")

    df = df.merge(fp_df, on=[TRIP_ID_COL, SEG_ID_COL, "split"], how="left", suffixes=("",""))
    df = df.merge(prior_df, on=[TRIP_ID_COL, SEG_ID_COL], how="left", suffixes=("",""))
    df = df.merge(t0_df, on=[TRIP_ID_COL, SEG_ID_COL], how="left", suffixes=("",""))

    df = df.loc[:, ~df.columns.duplicated()]
    df = df[df[TARGET_COL].notna() & df["split"].notna()].copy()

    fp_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("fp_")]
    prior_cols = [c for c in df.columns if isinstance(c, str) and (c.startswith("prior_") or c == "prior_seg_count")]
    drop = {TRIP_ID_COL, SEG_ID_COL, TARGET_COL, "split"} | set(fp_cols) | set(prior_cols)
    t0_cols = [c for c in df.columns if c not in drop]
    dt_like = df[t0_cols].select_dtypes(include=["datetime64[ns]", "timedelta64[ns]"]).columns.tolist()
    t0_cols = [c for c in t0_cols if c not in dt_like]

    blocks = {"t0": t0_cols, "fp": fp_cols, "prior": prior_cols}
    _print_shape(df, "Assembled (pre-encode)")
    return df, blocks

# ---------------- Base ----------------
def build_pre_t0(t0_cols: List[str], df_ref: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in t0_cols if df_ref[c].dtype.name in {"object","category"}]
    num_cols = [c for c in t0_cols if c not in cat_cols]
    num_t = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_t = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transformers = []
    if num_cols: transformers.append(("num", num_t, num_cols))
    if cat_cols: transformers.append(("cat", cat_t, cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

def base_candidates(pre: ColumnTransformer) -> List[Tuple[str, Pipeline]]:
    cands = []
    for max_leaf in [31, 63]:
        for lr in [0.03, 0.06]:
            for mleaf in [20, 100]:
                for l2 in [0.0, 0.5, 2.0]:
                    model = HistGradientBoostingRegressor(
                        loss="squared_error", max_leaf_nodes=max_leaf, learning_rate=lr,
                        min_samples_leaf=mleaf, l2_regularization=l2, early_stopping=True,
                        validation_fraction=0.15, n_iter_no_change=30, random_state=RANDOM_STATE
                    )
                    cands.append((f"HGBR_ml{max_leaf}_lr{lr}_msl{mleaf}_l2{l2}", Pipeline([("pre", pre), ("model", model)])))
    return cands

def kfold_oof_predictions_grouped(pipe_builder, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int = N_SPLITS_OOF) -> np.ndarray:
    uniq = np.unique(groups)
    n_groups = len(uniq)
    n_splits_eff = min(n_splits, n_groups) if n_groups >= 2 else 2
    gkf = GroupKFold(n_splits=n_splits_eff)
    oof = np.zeros(len(X), dtype=float)
    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr = y[tr_idx]
        pipe = pipe_builder()
        pipe.fit(Xtr, ytr)
        oof[va_idx] = pipe.predict(Xva)
    return oof

# ---------------- FP latents ----------------
def fit_fp_pls_latent(df: pd.DataFrame, fp_cols: List[str], y_resid: np.ndarray, n_comp: int):
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    X = pipe.fit_transform(df[fp_cols], y_resid)
    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(X, y_resid)
    return {"pipe": pipe, "pls": pls}


def transform_fp_latent(model, df: pd.DataFrame, fp_cols: List[str]) -> np.ndarray:
    """
    Augmented: returns [V, ΔV_k] where ΔV_k is per-trip lag-k difference in latent space (default k=3).
    This keeps causality because ΔV_k for row i only depends on previous rows in the same trip.
    """
    X = model["pipe"].transform(df[fp_cols])
    V = model["pls"].transform(X)
    V = np.asarray(V, dtype=float)

    # Compute ΔV_k within each trip ordered by segment id
    k = 3  # fast horizon
    if ("trip_id" in df.columns) and ("seg_id" in df.columns) and (len(V.shape) == 2):
        dV = np.zeros_like(V)
        # argsort index for stable group processing
        order = np.lexsort((df["seg_id"].to_numpy(), df["trip_id"].to_numpy()))
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))
        # Work on sorted view
        df_sorted = df.iloc[order]
        V_sorted = V[order]
        start = 0
        for _, g in df_sorted.groupby("trip_id", sort=False):
            n = len(g)
            idx = np.arange(start, start+n)
            # lag-k difference within group
            for j in range(k, n):
                dV[idx[j]] = V_sorted[idx[j]] - V_sorted[idx[j-k]]
            start += n
        # re-order back
        dV = dV[inv_order]
        V = np.concatenate([V, dV], axis=1)
    return V

def compute_fp_weights(df_blk: pd.DataFrame) -> np.ndarray:
    fc = df_blk.get("fp_count", pd.Series(0.0, index=df_blk.index)).astype(float).to_numpy()
    fs = df_blk.get("fp_stability", pd.Series(0.0, index=df_blk.index)).astype(float).to_numpy()
    w_count = np.clip(fc, 0.0, 6.0) / 6.0
    w_stab = (fs + 1.0) / 2.0
    w = w_count * w_stab
    w = 0.25 + 0.75 * w
    m = w.mean() if w.mean() > 0 else 1.0
    return (w / m).astype(float)

def apply_gating(res_pred: np.ndarray, meta_df: pd.DataFrame, count_thr: int, stab_thr: float, cap_q: float, shrink: float) -> np.ndarray:
    fc = meta_df.get("fp_count", pd.Series(0.0, index=meta_df.index)).to_numpy()
    fs = meta_df.get("fp_stability", pd.Series(0.0, index=meta_df.index)).fillna(0.0).to_numpy()
    mask = (fc >= count_thr) & (fs >= stab_thr)
    out = np.zeros_like(res_pred)
    if np.any(mask):
        cap = np.quantile(np.abs(res_pred[mask]), cap_q) if np.any(mask) else 0.0
        if cap <= 0: cap = np.quantile(np.abs(res_pred), cap_q)
        adj = np.clip(res_pred[mask], -cap, cap) * float(shrink)
        out[mask] = adj
    return out

# ---------------- Plotting ----------------
def parity_plot(y, yhat, title, out_path: Path):
    plt.figure(figsize=(6,6))
    plt.scatter(y, yhat, s=10, alpha=0.6)
    mn, mx = float(np.nanmin([y.min(), yhat.min()])), float(np.nanmax([y.max(), yhat.max()]))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True SoC end (%)"); plt.ylabel("Predicted SoC end (%)"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=220); plt.close()

def cdf_abs_err_plot(y, yhat, title, out_path: Path):
    ae = np.abs(y - yhat)
    xs = np.sort(ae); ys = np.linspace(0, 1, len(xs))
    plt.figure(figsize=(6,4)); plt.plot(xs, ys)
    plt.xlabel("|Error| (pct points)"); plt.ylabel("Cumulative fraction"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=220); plt.close()

def compare_bars(metrics_df: pd.DataFrame, out_prefix: str):
    order = list(metrics_df["variant"])
    x = np.arange(len(order)); width = 0.4
    plt.figure(figsize=(9,5)); plt.bar(x, metrics_df["test_MAE"], width)
    plt.xticks(x, order, rotation=15); plt.ylabel("MAE (pct points)"); plt.title(f"Final Test MAE — {out_prefix}")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / f"{out_prefix}_compare_test_mae.png", dpi=220); plt.close()
    plt.figure(figsize=(9,5)); plt.bar(x, metrics_df["test_RMSE"], width)
    plt.xticks(x, order, rotation=15); plt.ylabel("RMSE (pct points)"); plt.title(f"Final Test RMSE — {out_prefix}")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / f"{out_prefix}_compare_test_rmse.png", dpi=220); plt.close()

# ---------------- Main ----------------
def main():
    _ensure_dirs()

    # Load
    tgt_df = build_segment_targets(CLEANED_DATA_PATH, SEGMENTATION_DICT_PATH)
    fp_df = load_fingerprints(FINGERPRINTS_ALL)
    t0_df = load_t0_curated(NONBEHAV_T0_CURATED_PATH)
    prior_df = load_prior_aggs(BEHAVIOR_PRIOR_AGGS_PATH)
    df, blocks = assemble_dataset(tgt_df, fp_df, t0_df, prior_df)

    tr = df[df["split"] == "train"].reset_index(drop=True)
    va = df[df["split"] == "val"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)

    y_tr = tr[TARGET_COL].to_numpy()
    y_va = va[TARGET_COL].to_numpy()
    y_te = te[TARGET_COL].to_numpy()

    # Base selection on VAL (Train-only fit)
    pre_t0 = build_pre_t0(blocks["t0"], df_ref=df)
    base_cands = base_candidates(pre_t0)

    best_base_name, best_base_pipe, best_base_val = None, None, np.inf
    for name, pipe in base_cands:
        try:
            pipe.fit(tr[blocks["t0"]], y_tr)
            p_val = pipe.predict(va[blocks["t0"]])
            mae = mean_absolute_error(y_va, p_val)
            if mae < best_base_val:
                best_base_val = mae; best_base_name, best_base_pipe = name, pipe
        except Exception:
            continue
    Path(SELECTION_DIR / "base_selection.json").write_text(json.dumps({"best": best_base_name, "val_MAE": float(best_base_val)}, indent=2))

    # Residual targets
    def _fresh_base():
        for nm, pp in base_cands:
            if nm == best_base_name:
                return pickle.loads(pickle.dumps(pp))
        raise KeyError("Best base pipeline not found")

    oof_pred_tr = kfold_oof_predictions_grouped(_fresh_base, tr[blocks["t0"]], y_tr, tr[TRIP_ID_COL].to_numpy(), n_splits=N_SPLITS_OOF)
    resid_tr = y_tr - oof_pred_tr

    base_tr_only = _fresh_base(); base_tr_only.fit(tr[blocks["t0"]], y_tr)
    base_pred_va = base_tr_only.predict(va[blocks["t0"]])
    resid_va = y_va - base_pred_va

    # ------------- SOLO FP residual — selection -------------
    KV = [8, 16, 24, 32]
    DEG = [1, 2]          # 1: linear in V; 2: plus squares & pairwise
    FAMILY = ["ridge", "huber"]
    ALPH = [0.1, 1.0, 10.0]
    COUNT_THR = [1, 2, 3, 4, 5]
    STAB_THR = [-0.10, 0.00, 0.10, 0.20]
    CAP_Q = [0.90, 0.95]
    SHRINK = [0.5, 0.7, 0.85, 1.0]

    best_R1 = None; best_mae = np.inf

    for kv in KV:
        fp_lat_tr = fit_fp_pls_latent(tr, blocks["fp"], resid_tr, n_comp=kv)
        V_tr = transform_fp_latent(fp_lat_tr, tr, blocks["fp"])
        V_va = transform_fp_latent(fp_lat_tr, va, blocks["fp"])

        for deg in DEG:
            if deg == 1:
                fe_tr = V_tr; fe_va = V_va
            else:
                poly = PolynomialFeatures(degree=2, include_bias=False)
                fe_tr = poly.fit_transform(V_tr)
                fe_va = poly.transform(V_va)

            w_tr = compute_fp_weights(tr)

            for fam in FAMILY:
                if fam == "ridge":
                    for alpha in ALPH:
                        pipe = Pipeline([("scaler", StandardScaler()),
                                         ("model", Ridge(alpha=alpha, random_state=RANDOM_STATE))])
                        pipe.fit(fe_tr, resid_tr, model__sample_weight=w_tr)
                        r_va_raw = pipe.predict(fe_va)
                        for cthr in COUNT_THR:
                            for sthr in STAB_THR:
                                for capq in CAP_Q:
                                    for lam in SHRINK:
                                        r_va = apply_gating(r_va_raw, va, cthr, sthr, capq, lam)
                                        yhat_va = base_pred_va + r_va
                                        mae = mean_absolute_error(y_va, yhat_va)
                                        if mae < best_mae:
                                            best_mae = mae
                                            best_R1 = {"kv": kv, "deg": deg, "family": "ridge", "alpha": alpha,
                                                       "count_thr": cthr, "stab_thr": sthr, "cap_q": capq, "shrink": lam,
                                                       "fp_lat": fp_lat_tr}
                else:
                    pipe = Pipeline([("scaler", StandardScaler()),
                                     ("model", HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=2000))])
                    pipe.fit(fe_tr, resid_tr, model__sample_weight=w_tr)
                    r_va_raw = pipe.predict(fe_va)
                    for cthr in COUNT_THR:
                        for sthr in STAB_THR:
                            for capq in CAP_Q:
                                for lam in SHRINK:
                                    r_va = apply_gating(r_va_raw, va, cthr, sthr, capq, lam)
                                    yhat_va = base_pred_va + r_va
                                    mae = mean_absolute_error(y_va, yhat_va)
                                    if mae < best_mae:
                                        best_mae = mae
                                        best_R1 = {"kv": kv, "deg": deg, "family": "huber",
                                                   "count_thr": cthr, "stab_thr": sthr, "cap_q": capq, "shrink": lam,
                                                   "fp_lat": fp_lat_tr}

    Path(SELECTION_DIR / "residual_selection_R1_fp_solo_gated.json").write_text(
        json.dumps({"val_MAE": float(best_mae),
                    "best": {k: v for k, v in best_R1.items() if k in ["kv","deg","family","alpha","count_thr","stab_thr","cap_q","shrink"]}},
                   indent=2)
    )

    # ------------- FP cluster-offset head — selection -------------
    KM = [6, 10, 14]
    best_CL = None; best_mae_cl = np.inf

    for kv in KV:
        fp_lat_tr = fit_fp_pls_latent(tr, blocks["fp"], resid_tr, n_comp=kv)
        V_tr = transform_fp_latent(fp_lat_tr, tr, blocks["fp"])
        V_va = transform_fp_latent(fp_lat_tr, va, blocks["fp"])

        for k in KM:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
            km.fit(V_tr)
            cid_tr = km.predict(V_tr)
            cid_va = km.predict(V_va)

            # per-cluster residual mean on TRAIN only
            cl_off = {c: resid_tr[cid_tr == c].mean() if np.any(cid_tr == c) else 0.0 for c in range(k)}

            r_va_raw = np.array([cl_off.get(c, 0.0) for c in cid_va], dtype=float)

            for cthr in COUNT_THR:
                for sthr in STAB_THR:
                    for capq in CAP_Q:
                        for lam in SHRINK:
                            r_va = apply_gating(r_va_raw, va, cthr, sthr, capq, lam)
                            yhat_va = base_pred_va + r_va
                            mae = mean_absolute_error(y_va, yhat_va)
                            if mae < best_mae_cl:
                                best_mae_cl = mae
                                best_CL = {"kv": kv, "k": k, "count_thr": cthr, "stab_thr": sthr, "cap_q": capq, "shrink": lam,
                                           "fp_lat": fp_lat_tr, "km": km, "cl_off": cl_off}

    Path(SELECTION_DIR / "residual_selection_R1_fp_cluster_gated.json").write_text(
        json.dumps({"val_MAE": float(best_mae_cl),
                    "best": {k: v for k, v in best_CL.items() if k in ["kv","k","count_thr","stab_thr","cap_q","shrink"]}},
                   indent=2)
    )

    # ------------- Prior residual reference -------------
    pr_cols = blocks["prior"]
    best_R2 = None; best_mae2 = np.inf
    if len(pr_cols) > 0:
        pipe_pr = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        Xtr_pr = pipe_pr.fit_transform(tr[pr_cols]); Xva_pr = pipe_pr.transform(va[pr_cols])
        for loss in ["huber", "absolute_error", "squared_error"]:
            mdl = GradientBoostingRegressor(loss=loss, n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE)
            mdl.fit(Xtr_pr, resid_tr)
            r_va = mdl.predict(Xva_pr)
            yhat_va = base_pred_va + r_va
            mae = mean_absolute_error(y_va, yhat_va)
            if mae < best_mae2:
                best_mae2 = mae; best_R2 = {"loss": loss}
    Path(SELECTION_DIR / "residual_selection_R2_prior.json").write_text(json.dumps({"val_MAE": float(best_mae2)}, indent=2))

    # ------------- Final TEST -------------
    trva = pd.concat([tr, va], ignore_index=True)
    y_trva = trva[TARGET_COL].to_numpy()

    # Base on Train+Val
    base_full = None
    for nm, pp in base_cands:
        if nm == best_base_name: base_full = pickle.loads(pickle.dumps(pp)); break
    if base_full is None: raise RuntimeError("Best base not found")
    base_full.fit(trva[blocks["t0"]], y_trva)
    base_pred_te = base_full.predict(te[blocks["t0"]])

    # OOF base on Train+Val for residual targets
    oof_trva = kfold_oof_predictions_grouped(lambda: pickle.loads(pickle.dumps(base_full)),
                                             trva[blocks["t0"]], y_trva, trva[TRIP_ID_COL].to_numpy(), n_splits=N_SPLITS_OOF)
    resid_trva = y_trva - oof_trva

    # --- R3_none ---
    yhat_R3 = base_pred_te

    # --- R1_fp_solo_gated ---
    kv = best_R1["kv"]; deg = best_R1["deg"]; fam = best_R1["family"]
    fp_lat = fit_fp_pls_latent(trva, blocks["fp"], resid_trva, n_comp=kv)
    V_trva = transform_fp_latent(fp_lat, trva, blocks["fp"]); V_te = transform_fp_latent(fp_lat, te, blocks["fp"])
    if deg == 1:
        fe_trva = V_trva; fe_te = V_te
        fe_scaler = StandardScaler().fit(fe_trva)
    else:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        fe_trva = poly.fit_transform(V_trva); fe_te = poly.transform(V_te)
        fe_scaler = StandardScaler().fit(fe_trva)
    fe_trva_s = fe_scaler.transform(fe_trva); fe_te_s = fe_scaler.transform(fe_te)

    w_trva = compute_fp_weights(trva)

    if fam == "ridge":
        alpha = best_R1.get("alpha", 1.0)
        mdl_R1 = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    else:
        mdl_R1 = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=2000)

    mdl_R1.fit(fe_trva_s, resid_trva, sample_weight=w_trva)
    r_R1_raw = mdl_R1.predict(fe_te_s)
    r_R1 = apply_gating(r_R1_raw, te, best_R1["count_thr"], best_R1["stab_thr"], best_R1["cap_q"], best_R1["shrink"])
    yhat_R1 = base_pred_te + r_R1

    # --- R1_fp_cluster_gated ---
    kv = best_CL["kv"]; k = best_CL["k"]
    fp_lat_cl = fit_fp_pls_latent(trva, blocks["fp"], resid_trva, n_comp=kv)
    V_trva_cl = transform_fp_latent(fp_lat_cl, trva, blocks["fp"]); V_te_cl = transform_fp_latent(fp_lat_cl, te, blocks["fp"])
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto").fit(V_trva_cl)
    cid_trva = km.predict(V_trva_cl); cid_te = km.predict(V_te_cl)

    # per-cluster offsets from trva OOF resid
    cl_off = {c: resid_trva[cid_trva == c].mean() if np.any(cid_trva == c) else 0.0 for c in range(k)}
    r_CL_raw = np.array([cl_off.get(c, 0.0) for c in cid_te], dtype=float)
    r_CL = apply_gating(r_CL_raw, te, best_CL["count_thr"], best_CL["stab_thr"], best_CL["cap_q"], best_CL["shrink"])
    yhat_CL = base_pred_te + r_CL

    # --- Ensemble (val-tuned weight) ---
    # Rebuild components on Val to choose w (no leakage)
    # Use selection-time best models; compute on Val and pick w that minimizes Val MAE
    # (Here, implement a small shadow evaluation on Val using Train-only fits to avoid leakage)
    # Build reg & cluster raw residuals on Val
    # REG shadow:
    fp_lat_tr = best_R1["fp_lat"]
    V_tr = transform_fp_latent(fp_lat_tr, tr, blocks["fp"]); V_va = transform_fp_latent(fp_lat_tr, va, blocks["fp"])
    if best_R1["deg"] == 1:
        fe_tr = V_tr; fe_va = V_va
        scaler_shadow = StandardScaler().fit(fe_tr)
    else:
        poly_s = PolynomialFeatures(degree=2, include_bias=False)
        fe_tr = poly_s.fit_transform(V_tr); fe_va = poly_s.transform(V_va)
        scaler_shadow = StandardScaler().fit(fe_tr)
    fe_tr_s = scaler_shadow.transform(fe_tr); fe_va_s = scaler_shadow.transform(fe_va)
    w_tr = compute_fp_weights(tr)
    if best_R1["family"] == "ridge":
        alpha = best_R1.get("alpha", 1.0); mdl_s = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    else:
        mdl_s = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=2000)
    mdl_s.fit(fe_tr_s, resid_tr, sample_weight=w_tr)
    r_va_reg_raw = mdl_s.predict(fe_va_s)
    r_va_reg = apply_gating(r_va_reg_raw, va, best_R1["count_thr"], best_R1["stab_thr"], best_R1["cap_q"], best_R1["shrink"])

    # CL shadow:
    fp_lat_tr_cl = best_CL["fp_lat"]
    V_tr_cl = transform_fp_latent(fp_lat_tr_cl, tr, blocks["fp"]); V_va_cl = transform_fp_latent(fp_lat_tr_cl, va, blocks["fp"])
    km_s = best_CL["km"]
    cid_va = km_s.predict(V_va_cl)
    cl_off_s = best_CL["cl_off"]
    r_va_cl_raw = np.array([cl_off_s.get(c, 0.0) for c in cid_va], dtype=float)
    r_va_cl = apply_gating(r_va_cl_raw, va, best_CL["count_thr"], best_CL["stab_thr"], best_CL["cap_q"], best_CL["shrink"])

    # Val blend
    w_grid = [0.3, 0.5, 0.7]
    best_w, best_mae_w = 0.5, np.inf
    for w in w_grid:
        yhat_va = base_pred_va + (w * r_va_reg + (1 - w) * r_va_cl)
        mae = mean_absolute_error(y_va, yhat_va)
        if mae < best_mae_w:
            best_mae_w = mae; best_w = w

    r_ENS = best_w * r_R1 + (1 - best_w) * r_CL
    yhat_ENS = base_pred_te + r_ENS

    # --- R2_prior (reference) ---
    if best_R2 is not None:
        pr_cols = blocks["prior"]
        pipe_pr = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        Xtrva_pr = pipe_pr.fit_transform(trva[pr_cols]); Xte_pr = pipe_pr.transform(te[pr_cols])
        mdl_R2 = GradientBoostingRegressor(loss=best_R2["loss"], n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE)
        mdl_R2.fit(Xtrva_pr, resid_trva)
        r_R2 = mdl_R2.predict(Xte_pr)
        yhat_R2 = base_pred_te + r_R2
    else:
        yhat_R2 = base_pred_te

    # --- Metrics & plots ---
    rows = []
    def add_row(name, yhat):
        rows.append({
            "variant": name, "n_trainval": int(len(trva)), "n_test": int(len(te)),
            "test_MAE": mean_absolute_error(y_te, yhat),
            "test_RMSE": _rmse(y_te, yhat),
            "test_R2": r2_score(y_te, yhat),
        })

    add_row("R3_none", yhat_R3)
    add_row("R1_fp_solo_gated", yhat_R1)
    add_row("R1_fp_cluster_gated", yhat_CL)
    add_row("R1_fp_ens_gated", yhat_ENS)
    add_row("R2_prior", yhat_R2)

    final_df = pd.DataFrame(rows).sort_values("test_MAE").reset_index(drop=True)
    final_df.to_csv(OUT_DIR / "metrics_residual_final.csv", index=False)
    compare_bars(final_df, "residual_fp_solo")

    print("\n=== Final Test Metrics (Residual — SOLO FP) ===")
    print(final_df)

    # --- Save per-segment TEST predictions for richer evaluation ---
    try:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        pred_df = pd.DataFrame({
            TRIP_ID_COL: te[TRIP_ID_COL].to_numpy(),
            SEG_ID_COL: te[SEG_ID_COL].to_numpy(),
            "gt": y_te,
            "pred_R3_none": yhat_R3,
            "pred_R1_fp_solo_gated": yhat_R1,
            "pred_R1_fp_cluster_gated": yhat_CL,
            "pred_R1_fp_ens_gated": yhat_ENS,
            "pred_R2_prior": yhat_R2,
        })
        pred_path = PRED_DIR / "test_predictions.parquet"
        pred_df.to_parquet(pred_path, index=False)
        print(f"Saved per-segment predictions → {pred_path}")
    except Exception as e:
        print(f"[warn] Could not save per-segment predictions: {e}")

    # --- Generate simple relative-to-global plots (slide-ready) ---
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        mdf = pd.read_csv(OUT_DIR / "metrics_residual_final.csv")
        base_mae = float(mdf.loc[mdf["variant"] == "R3_none", "test_MAE"].iloc[0])
        mdf["impr_abs"] = base_mae - mdf["test_MAE"]
        mdf_sorted = mdf.sort_values("test_MAE")

        # Bar of MAE by variant
        plt.figure(figsize=(8,5))
        plt.bar(range(len(mdf_sorted)), mdf_sorted["test_MAE"].to_numpy())
        plt.axhline(base_mae, linestyle="--")
        plt.xticks(range(len(mdf_sorted)), mdf_sorted["variant"].tolist(), rotation=20, ha="right")
        plt.title("Test MAE by Variant (lower is better)")
        plt.ylabel("MAE")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "bar_mae_by_variant.png", dpi=180)
        plt.close()

        # Waterfall of absolute improvement vs baseline
        wf = mdf[mdf["variant"] != "R3_none"].copy().sort_values("impr_abs", ascending=False)
        labs = ["R3_none"] + wf["variant"].tolist()
        vals = [0.0] + wf["impr_abs"].tolist()
        plt.figure(figsize=(9,5))
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labs, rotation=20, ha="right")
        plt.title(f"Absolute MAE Improvement vs Global Baseline (R3_none = {base_mae:.3f})")
        plt.ylabel("ΔMAE (Global − Variant)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "waterfall_abs_improvement_vs_global.png", dpi=180)
        plt.close()
        print(f"Saved presentation plots → {PLOTS_DIR}")
    except Exception as e:
        print(f"[warn] Could not generate extra plots: {e}")

    print(final_df)

if __name__ == "__main__":
    main()
