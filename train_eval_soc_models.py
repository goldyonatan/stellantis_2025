#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leakage-proof SoC end-of-segment modeling with supervised fp reduction (PLS).

New in v8:
- For variants that include fingerprints (M1, M4), try **PLSRegression** on the fp block
  (train-only, inside the ColumnTransformer) with n_components in {8, 16, 24, 32},
  alongside the existing {None, PCA 16, PCA 32}. PLS aligns fp components to the
  regression target without leakage (supervised fit uses only the Train fold).
- Keeps the stronger-regularization grids and optional XGBoost/LightGBM/CatBoost.
- Same strict feature hygiene and causal target construction as v7.

Outputs under: End-to-End/model_soc_segment_selection_v8/
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet

# Try to make transformers output pandas for consistent feature names
_TRANSFORM_TO_PANDAS = False
try:
    from sklearn import set_config
    set_config(transform_output="pandas")
    _TRANSFORM_TO_PANDAS = True
except Exception:
    _TRANSFORM_TO_PANDAS = False

# Optional libraries
_has_xgb = False
_has_lgbm = False
_has_cat = False
try:
    from xgboost import XGBRegressor  # type: ignore
    _has_xgb = True
except Exception:
    pass
try:
    from lightgbm import LGBMRegressor  # type: ignore
    _has_lgbm = True
except Exception:
    pass
try:
    from catboost import CatBoostRegressor  # type: ignore
    _has_cat = True
except Exception:
    pass

# Reduce noisy warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# ---------------- Paths ----------------
BASE_DIR = Path(r"C:/Users/goldy/OneDrive/Documents/Y-DATA 24-25/Stellantis Project/End-to-End")

CLEANED_DATA_PATH = BASE_DIR / "clean_df" / "clean_df.parquet"
SEGMENTATION_DICT_PATH = BASE_DIR / "segmentation_dict" / "trips_seg_dict_v7.pickle"

DRIVER_EMB_DIR = BASE_DIR / "driver_emb_data"
# Prefer v3/v3_compat outputs if present; else fallback to original
FINGERPRINTS_ALL = None
for name in ["fingerprints_all.parquet", "fingerprints_all.csv"]:
    p = DRIVER_EMB_DIR / name
    if p.exists():
        FINGERPRINTS_ALL = p
if FINGERPRINTS_ALL is None:
    FINGERPRINTS_ALL = DRIVER_EMB_DIR / "fingerprints_all.parquet"

NONBEHAV_T0_CURATED_PATH = BASE_DIR / "nonbehavioral_t0_features" / "t0_nonbehavioral_features_curated.parquet"
BEHAVIOR_PRIOR_AGGS_PATH = BASE_DIR / "segment_behavior_agg" / "seg_behavior_prior_aggs.parquet"

OUT_DIR = BASE_DIR / "model_soc_segment_selection_v8"
PLOTS_DIR = OUT_DIR / "plots"
PRED_DIR = OUT_DIR / "predictions"
SELECTION_DIR = OUT_DIR / "selection_reports"
CHOSEN_DIR = OUT_DIR / "chosen_models"
ANALYSIS_DIR = OUT_DIR / "analysis"

TRIP_ID_COL = "trip_id"
SEG_ID_COL = "seg_id"
SOC_COL = "current_soc"
TARGET_COL = "soc_end_pct"

# ---------------- Utils ----------------
def _ensure_dirs():
    for p in [OUT_DIR, PLOTS_DIR, PRED_DIR, SELECTION_DIR, CHOSEN_DIR, ANALYSIS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _print_shape(df: pd.DataFrame, name: str):
    print(f"{name:>30s}: {df.shape[0]:,} rows x {df.shape[1]:,} cols")

# ---------------- Target construction ----------------
def build_segment_targets(clean_path: Path, segdict_path: Path) -> pd.DataFrame:
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing: {clean_path}")
    if not segdict_path.exists():
        raise FileNotFoundError(f"Missing: {segdict_path}")

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
            print(f"[warn] seg_dict length mismatch for trip {trip_id}: {len(labels)} vs {len(g)} -> skip")
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
            val = soc_vals[end_idx]
            rows.append({TRIP_ID_COL: trip_id, SEG_ID_COL: int(sid), TARGET_COL: float(val) if pd.notna(val) else np.nan})
    tgt = pd.DataFrame(rows)
    _print_shape(tgt, "Built targets")
    return tgt

# ---------------- Feature loading (strict keep) ----------------
def load_fingerprints(merged_all: Path) -> pd.DataFrame:
    def _read_any(p: Path) -> pd.DataFrame:
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    if merged_all.exists():
        df = _read_any(merged_all)
    else:
        # fallback: concatenating split files if present
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

    keep = [TRIP_ID_COL, SEG_ID_COL, "split", "fp_count", "fp_coldstart"]
    # include all fp_* columns (base + residual + stability + norm)
    fp_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("fp_")]
    # exclude timestamp-ish if any slipped in
    drop_dt = df.select_dtypes(include=["datetime64[ns]", "timedelta64[ns]"]).columns.tolist()
    df = df.drop(columns=drop_dt, errors="ignore")
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

# ---------------- Assembly & dtype separation ----------------
def assemble_dataset(
    target_df: pd.DataFrame,
    fp_df: pd.DataFrame,
    t0_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    use_fp: bool, use_priors: bool, use_t0: bool
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    base = fp_df[[TRIP_ID_COL, SEG_ID_COL, "split"]].drop_duplicates()
    df = target_df.merge(base, on=[TRIP_ID_COL, SEG_ID_COL], how="inner")

    if use_fp:
        fp_feats = [c for c in fp_df.columns if c not in [TRIP_ID_COL, SEG_ID_COL, "split"]]
        df = df.merge(fp_df[[TRIP_ID_COL, SEG_ID_COL, "split"] + fp_feats], on=[TRIP_ID_COL, SEG_ID_COL, "split"], how="left")
    if use_priors:
        prior_feats = [c for c in prior_df.columns if c not in [TRIP_ID_COL, SEG_ID_COL]]
        df = df.merge(prior_df[[TRIP_ID_COL, SEG_ID_COL] + prior_feats], on=[TRIP_ID_COL, SEG_ID_COL], how="left")
    if use_t0:
        t0_feats = [c for c in t0_df.columns if c not in [TRIP_ID_COL, SEG_ID_COL]]
        df = df.merge(t0_df[[TRIP_ID_COL, SEG_ID_COL] + t0_feats], on=[TRIP_ID_COL, SEG_ID_COL], how="left")

    df = df.loc[:, ~df.columns.duplicated()]

    drop_non_features = {TRIP_ID_COL, SEG_ID_COL, TARGET_COL, "split"}
    feat_cols = [c for c in df.columns if c not in drop_non_features]

    dt_like = df.select_dtypes(include=["datetime64[ns]", "timedelta64[ns]"]).columns.tolist()
    feat_cols = [c for c in feat_cols if c not in dt_like]
    all_nan_cols = [c for c in feat_cols if df[c].isna().all()]
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)
        feat_cols = [c for c in feat_cols if c not in all_nan_cols]

    fp_cols = [c for c in feat_cols if isinstance(c, str) and c.startswith("fp_")]
    cat_pool = df.select_dtypes(include=["object", "category"]).columns
    num_pool = df.select_dtypes(exclude=["object", "category", "datetime64[ns]", "timedelta64[ns]"]).columns
    num_cols = [c for c in feat_cols if (c in num_pool and c not in fp_cols)]
    cat_cols = [c for c in feat_cols if c in cat_pool]

    df = df[df[TARGET_COL].notna() & df["split"].notna()].copy()
    _print_shape(df, "Assembled (pre-encode)")
    return df, num_cols, cat_cols, fp_cols

# ---------------- Preprocessor factory ----------------
def build_preprocessor(num_cols: List[str], cat_cols: List[str], fp_cols: List[str],
                       fp_reduce: Optional[str], k: Optional[int]) -> ColumnTransformer:
    """
    fp_reduce ∈ {None, 'pca', 'pls'}
    k = n_components for PCA/PLS when applicable
    """
    transformers = []
    num_t = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_t = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    if fp_cols:
        if fp_reduce is None:
            # pass fp as numeric
            transformers.append(("num_all", num_t, num_cols + fp_cols))
        elif fp_reduce == "pca":
            fp_t = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("pca", PCA(n_components=k, random_state=42))
            ])
            transformers.append(("num_base", num_t, num_cols))
            transformers.append(("fp_pca", fp_t, fp_cols))
        elif fp_reduce == "pls":
            # Supervised reduction; ColumnTransformer will pass y into fit
            fp_t = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("pls", PLSRegression(n_components=k, scale=False))
            ])
            transformers.append(("num_base", num_t, num_cols))
            transformers.append(("fp_pls", fp_t, fp_cols))
        else:
            raise ValueError(f"Unknown fp_reduce: {fp_reduce}")
    else:
        transformers.append(("num_only", num_t, num_cols))

    if cat_cols:
        transformers.append(("cat", cat_t, cat_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    try:
        if _TRANSFORM_TO_PANDAS:
            pre.set_output(transform="pandas")
    except Exception:
        pass
    return pre

# ---------------- Candidate models ----------------
def build_candidates(num_cols: List[str], cat_cols: List[str], fp_cols: List[str], use_fp: bool) -> List[Tuple[str, Pipeline, Dict]]:
    fp_reduce_options: List[Tuple[Optional[str], Optional[int]]] = [(None, None)]
    if use_fp and len(fp_cols) > 0:
        fp_reduce_options = [(None, None), ("pca", 16), ("pca", 32), ("pls", 8), ("pls", 16), ("pls", 24), ("pls", 32)]

    cands = []
    for fp_reduce, k in fp_reduce_options:
        pre = build_preprocessor(num_cols, cat_cols, fp_cols, fp_reduce, k)

        # HistGradientBoostingRegressor
        for max_leaf in [31, 63]:
            for lr in [0.03, 0.06]:
                for mleaf in [20, 100]:
                    for l2 in [0.0, 0.5, 2.0]:
                        model = HistGradientBoostingRegressor(
                            loss="squared_error",
                            max_leaf_nodes=max_leaf,
                            learning_rate=lr,
                            min_samples_leaf=mleaf,
                            l2_regularization=l2,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=30,
                            random_state=42,
                        )
                        name = f"HGBR_fp{fp_reduce}_k{k}_ml{max_leaf}_lr{lr}_msl{mleaf}_l2{l2}"
                        cands.append((name, Pipeline([("pre", pre), ("model", model)]), {"family":"HGBR","fp_reduce":fp_reduce,"k":k}))

        # GradientBoostingRegressor
        for n_est in [300]:
            for max_depth in [2, 3]:
                for lr in [0.05]:
                    for subs in [0.8]:
                        for mf in [1.0, 0.7]:
                            gbr = GradientBoostingRegressor(
                                n_estimators=n_est, max_depth=max_depth, learning_rate=lr,
                                subsample=subs, max_features=mf, random_state=42
                            )
                            name = f"GBR_fp{fp_reduce}_k{k}_ne{n_est}_md{max_depth}_lr{lr}_sub{subs}_mf{mf}"
                            cands.append((name, Pipeline([("pre", pre), ("model", gbr)]), {"family":"GBR","fp_reduce":fp_reduce,"k":k}))

        # RandomForestRegressor
        for n_est in [400]:
            for md in [10, 15]:
                for msl in [5, 20]:
                    for mf in ["sqrt", 0.3]:
                        rf = RandomForestRegressor(
                            n_estimators=n_est, max_depth=md, min_samples_leaf=msl,
                            max_features=mf, n_jobs=-1, random_state=42
                        )
                        name = f"RF_fp{fp_reduce}_k{k}_ne{n_est}_md{md}_msl{msl}_mf{mf}"
                        cands.append((name, Pipeline([("pre", pre), ("model", rf)]), {"family":"RF","fp_reduce":fp_reduce,"k":k}))

        # ElasticNet
        for alpha in [1.0, 10.0, 50.0, 100.0]:
            for l1 in [0.1, 0.5, 0.9]:
                en = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=42, max_iter=6000)
                name = f"EN_fp{fp_reduce}_k{k}_a{alpha}_l1{l1}"
                cands.append((name, Pipeline([("pre", pre), ("scaler", StandardScaler(with_mean=True, with_std=True)), ("model", en)]), {"family":"ElasticNet","fp_reduce":fp_reduce,"k":k}))

        if _has_xgb:
            for n_est in [600]:
                for md in [3, 4, 5]:
                    for lr in [0.05]:
                        for subs in [0.6, 0.8]:
                            for cs in [0.5, 0.7, 0.9]:
                                xgb = XGBRegressor(
                                    objective="reg:squarederror",
                                    n_estimators=n_est,
                                    max_depth=md,
                                    learning_rate=lr,
                                    subsample=subs,
                                    colsample_bytree=cs,
                                    reg_lambda=5.0,
                                    tree_method="hist",
                                    random_state=42,
                                    n_jobs=-1,
                                )
                                name = f"XGB_fp{fp_reduce}_k{k}_ne{n_est}_md{md}_lr{lr}_sub{subs}_cs{cs}"
                                cands.append((name, Pipeline([("pre", pre), ("model", xgb)]), {"family":"XGB","fp_reduce":fp_reduce,"k":k}))

        if _has_lgbm:
            for n_est in [600]:
                for num_leaves in [15, 31]:
                    for md in [-1, 8]:
                        for lr in [0.05]:
                            for bag in [0.6, 0.8]:
                                for feat in [0.5, 0.7, 0.9]:
                                    lgbm = LGBMRegressor(
                                        boosting_type="gbdt",
                                        n_estimators=n_est,
                                        num_leaves=num_leaves,
                                        max_depth=md,
                                        learning_rate=lr,
                                        subsample=bag,
                                        feature_fraction=feat,
                                        lambda_l2=5.0,
                                        force_col_wise=True,
                                        random_state=42,
                                        n_jobs=-1,
                                        verbose=-1,
                                    )
                                    name = f"LGBM_fp{fp_reduce}_k{k}_ne{n_est}_nl{num_leaves}_md{md}_lr{lr}_bag{bag}_ff{feat}"
                                    cands.append((name, Pipeline([("pre", pre), ("model", lgbm)]), {"family":"LGBM","fp_reduce":fp_reduce,"k":k}))

        if _has_cat:
            for depth in [4, 6, 8]:
                for lr in [0.05]:
                    for n_est in [800]:
                        cat = CatBoostRegressor(
                            depth=depth,
                            learning_rate=lr,
                            iterations=n_est,
                            loss_function="RMSE",
                            l2_leaf_reg=10.0,
                            random_seed=42,
                            verbose=False,
                        )
                        name = f"CAT_fp{fp_reduce}_k{k}_it{n_est}_depth{depth}_lr{lr}"
                        cands.append((name, Pipeline([("pre", pre), ("model", cat)]), {"family":"CAT","fp_reduce":fp_reduce,"k":k}))

    return cands

# ---------------- Selection (Val) ----------------
def evaluate_on_val(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], fp_cols: List[str], use_fp: bool) -> pd.DataFrame:
    tr = df[df["split"] == "train"].copy()
    va = df[df["split"] == "val"].copy()
    Xtr = tr[num_cols + cat_cols + fp_cols]
    ytr = tr[TARGET_COL].to_numpy()
    Xva = va[num_cols + cat_cols + fp_cols]
    yva = va[TARGET_COL].to_numpy()

    rows = []
    for name, pipe, meta in build_candidates(num_cols, cat_cols, fp_cols, use_fp):
        try:
            pipe.fit(Xtr, ytr)
            p_va = pipe.predict(Xva)
            mae = float(mean_absolute_error(yva, p_va))
            rmse = float(math.sqrt(mean_squared_error(yva, p_va)))
            r2 = float(r2_score(yva, p_va))
        except Exception as e:
            mae, rmse, r2 = np.inf, np.inf, np.nan
        rows.append({"candidate": name, "family": meta["family"], "val_MAE": mae, "val_RMSE": rmse, "val_R2": r2, "meta": json.dumps(meta)})
    rep = pd.DataFrame(rows).sort_values("val_MAE", ascending=True).reset_index(drop=True)
    return rep

# ---------------- Final fit (Train+Val) & Test eval ----------------
def fit_final_and_test(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], fp_cols: List[str], best_name: str, use_fp: bool):
    trva = df[df["split"].isin(["train","val"])].copy()
    te = df[df["split"]=="test"].copy()

    Xtrva = trva[num_cols + cat_cols + fp_cols]
    ytrva = trva[TARGET_COL].to_numpy()
    Xte = te[num_cols + cat_cols + fp_cols]
    yte = te[TARGET_COL].to_numpy()

    cand_map = {name: (pipe, meta) for name, pipe, meta in build_candidates(num_cols, cat_cols, fp_cols, use_fp)}
    if best_name not in cand_map:
        raise KeyError(f"Best candidate not found: {best_name}")
    pipe, meta = cand_map[best_name]
    pipe.fit(Xtrva, ytrva)
    yhat = pipe.predict(Xte)

    stats = {
        "n_trainval": int(len(trva)),
        "n_test": int(len(te)),
        "test_MAE": float(mean_absolute_error(yte, yhat)),
        "test_RMSE": float(math.sqrt(mean_squared_error(yte, yhat))),
        "test_R2": float(r2_score(yte, yhat)),
    }
    return pipe, stats, yte, yhat, te

# ---------------- Plotting & Analysis ----------------
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

def compare_bars(metrics_df: pd.DataFrame):
    order = list(metrics_df["model"])
    x = np.arange(len(order)); width = 0.4
    plt.figure(figsize=(8,5)); plt.bar(x, metrics_df["test_MAE"], width)
    plt.xticks(x, order, rotation=15); plt.ylabel("MAE (pct points)"); plt.title("Final Test MAE by Variant")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "compare_test_mae.png", dpi=220); plt.close()
    plt.figure(figsize=(8,5)); plt.bar(x, metrics_df["test_RMSE"], width)
    plt.xticks(x, order, rotation=15); plt.ylabel("RMSE (pct points)"); plt.title("Final Test RMSE by Variant")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "compare_test_rmse.png", dpi=220); plt.close()

def feature_importance_plot_if_any(pipeline: Pipeline, title: str, out_path: Path):
    try:
        model = pipeline.named_steps.get("model", None)
        pre = pipeline.named_steps.get("pre", None)
        if model is None or pre is None or not hasattr(model, "feature_importances_"):
            return
        names = []
        if hasattr(pre, "get_feature_names_out"):
            fn = pre.get_feature_names_out()
            names = list(fn)
        imp = getattr(model, "feature_importances_", None)
        if imp is None:
            return
        df_imp = pd.DataFrame({"feature": names if len(names)==len(imp) else [f"f{i}" for i in range(len(imp))],
                               "importance": imp})
        df_imp = df_imp.sort_values("importance", ascending=False).head(20)[::-1]
        plt.figure(figsize=(7,6))
        plt.barh(df_imp["feature"], df_imp["importance"])
        plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
    except Exception:
        pass

def groupwise_mae(te_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, out_prefix: str):
    df = te_df[[TRIP_ID_COL, SEG_ID_COL]].copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["abs_err"] = np.abs(df["y_true"] - df["y_pred"])

    if "fp_coldstart" in te_df.columns:
        g = df.join(te_df[["fp_coldstart"]]).groupby("fp_coldstart")["abs_err"].mean().reset_index()
        g.to_csv(ANALYSIS_DIR / f"{out_prefix}_mae_by_fp_coldstart.csv", index=False)

    if "prior_seg_count" in te_df.columns:
        bins = [-np.inf, 1, 3, np.inf]
        labels = ["0-1", "2-3", "4+"]
        tmp = te_df[["prior_seg_count"]].copy()
        tmp["prior_bin"] = pd.cut(tmp["prior_seg_count"], bins=bins, labels=labels)
        g = df.join(tmp[["prior_bin"]]).groupby("prior_bin")["abs_err"].mean().reset_index()
        g.to_csv(ANALYSIS_DIR / f"{out_prefix}_mae_by_prior_bins.csv", index=False)

# ---------------- Main ----------------
def main():
    _ensure_dirs()

    tgt_df = build_segment_targets(CLEANED_DATA_PATH, SEGMENTATION_DICT_PATH)
    fp_df = load_fingerprints(FINGERPRINTS_ALL)
    t0_df = load_t0_curated(NONBEHAV_T0_CURATED_PATH)
    prior_df = load_prior_aggs(BEHAVIOR_PRIOR_AGGS_PATH)

    variants = {
        "M1_fp_t0":    dict(use_fp=True,  use_priors=False, use_t0=True),
        "M2_prior_t0": dict(use_fp=False, use_priors=True,  use_t0=True),
        "M3_t0_only":  dict(use_fp=False, use_priors=False, use_t0=True),
        "M4_all":      dict(use_fp=True,  use_priors=True,  use_t0=True),
    }

    assembled = {}
    for key, cfg in variants.items():
        df_k, num_k, cat_k, fp_k = assemble_dataset(tgt_df, fp_df, t0_df, prior_df, **cfg)
        assembled[key] = (df_k, num_k, cat_k, fp_k)

    # Align rows across variants for fair comparison
    common = None
    for key, (df_k, _, _, _) in assembled.items():
        kset = set(map(tuple, df_k[[TRIP_ID_COL, SEG_ID_COL, "split"]].to_numpy()))
        common = kset if common is None else (common & kset)

    def restrict(df):
        mask = df[[TRIP_ID_COL, SEG_ID_COL, "split"]].apply(tuple, axis=1).isin(common)
        return df.loc[mask].reset_index(drop=True)

    for key in list(assembled.keys()):
        df_k, num_k, cat_k, fp_k = assembled[key]
        df_k = restrict(df_k)
        assembled[key] = (df_k, num_k, cat_k, fp_k)

    # Selection per variant
    chosen = {}
    for key, (df_k, num_k, cat_k, fp_k) in assembled.items():
        use_fp = (len(fp_k) > 0)
        rep = evaluate_on_val(df_k, num_k, cat_k, fp_k, use_fp)
        rep.to_csv(SELECTION_DIR / f"val_report_{key}.csv", index=False)
        best = rep.iloc[0]["candidate"]
        chosen[key] = best
        meta = json.loads(rep.iloc[0]["meta"])
        with open(CHOSEN_DIR / f"{key}_chosen.json", "w") as f:
            json.dump({"best_candidate": best, "meta": meta, "val_best_MAE": float(rep.iloc[0]['val_MAE'])}, f, indent=2)
        print(f"[{key}] Best on Val: {best} | Val MAE={rep.iloc[0]['val_MAE']:.4f} | Meta={meta}")

    # Final Train+Val fit and Test eval + plots
    final_rows = []
    for key, (df_k, num_k, cat_k, fp_k) in assembled.items():
        use_fp = (len(fp_k) > 0)
        pipe, stats, y, yhat, te = fit_final_and_test(df_k, num_k, cat_k, fp_k, chosen[key], use_fp)
        pd.DataFrame({TRIP_ID_COL: te[TRIP_ID_COL].values, SEG_ID_COL: te[SEG_ID_COL].values, "y_true": y, "y_pred": yhat}).to_parquet(PRED_DIR / f"predictions_test_{key}.parquet", index=False)
        parity_plot(y, yhat, f"Parity — Test — {key}", PLOTS_DIR / f"parity_test_{key}.png")
        cdf_abs_err_plot(y, yhat, f"CDF |Error| — Test — {key}", PLOTS_DIR / f"cdf_abs_err_test_{key}.png")
        feature_importance_plot_if_any(pipe, f"Top-20 Feature Importances — {key}", PLOTS_DIR / f"feat_importance_top20_{key}.png")
        groupwise_mae(te, y, yhat, out_prefix=key)
        final_rows.append({"model": key, **stats})

    final_df = pd.DataFrame(final_rows).sort_values("test_MAE").reset_index(drop=True)
    final_df.to_csv(OUT_DIR / "metrics_final.csv", index=False)
    compare_bars(final_df)

    print("\n=== Final Test Metrics ===")
    print(final_df)


if __name__ == "__main__":
    main()
