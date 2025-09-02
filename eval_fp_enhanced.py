#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced evaluation for FP residual models — presentation-friendly.

Reads:
  - metrics_residual_final.csv (required)
  - test_predictions.parquet (optional: if present, makes richer plots)

Outputs:
  - CSV with absolute & % improvements vs global (R3_none)
  - Slide-ready PNGs: bar chart, waterfall, optional CDFs and per-driver improvements
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\model_soc_segment_residual_v6_fp_solo"
METRICS_CSV = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\model_soc_segment_residual_v6_fp_solo\metrics_residual_final.csv"
PREDICTIONS = BASE_DIR / "predictions" / "test_predictions.parquet"
OUT_DIR = BASE_DIR / "enhanced_eval_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_bar_and_waterfall(metrics: pd.DataFrame, out_dir: Path):
    base_mae = float(metrics.loc[metrics["variant"] == "R3_none", "test_MAE"].iloc[0])
    metrics = metrics.copy()
    metrics["impr_abs"] = base_mae - metrics["test_MAE"]
    metrics["impr_pct"] = 100.0 * metrics["impr_abs"] / base_mae
    metrics.sort_values("test_MAE", inplace=True)
    metrics.to_csv(out_dir / "fp_vs_global_improvements.csv", index=False)

    plt.figure(figsize=(8,5))
    plt.bar(range(len(metrics)), metrics["test_MAE"].to_numpy())
    plt.axhline(base_mae, linestyle="--")
    plt.xticks(range(len(metrics)), metrics["variant"].tolist(), rotation=20, ha="right")
    plt.title("Test MAE by Variant (lower is better)")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(out_dir / "bar_mae_by_variant.png", dpi=180)
    plt.close()

    wf = metrics[metrics["variant"] != "R3_none"].copy().sort_values("impr_abs", ascending=False)
    labs = ["R3_none"] + wf["variant"].tolist()
    vals = [0.0] + wf["impr_abs"].tolist()
    plt.figure(figsize=(9,5))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labs, rotation=20, ha="right")
    plt.title(f"Absolute MAE Improvement vs Global Baseline (R3_none = {base_mae:.3f})")
    plt.ylabel("ΔMAE (Global − Variant)")
    plt.tight_layout()
    plt.savefig(out_dir / "waterfall_abs_improvement_vs_global.png", dpi=180)
    plt.close()

def cdf_abs_errors(pred_df: pd.DataFrame, out_dir: Path):
    """Optional CDF of absolute errors by variant vs baseline."""
    if pred_df is None or pred_df.empty:
        return
    # Identify prediction columns by prefix 'pred_'
    pred_cols = [c for c in pred_df.columns if c.startswith("pred_")]
    if "gt" not in pred_df.columns or not pred_cols:
        return
    gt = pred_df["gt"].to_numpy()
    # Build CDF plots
    for col in pred_cols:
        abs_err = np.abs(gt - pred_df[col].to_numpy())
        abs_err = abs_err[np.isfinite(abs_err)]
        if abs_err.size == 0:
            continue
        xs = np.sort(abs_err)
        ys = np.linspace(0, 1, len(xs), endpoint=True)
        plt.figure(figsize=(6,4))
        plt.plot(xs, ys)
        plt.xlabel("Absolute Error")
        plt.ylabel("CDF")
        plt.title(f"CDF of |Error| — {col}")
        plt.tight_layout()
        fname = f"cdf_abs_error_{col}.png"
        plt.savefig(out_dir / fname, dpi=160)
        plt.close()

def main():
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {METRICS_CSV}")
    m = pd.read_csv(METRICS_CSV)
    if "R3_none" not in m["variant"].values:
        raise RuntimeError("Baseline 'R3_none' not found in metrics.")

    save_bar_and_waterfall(m, OUT_DIR)

    if PREDICTIONS.exists():
        try:
            pred_df = pd.read_parquet(PREDICTIONS)
            cdf_abs_errors(pred_df, OUT_DIR)
        except Exception as e:
            print(f"[warn] Could not read predictions for CDF plots: {e}")

    print(f"Done. Outputs saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
