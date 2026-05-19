"""Compute weighted metrics for two cases:

1. multi_protein datasets: weighted average across per-target rows in
   results/per_target_evaluation.csv (each protein within a single dataset is
   a separate series, weighted by N).

2. series_group datasets: weighted average across multiple rows in
   results/benchmark_summary.csv that share the same series_group YAML flag
   (e.g., CASP16 L1000 + L3000 → one weighted row per group).

Both emit rows into a single results/weighted_summary.csv with columns:
    Model, Key, Scope, N_total, N_series, RMSE, MSE, Pearson, Spearman, Kendall, CI, Rm2
where:
    Scope = "per_target" (case 1) or "series_group" (case 2)
    Key   = Dataset name (case 1) or series_group name (case 2)

Usage:
    python analysis/weighted_summary.py
"""

import os
import glob
import logging

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

METRIC_COLS = ["RMSE", "MSE", "Pearson", "Spearman", "Kendall", "CI", "Rm2"]


def load_dataset_config(dataset_name, config_dir="configs/dataset"):
    path = os.path.join(config_dir, f"{dataset_name}.yaml")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        log.warning(f"Failed to parse {path}: {e}")
        return {}


def weighted_row(group, n_col, scope, key, model):
    total_n = group[n_col].sum()
    row = {"Model": model, "Key": key, "Scope": scope,
           "N_total": int(total_n), "N_series": len(group)}
    for col in METRIC_COLS:
        vals = pd.to_numeric(group[col], errors="coerce")
        ns = group[n_col]
        mask = vals.notna()
        if mask.sum() > 0:
            row[col] = round((vals[mask] * ns[mask]).sum() / ns[mask].sum(), 3)
        else:
            row[col] = "NA"
    return row


def summarize_multi_protein():
    """Weighted summary for multi_protein datasets (per_target_evaluation.csv)."""
    pt_path = "results/per_target_evaluation.csv"
    if not os.path.exists(pt_path):
        log.info("No per_target_evaluation.csv found, skipping multi_protein summary.")
        return []
    df = pd.read_csv(pt_path)
    rows = []
    for (model, dataset), group in df.groupby(["Model", "Dataset"]):
        rows.append(weighted_row(group, "N", "per_target", dataset, model))
    return rows


def summarize_series_groups():
    """Weighted summary for series_group datasets (from benchmark_summary.csv)."""
    bs_path = "results/benchmark_summary.csv"
    if not os.path.exists(bs_path):
        log.info("No benchmark_summary.csv found, skipping series_group summary.")
        return []
    df = pd.read_csv(bs_path)
    # Map each Dataset → series_group from its YAML config
    ds_to_group = {}
    for dataset in df["Dataset"].unique():
        cfg = load_dataset_config(dataset)
        group = cfg.get("series_group")
        if group:
            ds_to_group[dataset] = group
    if not ds_to_group:
        return []
    df["series_group"] = df["Dataset"].map(ds_to_group)
    df = df[df["series_group"].notna()]

    rows = []
    for (model, group_name), group in df.groupby(["Model", "series_group"]):
        if len(group) < 2:
            # Single-member series_group: nothing to aggregate, skip
            continue
        rows.append(weighted_row(group, "N", "series_group", group_name, model))
    return rows


def main():
    os.makedirs("results", exist_ok=True)
    rows = summarize_multi_protein() + summarize_series_groups()

    if not rows:
        log.info("No weighted summary produced (no multi_protein or series_group data found).")
        return

    result = pd.DataFrame(rows)
    cols = ["Model", "Key", "Scope", "N_total", "N_series"] + METRIC_COLS
    for col in cols:
        if col not in result.columns:
            result[col] = "NA"
    result = result[cols].sort_values(by=["Scope", "Key", "Model"])

    out = "results/weighted_summary.csv"
    result.to_csv(out, index=False)
    print("=== Weighted Summary ===\n")
    print(result.to_markdown(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
