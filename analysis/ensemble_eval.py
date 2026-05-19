"""
Ensemble search: enumerate all subsets of methods, average predictions,
evaluate Kendall's tau, and find the best combination.

Usage:
    python analysis/ensemble_eval.py                       # default: boltz2 structure input
    python analysis/ensemble_eval.py --mode af3            # AF3 structure input
    python analysis/ensemble_eval.py --mode boltz2         # explicit boltz2
    python analysis/ensemble_eval.py --mode chembl35       # ChEMBL35 Dataset A (legacy, 10 targets)
    python analysis/ensemble_eval.py --mode chembl35_full  # ChEMBL35 full (84 targets / 7650),
                                                           #   per-target N-weighted Pearson + Kendall
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import kendalltau, pearsonr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Structure-based methods: paths differ by mode (boltz2 vs af3)
# Sequence-based methods: paths are the same regardless of mode
METHODS_BOLTZ2 = {
    "bapred":           {"l1000": "outputs/bapred/boltz2_casp16_l1000_stage1/latest/predictions.csv",
                         "l3000": "outputs/bapred/boltz2_casp16_l3000_stage1/latest/predictions.csv"},
    "boltz2":           {"l1000": "outputs/boltz2/boltz2_casp16_l1000/latest/predictions.csv",
                         "l3000": "outputs/boltz2/boltz2_casp16_l3000/latest/predictions.csv"},
    "deepdta":          {"l1000": "outputs/deepdta/sequence_casp16_l1000/latest/predictions.csv",
                         "l3000": "outputs/deepdta/sequence_casp16_l3000/latest/predictions.csv"},
    "flowdock":         {"l1000": "outputs/flowdock/boltz2_casp16_l1000_stage1/latest/predictions.csv",
                         "l3000": "outputs/flowdock/boltz2_casp16_l3000_stage1/latest/predictions.csv"},
    # flowr_root only has an AF3-input version; per user request, included with AF3 paths in both modes.
    "flowr_root":       {"l1000": "outputs/flowr_root/flowr_root_af3_casp16_l1000_stage1/latest/predictions.csv",
                         "l3000": "outputs/flowr_root/flowr_root_af3_casp16_l3000_stage1/latest/predictions.csv"},
    "llf":              {"l1000": "outputs/llf/sequence_casp16_l1000/latest/predictions.csv",
                         "l3000": "outputs/llf/sequence_casp16_l3000/latest/predictions.csv"},
    "llf_stage2":       {"l1000": "outputs/llf/sequence_casp16_l1000_stage2_finetune/latest/predictions.csv",
                         "l3000": "outputs/llf/sequence_casp16_l3000_stage2_finetune/latest/predictions.csv"},
    "mfe":              {"l1000": "outputs/mfe/mfe_casp16_l1000_stage1/latest/predictions.csv",
                         "l3000": "outputs/mfe/mfe_casp16_l3000_stage1/latest/predictions.csv"},
    "mixingdta":        {"l1000": "outputs/mixingdta/sequence_casp16_l1000/latest/predictions.csv",
                         "l3000": "outputs/mixingdta/sequence_casp16_l3000/latest/predictions.csv"},
    "mixingdta_stage2": {"l1000": "outputs/mixingdta_stage2/sequence_casp16_l1000/latest/predictions.csv",
                         "l3000": "outputs/mixingdta_stage2/sequence_casp16_l3000/latest/predictions.csv"},
}

METHODS_AF3 = {
    "bapred":           {"l1000": "outputs/bapred/af3_casp16_l1000_stage1/latest/predictions.csv",
                         "l3000": "outputs/bapred/af3_casp16_l3000_stage1/latest/predictions.csv"},
    "boltz2":           {"l1000": "outputs/boltz2/boltz2_casp16_l1000/latest/predictions.csv",
                         "l3000": "outputs/boltz2/boltz2_casp16_l3000/latest/predictions.csv"},
    "deepdta":          {"l1000": "outputs/deepdta/sequence_casp16_l1000/latest/predictions.csv",
                         "l3000": "outputs/deepdta/sequence_casp16_l3000/latest/predictions.csv"},
    "flowdock":         {"l1000": "outputs/flowdock/af3_casp16_l1000_stage1/latest/predictions.csv",
                         "l3000": "outputs/flowdock/af3_casp16_l3000_stage1/latest/predictions.csv"},
    "flowr_root":       {"l1000": "outputs/flowr_root/flowr_root_af3_casp16_l1000_stage1/latest/predictions.csv",
                         "l3000": "outputs/flowr_root/flowr_root_af3_casp16_l3000_stage1/latest/predictions.csv"},
    "haiping":          {"l1000": "outputs/haiping/haiping_af3_casp16_l1000_stage1/latest/predictions.csv",
                         "l3000": "outputs/haiping/haiping_af3_casp16_l3000_stage1/latest/predictions.csv"},
    "llf":              {"l1000": "outputs/llf/sequence_casp16_l1000/latest/predictions.csv",
                         "l3000": "outputs/llf/sequence_casp16_l3000/latest/predictions.csv"},
    "llf_stage2":       {"l1000": "outputs/llf/sequence_casp16_l1000_stage2_finetune/latest/predictions.csv",
                         "l3000": "outputs/llf/sequence_casp16_l3000_stage2_finetune/latest/predictions.csv"},
    "mfe":              {"l1000": "outputs/mfe/mfe_af3_casp16_l1000_stage1/latest/predictions.csv",
                         "l3000": "outputs/mfe/mfe_af3_casp16_l3000_stage1/latest/predictions.csv"},
    "mixingdta":        {"l1000": "outputs/mixingdta/sequence_casp16_l1000/latest/predictions.csv",
                         "l3000": "outputs/mixingdta/sequence_casp16_l3000/latest/predictions.csv"},
    "mixingdta_stage2": {"l1000": "outputs/mixingdta_stage2/sequence_casp16_l1000/latest/predictions.csv",
                         "l3000": "outputs/mixingdta_stage2/sequence_casp16_l3000/latest/predictions.csv"},
}

GT_FILES = {
    "l1000": "data/casp16_data/labels/L1000_exper_affinity.csv",
    "l3000": "data/casp16_data/labels/L3000_exper_affinity.csv",
}

# ChEMBL35 mode: one predictions.csv per method (spans all 10 targets).
# Splitting per-target happens at eval time by compound_id prefix.
METHODS_CHEMBL35 = {
    "boltz2":    "outputs/boltz2/boltz2_chembl35_A/latest/predictions.csv",
    "bapred":    "outputs/bapred/af3_chembl35_A/latest/predictions.csv",
    "flowdock":  "outputs/flowdock/af3_chembl35_A/latest/predictions.csv",
    "haiping":   "outputs/haiping/haiping_af3_chembl35_A/latest/predictions.csv",
    "mfe":       "outputs/mfe/mfe_af3_chembl35_A/latest/predictions.csv",
    "llf":       "outputs/llf/sequence_chembl35_A/latest/predictions.csv",
    "mixingdta": "outputs/mixingdta/sequence_chembl35_A/latest/predictions.csv",
}

# AF3-structure methods use compound_ids like "P08183_000" (from AF3 naming).
# All other methods (boltz2/llf/mixingdta) use "P08183_CHEMBL..." (from boltz2 input CSV).
# We bridge via row-order mapping within each target (rebuilt from dataset_A.csv).
CHEMBL35_AF3_METHODS = {"bapred", "flowdock", "haiping", "mfe"}
GT_CHEMBL35_AF3 = "data/chembl35/chembl35_af3_gt.csv"
GT_CHEMBL35_BOLTZ2 = "data/chembl35/chembl35_datasetA_boltz2_input.csv"

# --------------------------------------------------------------------------------------
# ChEMBL35 FULL mode (84 targets / 7650 compounds, unified compound_id naming).
# All 8 methods use {UniProt}_CHEMBL... IDs (the AF3 dirs were renamed in chembl35.md §1.5),
# so no bridge dict is needed. predictions.csv layout is uniform: `prediction,compound_id`,
# no header.
# --------------------------------------------------------------------------------------
METHODS_CHEMBL35_FULL = {
    "boltz2":            "outputs/boltz2/boltz2_chembl35_full/latest/predictions.csv",
    "flowr_root":        "outputs/flowr_root/flowr_root_af3_chembl35_full/latest/predictions.csv",
    "flowdock":          "outputs/flowdock/af3_chembl35_full/latest/predictions.csv",
    "mixingdta_stage2":  "outputs/mixingdta_stage2/sequence_chembl35_full/latest/predictions.csv",
    "haiping":           "outputs/haiping/haiping_af3_chembl35_full/latest/predictions.csv",
    "mfe":               "outputs/mfe/mfe_af3_chembl35_full/latest/predictions.csv",
    "bapred":            "outputs/bapred/af3_chembl35_full/latest/predictions.csv",
    "llf_stage2":        "outputs/llf/sequence_chembl35_full_stage2_finetune/latest/predictions.csv",
}
GT_CHEMBL35_FULL = "data/chembl35/chembl35_full_input.csv"


def load_predictions(methods_dict, method, dataset):
    """Load predictions for a method+dataset, return DataFrame with columns [target_id, prediction]."""
    path = os.path.join(PROJECT_ROOT, methods_dict[method][dataset])
    df = pd.read_csv(path, header=None, names=["prediction", "target_id"])
    # Handle duplicates by averaging
    df = df.groupby("target_id", as_index=False)["prediction"].mean()
    return df


def load_ground_truth(dataset):
    """Load GT, convert kJ/mol -> pKd, return DataFrame with columns [target_id, gt]."""
    path = os.path.join(PROJECT_ROOT, GT_FILES[dataset])
    df = pd.read_csv(path)
    df = df[["Target ID", "binding_affinity"]].copy()
    df.columns = ["target_id", "gt"]
    df["gt"] = df["gt"] * -0.733
    return df


def eval_ensemble(method_subset, pred_dfs, gt_df):
    """Evaluate a subset of methods: average predictions, compute Kendall's tau."""
    # Find common targets
    common_ids = set(gt_df["target_id"])
    for m in method_subset:
        common_ids &= set(pred_dfs[m]["target_id"])
    common_ids = sorted(common_ids)

    if len(common_ids) < 2:
        return None

    # Build aligned arrays
    gt_map = gt_df.set_index("target_id")["gt"]
    y_true = np.array([gt_map[tid] for tid in common_ids])

    # Average predictions
    pred_sum = np.zeros(len(common_ids))
    for m in method_subset:
        m_map = pred_dfs[m].set_index("target_id")["prediction"]
        pred_sum += np.array([m_map[tid] for tid in common_ids])
    y_pred = pred_sum / len(method_subset)

    tau, _ = kendalltau(y_true, y_pred)
    if len(common_ids) >= 3 and np.std(y_pred) > 0 and np.std(y_true) > 0:
        r_p, _ = pearsonr(y_true, y_pred)
    else:
        r_p = float("nan")
    return {
        "methods": "+".join(sorted(method_subset)),
        "n_methods": len(method_subset),
        "kendall": round(tau, 4),
        "pearson": round(float(r_p), 4) if r_p == r_p else float("nan"),
        "n_targets": len(common_ids),
    }


def run_dataset(methods_dict, dataset):
    """Evaluate all subsets for one dataset."""
    method_names = list(methods_dict.keys())

    # Load all predictions
    pred_dfs = {}
    for m in method_names:
        pred_dfs[m] = load_predictions(methods_dict, m, dataset)

    gt_df = load_ground_truth(dataset)

    n = len(method_names)
    total = 2**n - 1
    print(f"  {dataset.upper()}: {n} methods, {total} subsets")

    # Generate all non-empty subsets
    subsets = []
    for r in range(1, n + 1):
        for combo in combinations(method_names, r):
            subsets.append(combo)

    # Parallel evaluation
    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(eval_ensemble, subset, pred_dfs, gt_df): subset
                   for subset in subsets}
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)

    df = pd.DataFrame(results).sort_values("kendall", ascending=False).reset_index(drop=True)
    return df


def _build_chembl35_id_bridge():
    """Map boltz2-scheme compound IDs (UniProt_CHEMBL...) to AF3-scheme (UniProt_NNN).

    Both GT files list compounds in the same row order within each target group,
    so we align by (target, within-target index).
    """
    af3 = pd.read_csv(os.path.join(PROJECT_ROOT, GT_CHEMBL35_AF3))
    b2 = pd.read_csv(os.path.join(PROJECT_ROOT, GT_CHEMBL35_BOLTZ2))
    af3["target"] = af3["compound_id"].astype(str).str.split("_").str[0]
    b2["target"] = b2["compound_id"].astype(str).str.split("_").str[0]
    af3["row"] = af3.groupby("target").cumcount()
    b2["row"] = b2.groupby("target").cumcount()
    bridge = pd.merge(
        b2[["target", "row", "compound_id"]].rename(columns={"compound_id": "b2_id"}),
        af3[["target", "row", "compound_id"]].rename(columns={"compound_id": "af3_id"}),
        on=["target", "row"],
    )
    return dict(zip(bridge["b2_id"], bridge["af3_id"]))


def load_chembl35_predictions(method, b2_to_af3):
    """Load predictions.csv, normalize all compound_ids to AF3 scheme."""
    path = os.path.join(PROJECT_ROOT, METHODS_CHEMBL35[method])
    df = pd.read_csv(path, header=None, names=["prediction", "target_id"])
    df = df.groupby("target_id", as_index=False)["prediction"].mean()
    if method not in CHEMBL35_AF3_METHODS:
        df["target_id"] = df["target_id"].map(b2_to_af3)
        df = df.dropna(subset=["target_id"])
    return df


def run_chembl35():
    """ChEMBL35: for each of 10 targets, reuse eval_ensemble per target; then weight by N."""
    method_names = list(METHODS_CHEMBL35.keys())
    n = len(method_names)
    total = 2 ** n - 1
    print(f"  ChEMBL35: {n} methods, {total} subsets, 10 targets")

    # Bridge IDs and load all predictions once (normalized to AF3 scheme)
    bridge = _build_chembl35_id_bridge()
    all_preds = {m: load_chembl35_predictions(m, bridge) for m in method_names}

    # GT in AF3 scheme
    gt_full = pd.read_csv(os.path.join(PROJECT_ROOT, GT_CHEMBL35_AF3))
    gt_full = gt_full.rename(columns={"compound_id": "target_id", "affinity": "gt"})
    gt_full["target"] = gt_full["target_id"].astype(str).str.split("_").str[0]

    # Enumerate subsets once
    subsets = []
    for r in range(1, n + 1):
        for combo in combinations(method_names, r):
            subsets.append(combo)

    # For each target, filter preds/gt and evaluate all subsets
    targets = sorted(gt_full["target"].unique())
    per_target_dfs = {}

    def _eval_for_target(target):
        target_gt = gt_full[gt_full["target"] == target][["target_id", "gt"]]
        target_preds = {}
        for m in method_names:
            df = all_preds[m]
            target_preds[m] = df[df["target_id"].astype(str).str.startswith(f"{target}_")]
        rows = []
        for subset in subsets:
            res = eval_ensemble(subset, target_preds, target_gt)
            if res is not None:
                res["target"] = target
                rows.append(res)
        return pd.DataFrame(rows)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_eval_for_target, t): t for t in targets}
        for fut in as_completed(futures):
            t = futures[fut]
            per_target_dfs[t] = fut.result()

    # Aggregate: weighted Kendall per subset across all targets
    all_df = pd.concat(per_target_dfs.values(), ignore_index=True)
    agg_rows = []
    for methods_key, group in all_df.groupby("methods"):
        # Require at least 3 samples per target (eval_ensemble skips otherwise)
        total_n = group["n_targets"].sum()
        if total_n == 0:
            continue
        weighted = (group["kendall"] * group["n_targets"]).sum() / total_n
        agg_rows.append({
            "methods": methods_key,
            "n_methods": len(methods_key.split("+")),
            "weighted_kendall": round(weighted, 4),
            "n_total": int(total_n),
            "n_targets_groups": len(group),
        })
    agg = pd.DataFrame(agg_rows).sort_values("weighted_kendall", ascending=False).reset_index(drop=True)
    return agg, all_df


def run_chembl35_full(methods_subset=None):
    """ChEMBL35 full set (84 targets / 7650 compounds).

    All preds + GT joined once into one wide ndarray; per-target slices reused across
    every subset. Pearson is fully vectorized (numpy); Kendall is the only Python-level
    loop (scipy.stats.kendalltau on n<=727 vectors), parallelized via ThreadPoolExecutor.

    Returns (agg_df, per_target_df). Both have weighted_pearson and weighted_kendall.
    Aggregates are weighted by per-target compound counts N (drops targets with N<3
    where Kendall is degenerate).
    """
    method_names = list(methods_subset or METHODS_CHEMBL35_FULL.keys())
    n_methods = len(method_names)
    total_subsets = 2 ** n_methods - 1
    print(f"  ChEMBL35-full: {n_methods} methods, {total_subsets} subsets")

    # --- Load GT ---
    gt = pd.read_csv(os.path.join(PROJECT_ROOT, GT_CHEMBL35_FULL))
    gt = gt[["compound_id", "affinity"]].rename(columns={"affinity": "gt"})
    gt["target"] = gt["compound_id"].astype(str).str.split("_").str[0]

    # --- Load all method predictions; outer-merge into one wide DataFrame ---
    wide = gt[["compound_id", "target", "gt"]].copy()
    for m in method_names:
        path = os.path.join(PROJECT_ROOT, METHODS_CHEMBL35_FULL[m])
        df = pd.read_csv(path, header=None, names=[m, "compound_id"])
        df = df.groupby("compound_id", as_index=False)[m].mean()
        wide = wide.merge(df, on="compound_id", how="left")

    # Drop rows where any selected method is NaN (require full coverage for fair ensemble)
    before = len(wide)
    wide = wide.dropna(subset=method_names).reset_index(drop=True)
    print(f"  Coverage: {len(wide)}/{before} compounds with all {n_methods} methods present")

    # --- Build per-target arrays once: (gt_vec, pred_matrix [n_compounds × n_methods]) ---
    targets = sorted(wide["target"].unique())
    per_target_arrays = []
    for t in targets:
        sub = wide[wide["target"] == t]
        if len(sub) < 3:
            continue
        gt_vec = sub["gt"].to_numpy(dtype=np.float64)
        pmat = sub[method_names].to_numpy(dtype=np.float64)
        per_target_arrays.append((t, gt_vec, pmat))
    print(f"  Per-target arrays: {len(per_target_arrays)} targets (>=3 compounds)")

    # --- Enumerate subsets ---
    subsets = []
    for r in range(1, n_methods + 1):
        for combo in combinations(range(n_methods), r):
            subsets.append(combo)

    # Vectorized Pearson: corrcoef(gt, mean_pred). Precompute per-target gt centered/std.
    def _pearson(y, p):
        ym = y - y.mean()
        pm = p - p.mean()
        denom = np.sqrt((ym * ym).sum() * (pm * pm).sum())
        if denom == 0:
            return float("nan")
        return float((ym * pm).sum() / denom)

    # Each (subset, target) -> pearson, kendall, n
    rows = []

    def _eval_target(t, gt_vec, pmat):
        out = []
        for combo in subsets:
            mean_pred = pmat[:, list(combo)].mean(axis=1)
            r_p = _pearson(gt_vec, mean_pred)
            tau, _ = kendalltau(gt_vec, mean_pred)
            out.append({
                "target": t,
                "subset_idx": combo,
                "n_methods": len(combo),
                "n": len(gt_vec),
                "pearson": r_p,
                "kendall": float(tau) if tau == tau else float("nan"),
            })
        return out

    with ThreadPoolExecutor(max_workers=min(16, len(per_target_arrays))) as ex:
        futures = [ex.submit(_eval_target, t, g, p) for t, g, p in per_target_arrays]
        for fut in as_completed(futures):
            rows.extend(fut.result())

    per_target = pd.DataFrame(rows)
    per_target["methods"] = per_target["subset_idx"].apply(
        lambda c: "+".join(sorted(method_names[i] for i in c))
    )

    # --- Aggregate: N-weighted means across targets, per subset ---
    agg = []
    for methods_key, grp in per_target.groupby("methods"):
        g = grp.dropna(subset=["pearson", "kendall"])
        if g.empty:
            continue
        n_total = int(g["n"].sum())
        w = g["n"].to_numpy(dtype=np.float64)
        agg.append({
            "methods": methods_key,
            "n_methods": int(grp["n_methods"].iloc[0]),
            "weighted_pearson": float(np.average(g["pearson"], weights=w)),
            "weighted_kendall": float(np.average(g["kendall"], weights=w)),
            "n_total": n_total,
            "n_targets": int(len(g)),
        })
    agg_df = pd.DataFrame(agg).sort_values("weighted_pearson", ascending=False).reset_index(drop=True)
    return agg_df, per_target.drop(columns=["subset_idx"])


def main():
    parser = argparse.ArgumentParser(description="Ensemble search for PLA methods")
    parser.add_argument("--mode",
                        choices=["boltz2", "af3", "chembl35", "chembl35_full"],
                        default="boltz2",
                        help="Structure input mode: boltz2 (default), af3, chembl35, or chembl35_full")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated subset of methods to enumerate (chembl35_full only). "
                             "Defaults to all 8 listed in METHODS_CHEMBL35_FULL.")
    args = parser.parse_args()

    if args.mode == "chembl35_full":
        out_dir = os.path.join(PROJECT_ROOT, "outputs", "ensemble_search")
        os.makedirs(out_dir, exist_ok=True)
        subset = args.methods.split(",") if args.methods else None
        agg, per_target = run_chembl35_full(subset)
        agg.to_csv(os.path.join(out_dir, "ensemble_results_chembl35_full.csv"), index=False)
        per_target.to_csv(os.path.join(out_dir, "ensemble_results_chembl35_full_per_target.csv"),
                          index=False)

        print(f"\n{'=' * 100}")
        print("Top 10 ChEMBL35-full Ensemble Combinations (sorted by weighted Pearson)")
        print(f"{'=' * 100}")
        for i, row in agg.head(10).iterrows():
            print(f"{i + 1:2d}. {row['methods']}")
            print(f"    weighted_pearson={row['weighted_pearson']:.4f}  "
                  f"weighted_kendall={row['weighted_kendall']:.4f}  "
                  f"n_total={int(row['n_total'])}  n_targets={int(row['n_targets'])}")

        print(f"\n{'=' * 100}")
        print("Top 10 by weighted Kendall")
        print(f"{'=' * 100}")
        for i, row in agg.sort_values("weighted_kendall", ascending=False).head(10).reset_index(drop=True).iterrows():
            print(f"{i + 1:2d}. {row['methods']}")
            print(f"    weighted_pearson={row['weighted_pearson']:.4f}  "
                  f"weighted_kendall={row['weighted_kendall']:.4f}")

        print(f"\nResults saved to {out_dir}/ensemble_results_chembl35_full{{,_per_target}}.csv")
        return

    if args.mode == "chembl35":
        out_dir = os.path.join(PROJECT_ROOT, "outputs", "ensemble_search")
        os.makedirs(out_dir, exist_ok=True)
        agg, per_target = run_chembl35()
        agg.to_csv(os.path.join(out_dir, "ensemble_results_chembl35.csv"), index=False)
        per_target.to_csv(os.path.join(out_dir, "ensemble_results_chembl35_per_target.csv"), index=False)

        print(f"\n{'='*90}")
        print(f"Top 10 ChEMBL35 Ensemble Combinations (weighted Kendall across 10 targets)")
        print(f"{'='*90}")
        for i, row in agg.head(10).iterrows():
            print(f"{i+1:2d}. {row['methods']}")
            print(f"    weighted_kendall={row['weighted_kendall']:.4f}, n_total={int(row['n_total'])}")
        print(f"\nResults saved to {out_dir}/ensemble_results_chembl35.csv")
        return

    methods_dict = METHODS_AF3 if args.mode == "af3" else METHODS_BOLTZ2
    n = len(methods_dict)
    total = 2**n - 1
    print(f"Running ensemble evaluation (mode={args.mode}, {n} methods, {total} subsets per dataset)...")

    # Run both datasets in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_l1000 = executor.submit(run_dataset, methods_dict, "l1000")
        f_l3000 = executor.submit(run_dataset, methods_dict, "l3000")
        df_l1000 = f_l1000.result()
        df_l3000 = f_l3000.result()

    # Save per-dataset results
    suffix = f"_{args.mode}" if args.mode != "boltz2" else ""
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "ensemble_search")
    os.makedirs(out_dir, exist_ok=True)

    df_l1000.to_csv(os.path.join(out_dir, f"ensemble_results_l1000{suffix}.csv"), index=False)
    df_l3000.to_csv(os.path.join(out_dir, f"ensemble_results_l3000{suffix}.csv"), index=False)

    # Combined: weighted average by n_targets
    combined = pd.merge(df_l1000, df_l3000, on="methods", suffixes=("_l1000", "_l3000"))
    n_total = combined["n_targets_l1000"] + combined["n_targets_l3000"]
    combined["weighted_kendall"] = (
        (combined["kendall_l1000"] * combined["n_targets_l1000"] +
         combined["kendall_l3000"] * combined["n_targets_l3000"]) / n_total
    ).round(4)
    combined["weighted_pearson"] = (
        (combined["pearson_l1000"] * combined["n_targets_l1000"] +
         combined["pearson_l3000"] * combined["n_targets_l3000"]) / n_total
    ).round(4)
    combined = combined.sort_values("weighted_kendall", ascending=False).reset_index(drop=True)
    combined.to_csv(os.path.join(out_dir, f"ensemble_results_combined{suffix}.csv"), index=False)

    # Print top 10
    print(f"\n{'='*100}")
    print(f"Top 10 Ensemble Combinations - mode={args.mode} (sorted by weighted Kendall)")
    print(f"{'='*100}")
    for i, row in combined.head(10).iterrows():
        print(f"{i+1:2d}. {row['methods']}")
        print(f"    L1000: tau={row['kendall_l1000']:.4f} r={row['pearson_l1000']:.4f} "
              f"(n={int(row['n_targets_l1000'])})  |  "
              f"L3000: tau={row['kendall_l3000']:.4f} r={row['pearson_l3000']:.4f} "
              f"(n={int(row['n_targets_l3000'])})  |  "
              f"Weighted tau={row['weighted_kendall']:.4f} r={row['weighted_pearson']:.4f}")

    print(f"\n{'='*100}")
    print(f"Top 10 by weighted Pearson")
    print(f"{'='*100}")
    for i, row in combined.sort_values("weighted_pearson", ascending=False).head(10).reset_index(drop=True).iterrows():
        print(f"{i+1:2d}. {row['methods']}")
        print(f"    Weighted tau={row['weighted_kendall']:.4f} r={row['weighted_pearson']:.4f}")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
