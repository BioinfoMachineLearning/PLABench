"""Generate per-class performance summary for chembl35 (or other multi_protein datasets).

Reads `results/per_target_evaluation_chembl35.csv` (must already have a
Target_class column; for chembl35 this comes from `data/chembl35/target_class_map.csv`).

Targets with bi-functional class (e.g. CD38 = "Hydrolase|Transferase") are
counted in BOTH classes (the row is exploded on the `|` separator).

Outputs:
- results/per_class_summary_chembl35.csv (long form, one row per
  (Target_class, Model)). Each row has n_targets, n_compounds, and the
  N-weighted mean of each per-target metric (Pearson, Spearman, Kendall, RMSE,
  MSE, CI, Rm2). Weighting matches §5 (analysis/weighted_summary.py) and the
  CASP16 series_group convention used in the paper — targets with more
  compounds get larger weight because their per-target metric estimates are
  more stable (Var ∝ 1/N). NOTE: paper Table 1 uses an unweighted simple mean,
  so per-class numbers here are NOT directly comparable to paper Table 1
  values, although relative class ordering remains comparable.
- Also prints a wide pivot of N-weighted mean Pearson on stdout for quick
  inspection.

Usage:
    python scripts/per_class_summary.py [--input results/per_target_evaluation_chembl35.csv]
                                        [--output results/per_class_summary_chembl35.csv]
                                        [--dataset_filter chembl35_full]

Re-run this script after each new method's predictions are merged into
results/per_target_evaluation_chembl35.csv to refresh the summary table.
"""
import argparse
import os

import pandas as pd


METRICS = ["RMSE", "MSE", "Pearson", "Spearman", "Kendall", "CI", "Rm2"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/per_target_evaluation_chembl35.csv",
        help="Per-target eval CSV with Target_class column",
    )
    parser.add_argument(
        "--output",
        default="results/per_class_summary_chembl35.csv",
        help="Output long-form CSV path",
    )
    parser.add_argument(
        "--dataset_filter",
        default="chembl35_full",
        help="Substring filter on Dataset column (default chembl35_full to "
             "include all chembl35_full datasets across methods)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "Target_class" not in df.columns:
        raise SystemExit(
            f"Input CSV {args.input} missing Target_class column. "
            "Augment via target_class_map.csv first."
        )

    df = df[df["Dataset"].str.contains(args.dataset_filter, na=False)].copy()
    if df.empty:
        raise SystemExit(f"No rows match dataset_filter={args.dataset_filter!r}")

    # Explode bifunctional classes (pipe-separated)
    df["Target_class"] = df["Target_class"].str.split("|")
    df = df.explode("Target_class")
    df["Target_class"] = df["Target_class"].str.strip()

    # Per-class × per-method aggregation (N-weighted, matches §5 / CASP16).
    rows = []
    for (cls, model), grp in df.groupby(["Target_class", "Model"]):
        ns = pd.to_numeric(grp["N"], errors="coerce")
        row = {
            "Target_class": cls,
            "Model": model,
            "n_targets": grp["Target"].nunique(),
            "n_compounds": int(ns.sum()),
        }
        for m in METRICS:
            if m in grp.columns:
                vals = pd.to_numeric(grp[m], errors="coerce")
                mask = vals.notna() & ns.notna()
                if mask.sum() > 0 and ns[mask].sum() > 0:
                    row[f"mean_{m}"] = round(
                        (vals[mask] * ns[mask]).sum() / ns[mask].sum(), 4
                    )
                else:
                    row[f"mean_{m}"] = float("nan")
        rows.append(row)

    out = pd.DataFrame(rows)
    # Sort: by n_targets desc (most-populated class on top), then by mean_Pearson desc
    out = out.sort_values(
        by=["n_targets", "Target_class", "mean_Pearson"],
        ascending=[False, True, False],
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")

    # Quick inspection: wide-form pivot of mean Pearson
    piv = out.pivot_table(
        index="Target_class", columns="Model", values="mean_Pearson"
    )
    n_targets_per_class = (
        df.groupby("Target_class")["Target"].nunique().rename("n_targets")
    )
    piv = piv.join(n_targets_per_class)
    piv = piv.sort_values("n_targets", ascending=False)
    cols = ["n_targets"] + [c for c in piv.columns if c != "n_targets"]
    print("\n=== N-weighted mean Pearson per class (wide form for inspection) ===")
    print(piv[cols].round(3).to_string())


if __name__ == "__main__":
    main()
