"""ChEMBL35 cold-evaluation: warm / cold_target / cold_drug.

Reads per-method predictions from outputs/*/*chembl35*/latest/predictions.csv,
filters compound-protein pairs by novelty relative to the Boltz-2 training data,
then aggregates per-protein metrics. Three modes:

  - warm:        all pairs (baseline)
  - cold_target: Sequence_identity < 100  (target absent from training PDB at 100% identity)
  - cold_drug:   Compound_structural_similarity < 0.6  (compound far from ChEMBL v34)

Filtering is done at the pair level; metrics are then computed per protein
(UniProt_ID prefix of compound_id). A protein may appear in multiple modes
with different N, since warm/cold-drug subsets carry different pair counts.

Output (results/):
  - chembl35_warm_per_target.csv
  - chembl35_cold_target_per_target.csv
  - chembl35_cold_drug_per_target.csv

Format matches results/per_target_evaluation.csv:
  Model, Dataset, Target, N, RMSE, MSE, Pearson, Spearman, Kendall, CI, Rm2
"""
import argparse
import glob
import logging
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)
from plabench.analysis.metrics import calculate_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DEFAULT_GT = os.path.join(PROJECT_ROOT, "data/chembl35/chembl35_full_input.csv")

COLD_TARGET_MAX_SEQ_ID = 100   # strict <100% to match user spec
COLD_DRUG_MAX_SIM = 0.6        # Tanimoto Morgan FP

MIN_PAIRS_PER_PROTEIN = 3      # consistent with scripts/collect_results.py


def discover_predictions():
    pattern = os.path.join(
        PROJECT_ROOT, "outputs", "*", "*chembl35*", "latest", "predictions.csv"
    )
    return sorted(glob.glob(pattern))


def per_protein_metrics(pairs_df, model_name, dataset_name):
    if pairs_df.empty:
        return []
    pairs_df = pairs_df.copy()
    pairs_df["Target"] = pairs_df["compound_id"].str.split("_CHEMBL").str[0]
    rows = []
    for target, g in pairs_df.groupby("Target"):
        if len(g) < MIN_PAIRS_PER_PROTEIN:
            continue
        m = calculate_metrics(g["prediction"].tolist(), g["affinity"].tolist())
        m["Model"] = model_name
        m["Dataset"] = dataset_name
        m["Target"] = target
        m["N"] = len(g)
        rows.append(m)
    return rows


def write_csv(rows, path):
    cols = ["Model", "Dataset", "Target", "N",
            "RMSE", "MSE", "Pearson", "Spearman", "Kendall", "CI", "Rm2"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].sort_values(by=["Dataset", "Model", "Target"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Wrote {len(df)} rows -> {os.path.relpath(path, PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", default=DEFAULT_GT,
                        help="Ground truth CSV with Sequence_identity & Compound_structural_similarity")
    parser.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "results"))
    args = parser.parse_args()

    gt_df = pd.read_csv(args.gt)
    needed = {"compound_id", "affinity", "Sequence_identity", "Compound_structural_similarity"}
    missing = needed - set(gt_df.columns)
    if missing:
        raise SystemExit(f"GT {args.gt} missing columns: {missing}")

    pred_files = discover_predictions()
    if not pred_files:
        raise SystemExit("No chembl35 predictions found under outputs/*/*chembl35*/latest/")

    log.info(f"Found {len(pred_files)} prediction file(s):")
    for pf in pred_files:
        log.info(f"  {os.path.relpath(pf, PROJECT_ROOT)}")

    warm, ct, cd = [], [], []
    for pf in pred_files:
        parts = pf.split(os.sep)
        model_name = parts[-4]
        dataset_name = parts[-3]

        preds = pd.read_csv(pf, header=None, names=["prediction", "compound_id"])
        merged = preds.merge(
            gt_df[list(needed)], on="compound_id", how="inner"
        )
        if merged.empty:
            log.warning(f"{model_name}/{dataset_name}: 0 rows after GT merge, skipping")
            continue

        warm += per_protein_metrics(merged, model_name, dataset_name)

        ct_pairs = merged[merged["Sequence_identity"] < COLD_TARGET_MAX_SEQ_ID]
        ct += per_protein_metrics(ct_pairs, model_name, dataset_name)

        cd_pairs = merged[merged["Compound_structural_similarity"] < COLD_DRUG_MAX_SIM]
        cd += per_protein_metrics(cd_pairs, model_name, dataset_name)

        log.info(
            f"{model_name}/{dataset_name}: pairs warm={len(merged)} "
            f"cold_target={len(ct_pairs)} cold_drug={len(cd_pairs)}"
        )

    write_csv(warm, os.path.join(args.out_dir, "chembl35_warm_per_target.csv"))
    write_csv(ct,   os.path.join(args.out_dir, "chembl35_cold_target_per_target.csv"))
    write_csv(cd,   os.path.join(args.out_dir, "chembl35_cold_drug_per_target.csv"))


if __name__ == "__main__":
    main()
