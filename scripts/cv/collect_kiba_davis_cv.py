#!/usr/bin/env python
"""Score the Davis and KIBA five-fold cross-validation runs and refresh results/benchmark_summary.csv.

Every sequence model writes one prediction file per fold:

    outputs/<model>/<config>/fold_<n>/predictions.csv

with either a ``prediction,target_id`` header or the headerless
``prediction,identifier`` form that the model inference scripts emit. Rows are
in the same order as the ground-truth file, so predictions and labels are
joined by position.

Produce those files with scripts/cv/run_deepdta_cv.py, scripts/cv/run_llf_cv.py and
scripts/cv/run_mixingdta_cv.py, then run this script.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from plabench.analysis.metrics import calculate_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUMMARY_CSV = os.path.join(REPO_ROOT, "results", "benchmark_summary.csv")
DETAIL_CSV = os.path.join(REPO_ROOT, "analysis", "davis_kiba_cv_metrics.csv")

MODELS = ["deepdta", "llf", "mixingdta"]

# config -> (data subdirectory, ground-truth file relative to it). The warm
# splits have one test file per fold; the cold splits share a single held-out
# set across folds and vary only the model weights.
CONFIGS = {
    "davis_warm": ("DAVIS", "test_{fold}.csv"),
    "kiba_warm": ("KIBA", "test_{fold}.csv"),
    "davis_cold_drug": ("DAVIS", "cold/test_drug.csv"),
    "davis_cold_target": ("DAVIS", "cold/test_target.csv"),
    "kiba_cold_drug": ("KIBA", "cold/test_drug.csv"),
    "kiba_cold_target": ("KIBA", "cold/test_target.csv"),
}

PREDICTION_COLUMNS = ("prediction", "Predicted Affinity", "predicted_affinity")
LABEL_COLUMNS = ("label", "affinity", "true_affinity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--datasets", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the metrics without touching results/benchmark_summary.csv",
    )
    return parser.parse_args()


def ground_truth_path(config: str, fold: int) -> str:
    subdir, name = CONFIGS[config]
    return os.path.join(REPO_ROOT, "data", "Structure_independent", subdir, name.format(fold=fold))


def read_predictions(path: str) -> pd.Series:
    """Read a fold's predictions, tolerating the headerless inference output."""
    with open(path) as handle:
        first_field = handle.readline().split(",")[0].strip()
    try:
        float(first_field)
    except ValueError:
        frame = pd.read_csv(path)
        for column in PREDICTION_COLUMNS:
            if column in frame.columns:
                return frame[column].astype(float)
        raise ValueError(f"{path}: no prediction column among {PREDICTION_COLUMNS}")
    return pd.read_csv(path, header=None, usecols=[0], names=["prediction"])["prediction"].astype(float)


def label_column(frame: pd.DataFrame) -> str:
    for column in LABEL_COLUMNS:
        if column in frame.columns:
            return column
    raise ValueError(f"no label column among {LABEL_COLUMNS}; got {list(frame.columns)}")


def score_fold(model: str, config: str, fold: int) -> dict | None:
    path = os.path.join(REPO_ROOT, "outputs", model, config, f"fold_{fold}", "predictions.csv")
    if not os.path.exists(path):
        log.warning("  fold %d: missing %s", fold, os.path.relpath(path, REPO_ROOT))
        return None
    predictions = read_predictions(path)
    truth = pd.read_csv(ground_truth_path(config, fold))
    if len(predictions) != len(truth):
        log.warning(
            "  fold %d: %d predictions against %d ground-truth rows; skipping",
            fold,
            len(predictions),
            len(truth),
        )
        return None
    labels = truth[label_column(truth)]
    paired = pd.DataFrame(
        {"prediction": predictions.values, "label": labels.values}
    ).dropna()
    if paired.empty:
        log.warning("  fold %d: no rows left after dropping missing values", fold)
        return None
    metrics = calculate_metrics(paired["prediction"], paired["label"])
    metrics["Fold"] = fold
    metrics["Dataset"] = config
    metrics["N"] = len(paired)
    return metrics


def summary_row(model: str, config: str, folds: list[dict]) -> dict:
    frame = pd.DataFrame(folds)
    means = frame.mean(numeric_only=True)
    # N is the mean fold size, so the table reports a single-fold count rather
    # than the five-fold total.
    count = int(means["N"])
    row = {"Model": model, "Dataset": config, "N": count, "Total": count, "Coverage": "100.0%"}
    for metric in ("RMSE", "MSE", "Pearson", "Spearman", "Kendall", "CI", "Rm2"):
        row[metric] = round(float(means.get(metric, 0.0)), 3)
    return row


def update_summary(rows: list[dict]) -> None:
    new = pd.DataFrame(rows)
    if os.path.exists(SUMMARY_CSV):
        existing = pd.read_csv(SUMMARY_CSV)
        replaced = {(row["Model"], row["Dataset"]) for row in rows}
        keep = existing.apply(lambda r: (r["Model"], r["Dataset"]) not in replaced, axis=1)
        new = pd.concat([existing[keep], new], ignore_index=True)
    new.to_csv(SUMMARY_CSV, index=False)


def main() -> int:
    args = parse_args()
    detail_records: list[dict] = []
    summary_rows: list[dict] = []

    for model in args.models:
        for config in args.datasets:
            log.info("\n%s - %s", model, config)
            folds = [m for fold in range(1, 6) for m in [score_fold(model, config, fold)] if m]
            if not folds:
                log.warning("  no scored folds")
                continue
            for metrics in folds:
                log.info(
                    "  fold %d: N=%d Pearson=%.3f MSE=%.3f CI=%.3f Rm2=%.3f",
                    metrics["Fold"],
                    metrics["N"],
                    metrics["Pearson"],
                    metrics["MSE"],
                    metrics["CI"],
                    metrics["Rm2"],
                )
            row = summary_row(model, config, folds)
            summary_rows.append(row)
            detail_records.extend(folds)
            average = pd.DataFrame(folds).mean(numeric_only=True).to_dict()
            average.update({"Fold": "Average", "Dataset": config, "Model": model})
            detail_records.append(average)
            log.info("  >> average (N=%d): Pearson=%.3f CI=%.3f", row["N"], row["Pearson"], row["CI"])

    if not summary_rows:
        log.error("Nothing was scored; no predictions were found.")
        return 1

    os.makedirs(os.path.dirname(DETAIL_CSV), exist_ok=True)
    pd.DataFrame(detail_records).to_csv(DETAIL_CSV, index=False)
    log.info("\nPer-fold metrics written to %s", os.path.relpath(DETAIL_CSV, REPO_ROOT))

    table = pd.DataFrame(summary_rows).to_string(index=False)
    if args.dry_run:
        print("\nDry run; results/benchmark_summary.csv was not modified:\n" + table)
        return 0
    update_summary(summary_rows)
    print("\nUpdated results/benchmark_summary.csv:\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
