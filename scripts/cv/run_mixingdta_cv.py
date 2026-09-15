#!/usr/bin/env python
"""Run MixingDTA five-fold cross-validation on Davis and KIBA.

Writes outputs/mixingdta/<config>/fold_<n>/predictions.csv for the six
warm-start / cold-drug / cold-target configs. Score them afterwards with
scripts/cv/collect_kiba_davis_cv.py, which reads the same layout for DeepDTA and
LLF.

Warm-start folds read the per-fold test CSVs under data/Structure_independent/.
Cold-start folds read the authors' pickled test sets from the MixingDTA
checkout, because cold-start inference looks up per-compound and per-target
embeddings by the identifiers those pickles carry and the CSV copies do not.
The pickles are row-identical to the CSV ground truth, so the predictions stay
aligned either way.

Environment overrides:
    MIXINGDTA_PYTHON   interpreter of the MixingDTA conda environment
    MIXINGDTA_ROOT     MixingDTA checkout (default forks/MixingDTA)
    PLABENCH_DEVICE    torch device (default cuda)
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON = os.environ.get(
    "MIXINGDTA_PYTHON", "/home/lwfvx/miniforge3/envs/MixingDTA/bin/python"
)
MODEL_ROOT = os.environ.get(
    "MIXINGDTA_ROOT", os.path.join(REPO_ROOT, "forks", "MixingDTA")
)
DEVICE = os.environ.get("PLABENCH_DEVICE", "cuda")

WARM = os.path.join(REPO_ROOT, "data", "Structure_independent", "{dataset}", "test_{fold}.csv")
COLD = os.path.join(MODEL_ROOT, "DTA_DataBase", "cold", "{dataset}", "test_{split}.pkl")

# config -> (task, test-file template). The warm splits have one test file per
# fold; the cold splits share one held-out set across folds and vary only the
# model weights, which inference.py selects from --dataset_name and --fold.
EXPERIMENTS = {
    "davis_warm": ("warm_start", WARM.format(dataset="DAVIS", fold="{fold}")),
    "kiba_warm": ("warm_start", WARM.format(dataset="KIBA", fold="{fold}")),
    "davis_cold_drug": ("cold_start", COLD.format(dataset="DAVIS", split="Drug")),
    "davis_cold_target": ("cold_start", COLD.format(dataset="DAVIS", split="Target")),
    "kiba_cold_drug": ("cold_start", COLD.format(dataset="KIBA", split="Drug")),
    "kiba_cold_target": ("cold_start", COLD.format(dataset="KIBA", split="Target")),
}


def run_fold(config: str, task: str, fold: int, test_file: str) -> bool:
    output_file = os.path.join(
        REPO_ROOT, "outputs", "mixingdta", config, f"fold_{fold}", "predictions.csv"
    )
    command = [
        PYTHON, "-m", "plabench.models.mixingdta.inference",
        "--test_file", test_file,
        "--model_root", MODEL_ROOT,
        "--output_file", output_file,
        "--dataset_name", config,
        "--task", task,
        "--fold", str(fold),
        "--device", DEVICE,
    ]
    print(f"  fold {fold}...", flush=True)
    try:
        subprocess.run(command, check=True, cwd=REPO_ROOT)
        return True
    except subprocess.CalledProcessError as error:
        print(f"  fold {fold} failed: {error}", file=sys.stderr)
        return False


def main() -> int:
    configs = sys.argv[1:] or list(EXPERIMENTS)
    unknown = [name for name in configs if name not in EXPERIMENTS]
    if unknown:
        print(f"Unknown configs: {', '.join(unknown)}", file=sys.stderr)
        print(f"Choose from: {', '.join(EXPERIMENTS)}", file=sys.stderr)
        return 2
    if not os.path.isdir(MODEL_ROOT):
        print(
            f"MixingDTA checkout not found: {MODEL_ROOT}\n"
            "Run `git submodule update --init forks/MixingDTA` or set MIXINGDTA_ROOT.",
            file=sys.stderr,
        )
        return 2

    failures = 0
    for config in configs:
        task, template = EXPERIMENTS[config]
        print(f"\n--- {config} ({task}) ---")
        for fold in range(1, 6):
            test_file = template.format(fold=fold)
            if not os.path.exists(test_file):
                print(f"  fold {fold}: missing test file {test_file}", file=sys.stderr)
                failures += 1
                continue
            if not run_fold(config, task, fold, test_file):
                failures += 1

    if failures:
        print(f"\n{failures} fold(s) did not produce predictions.", file=sys.stderr)
        return 1
    print("\nAll folds written. Score them with scripts/cv/collect_kiba_davis_cv.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
