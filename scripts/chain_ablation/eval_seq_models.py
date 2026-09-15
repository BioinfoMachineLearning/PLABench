"""Score every LLF / DeepDTA chain-rule arm on every external test set.

Runs the PLABench inference modules -- plabench.models.{llf,deepdta}.inference -- so the
numbers come out of exactly the code path that produced the paper's, with the checkpoint and
the protein window passed in as overrides. Each model runs in its own conda environment.

Arms:
  A  published   longest chain, native window (LLF 1200, DeepDTA 2000), 1 seed
  B  L4700       longest chain, window 4700, seeds 0-2
  C  C4700       concatenated,  window 4700, seeds 0-2

B vs C is the chain rule with zero truncation on either side. A vs B prices the window.

Predictions land in results/seq/<model>/<arm>_<set>.csv; the metric table in
results/seq_metrics.csv.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

W = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation"
PLABENCH = "/bmlfast/Lyuwei/0.Projects/PLABench"
TEST_SRC = "/home/lwfvx/Lyuwei/1.Datasets/pdbbind_like_test_normalize"

LLF_PY = "/bmlfast/Lyuwei/0.Projects/LLF/.conda/bin/python"
DD_ENV = "/home/lwfvx/miniforge3/envs/deepdta"
DD_PY = f"{DD_ENV}/bin/python"

# the deepdta env has no libstdc++.so.6 symlink, so the loader falls back to /lib64 and
# pandas dies on GLIBCXX_3.4.29; preload the real library rather than modify the env
DD_ENV_VARS = {"LD_PRELOAD": f"{DD_ENV}/lib/libstdc++.so.6.0.34"}

TESTSETS = {
    "CASF2016_Std": "CASF-2016_standardized_test.csv",
    "CASF2013_Std": "CASF-2013_standardized_test.csv",
    "CSAR36_Std": "CSAR-HIQ_36_standardized_test.csv",
    "CSAR51_Std": "CSAR-HIQ_51_standardized_test.csv",
}

LLF_PUBLISHED = "/home/lwfvx/Lyuwei/0.Projects/LLF/models/pdbbind/best_model_full.pth"
DD_PUBLISHED = ("/home/lwfvx/Lyuwei/0.Projects/DeepDTA-Pytorch/"
                "pdbbind2020_refined91_results")


def arms(model):
    """(arm label, chain rule, window, checkpoint or model dir, seed)."""
    out = [("published", "longest",
            1200 if model == "llf" else 2000,
            LLF_PUBLISHED if model == "llf" else DD_PUBLISHED, None)]
    for rule in ("longest", "concat"):
        for seed in (0, 1, 2):
            d = os.path.join(W, "runs", model, f"{rule}_L4700_s{seed}")
            ckpt = os.path.join(d, "best_model_full.pth") if model == "llf" else d
            out.append((f"{rule}_L4700_s{seed}", rule, 4700, ckpt, seed))
    return out


def test_csv(rule, ts):
    """Longest-chain arms read the deployed test CSVs; concat arms read the rebuilt ones."""
    return (os.path.join(TEST_SRC, TESTSETS[ts]) if rule == "longest"
            else os.path.join(W, "data", f"{ts}_concat.csv"))


def predict(model, arm, rule, window, ckpt, ts, gpu):
    outdir = os.path.join(W, "results", "seq", model)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{arm}_{ts}.csv")
    if os.path.isfile(out):
        return out

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    if model == "llf":
        cmd = [LLF_PY, "-m", "plabench.models.llf.inference",
               "--test_file", test_csv(rule, ts), "--output_file", out,
               "--dataset_name", ts, "--task", "pdbbind_refined_91",
               "--model_path", ckpt, "--max_seq_len", str(window), "--device", "cuda"]
    else:
        env.update(DD_ENV_VARS)
        cmd = [DD_PY, "-m", "plabench.models.deepdta.inference",
               "--test_file", test_csv(rule, ts), "--output_file", out,
               "--dataset_name", ts, "--model_dir", ckpt,
               "--seqlen", str(window), "--device", "cuda"]

    r = subprocess.run(cmd, cwd=PLABENCH, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out):
        sys.stderr.write(f"FAILED {model}/{arm}/{ts}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}\n")
        return None
    return out


def score(pred_file, rule, ts):
    """Join predictions back onto the labels by compound id, then score."""
    pred = pd.read_csv(pred_file, header=None, names=["Prediction", "compound_id"])
    truth = pd.read_csv(test_csv(rule, ts))[["compound_id", "affinity"]]
    m = truth.merge(pred, on="compound_id", how="inner")
    y, p = m.affinity.values, m.Prediction.values
    kt = kendalltau(y, p)[0]
    return {
        "n": len(m),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAE": float(np.mean(np.abs(y - p))),
        "Pearson": float(pearsonr(y, p)[0]),
        "Spearman": float(spearmanr(y, p)[0]),
        "Kendall": float(kt),
        "CI": float((kt + 1) / 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llf", "deepdta"])
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    rows = []
    for model in a.models:
        for arm, rule, window, ckpt, seed in arms(model):
            probe = ckpt if model == "llf" else os.path.join(ckpt, "model_refined91.pt")
            if not os.path.exists(probe):
                print(f"skip {model}/{arm}: no checkpoint at {probe}", flush=True)
                continue
            for ts in TESTSETS:
                f = predict(model, arm, rule, window, ckpt, ts, a.gpu)
                if f is None:
                    continue
                rows.append({"model": model, "arm": arm, "chain_rule": rule,
                             "window": window, "seed": seed, "test_set": ts,
                             **score(f, rule, ts)})
                r = rows[-1]
                print(f"{model:8s} {arm:20s} {ts:14s} n={r['n']:4d} "
                      f"RMSE {r['RMSE']:.3f}  Pearson {r['Pearson']:.3f}  "
                      f"Spearman {r['Spearman']:.3f}  CI {r['CI']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(W, "results", "seq_metrics.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    sys.exit(main())
