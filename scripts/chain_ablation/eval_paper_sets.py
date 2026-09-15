"""Score every chain-rule arm on the two sets the paper reports besides the PDBbind family:
CASP16 (L1000 + L3000) and ChEMBL35 (84 targets / 7650 pairs).

Unlike CASF/CSAR, neither of these sets is chain-selected. CASP16 supplies one target
sequence per set and its prepared structures carry a single chain; ChEMBL35 supplies the
canonical UniProt sequence per target, which the originating benchmark defines as a monomer.
So the two arms receive byte-identical test input here and the only thing that differs is
the weights -- which is exactly the quantity the paper needs: how far does retraining under
the other chain rule move the published numbers.

Inputs are the paper's own files (configs/dataset/sequence_*.yaml), and scoring reuses
plabench.analysis.metrics.calculate_metrics plus the same two aggregations the paper uses:
  CASP16   -> series_group, N-weighted over L1000 (17) and L3000 (123)
  ChEMBL35 -> per_target over targets with >= 3 compounds, then N-weighted
so the output is directly comparable to results/{casp16/weighted_summary_casp16,chembl35/weighted_summary_chembl35}.csv.
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

W = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation"
PLABENCH = "/bmlfast/Lyuwei/0.Projects/PLABench"
sys.path.insert(0, PLABENCH)
from plabench.analysis.metrics import calculate_metrics  # noqa: E402

LLF_PY = "/bmlfast/Lyuwei/0.Projects/LLF/.conda/bin/python"
DD_ENV = "/home/lwfvx/miniforge3/envs/deepdta"
DD_PY = f"{DD_ENV}/bin/python"
MX_PY = "/home/lwfvx/miniforge3/envs/MixingDTA/bin/python"
# the deepdta env has no libstdc++.so.6 symlink, so the loader falls back to /lib64
DD_ENV_VARS = {"LD_PRELOAD": f"{DD_ENV}/lib/libstdc++.so.6.0.34"}
# MoLFormer's `main` has moved to a revision whose remote code needs transformers>=4.56,
# while the MixingDTA env pins 4.30; stay offline so the cached revision the paper's runs
# used is the one that loads
MX_ENV_VARS = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

DATASETS = {
    "casp16_l1000": f"{PLABENCH}/data/Structure_independent/L1000_casp16_test.csv",
    "casp16_l3000": f"{PLABENCH}/data/Structure_independent/L3000_casp16_test.csv",
    "chembl35_full": f"{PLABENCH}/data/chembl35/chembl35_full_input.csv",
}

LLF_PUBLISHED = "/home/lwfvx/Lyuwei/0.Projects/LLF/models/pdbbind/best_model_full.pth"
DD_PUBLISHED = f"{PLABENCH}/forks/DeepDTA-Pytorch/pdbbind2020_refined91_results"
MX_PUBLISHED = f"{PLABENCH}/forks/MixingDTA"
MX_CONCAT = f"{W}/meeta_roots/concat"


def arms(model):
    """(arm label, chain rule, protein window, checkpoint/root, seed)."""
    if model == "meeta":
        return [("published", "longest", 4700, MX_PUBLISHED, None),
                ("concat", "concat", 4700, MX_CONCAT, None)]
    out = [("published", "longest",
            1200 if model == "llf" else 2000,
            LLF_PUBLISHED if model == "llf" else DD_PUBLISHED, None)]
    for rule in ("longest", "concat"):
        for seed in (0, 1, 2):
            d = os.path.join(W, "runs", model, f"{rule}_L4700_s{seed}")
            ckpt = os.path.join(d, "best_model_full.pth") if model == "llf" else d
            out.append((f"{rule}_L4700_s{seed}", rule, 4700, ckpt, seed))
    return out


def predict(model, arm, window, ckpt, ds, gpu):
    outdir = os.path.join(W, "results", "paper_sets", model)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{arm}_{ds}.csv")
    if os.path.isfile(out):
        return out

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    common = ["--test_file", DATASETS[ds], "--output_file", out,
              "--dataset_name", ds, "--device", "cuda"]
    if model == "llf":
        cmd = [LLF_PY, "-m", "plabench.models.llf.inference", *common,
               "--task", "pdbbind_refined_91", "--model_path", ckpt,
               "--max_seq_len", str(window)]
    elif model == "deepdta":
        env.update(DD_ENV_VARS)
        cmd = [DD_PY, "-m", "plabench.models.deepdta.inference", *common,
               "--model_dir", ckpt, "--seqlen", str(window)]
    else:
        env.update(MX_ENV_VARS)
        cmd = [MX_PY, "-m", "plabench.models.mixingdta.inference", *common,
               "--model_root", ckpt, "--task", "pdbbind_refined_91"]

    r = subprocess.run(cmd, cwd=PLABENCH, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out):
        sys.stderr.write(f"FAILED {model}/{arm}/{ds}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}\n")
        return None
    return out


def truth(ds):
    d = pd.read_csv(DATASETS[ds])
    id_col = "compound_id" if "compound_id" in d.columns else d.columns[0]
    out = d[[id_col, "affinity"]].rename(columns={id_col: "name"})
    return out.assign(name=out.name.astype(str))


def score(pred_file, ds):
    """Overall metrics, plus the per-target block ChEMBL35 is aggregated from."""
    pred = pd.read_csv(pred_file, header=None, names=["prediction", "name"])
    t = truth(ds)
    # MixingDTA falls back to a row counter when the CSV keys on molecule_id rather than
    # compound_id, so join on the id when the two sides share one and positionally otherwise
    # (the writer preserves input order and drops nothing).
    if set(pred.name.astype(str)) & set(t.name.astype(str)):
        m = t.merge(pred.assign(name=pred.name.astype(str)).astype({"name": str}),
                    on="name", how="inner")
    else:
        assert len(pred) == len(t), f"{pred_file}: {len(pred)} preds vs {len(t)} labels"
        m = t.assign(prediction=pred.prediction.values)
    overall = calculate_metrics(m.prediction, m.affinity)
    overall["N"] = len(m)

    per_target = []
    if ds == "chembl35_full":
        # target = compound_id prefix before the first "_", matching collect_results.py
        m["target"] = m["name"].astype(str).str.split("_").str[0]
        for tgt, g in m.groupby("target"):
            if len(g) < 3:
                continue
            r = calculate_metrics(g.prediction, g.affinity)
            r.update({"Target": tgt, "N": len(g)})
            per_target.append(r)
    return overall, per_target


METRICS = ("RMSE", "MSE", "Pearson", "Spearman", "Kendall", "CI", "Rm2")


def weighted(group, n_col="N"):
    """N-weighted mean of each metric -- the aggregation scripts/weighted_summary.py uses."""
    row = {"N_total": int(group[n_col].sum()), "N_series": len(group)}
    for c in METRICS:
        v = pd.to_numeric(group[c], errors="coerce")
        n = group[n_col]
        ok = v.notna()
        row[c] = round((v[ok] * n[ok]).sum() / n[ok].sum(), 3) if ok.any() else float("nan")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llf", "deepdta", "meeta"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    raw, pt_rows = [], []
    for model in a.models:
        for arm, rule, window, ckpt, seed in arms(model):
            probe = ckpt
            if model == "deepdta":
                probe = os.path.join(ckpt, "model_refined91.pt")
            if not os.path.exists(probe):
                print(f"skip {model}/{arm}: nothing at {probe}", flush=True)
                continue
            for ds in a.datasets:
                f = predict(model, arm, window, ckpt, ds, a.gpu)
                if f is None:
                    continue
                overall, per_target = score(f, ds)
                raw.append({"model": model, "arm": arm, "chain_rule": rule,
                            "window": window, "seed": seed, "dataset": ds, **overall})
                for r in per_target:
                    pt_rows.append({"model": model, "arm": arm, "dataset": ds, **r})
                print(f"{model:8s} {arm:20s} {ds:14s} n={overall['N']:5d} "
                      f"RMSE {overall['RMSE']}  Pearson {overall['Pearson']}  "
                      f"CI {overall['CI']}", flush=True)

    res = os.path.join(W, "results")
    raw = pd.DataFrame(raw)
    raw.to_csv(os.path.join(res, "paper_sets_raw.csv"), index=False)
    if pt_rows:
        pd.DataFrame(pt_rows).to_csv(os.path.join(res, "paper_sets_per_target.csv"),
                                     index=False)

    # --- aggregate the two ways the paper does ---
    agg = []
    for (model, arm), g in raw.groupby(["model", "arm"], sort=False):
        c16 = g[g.dataset.str.startswith("casp16")]
        if len(c16) == 2:
            agg.append({"model": model, "arm": arm, "key": "casp16",
                        "scope": "series_group", **weighted(c16)})
    if pt_rows:
        pt = pd.DataFrame(pt_rows)
        for (model, arm), g in pt.groupby(["model", "arm"], sort=False):
            agg.append({"model": model, "arm": arm, "key": "chembl35",
                        "scope": "per_target", **weighted(g)})
    agg = pd.DataFrame(agg)
    agg.to_csv(os.path.join(res, "paper_sets_weighted.csv"), index=False)
    print("\n=== weighted, comparable to results/{casp16/weighted_summary_casp16,chembl35/weighted_summary_chembl35}.csv ===")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
