"""Chain-rule ablation, step 5: score either arm on all four external test sets.

One code path for both arms -- the only thing that changes is which dataset directories and
which checkpoint root are read -- so any difference in the numbers is attributable to the
chain rule and not to the evaluation. Run with --arm longest first: it must reproduce the
already-published longest-chain CASF-2016 figures (Pearson 0.781 / RMSE 1.378), which is the
check that the generalisation did not silently alter anything.

Reports rank metrics with bootstrap confidence intervals alongside the error metrics, since
error alone cannot establish that one arm is better than the other.
"""
import argparse
import os
import pickle
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr

ROOT = "/bmlfast/Lyuwei/0.Projects/MixingDTA"
OUT = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation/results"
sys.path.insert(0, os.path.join(ROOT, "MEETA"))

from dataset import Index_dataset, load_dataset  # noqa: E402
from model import DTA, Regress  # noqa: E402
from utils import dict2namespace, pad_tensor_list  # noqa: E402
from config_pdbbind2020_refined import cases  # noqa: E402
from config_pdbbind2020_refined import configuration as base_config  # noqa: E402

ARMS = {
    # arm -> (training dataset name, checkpoint root, test-set suffix)
    "longest": ("PDBbind_Refined_91", "results_refined_91", ""),
    "concat": ("PDBbind_Refined_91_concat", "results_refined_91_concat", "_concat"),
}
TESTSETS = ["CASF2016_Std", "CASF2013_Std", "CSAR36_Std", "CSAR51_Std"]
SUBMODELS = [("results_none", 0), ("results_all_pair", 1),
             ("results_drug", 2), ("results_reversed", 5)]


def concordance_index(y, p):
    """Fraction of discordant-eligible label pairs ranked correctly; ties count a half."""
    tau, _ = kendalltau(y, p)
    return (tau + 1) / 2


def boot(y, p, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 3:
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def metrics(y, p):
    r = pearsonr(y, p)[0]
    s = spearmanr(y, p)[0]
    rlo, rhi = boot(y, p, lambda a, b: pearsonr(a, b)[0])
    slo, shi = boot(y, p, lambda a, b: spearmanr(a, b)[0])
    return {
        "n": len(y),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAE": float(np.mean(np.abs(y - p))),
        "Pearson": r, "P_lo": rlo, "P_hi": rhi,
        "Spearman": s, "S_lo": slo, "S_hi": shi,
        "Kendall": float(kendalltau(y, p)[0]),
        "CI": float(concordance_index(y, p)),
    }


def load_models(train_ds, ckpt_root, device):
    subs = []
    for sub_dir, case_idx in SUBMODELS:
        cfg = base_config.copy()
        cfg.update(cases[case_idx])
        path = os.path.join(ROOT, "MEETA", ckpt_root, sub_dir, train_ds,
                            "1_fold_valid_best_checkpoint.pth")
        m = DTA(dict2namespace(cfg)).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        subs.append(m)

    int_cfg = base_config.copy()
    int_cfg["Integration_mode"] = "total"
    reg = Regress(dict2namespace(int_cfg), 257).to(device)
    reg.load_state_dict(torch.load(
        os.path.join(ROOT, "MEETA", ckpt_root, "result_integration", train_ds,
                     "1_fold_valid_best_checkpoint.pth"), map_location=device))
    reg.eval()
    return subs, reg


def predict(ds_name, subs, reg, device):
    db = os.path.join(ROOT, "DTA_DataBase")
    d = os.path.join(db, "datasets", ds_name)
    with open(os.path.join(d, "drug_id_2_idx.pkl"), "rb") as f:
        drug2i = pickle.load(f)
    with open(os.path.join(d, "protein_id_2_idx.pkl"), "rb") as f:
        prot2i = pickle.load(f)
    with open(os.path.join(d, "test_1.pkl"), "rb") as f:
        rows = pickle.load(f)

    mol = torch.load(os.path.join(db, "MolFormer", f"{ds_name}.pt"), map_location=device)
    esm = torch.load(os.path.join(db, "ESM3_open_small", f"{ds_name}.pt"),
                     map_location=device)
    mol_i, esm_i = OrderedDict(), OrderedDict()
    for k, v in mol.items():
        mol_i[k if isinstance(k, int) else drug2i[k]] = v
    for k, v in esm.items():
        esm_i[k if isinstance(k, int) else prot2i[k]] = v

    cfg = dict2namespace(base_config.copy())
    cfg.max_seq_len = 4700
    dr, pr, ys = load_dataset(cfg, rows, drug2i, prot2i)
    loader = torch.utils.data.DataLoader(Index_dataset(dr, pr, ys), batch_size=64,
                                         shuffle=False)

    preds, labels = [], []
    with torch.no_grad():
        for dd, pp, yy in loader:
            bd, md = pad_tensor_list([mol_i[i] for i in dd.cpu().numpy().flatten()])
            bp, mp = pad_tensor_list([esm_i[i] for i in pp.cpu().numpy().flatten()])
            bd, bp, md, mp = bd.to(device), bp.to(device), md.to(device), mp.to(device)
            acc = None
            for m in subs:
                p_ba, b_emb = m(bd, bp, md, mp)
                vec = torch.cat([b_emb, p_ba], dim=1)
                acc = vec if acc is None else acc + vec
            out, _ = reg(acc)
            preds.append(out.cpu().numpy())
            labels.append(yy.numpy())
    return (np.concatenate(labels).flatten(), np.concatenate(preds).flatten(),
            [(r[0], r[1]) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_ds, ckpt_root, suffix = ARMS[args.arm]
    subs, reg = load_models(train_ds, ckpt_root, device)

    rowsout = []
    for base in TESTSETS:
        ds = base + suffix
        y, p, ids = predict(ds, subs, reg, device)
        m = metrics(y, p)
        m["test_set"], m["arm"] = base, args.arm
        rowsout.append(m)
        pd.DataFrame({"Drug_ID": [i[0] for i in ids],
                      "Protein_ID": [i[1] for i in ids],
                      "Label": y, "Prediction": p}).to_csv(
            os.path.join(OUT, f"pred_{args.arm}_{base}.csv"), index=False)
        print(f"{base:14s} n={m['n']:4d} RMSE {m['RMSE']:.3f} MAE {m['MAE']:.3f}  "
              f"Pearson {m['Pearson']:.3f} [{m['P_lo']:.3f},{m['P_hi']:.3f}]  "
              f"Spearman {m['Spearman']:.3f} [{m['S_lo']:.3f},{m['S_hi']:.3f}]  "
              f"Kendall {m['Kendall']:.3f}  CI {m['CI']:.3f}", flush=True)

    df = pd.DataFrame(rowsout)[["arm", "test_set", "n", "RMSE", "MAE", "Pearson", "P_lo",
                                "P_hi", "Spearman", "S_lo", "S_hi", "Kendall", "CI"]]
    df.to_csv(os.path.join(OUT, f"metrics_{args.arm}.csv"), index=False)
    print(f"\nwrote {OUT}/metrics_{args.arm}.csv")


if __name__ == "__main__":
    sys.exit(main())
