"""Chain-rule ablation, step 7: how much of the gap is actually about the chain rule?

Concatenation changes two things at once: the test input for multi-chain entries, and the
training corpus for every entry. Single-chain test entries isolate the second -- their input
is byte-identical between arms, so any movement there is the model, not the chain rule.

The difference-in-differences

    (concat - longest) on multi-chain  -  (concat - longest) on single-chain

therefore estimates the part attributable to supplying the missing chains at prediction
time. If concatenation helped because single-chain input omits binding-site residues, this
is where it has to show up. The bootstrap resamples the two strata independently.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

W = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation"
RES = os.path.join(W, "results")
NBOOT = 5000
# CSAR-36/51 hold one multi-chain entry each; a stratum of one cannot be bootstrapped
TESTSETS = ["CASF2016_Std", "CASF2013_Std"]


def stat(y, p, key):
    if key == "Pearson":
        return pearsonr(y, p)[0]
    if key == "Spearman":
        return spearmanr(y, p)[0]
    if key == "RMSE":
        return np.sqrt(np.mean((y - p) ** 2))
    if key == "CI":
        return (kendalltau(y, p)[0] + 1) / 2
    raise KeyError(key)


def main():
    chains = pd.read_csv(os.path.join(W, "data", "testset_chain_counts.csv"))
    print("difference-in-differences: (concat - longest) on multi-chain entries,")
    print("minus the same contrast on entries whose input is identical between arms\n")
    print(f"{'test set':14s} {'metric':9s} {'multi':>8s} {'single':>8s} {'DiD':>8s} "
          f"{'95% CI':>18s} {'p':>7s}")

    out = []
    for ts in TESTSETS:
        a = pd.read_csv(os.path.join(RES, f"pred_longest_{ts}.csv"))
        b = pd.read_csv(os.path.join(RES, f"pred_concat_{ts}.csv"))
        nc = chains[chains.test_set == ts].n_chains.values
        assert len(nc) == len(a)
        mi, si = np.where(nc > 1)[0], np.where(nc == 1)[0]

        for key in ("Pearson", "Spearman", "RMSE", "CI"):
            def contrast(idx):
                return (stat(a.Label.values[idx], b.Prediction.values[idx], key)
                        - stat(a.Label.values[idx], a.Prediction.values[idx], key))

            dm, ds = contrast(mi), contrast(si)
            rng = np.random.default_rng(0)
            vals = []
            for _ in range(NBOOT):
                bm = rng.choice(mi, len(mi), replace=True)
                bs = rng.choice(si, len(si), replace=True)
                if len(np.unique(a.Label.values[bm])) < 3:
                    continue
                if len(np.unique(a.Label.values[bs])) < 3:
                    continue
                vals.append(contrast(bm) - contrast(bs))
            v = np.array(vals)
            lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
            p = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))
            sig = "  *" if not (lo <= 0 <= hi) else ""
            print(f"{ts:14s} {key:9s} {dm:+8.3f} {ds:+8.3f} {dm - ds:+8.3f} "
                  f"[{lo:+.3f},{hi:+.3f}] {p:7.3f}{sig}")
            out.append({"test_set": ts, "metric": key, "n_multi": len(mi),
                        "n_single": len(si), "diff_multi": dm, "diff_single": ds,
                        "DiD": dm - ds, "lo": lo, "hi": hi, "p": p})

    pd.DataFrame(out).to_csv(os.path.join(RES, "diff_in_diff.csv"), index=False)
    print("\n*  = 95% CI excludes zero")
    print(f"wrote {RES}/diff_in_diff.csv")


if __name__ == "__main__":
    sys.exit(main())
