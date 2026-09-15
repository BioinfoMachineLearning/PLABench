"""Chain-rule ablation, step 6: compare the two arms.

Both arms predict the same rows in the same order, so the comparison is paired and the
bootstrap resamples row indices once and applies them to both arms. That is the difference
between "the two intervals overlap" -- which is not a test -- and a confidence interval on
the difference itself, which is.

Reported per test set and, for the sets that have enough of them, separately on the
multi-chain and single-chain strata (see chain_counts.py for why they answer different
questions).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

W = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation"
RES = os.path.join(W, "results")
TESTSETS = ["CASF2016_Std", "CASF2013_Std", "CSAR36_Std", "CSAR51_Std"]
NBOOT = 5000


def stats(y, p):
    return {
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAE": float(np.mean(np.abs(y - p))),
        "Pearson": float(pearsonr(y, p)[0]),
        "Spearman": float(spearmanr(y, p)[0]),
        "Kendall": float(kendalltau(y, p)[0]),
        "CI": float((kendalltau(y, p)[0] + 1) / 2),
    }


def paired_ci(y, pa, pb, key, seed=0):
    """Bootstrap CI on stat(arm B) - stat(arm A), resampling rows once for both arms."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(NBOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 3:
            continue
        diffs.append(stats(y[i], pb[i])[key] - stats(y[i], pa[i])[key])
    d = np.array(diffs)
    # two-sided bootstrap p: how often the resampled difference crosses zero
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(min(p, 1.0)))


def main():
    chains = pd.read_csv(os.path.join(W, "data", "testset_chain_counts.csv"))
    out = []
    for ts in TESTSETS:
        a = pd.read_csv(os.path.join(RES, f"pred_longest_{ts}.csv"))
        b = pd.read_csv(os.path.join(RES, f"pred_concat_{ts}.csv"))
        assert len(a) == len(b), ts
        assert np.allclose(a.Label.values, b.Label.values), f"{ts}: labels differ"

        # Protein_ID in the prediction dumps is the literal "dummy" the pkl carries, so the
        # chain counts are matched by row order instead. That is sound because
        # build_datasets.py asserted, for every one of these sets, that the deployed pkl
        # lines up row-for-row with the source CSV, and chain_counts.py walks that same CSV.
        nc = chains[chains.test_set == ts].n_chains.values
        assert len(nc) == len(a), f"{ts}: {len(nc)} chain counts vs {len(a)} predictions"

        strata = [("all", np.ones(len(a), bool)), ("multi", nc > 1), ("single", nc == 1)]

        for label, m in strata:
            if m.sum() < 10:
                continue
            y, pa, pb = a.Label.values[m], a.Prediction.values[m], b.Prediction.values[m]
            sa, sb = stats(y, pa), stats(y, pb)
            row = {"test_set": ts, "stratum": label, "n": int(m.sum())}
            for k in ("RMSE", "Pearson", "Spearman", "CI"):
                lo, hi, pv = paired_ci(y, pa, pb, k)
                row[f"{k}_longest"] = sa[k]
                row[f"{k}_concat"] = sb[k]
                row[f"{k}_diff"] = sb[k] - sa[k]
                row[f"{k}_lo"], row[f"{k}_hi"], row[f"{k}_p"] = lo, hi, pv
            out.append(row)

    df = pd.DataFrame(out)
    df.to_csv(os.path.join(RES, "arm_comparison.csv"), index=False)

    for k in ("Pearson", "Spearman", "RMSE"):
        print(f"\n=== {k}   (concat - longest, paired bootstrap 95% CI) ===")
        print(f"{'test set':14s} {'stratum':8s} {'n':>4s} {'longest':>8s} {'concat':>8s} "
              f"{'diff':>7s} {'95% CI':>18s} {'p':>7s}")
        for r in df.itertuples():
            lo, hi = getattr(r, f"{k}_lo"), getattr(r, f"{k}_hi")
            sig = "" if lo <= 0 <= hi else "  *"
            print(f"{r.test_set:14s} {r.stratum:8s} {r.n:4d} "
                  f"{getattr(r, f'{k}_longest'):8.3f} {getattr(r, f'{k}_concat'):8.3f} "
                  f"{getattr(r, f'{k}_diff'):+7.3f} [{lo:+.3f},{hi:+.3f}]"
                  f" {getattr(r, f'{k}_p'):7.3f}{sig}")
    print("\n*  = 95% CI on the paired difference excludes zero")
    print(f"wrote {RES}/arm_comparison.csv")


if __name__ == "__main__":
    sys.exit(main())
