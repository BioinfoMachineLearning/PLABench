"""Chain-rule ablation for LLF and DeepDTA: turn 56 prediction dumps into a verdict.

Three questions, three sections:

  1. Seed spread. Three seeds per arm, so the arm difference can be read against the noise
     it has to clear. A single seed per arm -- the position the MEETA ablation is stuck in --
     cannot do this.
  2. Chain rule. The three seeds of each arm are averaged into one prediction (a 3-seed
     ensemble), then compared to the other arm by paired bootstrap over test rows. Both arms
     sit at window 4700, where neither truncates, so the chain rule is the only variable.
  3. Where the difference lives. Single-chain test entries receive byte-identical input in
     both arms, so any movement there is the retrained corpus, not the chain rule. The
     difference-in-differences against multi-chain entries isolates the part attributable to
     supplying the missing chains at prediction time.

The published arm (LLF window 1200, DeepDTA 2000) is reported alongside so the cost of
widening the window is visible separately from the chain rule.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

W = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation"
RES = os.path.join(W, "results")
SEQ = os.path.join(RES, "seq")
TEST_SRC = "/home/lwfvx/Lyuwei/1.Datasets/pdbbind_like_test_normalize"
TESTSETS = ["CASF2016_Std", "CASF2013_Std", "CSAR36_Std", "CSAR51_Std"]
CSVS = {"CASF2016_Std": "CASF-2016_standardized_test.csv",
        "CASF2013_Std": "CASF-2013_standardized_test.csv",
        "CSAR36_Std": "CSAR-HIQ_36_standardized_test.csv",
        "CSAR51_Std": "CSAR-HIQ_51_standardized_test.csv"}
SEEDS = (0, 1, 2)
NBOOT = 5000
METRICS = ("RMSE", "Pearson", "Spearman", "CI")


def stat(y, p, key):
    if key == "Pearson":
        return pearsonr(y, p)[0]
    if key == "Spearman":
        return spearmanr(y, p)[0]
    if key == "RMSE":
        return float(np.sqrt(np.mean((y - p) ** 2)))
    if key == "CI":
        return (kendalltau(y, p)[0] + 1) / 2
    raise KeyError(key)


def load(model, arm, ts):
    f = os.path.join(SEQ, model, f"{arm}_{ts}.csv")
    return pd.read_csv(f, header=None, names=["Prediction", "compound_id"])


def aligned(model, ts):
    """Truth plus one averaged prediction column per arm, all on a common row order.

    Rows are keyed by compound_id rather than position: the harness silently drops entries
    whose SMILES fail to parse, so positions are not guaranteed to line up across dumps.
    """
    truth = pd.read_csv(os.path.join(TEST_SRC, CSVS[ts]))[["compound_id", "affinity"]]
    out = truth
    for rule in ("longest", "concat"):
        cols = []
        for s in SEEDS:
            d = load(model, f"{rule}_L4700_s{s}", ts).rename(
                columns={"Prediction": f"{rule}_{s}"})
            out = out.merge(d, on="compound_id", how="inner")
            cols.append(f"{rule}_{s}")
        out[rule] = out[cols].mean(axis=1)
    pub = load(model, "published", ts).rename(columns={"Prediction": "published"})
    return out.merge(pub, on="compound_id", how="inner")


def paired_ci(y, pa, pb, key, seed=0):
    """Bootstrap CI on stat(B) - stat(A), resampling rows once and applying to both arms."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(NBOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 3:
            continue
        d.append(stat(y[i], pb[i], key) - stat(y[i], pa[i], key))
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(min(p, 1.0))


def section_seed_spread(m):
    print("\n" + "=" * 92)
    print("1. SEED SPREAD -- mean (sd) over 3 seeds, both arms at window 4700")
    print("=" * 92)
    print(f"{'model':9s} {'test set':14s} {'metric':9s} {'longest':>16s} {'concat':>16s} "
          f"{'published':>10s}")
    rows = []
    for model in ("llf", "deepdta"):
        for ts in TESTSETS:
            a = aligned(model, ts)
            y = a.affinity.values
            for key in METRICS:
                vals = {r: np.array([stat(y, a[f"{r}_{s}"].values, key) for s in SEEDS])
                        for r in ("longest", "concat")}
                pub = stat(y, a.published.values, key)
                print(f"{model:9s} {ts:14s} {key:9s} "
                      f"{vals['longest'].mean():8.3f} ({vals['longest'].std(ddof=1):.3f}) "
                      f"{vals['concat'].mean():8.3f} ({vals['concat'].std(ddof=1):.3f}) "
                      f"{pub:10.3f}")
                rows.append({"model": model, "test_set": ts, "metric": key,
                             "longest_mean": vals["longest"].mean(),
                             "longest_sd": vals["longest"].std(ddof=1),
                             "concat_mean": vals["concat"].mean(),
                             "concat_sd": vals["concat"].std(ddof=1),
                             "published": pub})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "seq_seed_spread.csv"), index=False)


def section_chain_rule():
    print("\n" + "=" * 92)
    print("2. CHAIN RULE -- 3-seed ensemble, concat minus longest, paired bootstrap")
    print("=" * 92)
    print(f"{'model':9s} {'test set':14s} {'metric':9s} {'longest':>8s} {'concat':>8s} "
          f"{'diff':>7s} {'95% CI':>18s} {'p':>7s}")
    rows = []
    for model in ("llf", "deepdta"):
        for ts in TESTSETS:
            a = aligned(model, ts)
            y, pl, pc = a.affinity.values, a.longest.values, a.concat.values
            for key in METRICS:
                sa, sb = stat(y, pl, key), stat(y, pc, key)
                lo, hi, p = paired_ci(y, pl, pc, key)
                sig = "" if lo <= 0 <= hi else "  *"
                print(f"{model:9s} {ts:14s} {key:9s} {sa:8.3f} {sb:8.3f} {sb - sa:+7.3f} "
                      f"[{lo:+.3f},{hi:+.3f}] {p:7.3f}{sig}")
                rows.append({"model": model, "test_set": ts, "metric": key, "n": len(y),
                             "longest": sa, "concat": sb, "diff": sb - sa,
                             "lo": lo, "hi": hi, "p": p})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "seq_chain_rule.csv"), index=False)
    print("\n*  = 95% CI on the paired difference excludes zero")


def section_did():
    chains = pd.read_csv(os.path.join(W, "data", "testset_chain_counts.csv"))
    print("\n" + "=" * 92)
    print("3. DIFFERENCE-IN-DIFFERENCES -- (concat-longest) on multi-chain entries minus")
    print("   the same contrast on entries whose input is identical between arms")
    print("=" * 92)
    print(f"{'model':9s} {'test set':14s} {'metric':9s} {'multi':>8s} {'single':>8s} "
          f"{'DiD':>8s} {'95% CI':>18s} {'p':>7s}")
    rows = []
    # CSAR-36/51 hold one multi-chain entry each; a stratum of one cannot be bootstrapped
    for model in ("llf", "deepdta"):
        for ts in ("CASF2016_Std", "CASF2013_Std"):
            a = aligned(model, ts).merge(
                chains[chains.test_set == ts][["pdb_id", "n_chains"]],
                left_on="compound_id", right_on="pdb_id", how="left")
            assert a.n_chains.notna().all(), f"{model}/{ts}: unmatched chain counts"
            y, pl, pc = a.affinity.values, a.longest.values, a.concat.values
            nc = a.n_chains.values
            mi, si = np.where(nc > 1)[0], np.where(nc == 1)[0]
            for key in METRICS:
                def contrast(idx):
                    return stat(y[idx], pc[idx], key) - stat(y[idx], pl[idx], key)

                dm, ds = contrast(mi), contrast(si)
                rng = np.random.default_rng(0)
                v = []
                for _ in range(NBOOT):
                    bm = rng.choice(mi, len(mi), replace=True)
                    bs = rng.choice(si, len(si), replace=True)
                    if len(np.unique(y[bm])) < 3 or len(np.unique(y[bs])) < 3:
                        continue
                    v.append(contrast(bm) - contrast(bs))
                v = np.array(v)
                lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
                p = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))
                sig = "" if lo <= 0 <= hi else "  *"
                print(f"{model:9s} {ts:14s} {key:9s} {dm:+8.3f} {ds:+8.3f} {dm - ds:+8.3f} "
                      f"[{lo:+.3f},{hi:+.3f}] {p:7.3f}{sig}")
                rows.append({"model": model, "test_set": ts, "metric": key,
                             "n_multi": len(mi), "n_single": len(si),
                             "diff_multi": dm, "diff_single": ds, "DiD": dm - ds,
                             "lo": lo, "hi": hi, "p": p})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "seq_did.csv"), index=False)


def main():
    m = pd.read_csv(os.path.join(RES, "seq_metrics.csv"))
    section_seed_spread(m)
    section_chain_rule()
    section_did()
    print(f"\nwrote {RES}/seq_{{seed_spread,chain_rule,did}}.csv")


if __name__ == "__main__":
    sys.exit(main())
