"""Chain counts per test-set entry, so the ablation can be stratified.

The two arms differ in the training corpus as well as in the test input, so predictions
differ on every row -- but the two strata answer different questions:

  single-chain test rows  identical input, different model  -> effect of the chain rule on
                                                               the training corpus alone
  multi-chain test rows   different input and model         -> total effect

CSAR-36 and CSAR-51 turn out to hold one multi-chain entry each, so neither can say anything
about the input side; only CASF-2016 (98) and CASF-2013 (76) carry that signal.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_concat_testsets import SETS, chain_seqs  # noqa: E402

SRC_CSV = "/home/lwfvx/Lyuwei/1.Datasets/pdbbind_like_test_normalize"
TRAIN_STATS = ("/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation/data/"
               "concat_stats.csv")
OUT = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation/data"


def main():
    rows = []
    for name, (csv_name, tree) in SETS.items():
        df = pd.read_csv(os.path.join(SRC_CSV, csv_name))
        for pid in df.compound_id:
            cs = chain_seqs(os.path.join(tree, pid, f"{pid}_protein.pdb"))
            rows.append({"test_set": name, "pdb_id": pid, "n_chains": len(cs),
                         "len_longest": max(len(s) for _, s in cs),
                         "len_concat": sum(len(s) for _, s in cs)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "testset_chain_counts.csv"), index=False)

    print(f"{'test set':14s} {'n':>4s} {'multi':>6s} {'%multi':>7s} "
          f"{'max concat':>11s}")
    for name, g in out.groupby("test_set", sort=False):
        multi = (g.n_chains > 1).sum()
        print(f"{name:14s} {len(g):4d} {multi:6d} {100*multi/len(g):6.1f}% "
              f"{g.len_concat.max():11d}")

    tr = pd.read_csv(TRAIN_STATS)
    print(f"\ntraining corpus: {len(tr)} entries, multi-chain "
          f"{(tr.n_chains > 1).sum()} ({100*(tr.n_chains > 1).mean():.1f}%), "
          f"max concat {tr.len_concat.max()}")


if __name__ == "__main__":
    sys.exit(main())
