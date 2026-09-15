"""Chain-rule ablation, step 1b: concatenated protein strings for the external test sets.

Same extraction as build_concat_csvs.py. Each test set is resolved to the single source
tree that covers all of its entries, and the longest-chain string rebuilt from that tree
is checked against the string actually used in the deployed evaluation -- if the two agree
for every row, the concatenated string from the same tree is trustworthy.
"""
import os
import sys

import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

SRC_CSV = "/home/lwfvx/Lyuwei/1.Datasets/pdbbind_like_test_normalize"
OUT = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation/data"

# test set -> (csv name, structure tree that covers 100% of its entries)
SETS = {
    "CASF2016_Std": ("CASF-2016_standardized_test.csv", "/bmlfast/Lyuwei/1.Datasets/v2016"),
    "CASF2013_Std": ("CASF-2013_standardized_test.csv", "/bmlfast/Lyuwei/1.Datasets/v2016"),
    "CSAR36_Std": ("CSAR-HIQ_36_standardized_test.csv", "/bmlfast/Lyuwei/1.Datasets/CSAR-HIQ_36"),
    "CSAR51_Std": ("CSAR-HIQ_51_standardized_test.csv", "/bmlfast/Lyuwei/1.Datasets/CSAR-HIQ_51"),
}


def chain_seqs(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    out = []
    for model in structure:
        for chain in model:
            residues = [res for res in chain if res.id[0] == " "]
            current_seq = ""
            for res in residues:
                try:
                    current_seq += seq1(res.get_resname())
                except Exception:
                    continue
            if current_seq:
                out.append((str(chain.id), current_seq))
        break
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = []
    for name, (csv_name, tree) in SETS.items():
        df = pd.read_csv(os.path.join(SRC_CSV, csv_name))
        concat, longest, nchain, failed = {}, {}, {}, []
        for pid in df.compound_id:
            f = os.path.join(tree, pid, f"{pid}_protein.pdb")
            try:
                cs = chain_seqs(f)
            except Exception as e:
                failed.append((pid, repr(e)[:60]))
                continue
            if not cs:
                failed.append((pid, "no chains"))
                continue
            concat[pid] = "".join(s for _, s in cs)
            best, blen = "", -1
            for _, s in cs:
                if len(s) > blen:
                    blen, best = len(s), s
            longest[pid] = best
            nchain[pid] = len(cs)

        ok = mism = 0
        bad_ids = []
        for r in df.itertuples():
            if r.compound_id not in longest:
                continue
            if longest[r.compound_id] == str(r.target_sequence):
                ok += 1
            else:
                mism += 1
                bad_ids.append(r.compound_id)

        out = df.copy()
        out["target_sequence"] = out.compound_id.map(concat)
        path = os.path.join(OUT, f"{name}_concat.csv")
        out.to_csv(path, index=False)
        L = out.target_sequence.dropna().str.len()
        multi = sum(1 for p in nchain if nchain[p] > 1)
        print(f"{name:14s} n={len(df):4d}  tree={os.path.basename(tree):12s} "
              f"reproduced {ok:4d}  mismatch {mism:3d}  failed {len(failed):3d}  "
              f"multi-chain {multi:4d}  concat len max {L.max() if len(L) else 0:.0f}")
        if bad_ids:
            print(f"    mismatching: {bad_ids[:12]}")
        if failed:
            print(f"    failed: {failed[:6]}")
        summary.append((name, len(df), ok, mism, len(failed)))

    print("\nA test set is usable for the ablation only if mismatch = 0 and failed = 0.")


if __name__ == "__main__":
    sys.exit(main())
