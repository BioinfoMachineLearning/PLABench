"""Chain-rule ablation, step 1: rebuild the protein strings by concatenating every chain.

The production rule (LLF prepare_standard_training_data.py) keeps only the longest chain.
This produces the alternative the field actually uses -- all chains, file order, no
delimiter, which is MixingDTA's own convention -- while holding every other input fixed.

Residue extraction is copied verbatim from the production selector so that the only
difference between the two arms is the set of chains included: same parser, same
`res.id[0] == ' '` filter, same seq1() mapping, first model only.

Writes CSVs with the same columns and the same row order as the originals, so the split
stays bit-identical downstream.
"""
import os
import sys

import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from tqdm import tqdm

PDBBIND = "/bmlfast/Lyuwei/1.Datasets/pbpp-2020"
SRC = "/bmlfast/Lyuwei/1.Datasets/pdbbind_refined_91"
OUT = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation/data"


def chain_seqs(pdb_file):
    """Per-chain strings, file order. Same extraction as the production selector."""
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
        break  # only the first model, as in production
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    frames = {}
    for split in ("train", "val"):
        frames[split] = pd.read_csv(os.path.join(SRC, f"pdbbind_{split}_full.csv"))
    ids = sorted(set(pd.concat(frames.values()).compound_id))
    print(f"unique PDB entries: {len(ids)}")

    concat, longest, nchain, bad = {}, {}, {}, []
    for pid in tqdm(ids, desc="chains"):
        f = os.path.join(PDBBIND, pid, f"{pid}_protein.pdb")
        try:
            cs = chain_seqs(f)
        except Exception as e:
            bad.append((pid, repr(e)))
            continue
        if not cs:
            bad.append((pid, "no chains"))
            continue
        concat[pid] = "".join(s for _, s in cs)
        # strictly-greater comparison => first of the equal-longest wins, as in production
        best, blen = "", -1
        for _, s in cs:
            if len(s) > blen:
                blen, best = len(s), s
        longest[pid] = best
        nchain[pid] = len(cs)

    if bad:
        print(f"FAILED to parse {len(bad)}: {bad[:10]}")

    # --- verification against the strings actually used in the longest-chain arm -----
    n_ok = n_mismatch = n_single_exact = n_multi_substr = 0
    mismatches = []
    for split, df in frames.items():
        for r in df.itertuples():
            pid = r.compound_id
            if pid not in longest:
                continue
            used = str(r.target_sequence)
            if longest[pid] == used:
                n_ok += 1
                if nchain[pid] == 1:
                    n_single_exact += 1
                if nchain[pid] > 1 and used in concat[pid]:
                    n_multi_substr += 1
            else:
                n_mismatch += 1
                mismatches.append(pid)
    print(f"\nreproduced the deployed longest-chain string for {n_ok} rows; "
          f"{n_mismatch} mismatch")
    print(f"  single-chain entries where concat == longest exactly : {n_single_exact}")
    print(f"  multi-chain entries where longest is a substring of concat: {n_multi_substr}")
    if mismatches:
        print(f"  mismatching ids (first 10): {mismatches[:10]}")

    # --- emit the concat CSVs, same columns, same row order -------------------------
    for split, df in frames.items():
        out = df.copy()
        out["target_sequence"] = out.compound_id.map(concat)
        missing = out.target_sequence.isna().sum()
        if missing:
            print(f"  WARNING {split}: {missing} rows without a concat sequence")
        path = os.path.join(OUT, f"pdbbind_{split}_full_concat.csv")
        out.to_csv(path, index=False)
        L = out.target_sequence.dropna().str.len()
        print(f"{split:5s} n={len(out)}  concat len median {L.median():.0f} "
              f"mean {L.mean():.0f} max {L.max()}  -> {path}")

    stats = pd.DataFrame({
        "pdb_id": list(concat),
        "n_chains": [nchain[p] for p in concat],
        "len_longest": [len(longest[p]) for p in concat],
        "len_concat": [len(concat[p]) for p in concat],
    })
    stats.to_csv(os.path.join(OUT, "concat_stats.csv"), index=False)
    print(f"\nentries with >1 chain: {(stats.n_chains > 1).sum()} "
          f"({100*(stats.n_chains > 1).mean():.1f}%)")
    for cap in (2000, 3000, 4000, 4700):
        print(f"  concat length > {cap}: {(stats.len_concat > cap).sum()}")


if __name__ == "__main__":
    sys.exit(main())
