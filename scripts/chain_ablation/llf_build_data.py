"""LLF chain-rule ablation, step 1: build the PyG graph datasets for both arms.

LLF bakes the protein window into the stored tensor: data_creation_pdbbind_full.py encodes
every sequence into a fixed-width vector of length `max_seq_len` (1200 for the deployed
model), so changing the window means rebuilding the .pt files. The network itself is
length-agnostic -- gcn.py runs Embedding -> three Conv1d(padding=1, stride=1) ->
AdaptiveMaxPool1d(1) -> Linear(384, 768) -- so no weight shape depends on the window and a
wider one costs only memory.

Four datasets are produced, all at window 4700 (concat maxes out at 4638, longest at 1287,
so neither arm is truncated):

    llf_{longest,concat}_{train,val}_4700.pt

Ligand graphs are identical across arms -- same SMILES, same row order -- so only the
`target` tensor differs.

Everything lands in scratch/concat_ablation/llf_data/; the LLF tree is read-only here.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

SRC = "/home/lwfvx/Lyuwei/0.Projects/LLF"
sys.path.insert(0, SRC)
from utils import TestbedDataset  # noqa: E402

# atom featurisation and graph construction copied verbatim from
# data_creation_pdbbind_full.py so the graphs are byte-identical to the deployed ones
from rdkit import Chem, RDLogger  # noqa: E402
import networkx as nx             # noqa: E402

RDLogger.DisableLog("rdApp.*")

W = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation"
SPLIT = "/home/lwfvx/Lyuwei/1.Datasets/pdbbind_refined_91"
ROOT = os.path.join(W, "llf_data")

ARMS = {
    "longest": (f"{SPLIT}/pdbbind_train_full.csv", f"{SPLIT}/pdbbind_val_full.csv"),
    "concat": (f"{W}/data/pdbbind_train_full_concat.csv",
               f"{W}/data/pdbbind_val_full_concat.csv"),
}

SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {v: i + 1 for i, v in enumerate(SEQ_VOC)}


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set{allowable_set}:")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(),
                              ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                               'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                               'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                               'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                               'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
        + one_of_k_encoding(atom.GetDegree(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))
        + [atom.GetIsAromatic()])


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    features = [atom_features(a) / sum(atom_features(a)) for a in mol.GetAtoms()]
    edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    g = nx.Graph(edges).to_directed()
    return mol.GetNumAtoms(), features, [[e1, e2] for e1, e2 in g.edges]


def seq_cat(prot, max_seq_len):
    x = np.zeros(max_seq_len, dtype=int)
    for i, ch in enumerate(prot[:max_seq_len]):
        if ch in SEQ_DICT:
            x[i] = SEQ_DICT[ch]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_seq_len", type=int, default=4700)
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "processed"), exist_ok=True)

    # one graph cache for both arms: the SMILES column is identical between them
    frames = {(arm, split): pd.read_csv(csv)
              for arm, csvs in ARMS.items()
              for split, csv in zip(("train", "val"), csvs)}
    smiles = sorted({s for df in frames.values() for s in df.compound_iso_smiles})
    print(f"building graphs for {len(smiles)} unique SMILES ...", flush=True)
    graphs, failed = {}, []
    for s in smiles:
        g = smile_to_graph(s)
        if g is None:
            failed.append(s)
        else:
            graphs[s] = g
    print(f"  {len(graphs)} ok, {len(failed)} failed", flush=True)

    for (arm, split), df in frames.items():
        keep = [i for i, s in enumerate(df.compound_iso_smiles) if s not in failed]
        sub = df.iloc[keep]
        seqs = sub.target_sequence.astype(str)
        n_trunc = int((seqs.str.len() > a.max_seq_len).sum())
        name = f"llf_{arm}_{split}_{a.max_seq_len}"
        path = os.path.join(ROOT, "processed", f"{name}.pt")
        print(f"{name}: n={len(sub)} longest={seqs.str.len().max()} "
              f"truncated={n_trunc}", flush=True)
        if os.path.isfile(path):
            print("  exists, skipping", flush=True)
            continue
        TestbedDataset(root=ROOT, dataset=name,
                       xd=list(sub.compound_iso_smiles),
                       xt=np.asarray([seq_cat(t, a.max_seq_len) for t in seqs]),
                       y=np.asarray(list(sub.affinity)),
                       smile_graph=graphs)


if __name__ == "__main__":
    sys.exit(main())
