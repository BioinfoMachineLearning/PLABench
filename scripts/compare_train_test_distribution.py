#!/usr/bin/env python3
"""Train-vs-test distribution comparison for LLF / sequence-based PLA models.

Three measures per (train, test) pair:

  1. Compound chemical-space overlap — Morgan-2048 Tanimoto similarity:
     for each unique test compound, find the max similarity to any train
     compound; report distribution of these max-similarities (mean, median,
     percentiles, fraction >= {0.7, 0.5, 0.3}).

  2. Protein sequence overlap — k-mer Jaccard similarity (k=10):
     for each unique test sequence, find max Jaccard to any train sequence;
     report distribution.  k=10 is short enough to tolerate small indels but
     specific enough that random-sequence overlap stays near zero.

  3. Affinity (-log10 M, pX) summary statistics: mean, std, percentiles —
     gauges absolute-prediction shift between training and test.

Inputs are CSV files with columns 'compound_iso_smiles' (SMILES),
'target_sequence' (one-letter aa), and 'affinity' (pX float).  Column names
can be overridden per-dataset.

Outputs to stdout.  No file writes — one-off analysis tool.

Usage:
    python scripts/compare_train_test_distribution.py \
        --train v1_refined=/path/to/pdbbind_train_full.csv \
        --train v6_filtered=/path/to/v6/train.csv \
        --test  chembl35=/path/to/chembl35_full_input.csv \
        --test  casp16_l1000=/path/to/L1000_casp16_test.csv

All --train and --test args may be repeated; cross-product is computed.
"""

import argparse
import time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


DEFAULT_SMI_COL = 'compound_iso_smiles'
DEFAULT_SEQ_COL = 'target_sequence'
DEFAULT_AFF_COL = 'affinity'

KMER_K = 10
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
TANIMOTO_THRESHOLDS = (0.7, 0.5, 0.3)
JACCARD_THRESHOLDS = (0.9, 0.5, 0.1)


def get_fp(smi):
    """Morgan radius-2 2048-bit fingerprint, or None if RDKit can't parse."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, MORGAN_RADIUS, MORGAN_BITS)


def kmer_set(seq, k=KMER_K):
    """k-mer multiset (as set) for a protein sequence string."""
    if not seq or not isinstance(seq, str):
        return set()
    if len(seq) < k:
        return {seq}  # short sequence: itself is the only k-mer
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def load_dataset(path, smi_col, seq_col, aff_col):
    df = pd.read_csv(path)
    smis = df[smi_col].dropna().tolist() if smi_col in df.columns else []
    seqs = df[seq_col].dropna().tolist() if seq_col in df.columns else []
    affs = df[aff_col].dropna().to_numpy() if aff_col in df.columns else np.array([])
    return smis, seqs, affs


def compute_train_fps(train_smiles):
    fps = []
    n_failed = 0
    for s in set(train_smiles):
        fp = get_fp(s)
        if fp is None:
            n_failed += 1
        else:
            fps.append(fp)
    return fps, n_failed


def compute_train_kmers(train_seqs):
    out = []
    for s in set(train_seqs):
        ks = kmer_set(s)
        if ks:
            out.append(ks)
    return out


def compound_max_tanimoto(test_smiles, train_fps):
    """Per unique test SMILES, max Tanimoto to any train FP."""
    if not train_fps:
        return np.array([]), 0
    sims = []
    n_failed = 0
    for s in set(test_smiles):
        fp = get_fp(s)
        if fp is None:
            n_failed += 1
            continue
        sims.append(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
    return np.array(sims), n_failed


def protein_max_jaccard(test_seqs, train_kmer_sets):
    if not train_kmer_sets:
        return np.array([])
    jaccs = []
    for s in set(test_seqs):
        ks = kmer_set(s)
        if not ks:
            continue
        best = 0.0
        for tk in train_kmer_sets:
            inter = len(ks & tk)
            if inter == 0:
                continue
            j = inter / len(ks | tk)
            if j > best:
                best = j
                if best == 1.0:
                    break
        jaccs.append(best)
    return np.array(jaccs)


def affinity_summary(arr):
    if len(arr) == 0:
        return None
    return dict(
        N=len(arr),
        mean=float(arr.mean()),
        std=float(arr.std()),
        min=float(arr.min()),
        p25=float(np.percentile(arr, 25)),
        p50=float(np.percentile(arr, 50)),
        p75=float(np.percentile(arr, 75)),
        max=float(arr.max()),
    )


def parse_dataset_arg(arg):
    """Parse 'name=path' or 'name=path,smi_col=X,seq_col=Y,aff_col=Z'."""
    parts = arg.split(',')
    name_path = parts[0]
    if '=' not in name_path:
        raise argparse.ArgumentTypeError(f"expected name=path, got {name_path!r}")
    name, path = name_path.split('=', 1)
    spec = dict(name=name, path=path,
                smi_col=DEFAULT_SMI_COL,
                seq_col=DEFAULT_SEQ_COL,
                aff_col=DEFAULT_AFF_COL)
    for kv in parts[1:]:
        if '=' in kv:
            k, v = kv.split('=', 1)
            spec[k] = v
    return spec


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--train', action='append', required=True, type=parse_dataset_arg,
                   help='train set, name=path[,smi_col=X][,seq_col=Y][,aff_col=Z]; can repeat')
    p.add_argument('--test', action='append', required=True, type=parse_dataset_arg,
                   help='test set, same syntax; can repeat')
    args = p.parse_args()

    print('=' * 90)
    print('TRAIN-vs-TEST DISTRIBUTION COMPARISON')
    print('=' * 90, '\n')

    # Load datasets
    train_data = {}
    for spec in args.train:
        smis, seqs, affs = load_dataset(spec['path'], spec['smi_col'], spec['seq_col'], spec['aff_col'])
        train_data[spec['name']] = dict(smis=smis, seqs=seqs, affs=affs, path=spec['path'])
        print(f"  TRAIN [{spec['name']:20s}]  rows={len(smis):>6d}  "
              f"unique_smiles={len(set(smis)):>6d}  unique_seqs={len(set(seqs)):>5d}  "
              f"({spec['path']})")

    test_data = {}
    for spec in args.test:
        smis, seqs, affs = load_dataset(spec['path'], spec['smi_col'], spec['seq_col'], spec['aff_col'])
        test_data[spec['name']] = dict(smis=smis, seqs=seqs, affs=affs, path=spec['path'])
        print(f"  TEST  [{spec['name']:20s}]  rows={len(smis):>6d}  "
              f"unique_smiles={len(set(smis)):>6d}  unique_seqs={len(set(seqs)):>5d}  "
              f"({spec['path']})")

    # Pre-compute train fingerprints + kmer sets
    print('\n=== preparing train representations ===')
    for name, td in train_data.items():
        t0 = time.time()
        td['fps'], n_fp_fail = compute_train_fps(td['smis'])
        td['kmers'] = compute_train_kmers(td['seqs'])
        print(f"  {name:20s}  train_fps={len(td['fps']):>5d} (parse_fail={n_fp_fail})  "
              f"train_kmer_sets={len(td['kmers']):>5d}  ({time.time()-t0:.1f}s)")

    # =======================================================
    # TEST 1: Compound Tanimoto
    # =======================================================
    print('\n=== TEST 1: max Morgan-2048 Tanimoto similarity (test compound → nearest train compound) ===')
    print()
    hdr_thresh = '  '.join(f'%≥{t}' for t in TANIMOTO_THRESHOLDS)
    print(f'{"train":18s} {"test":18s} {"nT":>5s}  '
          f'{"mean":>5s} {"med":>5s} {"p10":>5s} {"p25":>5s} {"p75":>5s} {"p90":>5s}  '
          f'{hdr_thresh}  {"fail":>4s}')
    for tr_name, tr in train_data.items():
        for ts_name, ts in test_data.items():
            sims, n_fail = compound_max_tanimoto(ts['smis'], tr['fps'])
            if len(sims) == 0:
                print(f'{tr_name:18s} {ts_name:18s} {0:>5d}  (no comparisons)')
                continue
            thresh_strs = '  '.join(f'{(sims >= t).mean()*100:>4.1f}%' for t in TANIMOTO_THRESHOLDS)
            print(f'{tr_name:18s} {ts_name:18s} {len(sims):>5d}  '
                  f'{sims.mean():>5.3f} {np.median(sims):>5.3f} '
                  f'{np.percentile(sims, 10):>5.3f} {np.percentile(sims, 25):>5.3f} '
                  f'{np.percentile(sims, 75):>5.3f} {np.percentile(sims, 90):>5.3f}  '
                  f'{thresh_strs}  {n_fail:>4d}')

    # =======================================================
    # TEST 2: Protein Jaccard
    # =======================================================
    print('\n=== TEST 2: max k-mer (k=10) Jaccard similarity (test protein → nearest train protein) ===')
    print()
    hdr_thresh = '  '.join(f'%≥{t}' for t in JACCARD_THRESHOLDS)
    print(f'{"train":18s} {"test":18s} {"nT":>5s}  '
          f'{"mean":>5s} {"med":>5s} {"min":>5s} {"max":>5s}  {hdr_thresh}')
    for tr_name, tr in train_data.items():
        for ts_name, ts in test_data.items():
            jaccs = protein_max_jaccard(ts['seqs'], tr['kmers'])
            if len(jaccs) == 0:
                print(f'{tr_name:18s} {ts_name:18s} {0:>5d}  (no comparisons)')
                continue
            thresh_strs = '  '.join(f'{(jaccs >= t).mean()*100:>4.1f}%' for t in JACCARD_THRESHOLDS)
            print(f'{tr_name:18s} {ts_name:18s} {len(jaccs):>5d}  '
                  f'{jaccs.mean():>5.3f} {np.median(jaccs):>5.3f} '
                  f'{jaccs.min():>5.3f} {jaccs.max():>5.3f}  {thresh_strs}')

    # =======================================================
    # TEST 3: Affinity distribution
    # =======================================================
    print('\n=== TEST 3: affinity (-log10 M, pX) distribution ===')
    print()
    print(f'{"set":20s} {"role":6s} {"N":>6s}  {"mean":>5s} {"std":>4s} {"min":>5s} '
          f'{"p25":>5s} {"p50":>5s} {"p75":>5s} {"max":>5s}')
    for name, td in train_data.items():
        s = affinity_summary(td['affs'])
        if s:
            print(f'{name:20s} {"TRAIN":6s} {s["N"]:>6d}  '
                  f'{s["mean"]:>5.2f} {s["std"]:>4.2f} {s["min"]:>5.2f} '
                  f'{s["p25"]:>5.2f} {s["p50"]:>5.2f} {s["p75"]:>5.2f} {s["max"]:>5.2f}')
    for name, td in test_data.items():
        s = affinity_summary(td['affs'])
        if s:
            print(f'{name:20s} {"TEST":6s} {s["N"]:>6d}  '
                  f'{s["mean"]:>5.2f} {s["std"]:>4.2f} {s["min"]:>5.2f} '
                  f'{s["p25"]:>5.2f} {s["p50"]:>5.2f} {s["p75"]:>5.2f} {s["max"]:>5.2f}')


if __name__ == '__main__':
    main()
