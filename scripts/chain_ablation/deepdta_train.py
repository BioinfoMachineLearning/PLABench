"""DeepDTA chain-rule ablation: retrain the refined-91 model under a given chain rule.

A faithful clone of the deployed /home/lwfvx/Lyuwei/0.Projects/DeepDTA-Pytorch/train_refined91.py
-- same Trainer, same architecture, same optimizer, same 90/10 split -- with three things
added: the input CSVs are a parameter, seqlen is a parameter, and the run is seeded.

Three arms are defined for the paper:

  A  longest chain, seqlen 2000   the deployed model; nothing here reproduces it, it is the
                                  published checkpoint and is left untouched
  B  longest chain, seqlen 4700   control: isolates the effect of widening the window alone
  C  concatenated,  seqlen 4700   treatment

B vs C is the chain rule with zero truncation on either side (longest max 1287, concat max
4638), which is the same condition MEETA was compared under. A vs B prices the window.

Nothing is written outside scratch/concat_ablation/runs/deepdta/.
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

SRC = "/home/lwfvx/Lyuwei/0.Projects/DeepDTA-Pytorch"
sys.path.insert(0, SRC)
from model import DeepDTA          # noqa: E402
from trainer_gpu import Trainer    # noqa: E402

W = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation"

# Hyperparameters copied verbatim from train_refined91.py
CHANNEL, PROTEIN_KERNEL, LIGAND_KERNEL = 32, 12, 8
NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE = 50, 256, 0.001
SMILEN = 250

SPLIT = "/home/lwfvx/Lyuwei/1.Datasets/pdbbind_refined_91"   # the deployed 90/10 split
ARMS = {
    # arm: (train csv, val csv). Both arms carry identical ids, SMILES, labels and row
    # order; target_sequence is the only column that differs.
    "longest": (f"{SPLIT}/pdbbind_train_full.csv", f"{SPLIT}/pdbbind_val_full.csv"),
    "concat": (f"{W}/data/pdbbind_train_full_concat.csv",
               f"{W}/data/pdbbind_val_full_concat.csv"),
}


def load_data(train_csv, val_csv):
    """Concatenate train and val into one frame; Trainer indexes into it by position."""
    train_df, val_df = pd.read_csv(train_csv), pd.read_csv(val_csv)
    full = pd.concat([train_df, val_df], ignore_index=True)
    full.rename(columns={"target_sequence": "proteins",
                         "compound_iso_smiles": "ligands"}, inplace=True)
    return (full, list(range(len(train_df))),
            list(range(len(train_df), len(full))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--seqlen", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    tag = f"{a.arm}_L{a.seqlen}_s{a.seed}"
    out = os.path.join(W, "runs", "deepdta", tag)
    os.makedirs(out, exist_ok=True)

    train_csv, val_csv = ARMS[a.arm]
    df, train_idx, val_idx = load_data(train_csv, val_csv)
    longest = df.proteins.astype(str).str.len().max()
    n_trunc = int((df.proteins.astype(str).str.len() > a.seqlen).sum())
    print(f"[{tag}] train {len(train_idx)} val {len(val_idx)} | longest sequence {longest} "
          f"| truncated at seqlen={a.seqlen}: {n_trunc}", flush=True)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    os.chdir(out)  # Trainer writes the vocab JSONs and test-result dump to cwd
    trainer = Trainer(
        model=DeepDTA, channel=CHANNEL,
        protein_kernel=PROTEIN_KERNEL, ligand_kernel=LIGAND_KERNEL,
        df=df, train_idx=train_idx, val_idx=val_idx,
        test_idx=val_idx,               # as in train_refined91.py: val doubles as the monitor
        log_file="training.log", gpu_id=a.gpu,
        seqlen=a.seqlen, smilen=SMILEN,
    )
    trainer.train(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                  lr=LEARNING_RATE, save_path="model_refined91.pt")
    print(f"[{tag}] done -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
