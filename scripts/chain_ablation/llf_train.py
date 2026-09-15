"""LLF chain-rule ablation, step 2: train one arm.

A faithful clone of /home/lwfvx/Lyuwei/0.Projects/LLF/train_pdbbind_full.py -- same GCNNet,
same optimiser, same cosine schedule, same early-stopping rule, same 90/10 split -- with the
dataset paths, the window and the seed made parameters.

Arms (see llf_build_data.py):
  A  longest chain, window 1200   the deployed checkpoint; not reproduced here
  B  longest chain, window 4700   control for the window
  C  concatenated,  window 4700   treatment

Writes only under scratch/concat_ablation/runs/llf/.
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

SRC = "/home/lwfvx/Lyuwei/0.Projects/LLF"
sys.path.insert(0, SRC)
from gcn import GCNNet                                                   # noqa: E402
from emetrics import get_cindex, get_mse, get_pearson, get_rmse, get_spearman  # noqa: E402

W = "/bmlfast/Lyuwei/0.Projects/PLABench/scratch/concat_ablation"

# hyperparameters copied verbatim from train_pdbbind_full.py
BATCH_SIZE, LR, DROPOUT, EPOCHS, PATIENCE, ETA_MIN = 128, 0.0005, 0.2, 200, 66, 1e-6


class SimpleDataset(InMemoryDataset):
    def __init__(self, data_path):
        super().__init__()
        self.data, self.slices = torch.load(data_path, weights_only=False)
        self._indices = None


def train_epoch(model, device, loader, optimizer, criterion):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch, None, None), batch.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / n


def validate(model, device, loader):
    model.eval()
    Y, P = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            Y.extend(batch.y.cpu().numpy().tolist())
            P.extend(model(batch, None, None).cpu().numpy().tolist())
    Y, P = np.array(Y).flatten(), np.array(P).flatten()
    return (get_mse(Y, P), get_rmse(Y, P), get_cindex(Y, P),
            get_pearson(Y, P), get_spearman(Y, P))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("longest", "concat"), required=True)
    ap.add_argument("--max_seq_len", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    tag = f"{a.arm}_L{a.max_seq_len}_s{a.seed}"
    out = os.path.join(W, "runs", "llf", tag)
    os.makedirs(out, exist_ok=True)
    proc = os.path.join(W, "llf_data", "processed")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    train_data = SimpleDataset(f"{proc}/llf_{a.arm}_train_{a.max_seq_len}.pt")
    val_data = SimpleDataset(f"{proc}/llf_{a.arm}_val_{a.max_seq_len}.pt")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    print(f"[{tag}] device={device} train={len(train_data)} val={len(val_data)}", flush=True)

    model = GCNNet(k1=1, k2=2, k3=3, embed_dim=256, num_layer=1, device=device,
                   num_feature_xd=78, n_output=1, num_feature_xt=25, output_dim=128,
                   dropout=DROPOUT).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS,
                                                           eta_min=ETA_MIN)

    best_mse, best_epoch, stale = float("inf"), 0, 0
    history = {"train_loss": [], "val_mse": [], "val_ci": [], "val_pearson": []}
    ckpt = os.path.join(out, "best_model_full.pth")

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, device, train_loader, optimizer, criterion)
        mse, _, ci, pear, _ = validate(model, device, val_loader)
        scheduler.step()
        history["train_loss"].append(loss)
        history["val_mse"].append(mse)
        history["val_ci"].append(ci)
        history["val_pearson"].append(pear)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{tag}] epoch {epoch:3d}/{EPOCHS} loss {loss:.4f} "
                  f"val MSE {mse:.4f} CI {ci:.4f} Pearson {pear:.4f}", flush=True)
        # same rule as train_pdbbind_full.py: 0.001 MSE improvement threshold
        if mse < best_mse - 0.001:
            best_mse, best_epoch, stale = mse, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "mse": mse, "ci": ci, "pearson": pear}, ckpt)
        else:
            stale += 1
        if stale >= PATIENCE:
            print(f"[{tag}] early stopping at epoch {epoch}", flush=True)
            break

    with open(os.path.join(out, "training_history.json"), "w") as f:
        json.dump({"arm": a.arm, "max_seq_len": a.max_seq_len, "seed": a.seed,
                   "history": history, "best_epoch": best_epoch,
                   "best_mse": float(best_mse),
                   "timestamp": datetime.now().isoformat()}, f, indent=2)
    print(f"[{tag}] done: best epoch {best_epoch}, best val MSE {best_mse:.4f} -> {ckpt}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
