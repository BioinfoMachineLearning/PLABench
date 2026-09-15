"""
DeepDTA Unified Inference Script
Handles all Structure_independent datasets with a single function.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import sys
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# DeepDTA hyperparameters (default for Refined 9:1 model)
CHANNEL = 32
PROTEIN_KERNEL = 12
LIGAND_KERNEL = 8


class TestDataset(Dataset):
    """Dataset that encodes protein sequences and SMILES strings as integer tensors."""

    def __init__(self, df, protein_dict, ligand_dict, seq_col, smi_col, aff_col, seqlen=2000, smilen=250):
        self.df = df
        self.protein_dict = protein_dict
        self.ligand_dict = ligand_dict
        self.seq_col = seq_col
        self.smi_col = smi_col
        self.aff_col = aff_col
        self.seqlen = seqlen
        self.smilen = smilen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row[self.seq_col])
        smi = str(row[self.smi_col])
        label = float(row[self.aff_col])

        p_enc = [self.protein_dict.get(a, 0) for a in seq[:self.seqlen]]
        if len(p_enc) < self.seqlen:
            p_enc += [self.protein_dict.get('dummy', 0)] * (self.seqlen - len(p_enc))

        l_enc = [self.ligand_dict.get(a, 0) for a in smi[:self.smilen]]
        if len(l_enc) < self.smilen:
            l_enc += [self.ligand_dict.get('dummy', 0)] * (self.smilen - len(l_enc))

        return (torch.tensor(p_enc),
                torch.tensor(l_enc),
                torch.tensor(label, dtype=torch.float))


def run_inference(test_file, model_dir, output_file, device_str="cuda", dataset_name="unknown", fold=1, task="warm_start", seqlen_override=None):
    """
    Run DeepDTA inference on a single CSV dataset.

    Args:
        test_file: Path to CSV with columns containing target_sequence,
                   compound_iso_smiles, and affinity.
        model_dir: Directory containing model_refined91.pt and dict JSONs (base dir for CV).
        output_file: Path to write predictions.csv (prediction,target_id).
        device_str: 'cuda' or 'cpu'.
        dataset_name: Name of the dataset for CV logic.
        fold: Fold number for CV logic.
        task: Task type ('warm_start', 'cold_drug', 'cold_target') for CV logic.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # --- Load data ---
    df = pd.read_csv(test_file)
    log.info(f"Loaded {len(df)} samples from {test_file}")

    # Auto-detect column names
    cols = list(df.columns)
    id_col = cols[0]  # First column is always the ID

    seq_col = 'target_sequence'
    smi_col = 'compound_iso_smiles'
    aff_col = 'affinity' if 'affinity' in cols else 'label'

    for c in [seq_col, smi_col, aff_col]:
        if c not in cols:
            raise ValueError(f"Required column '{c}' not found in {test_file}. Columns: {cols}")

    log.info(f"ID column: '{id_col}', Seq: '{seq_col}', SMILES: '{smi_col}', Affinity: '{aff_col}'")

    # --- Determine paths based on CV logic ---
    base_proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../forks/DeepDTA-Pytorch'))
    seqlen = 2000
    smilen = 250
    pro_vocab_size = None
    lig_vocab_size = None
    
    if "davis" in dataset_name.lower() or "kiba" in dataset_name.lower():
        # CV Logic
        # Override model_dir to the FULL project directory where checkpoints live
        model_dir = base_proj_dir
        
        # Sequence lengths
        if "davis" in dataset_name.lower():
            if "warm" in task:
                seqlen = 1000
                smilen = 100
            else:
                seqlen = 3000
                smilen = 300
        else:
            seqlen = 8000
            smilen = 600

        # Determine specific folder and filename based on dataset and task
        if "davis" in dataset_name.lower():
            if "warm" in task:
                target_dir = os.path.join(model_dir, "davis_warm_results", f"davis_warm_fold_{fold}")
                model_filename = f"best_model_fold_{fold}.pt"
            elif "drug" in task:
                target_dir = os.path.join(model_dir, "davis_drug_results", f"davis_drug_fold_{fold}")
                model_filename = f"deepdta-fold_{fold}-prk12-ldk8.pt"
            else: # target
                target_dir = os.path.join(model_dir, "davis_results", f"davis_target_fold_{fold}")
                model_filename = f"deepdta-fold_{fold}-prk12-ldk8.pt"
        else: # KIBA
            if "warm" in task:
                target_dir = os.path.join(model_dir, "kiba_warm_results", f"fold_{fold}")
                model_filename = f"deepdta-warm-fold_{fold}-prk12-ldk8.pt"
            elif "drug" in task:
                target_dir = os.path.join(model_dir, "kiba_drug_results", f"kiba_drug_fold_{fold}")
                model_filename = f"deepdta-fold_{fold}-prk12-ldk8.pt"
            else: # target
                target_dir = os.path.join(model_dir, "kiba_target_cold_results", f"kiba_target_cold_fold_{fold}")
                model_filename = f"best_model_fold_{fold}.pt"
        
        # Check if the computed target directory exists, fallback to standard model if not
        if not os.path.exists(target_dir):
            if "target_fold" in target_dir: # Handle naming inconsistency
                 target_dir = target_dir.replace("davis_target_fold", "fold")
                 model_filename = model_filename.replace("fold", "davis-target-fold")
                 
        if os.path.exists(target_dir):
             model_path = os.path.join(target_dir, model_filename)
             p_dict_path = os.path.join(target_dir, 'protein_dict-prk12-ldk8.json')
             l_dict_path = os.path.join(target_dir, 'ligand_dict-prk12-ldk8.json')
             
             # Hardcoded KIBA datasets or Davis Warm which never saved JSONs
             use_hardcoded = False
             if "kiba" in dataset_name.lower() and "target" in task:
                 use_hardcoded = True
             if "davis" in dataset_name.lower() and "warm" in task:
                 use_hardcoded = True

             # If dict JSONs are missing for dynamic Trainer folds, attempt to fallback to Fold 1 dicts
             if not os.path.exists(p_dict_path) and not use_hardcoded:
                 print(f"Dictionaries missing in {target_dir}. Falling back to fold 1 dictionaries.")
                 target_dir_fold1 = target_dir.replace(f"fold_{fold}", "fold_1").replace(f"fold_{fold}", "fold_1")
                 p_dict_path = os.path.join(target_dir_fold1, 'protein_dict-prk12-ldk8.json')
                 l_dict_path = os.path.join(target_dir_fold1, 'ligand_dict-prk12-ldk8.json')

             if use_hardcoded:
                 protein_dict = {
                     "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9,
                     "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17,
                     "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
                     "dummy": 0
                 }
                 ligand_dict = {
                     "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, ".": 7, "1": 8, "0": 9,
                     "3": 10, "2": 11, "5": 12, "4": 13, "7": 14, "6": 15, "9": 16, "8": 17,
                     "=": 18, "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24, "F": 25,
                     "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, "O": 31, "N": 32, "P": 33,
                     "S": 34, "R": 35, "U": 36, "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41,
                     "Z": 42, "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, "d": 49,
                     "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, "l": 55, "o": 56, "n": 57,
                     "s": 58, "r": 59, "u": 60, "t": 61, "y": 62, "dummy": 0
                 }
                 pro_vocab_size = 100
                 lig_vocab_size = 100
             elif os.path.exists(p_dict_path) and os.path.exists(l_dict_path):
                 with open(p_dict_path) as f:
                     protein_dict = json.load(f)
                 with open(l_dict_path) as f:
                     ligand_dict = json.load(f)
             else:
                 print(f"Dictionaries not found for {dataset_name} {task}, falling back to default PDBbind dictionaries")
                 p_dict_path = os.path.join(model_dir, 'pdbbind2020_refined91_results', 'protein_dict-prk12-ldk8.json')
                 l_dict_path = os.path.join(model_dir, 'pdbbind2020_refined91_results', 'ligand_dict-prk12-ldk8.json')
                 with open(p_dict_path) as f:
                     protein_dict = json.load(f)
                 with open(l_dict_path) as f:
                     ligand_dict = json.load(f)

        else:
             logging.warning(f"Could not find CV target dir {target_dir}. Falling back to default.")
             model_path = os.path.join(model_dir, 'pdbbind2020_refined91_results', 'model_refined91.pt')
             p_dict_path = os.path.join(model_dir, 'pdbbind2020_refined91_results', 'protein_dict-prk12-ldk8.json')
             l_dict_path = os.path.join(model_dir, 'pdbbind2020_refined91_results', 'ligand_dict-prk12-ldk8.json')
             with open(p_dict_path) as f:
                 protein_dict = json.load(f)
             with open(l_dict_path) as f:
                 ligand_dict = json.load(f)
             
    else:
        # Default Logic (CASP, CSAR, CASF)
        model_path = os.path.join(model_dir, 'model_refined91.pt')
        p_dict_path = os.path.join(model_dir, 'protein_dict-prk12-ldk8.json')
        l_dict_path = os.path.join(model_dir, 'ligand_dict-prk12-ldk8.json')
        with open(p_dict_path) as f:
            protein_dict = json.load(f)
        with open(l_dict_path) as f:
            ligand_dict = json.load(f)

    log.info(f"Protein dict: {len(protein_dict)} entries, Ligand dict: {len(ligand_dict)} entries")

    if pro_vocab_size is None:
        pro_vocab_size = len(protein_dict) + 1
    if lig_vocab_size is None:
        lig_vocab_size = len(ligand_dict) + 1
        
    # --- Load model ---
    # Import DeepDTA model from forks
    source_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), 'forks', 'DeepDTA-Pytorch')
    sys.path.insert(0, source_dir)
    from model import DeepDTA

    model = DeepDTA(
        pro_vocab_size,
        lig_vocab_size,
        CHANNEL,
        PROTEIN_KERNEL,
        LIGAND_KERNEL
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    log.info(f"Model loaded from {model_path}")

    # --- Inference ---
    # DeepDTA global-max-pools over the sequence axis, so the window is a data-prep choice
    # rather than an architectural one; an override lets a checkpoint trained at a different
    # window be scored at the window it was trained on.
    if seqlen_override is not None:
        seqlen = int(seqlen_override)
        log.info(f"seqlen overridden to {seqlen}")

    ds = TestDataset(df, protein_dict, ligand_dict, seq_col, smi_col, aff_col, seqlen, smilen)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    preds = []
    swap_inputs = ("davis" in dataset_name.lower() and "warm" in task)
    
    with torch.no_grad():
        for p, l, aff in loader: # Changed from `_` to `aff` as per instruction, `meta` removed as TestDataset only returns 3 items
            p, l = p.to(device), l.to(device)
            if swap_inputs:
                out = model(l, p)  # train_davis_warm swapped args
            else:
                out = model(p, l)
            preds.extend(out.cpu().numpy().flatten())

    preds = np.array(preds)
    ids = df[id_col].tolist()

    # --- Save predictions.csv ---
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    with open(output_file, 'w') as f:
        for pred, tid in zip(preds, ids):
            f.write(f"{pred:.4f},{tid}\n")

    log.info(f"Saved {len(preds)} predictions to {output_file}")

    return preds, ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepDTA Unified Inference')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test CSV')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing model_refined91.pt and dict JSONs')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save predictions.csv')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--dataset_name', type=str, default='unknown',
                        help='Dataset name for CV indexing')
    parser.add_argument('--fold', type=int, default=1,
                        help='Fold number for CV')
    parser.add_argument('--task', type=str, default='warm_start',
                        help='Task type for CV (warm_start, cold_drug, cold_target)')
    parser.add_argument('--seqlen', type=int, default=None,
                        help='Override max protein sequence length (must match training)')
    args = parser.parse_args()

    run_inference(args.test_file, args.model_dir, args.output_file, args.device,
                  args.dataset_name, args.fold, args.task, args.seqlen)
