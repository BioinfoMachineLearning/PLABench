
"""
LLF (GCNNet) Unified Inference Script
Handles on-the-fly SMILES to Graph conversion and inference for all datasets.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse
import networkx as nx
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Check if model class can be imported
# Models are in forks/LLF
current_dir = os.path.dirname(os.path.abspath(__file__))
model_source_dir = os.path.abspath(os.path.join(current_dir, "../../../forks/LLF"))
if model_source_dir not in sys.path:
    sys.path.append(model_source_dir)

try:
    from gcn import GCNNet
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"Error: Could not import GCNNet from {model_source_dir}: {e}")
    sys.exit(1)

# --- Data Processing Logic (Ported from data_creation.py) ---

atom_fdim = 78
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: i for i, v in enumerate(seq_voc, 1)}
# Default, will be overridden
max_seq_len = 1200 

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                           'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                           'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                                           'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                           'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def smile_to_graph(smile):
    try:
        mol = MolFromSmiles(smile)
        if mol is None:
            mol = MolFromSmiles(smile, sanitize=False)
            if mol is not None:
                try: mol.UpdatePropertyCache(strict=False)
                except: pass
    except:
        mol = None

    if mol is None:
        return None

    features = []
    for atom in mol.GetAtoms():
        try:
            feature = atom_features(atom)
            features.append(feature / sum(feature))
        except:
            features.append(np.zeros(atom_fdim))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return max(1, len(features)), features, edge_index

def seq_cat(prot, max_len=1200):
    x = np.zeros(max_len, dtype=np.int64)
    # Truncate or pad
    for i, ch in enumerate(prot[:max_len]):
        x[i] = seq_dict.get(ch, 0)
    return x

def process_csv(csv_file, max_len=1200):
    print(f"Loading data from {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Detect ID column
    id_col = 'compound_id'
    if 'compound_id' not in df.columns: 
        if 'molecule_id' in df.columns:
            id_col = 'molecule_id'
        elif 'cid' in df.columns:
            id_col = 'cid'
    
    data_list = []
    ids = []
    
    print(f"Processing {len(df)} samples...")
    for _, row in df.iterrows():
        cid = row.get(id_col, f"Unknown_{_}")
        smi = row['compound_iso_smiles']
        seq = row['target_sequence']
        aff = row.get('affinity', 0.0)
        
        graph = smile_to_graph(smi)
        if graph is None:
            # print(f"Warning: Failed to process SMILES for {cid}")
            # Skip invalid smiles to match training behavior usually
            continue
            
        _, features, edge_index = graph
        
        try:
            # Ensure edge_index is not empty for PyG
            if len(edge_index) == 0:
                 # Single atom or disconnect
                 edge_index = [[0],[0]] # dummy self loop?
            
            data = Data(
                x=torch.FloatTensor(np.array(features)),
                edge_index=torch.LongTensor(edge_index).transpose(0, 1),
                y=torch.FloatTensor([aff]),
                target=torch.LongTensor([seq_cat(seq, max_len=max_len)]),
                compound_id=cid
            )
            data_list.append(data)
            ids.append(cid)
        except Exception as e:
            print(f"Warning: Failed to create Data object for {cid}: {e}")
            
    return data_list, ids

# --- Inference Logic ---

def get_model_path(dataset_name, fold, task):
    # Base path for LLF models
    base_proj_dir = os.path.abspath(os.path.join(model_source_dir, "models"))

    # Map task/dataset to directory structure
    # Scenarios:
    # 1. Davis Warm: davis_warm/fold_{fold}/best_model.pth
    # 2. Key: warm_start, dataset: davis_warm
    # 3. v2 (HiQBind+PDBbind+SPINDR): hiqbind_pdbbind_spider_v2/best_model_full.pth
    # 4. Default (non warm/cold): PDBbind-trained full model

    path = ""

    if task == "hiqbind_pdbbind_spider_v3" or "hiqbind_pdbbind_spider_v3" in dataset_name:
        # v3 ckpt lives in PLABench/forks/LLF/models — resolve relative to this file.
        plabench_llf_models = os.path.abspath(os.path.join(model_source_dir, "models"))
        path = os.path.join(plabench_llf_models, "hiqbind_pdbbind_spider_v3", "best_model_full.pth")

    elif task == "hiqbind_pdbbind_spider_v2" or "hiqbind_pdbbind_spider_v2" in dataset_name:
        # v2 ckpt lives in PLABench/forks/LLF/models — resolve relative to this file.
        plabench_llf_models = os.path.abspath(os.path.join(model_source_dir, "models"))
        path = os.path.join(plabench_llf_models, "hiqbind_pdbbind_spider_v2", "best_model_full.pth")

    elif "warm" in dataset_name and ("davis" in dataset_name or "kiba" in dataset_name):
        # e.g. davis_warm, kiba_warm
        path = os.path.join(base_proj_dir, f"{dataset_name}/fold_{fold}/best_model.pth")

    elif "cold" in dataset_name:
        # Cold start naming conventions are specific
        # davis_cold_drug -> davis_drug_cv
        # davis_cold_target -> davis_target_cv
        
        if "drug" in dataset_name:
            suffix = "drug_cv"
            # Naming: models/davis_drug_cv/best_model_fold_1.model (Note .model extension and position)
            # Check structure again:
            # list output: models/davis_drug_cv/best_model_fold_1.model
            
            prefix = "davis" if "davis" in dataset_name else "kiba"
            dir_name = f"{prefix}_{suffix}"
            filename = f"best_model_fold_{fold}.model"
            path = os.path.join(base_proj_dir, dir_name, filename)
            
        elif "target" in dataset_name:
            suffix = "target_cv"
            # Naming: models/davis_target_cv/fold_1/best_model.pth (Standardized?)
            # list output: models/davis_target_cv/fold_1/best_model.pth
            
            prefix = "davis" if "davis" in dataset_name else "kiba"
            dir_name = f"{prefix}_{suffix}"
            path = os.path.join(base_proj_dir, dir_name, f"fold_{fold}/best_model.pth")

    # Default: PDBbind-trained full model (for CASF/CSAR/CASP16/ChEMBL35 etc.)
    if not path:
        path = os.path.join(base_proj_dir, "pdbbind", "best_model_full.pth")

    return path

def run_inference(test_file, output_file, dataset_name, fold, task, device_name,
                  model_path_override=None, max_seq_len_override=None):
    model_path = model_path_override or get_model_path(dataset_name, fold, task)
    print(f"Looking for model at: {model_path}")

    # Determine max_seq_len (LLF is strict about this — must match training):
    #   warm_start / davis-kiba warm → 2000
    #   cold (davis-kiba cold_drug/cold_target) → 800
    #   v2 (HiQBind+PDBbind+SPINDR) → 1500
    #   PDBbind v1 → 1200
    if max_seq_len_override is not None:
        max_len = int(max_seq_len_override)
    elif task == "hiqbind_pdbbind_spider_v3" or "hiqbind_pdbbind_spider_v3" in dataset_name:
        max_len = 1500
    elif task == "hiqbind_pdbbind_spider_v2" or "hiqbind_pdbbind_spider_v2" in dataset_name:
        max_len = 1500
    elif "warm" in dataset_name or task == "warm_start":
        max_len = 2000
    elif "cold" in dataset_name:
        max_len = 800
    else:
        max_len = 1200

    print(f"Using max_seq_len={max_len} for {dataset_name}")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Load Data
    data_list, ids = process_csv(test_file, max_len=max_len)
    if not data_list:
        print("Error: No valid data processed.")
        return
        
    loader = DataLoader(dataset=data_list, batch_size=512, shuffle=False)
    print(f"Processed {len(data_list)} samples")
    
    # Load Model
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    
    # GCNNet Parameters (Standardized from reports)
    model = GCNNet(
        k1=1, k2=2, k3=3,
        embed_dim=256,
        num_layer=1,
        device=device,
        num_feature_xd=78,
        n_output=1,
        num_feature_xt=25,
        output_dim=128,
        dropout=0.2
    ).to(device)
    
    # Load Weights
    # Weights might be full checkpoint or just state_dict?
    # Common format is checkpoint dict with 'model_state_dict'
    try:
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            # Maybe saved directly as state_dict or model
            model.load_state_dict(ckpt)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
        
    model.eval()
    preds = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch, None, None)
            preds.extend(output.cpu().numpy().flatten())
            
    # Save Results
    print(f"Saving {len(preds)} predictions to {output_file}")
    df_out = pd.DataFrame({
        "prediction": preds,
        "target_id": ids
    })
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # plabench standard: no header, columns prediction, compound_id
    df_out.to_csv(output_file, index=False, header=False)
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--dataset_name", required=True, help="davis_warm, kiba_cold_drug, etc.")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--task", default="warm_start")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_path", default=None, help="Override auto-resolved model path")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Override max protein sequence length")
    args = parser.parse_args()

    run_inference(args.test_file, args.output_file, args.dataset_name, args.fold,
                  args.task, args.device, args.model_path, args.max_seq_len)
