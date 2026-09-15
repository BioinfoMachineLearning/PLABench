"""
MFE (Multimodal Fusion Encoder) inference script for PLABench.

Runs inside the MFE conda environment. Handles:
1. Sandbox creation (SEQRES injection, pocket cutting, ligand symlink)
2. INDEX file generation + process_raw_data() for pkl
3. Surface precompute + model inference
4. Output predictions.csv

Usage:
    python -m plabench.models.mfe.inference \
        --input_dir data/casp16_data/stage2_input/L3000_prepared \
        --output_dir outputs/mfe/experimental_casp16_l3000_stage2 \
        --checkpoint checkpoints/structure/mfe/best_model.pt \
        --device cuda
"""

import os
import sys
import argparse
import shutil
import logging
import pickle
import csv
import numpy as np
import torch

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# ── SEQRES helpers ──────────────────────────────────────────────────

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M',
}


def extract_sequence_from_atoms(pdb_path):
    """Extract amino acid sequence from ATOM records (ordered by chain, resSeq).

    Returns dict: {chain_id: [(resSeq, resName_3letter), ...]}
    """
    seen = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            res_name = line[17:20].strip()
            if res_name not in AA_3TO1:
                continue
            chain = line[21]
            res_seq = line[22:27]  # resSeq + iCode
            key = (chain, res_seq)
            if key not in seen:
                seen[key] = res_name
    # Group by chain, preserving order
    chains = {}
    for (chain, _), res_name in seen.items():
        chains.setdefault(chain, []).append(res_name)
    return chains


def generate_seqres_lines(chains):
    """Generate SEQRES records from chain→residue_list dict."""
    lines = []
    for chain_id, residues in sorted(chains.items()):
        num_res = len(residues)
        for i in range(0, num_res, 13):
            chunk = residues[i:i+13]
            serial = (i // 13) + 1
            res_str = ' '.join(f'{r:>3s}' for r in chunk)
            line = f"SEQRES {serial:>3d} {chain_id} {num_res:>4d}  {res_str}"
            lines.append(line.rstrip() + '\n')
    return lines


def inject_seqres(input_pdb, output_pdb):
    """Read a PDB, inject SEQRES from ATOM records if missing, write output."""
    with open(input_pdb, 'r') as f:
        content = f.readlines()

    has_seqres = any(l.startswith('SEQRES') for l in content)
    if has_seqres:
        # Already has SEQRES, just copy
        if input_pdb != output_pdb:
            shutil.copy2(input_pdb, output_pdb)
        return

    chains = extract_sequence_from_atoms(input_pdb)
    if not chains:
        log.warning(f"No standard residues found in {input_pdb}")
        shutil.copy2(input_pdb, output_pdb)
        return

    seqres_lines = generate_seqres_lines(chains)

    with open(output_pdb, 'w') as f:
        # Write SEQRES before first ATOM/HETATM
        seqres_written = False
        for line in content:
            if not seqres_written and line.startswith(('ATOM', 'HETATM')):
                for sl in seqres_lines:
                    f.write(sl)
                seqres_written = True
            f.write(line)


# ── Sandbox preparation ─────────────────────────────────────────────

def prepare_sandbox(input_dir, sandbox_dir, target_pattern='L*', filter_file=None):
    """Prepare MFE sandbox from PLABench stage2 input.

    For each target directory, creates:
      sandbox/{target_id}/{target_id}_protein.pdb  (full protein with SEQRES)
      sandbox/{target_id}/{target_id}_pocket.pdb   (pocket cut at 10A)
      sandbox/{target_id}/{target_id}_ligand.mol2  (copy/symlink)

    Args:
        input_dir: directory holding {target_id}/ subdirs
        sandbox_dir: where to write per-target prepared files
        target_pattern: glob pattern within input_dir
        filter_file: optional CSV/DAT path; if given, only target_ids appearing
            in the first column are processed. Implements the agent.md §2.1
            GT-filter convention for multi-protein datasets.

    Returns:
        list of (target_id, dummy_score) tuples for INDEX generation
    """
    from pathlib import Path
    from plabench.utils.pocket_utils import extract_pocket

    input_path = Path(input_dir)
    targets = []

    # Optional GT-filter (matches bapred/inference.py:run_inference_loop signature)
    valid_targets = None
    if filter_file and os.path.exists(filter_file):
        valid_targets = set()
        try:
            if filter_file.endswith('.dat'):
                with open(filter_file) as f:
                    for line in f:
                        if line.startswith('#') or not line.strip():
                            continue
                        parts = line.split()
                        if parts:
                            valid_targets.add(parts[0])
            else:
                import pandas as pd
                fdf = pd.read_csv(filter_file)
                id_col = fdf.columns[0]
                for c in ('Target ID', 'Target_ID', 'compound_id', 'PDBID', 'TargetID'):
                    if c in fdf.columns:
                        id_col = c
                        break
                valid_targets = set(fdf[id_col].astype(str))
            log.info(f"GT-filter: {len(valid_targets)} target_ids loaded from {filter_file}")
        except Exception as e:
            log.warning(f"Failed to load filter_file {filter_file}: {e}; no filtering applied")
            valid_targets = None

    candidate_dirs = sorted(input_path.glob(target_pattern))
    if valid_targets is not None:
        original_n = len(candidate_dirs)
        candidate_dirs = [d for d in candidate_dirs if d.name in valid_targets]
        log.info(f"GT-filter: {len(candidate_dirs)}/{original_n} target dirs after filter")

    failures = []  # (target_id, ErrorType) — agent.md §3.1

    for target_dir in candidate_dirs:
        if not target_dir.is_dir():
            continue
        target_id = target_dir.name

        # Find input files
        protein_pdb = target_dir / 'protein.pdb'
        # Also check protein_aligned.pdb (some datasets)
        if not protein_pdb.exists():
            protein_pdb = target_dir / 'protein_aligned.pdb'
        ligand_mol2 = target_dir / 'ligand.mol2'

        if not protein_pdb.exists():
            log.warning(f"[{target_id}] Missing protein PDB, skipping")
            failures.append((target_id, 'missing_protein_pdb'))
            continue
        if not ligand_mol2.exists():
            log.warning(f"[{target_id}] Missing ligand.mol2, skipping")
            failures.append((target_id, 'missing_ligand_mol2'))
            continue

        # Create sandbox subdirectory
        sb_dir = os.path.join(sandbox_dir, target_id)
        os.makedirs(sb_dir, exist_ok=True)

        out_protein = os.path.join(sb_dir, f"{target_id}_protein.pdb")
        out_pocket = os.path.join(sb_dir, f"{target_id}_pocket.pdb")
        out_ligand = os.path.join(sb_dir, f"{target_id}_ligand.mol2")

        try:
            # 1. Inject SEQRES into full protein
            inject_seqres(str(protein_pdb), out_protein)

            # 2. Cut pocket (PDBbind standard: heavy atoms, 10A, whole residues)
            n_res = extract_pocket(str(protein_pdb), str(ligand_mol2),
                                   out_pocket, cutoff=10.0)
            if n_res == 0:
                log.warning(f"[{target_id}] Empty pocket, skipping")
                failures.append((target_id, 'empty_pocket'))
                continue

            # 3. Copy ligand mol2
            shutil.copy2(str(ligand_mol2), out_ligand)

            # Use dummy score 0.0 (we only need predictions)
            targets.append((target_id, 0.0))
            log.info(f"[{target_id}] OK ({n_res} pocket residues)")

        except Exception as e:
            log.error(f"[{target_id}] Sandbox prep failed: {e}")
            failures.append((target_id, f'sandbox_prep_failed:{type(e).__name__}'))
            continue

    # Persist failures to sandbox so the caller can copy out (or for debugging)
    if failures:
        fail_path = os.path.join(sandbox_dir, 'failures_sandbox.csv')
        with open(fail_path, 'w') as f:
            f.write('TargetID,ErrorType\n')
            for tid, err in failures:
                f.write(f'{tid},{err}\n')
        log.warning(f"[sandbox] {len(failures)} sandbox-prep failures recorded "
                    f"in {fail_path}")

    return targets


def write_index_file(targets, index_path):
    """Write MFE-compatible INDEX file."""
    with open(index_path, 'w') as f:
        for target_id, score in targets:
            # Format: "{target_id}  //  -  {score}"
            f.write(f"{target_id}  //  -  {score:.3f}\n")


# ── MFE data processing + inference ─────────────────────────────────

def run_mfe_inference(sandbox_dir, index_path, output_dir, checkpoint_path,
                      mfe_root, protbert_model, device='cuda'):
    """Run full MFE pipeline: process → surface precompute → inference."""

    # Add MFE fork to path
    if mfe_root not in sys.path:
        sys.path.insert(0, mfe_root)

    from easydict import EasyDict
    import yaml
    from process import (process_raw_data, GetPDBDict, GetPDBList,
                         Mult_graph, load_protein_atoms, load_protein_graph)
    from binding_data import PLBA_Dataset
    from models.lba_model import LBAPredictor
    from dmasif_encoder.data_iteration import iterate_surface_precompute
    from torch_geometric.data import DataLoader
    from transformers import BertModel, BertTokenizer, pipeline
    from tqdm import tqdm
    import re

    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Process raw data into pkl ──
    pkl_path = os.path.join(output_dir, 'test.pkl')

    if os.path.exists(pkl_path):
        log.info(f"Reusing existing pkl: {pkl_path}")
    else:
        log.info("Processing raw data...")
        # Parse INDEX
        res_dict = {}
        protein_list = []
        with open(index_path, 'r') as f:
            for line in f:
                if '//' in line:
                    parts = line.split()
                    name, score = parts[0], float(parts[3])
                    res_dict[name] = score
                    protein_list.append(name)

        # Load ProtBert
        log.info(f"Loading ProtBert from {protbert_model}...")
        tokenizer = BertTokenizer.from_pretrained(protbert_model,
                                                  do_lower_case=False)
        model = BertModel.from_pretrained(protbert_model)
        fe = pipeline('feature-extraction', model=model,
                      tokenizer=tokenizer, device=0 if device == 'cuda' else -1)

        dataset_path = sandbox_dir.rstrip('/') + '/'
        G_list = []

        for item in tqdm(protein_list, desc="Processing targets"):
            score = res_dict[item]
            lig_file = dataset_path + item + '/' + item + '_ligand.mol2'
            pocket_file = dataset_path + item + '/' + item + '_pocket.pdb'
            protein_path = dataset_path + item + '/'

            if not os.path.exists(protein_path):
                continue

            G = Mult_graph(lig_file, pocket_file, item, score, fe)
            protein_atoms = load_protein_atoms(protein_path, item)
            protein_graph = load_protein_graph(protein_path, item)

            if G is not None and protein_atoms is not None and protein_graph is not None:
                G.append(protein_atoms)
                G.append(protein_graph)
                G_list.append(G)
            else:
                log.warning(f"[{item}] Processing failed (G={G is not None}, "
                            f"atoms={protein_atoms is not None}, "
                            f"graph={protein_graph is not None})")

        log.info(f"Processed {len(G_list)}/{len(protein_list)} targets")
        with open(pkl_path, 'wb') as f:
            pickle.dump(G_list, f)

    # ── Step 2: Load model + surface precompute ──
    log.info("Loading model and config...")
    config_path = os.path.join(mfe_root, 'configs', 'config.yml')
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    test_set = PLBA_Dataset('file', pkl_path)
    if len(test_set) == 0:
        log.error("No valid samples in pkl. Aborting.")
        return

    metadata = test_set[0][1].metadata()
    dev = torch.device(device)

    model = LBAPredictor(metadata=metadata, config=config, device=dev).to(dev)

    # Materialize lazy parameters with a dummy forward pass
    # (SAGEConv uses lazy init, needs one pass before load_state_dict)
    batch_vars = ["atom_coords", "seq"]
    dummy_loader = DataLoader(dataset=test_set, batch_size=1,
                              follow_batch=batch_vars)
    with torch.no_grad():
        dummy_data = next(iter(dummy_loader))
        dummy_data = [g.to(dev) for g in dummy_data]
        # Only need forward through heterognn to materialize SAGEConv params
        try:
            model(dummy_data)
        except Exception:
            pass  # May fail on surface encoder, that's fine

    # Load checkpoint
    log.info(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(state_dict)
    model.eval()

    # Surface precompute (batch_size=1)
    batch_vars = ["atom_coords", "seq"]
    loader = DataLoader(dataset=test_set, batch_size=1,
                        follow_batch=batch_vars)
    log.info("Running surface precompute...")
    processed = iterate_surface_precompute(loader,
                                           model.pmn.protein_surface_encoder,
                                           dev)

    # ── Step 3: Inference ──
    log.info("Running inference...")
    batch_size = 1
    test_loader = DataLoader(dataset=processed, batch_size=batch_size,
                             follow_batch=batch_vars, shuffle=False)

    predictions = []
    target_ids = []

    # Extract target IDs from pkl (G[3] = id)
    with open(pkl_path, 'rb') as f:
        G_list = pickle.load(f)
    for G in G_list:
        target_ids.append(G[3])

    idx = 0
    for data in tqdm(test_loader, desc="Inference"):
        with torch.no_grad():
            data_gpu = [g.to(dev) for g in data]
            pred = model(data_gpu)
            preds = pred.cpu().tolist()
            if isinstance(preds, float):
                preds = [preds]
            predictions.extend(preds)
            idx += len(preds)

    # ── Step 4: Write predictions.csv ──
    pred_path = os.path.join(output_dir, 'predictions.csv')
    with open(pred_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for pred_val, tid in zip(predictions, target_ids):
            writer.writerow([pred_val, tid])

    log.info(f"Wrote {len(predictions)} predictions to {pred_path}")


# ── CLI entry point ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MFE inference for PLABench')
    parser.add_argument('--input_dir', required=True,
                        help='Path to input data (e.g. data/casp16_data/stage2_input/L3000_prepared)')
    parser.add_argument('--output_dir', required=True,
                        help='Path to output directory')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to MFE model checkpoint')
    parser.add_argument('--mfe_root', default='forks/MFE',
                        help='Path to MFE fork root')
    parser.add_argument('--protbert_model', default='Rostlab/prot_bert',
                        help='ProtBert model name or local path '
                             '(default: Rostlab/prot_bert, auto-downloads from HuggingFace)')
    parser.add_argument('--device', default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--target_pattern', default='L*',
                        help='Glob pattern for target directories')
    parser.add_argument('--filter_file', default=None,
                        help='Optional CSV/DAT path; if given, only target_ids '
                             'appearing in the first column are processed. '
                             'Implements the agent.md §2.1 GT-filter convention.')
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    os.chdir(project_root)

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    checkpoint = os.path.abspath(args.checkpoint)
    mfe_root = os.path.abspath(args.mfe_root)

    os.makedirs(output_dir, exist_ok=True)

    # Sandbox in tmp/ — use parent dir name (dataset name) for uniqueness
    dataset_name = os.path.basename(os.path.dirname(output_dir))
    sandbox_dir = os.path.join(project_root, 'tmp',
                               f'mfe_{dataset_name}_sandbox')
    os.makedirs(sandbox_dir, exist_ok=True)

    # Step 1: Prepare sandbox
    index_path = os.path.join(sandbox_dir, 'INDEX')
    if os.path.exists(index_path):
        log.info(f"Reusing existing sandbox: {sandbox_dir}")
        # Re-read targets from INDEX
        targets = []
        with open(index_path, 'r') as f:
            for line in f:
                if '//' in line:
                    parts = line.split()
                    targets.append((parts[0], float(parts[3])))
    else:
        log.info(f"Preparing sandbox from {input_dir}...")
        targets = prepare_sandbox(
            input_dir, sandbox_dir, args.target_pattern,
            filter_file=os.path.abspath(args.filter_file) if args.filter_file else None,
        )
        if not targets:
            log.error("No valid targets found. Aborting.")
            sys.exit(1)
        write_index_file(targets, index_path)

    log.info(f"Total targets: {len(targets)}")

    # Step 2: Run MFE inference
    run_mfe_inference(
        sandbox_dir=sandbox_dir,
        index_path=index_path,
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        mfe_root=mfe_root,
        protbert_model=args.protbert_model,
        device=args.device,
    )


if __name__ == '__main__':
    main()
