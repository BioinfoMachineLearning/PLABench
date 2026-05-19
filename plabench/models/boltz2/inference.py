"""Boltz2 affinity prediction inference for PLABench.

Two input modes:
  1. CASP16 mode (--sequence_file + --smiles_dir):
     Single protein FASTA + per-target SMILES TSV directory.
  2. Standardized CSV mode (--standardized_csv):
     CSV with columns: compound_id, target_sequence, compound_iso_smiles.
     Each row is an independent protein-ligand pair.

Generates per-target YAML configs, runs `boltz predict` per target with
MSA + MW correction, and collects affinity predictions into predictions.csv.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def parse_fasta(fasta_path):
    """Return the first sequence from a FASTA file."""
    sequence_lines = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if sequence_lines:
                    break
                continue
            sequence_lines.append(line)
    return "".join(sequence_lines)


def load_smiles_from_dir(smiles_dir, ground_truth_limit=None):
    """Load {target_id: SMILES} from a directory of TSV files."""
    valid_targets = None
    if ground_truth_limit and os.path.exists(ground_truth_limit):
        valid_targets = set()
        with open(ground_truth_limit, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if 'Target ID' in row:
                    valid_targets.add(row['Target ID'])

    targets = {}
    for fname in sorted(os.listdir(smiles_dir)):
        if not fname.endswith(".tsv"):
            continue
        target_id = fname.replace(".tsv", "")
        if valid_targets is not None and target_id not in valid_targets:
            continue
            
        fpath = os.path.join(smiles_dir, fname)
        with open(fpath) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                targets[target_id] = row["SMILES"]
                break
    return targets


def load_targets_from_csv(csv_path):
    """Load {target_id: (sequence, smiles)} from a standardized CSV.

    Expected columns: compound_id, target_sequence, compound_iso_smiles.
    Rows with status != 'OK' (if column exists) are skipped.
    """
    targets = {}
    with open(csv_path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if 'status' in row and row['status'] != 'OK':
                continue
            tid = row['compound_id']
            seq = row['target_sequence']
            smi = row['compound_iso_smiles']
            if seq and smi:
                targets[tid] = (seq, smi)
    return targets


def generate_yaml(target_id, sequence, smiles, out_path, msa_path=None):
    """Write a Boltz2 input YAML for one target.

    If msa_path is provided, inject it into the protein entry so that
    Boltz2 skips the MSA server call and reuses the pre-computed MSA.
    """
    protein_entry = {"id": "A", "sequence": sequence}
    if msa_path:
        protein_entry["msa"] = msa_path
    config = {
        "version": 1,
        "sequences": [
            {"protein": protein_entry},
            {"ligand": {"id": "B", "smiles": smiles}},
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }
    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_boltz_single(yaml_path, output_dir, boltz_bin,
                     diffusion_samples_affinity=10, sampling_steps_affinity=500):
    """Run boltz predict on a single YAML file."""
    cmd = [
        boltz_bin, "predict", yaml_path,
        "--use_msa_server", "--affinity_mw_correction",
        "--diffusion_samples_affinity", str(diffusion_samples_affinity),
        "--sampling_steps_affinity", str(sampling_steps_affinity),
        "--out_dir", output_dir,
    ]
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0



def get_affinity_path(output_dir, target_id):
    """Return expected affinity JSON path for a target."""
    # boltz outputs: <out_dir>/boltz_results_<stem>/predictions/B/affinity_B.json
    return os.path.join(
        output_dir, f"boltz_results_{target_id}", "predictions", "B", "affinity_B.json")


def read_affinity(json_path):
    """Read affinity_pred_value from a JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    return data.get("affinity_pred_value")


def write_predictions_csv(predictions, output_path):
    """Write predictions.csv in PLABench format: prediction,target_id.

    Boltz2 affinity_pred_value is log10(IC50/uM): -3 for strong binders,
    0 for moderate, +2 for weak.  Convert to pKd via  pred_pKd = 6 - y
    so that higher values = stronger binding, matching GT scale.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        for tid in sorted(predictions.keys()):
            writer.writerow([6 - predictions[tid], tid])
    log.info(f"Wrote {len(predictions)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Boltz2 affinity prediction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence_file", help="Protein FASTA file (CASP16 mode)")
    group.add_argument("--standardized_csv", help="Standardized CSV with compound_id, target_sequence, compound_iso_smiles")
    parser.add_argument("--smiles_dir", help="Directory of per-target SMILES TSV files (CASP16 mode)")
    parser.add_argument("--output_dir", required=True, help="Output directory for predictions.csv")
    parser.add_argument("--ground_truth_path", default=None, help="Optional CSV path to limit the targets")
    parser.add_argument("--boltz_bin", default="/bmlfast/Lyuwei/miniconda3/envs/boltz/bin/boltz",
                        help="Path to boltz binary")
    parser.add_argument("--diffusion_samples_affinity", type=int, default=10,
                        help="Number of diffusion samples for affinity (default: 10)")
    parser.add_argument("--sampling_steps_affinity", type=int, default=500,
                        help="Number of sampling steps for affinity (default: 500)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pred_file = os.path.join(args.output_dir, "predictions.csv")
    if os.path.exists(pred_file):
        log.info(f"predictions.csv already exists at {pred_file}, skipping.")
        return

    # 1. Read inputs — two modes
    if args.standardized_csv:
        # Standardized CSV mode: each row has its own sequence + SMILES
        csv_targets = load_targets_from_csv(args.standardized_csv)
        log.info(f"Loaded {len(csv_targets)} targets from {args.standardized_csv}")
        # targets dict maps tid -> smiles (for YAML generation)
        targets = {tid: smi for tid, (seq, smi) in csv_targets.items()}
        # sequences dict maps tid -> sequence
        sequences = {tid: seq for tid, (seq, smi) in csv_targets.items()}
    else:
        # CASP16 mode: single protein FASTA + SMILES directory
        if not args.smiles_dir:
            log.error("--smiles_dir required in CASP16 mode")
            sys.exit(1)
        sequence = parse_fasta(args.sequence_file)
        log.info(f"Protein sequence: {len(sequence)} residues")
        targets = load_smiles_from_dir(args.smiles_dir, args.ground_truth_path)
        sequences = {tid: sequence for tid in targets}
        log.info(f"Loaded {len(targets)} targets from {args.smiles_dir}")

    if not targets:
        log.error("No targets found.")
        sys.exit(1)

    # 2. Setup directories
    sandbox_dir = os.path.join(args.output_dir, "sandbox")
    os.makedirs(sandbox_dir, exist_ok=True)
    boltz_output = os.path.join(args.output_dir, "boltz_output")
    os.makedirs(boltz_output, exist_ok=True)
    msa_cache_dir = os.path.join(args.output_dir, "msa_cache")
    os.makedirs(msa_cache_dir, exist_ok=True)

    # Group compounds by protein sequence for MSA reuse
    seq_groups = defaultdict(list)
    for tid in sorted(targets.keys()):
        seq_groups[sequences[tid]].append(tid)

    multi_compound = len(seq_groups) < len(targets)
    if multi_compound:
        log.info(f"MSA reuse mode: {len(targets)} compounds, {len(seq_groups)} unique sequences")

    # 3. For each sequence group: run first compound with MSA server,
    #    then reuse its MSA for remaining compounds.
    predictions = {}

    for group_idx, (seq, tids) in enumerate(sorted(seq_groups.items(), key=lambda x: len(x[1]))):
        # Determine MSA cache key from sequence hash (dataset-agnostic)
        seq_hash = hashlib.md5(seq.encode()).hexdigest()[:12]
        msa_cache_path = os.path.join(msa_cache_dir, f"{seq_hash}_msa.csv")

        # Check which compounds are already done
        pending = []
        for tid in tids:
            aff_path = get_affinity_path(boltz_output, tid)
            if os.path.exists(aff_path):
                val = read_affinity(aff_path)
                if val is not None:
                    predictions[tid] = val
                else:
                    pending.append(tid)
            else:
                pending.append(tid)

        cached_count = len(tids) - len(pending)
        if not pending:
            log.info(f"[Group {group_idx+1}/{len(seq_groups)}] {seq_hash}: all {len(tids)} done (cached)")
            continue

        log.info(f"[Group {group_idx+1}/{len(seq_groups)}] {seq_hash}: {len(pending)} pending, {cached_count} cached")

        # Phase 1: If no MSA cache, run first compound WITH MSA server to generate it
        if not os.path.exists(msa_cache_path):
            first_tid = pending[0]
            yaml_path = os.path.join(sandbox_dir, f"{first_tid}.yaml")
            if not os.path.exists(yaml_path):
                generate_yaml(first_tid, seq, targets[first_tid], yaml_path)

            log.info(f"  MSA generation: running {first_tid} with MSA server...")
            ok = run_boltz_single(yaml_path, boltz_output, args.boltz_bin,
                                  args.diffusion_samples_affinity,
                                  args.sampling_steps_affinity)

            # Extract MSA from the result
            first_msa = os.path.join(
                boltz_output, f"boltz_results_{first_tid}", "msa", "B_0.csv")
            if os.path.exists(first_msa):
                shutil.copy2(first_msa, msa_cache_path)
                log.info(f"  MSA cached: {msa_cache_path}")
            else:
                log.error(f"  MSA generation failed for {first_tid}, no B_0.csv found")
                continue

            # Collect first compound's result
            aff_path = get_affinity_path(boltz_output, first_tid)
            if ok and os.path.exists(aff_path):
                val = read_affinity(aff_path)
                if val is not None:
                    predictions[first_tid] = val
                    log.info(f"  {first_tid}: affinity={val}")
            pending = pending[1:]  # Remove first from pending

        if not pending:
            continue

        # Phase 2: Run remaining compounds with pre-computed MSA (no server calls)
        abs_msa_path = os.path.abspath(msa_cache_path)
        log.info(f"  Running {len(pending)} compounds with cached MSA...")

        for i, tid in enumerate(pending):
            aff_path = get_affinity_path(boltz_output, tid)
            if os.path.exists(aff_path):
                val = read_affinity(aff_path)
                if val is not None:
                    predictions[tid] = val
                continue

            # Generate YAML with MSA path injected
            yaml_path = os.path.join(sandbox_dir, f"{tid}.yaml")
            generate_yaml(tid, seq, targets[tid], yaml_path, msa_path=abs_msa_path)

            log.info(f"  [{i+1}/{len(pending)}] {tid}...")
            cmd = [
                args.boltz_bin, "predict", yaml_path,
                "--affinity_mw_correction",
                "--diffusion_samples_affinity", str(args.diffusion_samples_affinity),
                "--sampling_steps_affinity", str(args.sampling_steps_affinity),
                "--out_dir", boltz_output,
            ]
            result = subprocess.run(cmd)

            if result.returncode == 0 and os.path.exists(aff_path):
                val = read_affinity(aff_path)
                if val is not None:
                    predictions[tid] = val
                    log.info(f"    affinity={val}")
                else:
                    log.warning(f"    no affinity_pred_value")
            else:
                log.warning(f"    FAILED")

    log.info(f"Collected {len(predictions)}/{len(targets)} predictions")

    if not predictions:
        log.error("No predictions collected.")
        sys.exit(1)

    # 4. Write predictions.csv
    write_predictions_csv(predictions, pred_file)

    missing = set(targets.keys()) - set(predictions.keys())
    if missing:
        log.warning(f"Missing predictions for: {sorted(missing)}")


if __name__ == "__main__":
    main()
