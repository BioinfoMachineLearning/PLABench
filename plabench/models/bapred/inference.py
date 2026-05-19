
import sys
import os
import glob
import logging
import argparse
import pandas as pd
import time
import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def run_inference_loop(input_dir, output_file, inference_func, model_path, device="cuda", filter_file=None):
    """
    Iterate over input directory and run inference for each target using provided inference_func.
    Assumes CWD is already set to BAPred root if necessary for inference_func.
    """
    # 2. Scan Targets
    # Structure: input_dir/Lxxxx_*/[protein.pdb, ligand.sdf]
    if not os.path.exists(input_dir):
        log.error(f"Input dir not found: {input_dir}")
        return

    subdirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
    # Optional Filtering
    if filter_file and os.path.exists(filter_file):
        try:
            if filter_file.endswith('.dat'):
                # CASF .dat format: space-delimited, first column is PDB ID, skip # comments
                valid_targets = set()
                with open(filter_file) as f:
                    for line in f:
                        if line.startswith('#') or not line.strip():
                            continue
                        parts = line.split()
                        if parts:
                            valid_targets.add(parts[0])
            else:
                filter_df = pd.read_csv(filter_file)
                id_col = filter_df.columns[0]
                if "Target ID" in filter_df.columns: id_col = "Target ID"
                valid_targets = set(filter_df[id_col].astype(str))
            original_count = len(subdirs)
            subdirs = [d for d in subdirs if d in valid_targets]
            log.info(f"Filtered targets: {len(subdirs)}/{original_count} using {filter_file}")
        except Exception as e:
            log.warning(f"Failed to read filter file {filter_file}: {e}")

    log.info(f"Found {len(subdirs)} potential targets in {input_dir}")
    
    results = []
    
    # Iterate
    for target in tqdm.tqdm(subdirs, desc="Inference"):
        target_dir = os.path.join(input_dir, target)
        # Protein: prefer _protein_clean.pdb > protein.pdb > protein_aligned.pdb > *_protein.pdb
        prot_path = None
        for candidate in [
            os.path.join(target_dir, "protein_clean.pdb"),
            os.path.join(target_dir, "protein.pdb"),
            os.path.join(target_dir, "protein_aligned.pdb"),
        ]:
            if os.path.exists(candidate):
                prot_path = candidate
                break
        if not prot_path:
            # CASF naming: {pdb_id}_protein_clean.pdb > {pdb_id}_protein.pdb
            matches = glob.glob(os.path.join(target_dir, "*_protein_clean.pdb"))
            if matches:
                prot_path = matches[0]
            else:
                matches = glob.glob(os.path.join(target_dir, "*_protein.pdb"))
                if matches:
                    prot_path = matches[0]

        # Ligand: prefer _ligand_clean.sdf > ligand.sdf > _ligand.sdf > any .sdf
        lig_candidates = []
        for f in sorted(glob.glob(os.path.join(target_dir, "*_ligand_clean.sdf"))):
            lig_candidates.append(f)
        exact = os.path.join(target_dir, "ligand.sdf")
        if os.path.exists(exact):
            lig_candidates.append(exact)
        for f in sorted(glob.glob(os.path.join(target_dir, "*_ligand.sdf"))):
            if "_clean" not in os.path.basename(f) and f not in lig_candidates:
                lig_candidates.append(f)
        for f in sorted(glob.glob(os.path.join(target_dir, "*.sdf"))):
            if f not in lig_candidates:
                lig_candidates.append(f)

        if not prot_path or not lig_candidates:
            continue

        temp_out = f"/tmp/bapred_{os.getpid()}_{target}.csv"

        # Try each ligand candidate until one succeeds
        success = False
        for lig_path in lig_candidates:
            try:
                inference_func(
                    protein_pdb=prot_path,
                    ligand_file=lig_path,
                    output=temp_out,
                    batch_size=1,
                    model_path=model_path,
                    device=device
                )

                if os.path.exists(temp_out):
                    try:
                        df = pd.read_csv(temp_out, sep='\t')
                        if len(df.columns) < 2: df = pd.read_csv(temp_out)
                    except:
                        df = pd.read_csv(temp_out)

                    if 'pKd' in df.columns:
                        val = df['pKd'].iloc[0]
                        if pd.notna(val):
                            results.append({"name": target, "prediction": val})
                            success = True

                    os.remove(temp_out)

                if success:
                    break
            except Exception as e:
                log.debug(f"Failed {target} with {os.path.basename(lig_path)}: {e}")
                if os.path.exists(temp_out):
                    os.remove(temp_out)

        if not success:
            log.warning(f"All ligand candidates failed for {target}")

    # Save Final
    if results:
        res_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        res_df[['prediction', 'name']].to_csv(output_file, index=False, header=False)
        log.info(f"Saved {len(results)} predictions to {output_file}")
    else:
        log.warning("No predictions generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--bapred_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--filter_file", default=None, help="Optional CSV to filter targets (Target ID column)")
    
    args = parser.parse_args()
    
    # Resolve absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_file = os.path.abspath(args.output_file)
    model_path = os.path.abspath(args.model_path)
    bapred_root = os.path.abspath(args.bapred_root)
    filter_file = os.path.abspath(args.filter_file) if args.filter_file else None

    # Switch to BAPred root
    if not os.path.exists(bapred_root):
        log.error(f"BAPred root not found: {bapred_root}")
        sys.exit(1)
        
    print(f"DEBUG: Switching CWD to {bapred_root}")
    os.chdir(bapred_root)
    if bapred_root not in sys.path:
        sys.path.insert(0, bapred_root)
        
    print("DEBUG: Importing bapred.inference...")
    from bapred.inference import inference
    print("DEBUG: Import successful")
    
    run_inference_loop(
        input_dir,
        output_file,
        inference,
        model_path,
        args.device,
        filter_file
    )
