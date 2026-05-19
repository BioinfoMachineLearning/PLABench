
import os
import sys
import csv
import glob
import logging
import subprocess
from omegaconf import DictConfig

log = logging.getLogger(__name__)

class BapredWrapper:
    def __init__(self, config: DictConfig):
        self.cfg = config
        self.output_dir = self.cfg.output_dir

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def _prepare_boltz2_sandbox(self, boltz2_dir, gt_path, prot_pattern, lig_pattern):
        """Create sandbox with subdirectory structure for BAPred inference.
        Boltz2 data is flat; BAPred needs {pdb_id}/ subdirs with protein + ligand.
        Sandbox is placed in project tmp/ for reuse across runs.
        """
        dataset_name = self.cfg.name
        sandbox = os.path.join("tmp", f"bapred_{dataset_name}_sandbox")
        os.makedirs(sandbox, exist_ok=True)

        # Read PDB IDs from ground truth
        pdbs = []
        with open(gt_path) as f:
            if gt_path.endswith('.dat'):
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.split()
                    if parts:
                        pdbs.append(parts[0])
            else:
                for row in csv.DictReader(f):
                    col = row.get('PDBID') or row.get(list(row.keys())[0])
                    pdbs.append(col.strip())

        ok, skip = 0, 0
        for pdb_id in sorted(pdbs):
            target_dir = os.path.join(sandbox, pdb_id)
            prot_link = os.path.join(target_dir, f"{pdb_id}_protein.pdb")
            lig_sdf = os.path.join(target_dir, f"{pdb_id}_ligand.sdf")

            # Skip if already prepared
            if os.path.exists(prot_link) and os.path.exists(lig_sdf):
                ok += 1
                continue

            # Find Boltz2 protein
            pat = prot_pattern.replace("{pdb_id}", pdb_id)
            matches = glob.glob(os.path.join(boltz2_dir, pat))
            if not matches:
                log.warning(f"SKIP {pdb_id}: no Boltz2 protein")
                skip += 1
                continue

            # Find Boltz2 ligand mol2
            lig_pat = lig_pattern.replace("{pdb_id}", pdb_id)
            lig_matches = glob.glob(os.path.join(boltz2_dir, lig_pat))
            if not lig_matches:
                log.warning(f"SKIP {pdb_id}: no Boltz2 ligand mol2")
                skip += 1
                continue

            os.makedirs(target_dir, exist_ok=True)

            # Symlink protein
            if os.path.exists(prot_link):
                os.remove(prot_link)
            os.symlink(os.path.abspath(matches[0]), prot_link)

            # Use pre-generated SDF if available, otherwise convert mol2 -> SDF
            # Derive SDF path from mol2 path (handles prefixed naming like csar51_XXXX_)
            pre_sdf = lig_matches[0].rsplit(".", 1)[0] + ".sdf"
            if os.path.exists(lig_sdf):
                os.remove(lig_sdf)
            if os.path.exists(pre_sdf) and os.path.getsize(pre_sdf) > 0:
                os.symlink(os.path.abspath(pre_sdf), lig_sdf)
            else:
                try:
                    subprocess.run(
                        ["obabel", lig_matches[0], "-O", lig_sdf],
                        capture_output=True, timeout=30
                    )
                except Exception as e:
                    log.warning(f"SKIP {pdb_id}: obabel failed: {e}")
                    skip += 1
                    continue

                if not os.path.exists(lig_sdf) or os.path.getsize(lig_sdf) == 0:
                    log.warning(f"SKIP {pdb_id}: obabel produced empty SDF")
                    skip += 1
                    continue

            ok += 1

        log.info(f"Boltz2 sandbox: {ok} ready, {skip} skipped -> {sandbox}")
        return os.path.abspath(sandbox)

    def run(self, input_dir):
        """
        Run BAPred inference.
        input_dir: Path to directory containing target folders with protein.pdb and ligand.sdf
        """
        log.info(f"Initialized BAPred Wrapper for {input_dir}")
        log.info(f"Output Directory: {self.output_dir}")

        # Boltz2 mode: create sandbox from flat Boltz2 data
        boltz2_mode = getattr(self.cfg, 'boltz2_mode', False)
        if boltz2_mode:
            gt_path = self.cfg.ground_truth_path
            prot_pattern = getattr(self.cfg, 'prot_pattern', '{pdb_id}_protein.pdb')
            lig_pattern = getattr(self.cfg, 'lig_pattern', '{pdb_id}_ligand.mol2')
            input_dir = self._prepare_boltz2_sandbox(
                input_dir, gt_path, prot_pattern, lig_pattern
            )

        script_path = "plabench/models/bapred/inference.py"

        if not os.path.exists(script_path):
            log.error(f"Batch inference script not found at {script_path}")
            raise FileNotFoundError(f"{script_path} missing")

        output_file = os.path.join(self.output_dir, "predictions.csv")
        bapred_root = self.cfg.models_source_dir
        model_weights = self.cfg.model_path
        if not os.path.isabs(model_weights):
            model_weights = os.path.abspath(model_weights)
        device = self.cfg.device

        cmd = [
            sys.executable, script_path,
            "--input_dir", input_dir,
            "--output_file", output_file,
            "--bapred_root", bapred_root,
            "--model_path", os.path.dirname(model_weights),
            "--device", device
        ]

        filter_file = getattr(self.cfg, 'labels_file', None) or getattr(self.cfg, 'ground_truth_path', None)
        if filter_file:
            cmd.extend(["--filter_file", filter_file])

        log.info(f"Executing: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            log.info("BAPred inference completed.")
        except subprocess.CalledProcessError as e:
            log.error(f"BAPred inference failed: {e}")
            raise e
