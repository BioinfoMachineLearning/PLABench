import os
import ast
import csv
import glob
import logging
import subprocess
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class FlowDockWrapper:
    def __init__(self, config: DictConfig):
        self.cfg = config
        self.output_dir = self.cfg.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.flowdock_env = self.cfg.flowdock_env
        self.flowdock_root = os.path.abspath(self.cfg.models_source_dir)
        self.model_checkpoint = self.cfg.model_checkpoint
        self.flowdock_exec_dir = self.cfg.flowdock_exec_dir
        self.n_samples = getattr(self.cfg, 'n_samples', 1)
        self.chunk_size = getattr(self.cfg, 'chunk_size', 1)
        self.num_steps = getattr(self.cfg, 'num_steps', 40)
        self.sampler = getattr(self.cfg, 'sampler', 'VDODE')
        self.cuda_device_index = getattr(self.cfg, 'cuda_device_index', 0)
        # CSV-batch mode: load model once, iterate all targets in a single
        # subprocess (sample.py csv_path mode). Saves ~10s/target ckpt-load
        # overhead vs. spawning N subprocesses. Default on for large datasets.
        self.csv_batch_mode = bool(getattr(self.cfg, 'csv_batch_mode', False))

    def _parse_pdb_ids(self, gt_path):
        """Parse PDB IDs from ground truth file."""
        pdb_ids = []
        with open(gt_path) as f:
            if gt_path.endswith('.dat'):
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.split()
                    if parts:
                        pdb_ids.append(parts[0])
            else:
                for row in csv.DictReader(f):
                    col = row.get('PDBID') or row.get(list(row.keys())[0])
                    pdb_ids.append(col.strip())
        return pdb_ids

    def _find_file(self, base_dir, pdb_id, pattern):
        """Find a file matching pattern in base_dir (flat, nested, glob)."""
        pat = pattern.replace("{pdb_id}", pdb_id)
        # Flat
        candidate = os.path.join(base_dir, pat)
        if os.path.exists(candidate):
            return candidate
        # Nested
        candidate = os.path.join(base_dir, pdb_id, pat)
        if os.path.exists(candidate):
            return candidate
        # Glob (for prefixed naming like csar51_XXXX_)
        if '*' in pat:
            matches = glob.glob(os.path.join(base_dir, pat))
            if matches:
                return matches[0]
        return None

    def _run_flowdock(self, pdb_id, protein_pdb, ligand_sdf, out_path):
        """Run FlowDock auxiliary estimation for one complex via sample.py."""
        # Resolve paths to absolute
        protein_pdb = os.path.abspath(protein_pdb)
        ligand_sdf = os.path.abspath(ligand_sdf)
        out_path = os.path.abspath(out_path)

        # sample.py is in forks/FlowDock/src/sample.py relative to flowdock_root
        sample_script = os.path.join(
            self.flowdock_root, self.flowdock_exec_dir, "src", "sample.py"
        )
        ckpt_path = os.path.join(
            self.flowdock_root, self.model_checkpoint
        )

        cmd = [
            self.flowdock_env, sample_script,
            "sampling_task=batched_structure_sampling",
            f"sample_id={pdb_id}_rank1",
            f"input_ligand={ligand_sdf}",
            f"input_receptor={protein_pdb}",
            f"input_template={protein_pdb}",
            f"ckpt_path={ckpt_path}",
            f"out_path={out_path}",
            f"n_samples={self.n_samples}",
            f"chunk_size={self.chunk_size}",
            f"num_steps={self.num_steps}",
            f"sampler={self.sampler}",
            "sampler_eta=1.0",
            "start_time=1.0",
            "max_chain_encoding_k=-1",
            "exact_prior=false",
            "discard_ligand=false",
            "discard_sdf_coords=false",
            "detect_covalent=false",
            "use_template=true",
            "separate_pdb=true",
            "rank_outputs_by_confidence=true",
            "plddt_ranking_type=ligand",
            "visualize_sample_trajectories=false",
            "auxiliary_estimation_only=true",
            f"trainer=gpu",
            f"trainer.devices=[{self.cuda_device_index}]",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=self.flowdock_root, timeout=600,
            )
            if result.returncode != 0:
                log.error(f"FlowDock stderr for {pdb_id}: {result.stderr[-500:]}")
            return result.returncode == 0
        except Exception as e:
            log.error(f"FlowDock failed for {pdb_id}: {e}")
            return False

    def _run_flowdock_batch(self, manifest_csv, sandbox_root):
        """CSV-batch mode: ONE sample.py subprocess processes ALL rows in
        manifest_csv. Model loads once. Per-row outputs land in
        sandbox_root/{sample_id}/auxiliary_estimation.csv (requires the
        per-sample-out fork patch in flowdock_fm_module.py).

        manifest_csv columns: id, input_receptor, input_ligand, input_template
        """
        sample_script = os.path.join(
            self.flowdock_root, self.flowdock_exec_dir, "src", "sample.py"
        )
        ckpt_path = os.path.join(self.flowdock_root, self.model_checkpoint)

        cmd = [
            self.flowdock_env, sample_script,
            "sampling_task=batched_structure_sampling",
            f"csv_path={os.path.abspath(manifest_csv)}",
            f"ckpt_path={ckpt_path}",
            f"out_path={os.path.abspath(sandbox_root)}",
            f"n_samples={self.n_samples}",
            f"chunk_size={self.chunk_size}",
            f"num_steps={self.num_steps}",
            f"sampler={self.sampler}",
            "sampler_eta=1.0",
            "start_time=1.0",
            "max_chain_encoding_k=-1",
            "exact_prior=false",
            "discard_ligand=false",
            "discard_sdf_coords=false",
            "detect_covalent=false",
            "use_template=true",
            "separate_pdb=true",
            "rank_outputs_by_confidence=true",
            "plddt_ranking_type=ligand",
            "visualize_sample_trajectories=false",
            "auxiliary_estimation_only=true",
            "trainer=gpu",
            f"trainer.devices=[{self.cuda_device_index}]",
        ]
        log.info(f"FlowDock batch cmd: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=False, text=True,
                cwd=self.flowdock_root,
            )
            return result.returncode == 0
        except Exception as e:
            log.error(f"FlowDock batch failed: {e}")
            return False

    def _get_prediction(self, out_path):
        """Parse predicted pKd from auxiliary_estimation.csv."""
        # sample.py writes auxiliary_estimation.csv in out_path
        aux_csv = os.path.join(out_path, "auxiliary_estimation.csv")
        if not os.path.exists(aux_csv):
            # Also check subdirectories (sample.py may nest by sample_id)
            for root, dirs, files in os.walk(out_path):
                if "auxiliary_estimation.csv" in files:
                    aux_csv = os.path.join(root, "auxiliary_estimation.csv")
                    break
        if not os.path.exists(aux_csv):
            return None
        try:
            import pandas as pd
            df = pd.read_csv(aux_csv)
            if 'affinity_ligs' not in df.columns or len(df) == 0:
                return None
            val = df['affinity_ligs'].values[0]
            if isinstance(val, str):
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return float(parsed[0])
            return float(val)
        except Exception as e:
            log.warning(f"Failed to parse prediction: {e}")
            return None

    def run(self, input_dir):
        """Run FlowDock auxiliary estimation on CASF/CSAR datasets."""
        log.info(f"FlowDock Wrapper: input_dir={input_dir}")

        struct_dir = os.path.abspath(input_dir)
        gt_path = os.path.abspath(self.cfg.ground_truth_path)
        boltz2_mode = getattr(self.cfg, 'boltz2_mode', False)

        pdb_ids = self._parse_pdb_ids(gt_path)
        log.info(f"Found {len(pdb_ids)} PDB IDs in ground truth")

        # Sandbox for auxiliary input dirs
        dataset_name = self.cfg.name
        sandbox = os.path.join("tmp", f"flowdock_{dataset_name}_sandbox")
        os.makedirs(sandbox, exist_ok=True)

        # Protein/ligand patterns
        prot_pattern = getattr(self.cfg, 'prot_pattern', '{pdb_id}_protein.pdb')
        lig_pattern = getattr(self.cfg, 'lig_pattern', '{pdb_id}_ligand.mol2')

        # Determine SDF source
        exp_sdf_dir = os.path.abspath(
            self.cfg.experimental_sdf_dir
        ) if hasattr(self.cfg, 'experimental_sdf_dir') else None

        predictions = []
        failures = []

        # Resolve protein/ligand paths for every pdb_id. Used by both per-target
        # and CSV-batch modes; we collect first so we can either skip per-target
        # (cached) or build a manifest CSV for batch mode.
        resolved = []  # list of (pdb_id, protein_pdb, ligand_sdf)
        for pdb_id in sorted(pdb_ids):
            protein_pdb = self._find_file(struct_dir, pdb_id, prot_pattern)
            if not protein_pdb or not os.path.exists(protein_pdb):
                for fallback in ["protein.pdb", "protein_aligned.pdb",
                                 f"{pdb_id}_protein_clean.pdb",
                                 f"{pdb_id}_protein.pdb"]:
                    protein_pdb = self._find_file(struct_dir, pdb_id, fallback)
                    if protein_pdb and os.path.exists(protein_pdb):
                        break
            if not protein_pdb or not os.path.exists(protein_pdb):
                log.warning(f"SKIP {pdb_id}: no protein PDB")
                continue

            ligand_sdf = None
            if boltz2_mode:
                sdf_pat = lig_pattern.replace("{pdb_id}", pdb_id)
                sdf_pat = sdf_pat.rsplit(".", 1)[0] + ".sdf"
                ligand_sdf = self._find_file(struct_dir, pdb_id, sdf_pat)
            else:
                if exp_sdf_dir:
                    for suffix in [f"{pdb_id}_ligand_clean.sdf",
                                   f"{pdb_id}_ligand.sdf"]:
                        candidate = os.path.join(exp_sdf_dir, pdb_id, suffix)
                        if os.path.exists(candidate):
                            ligand_sdf = candidate
                            break
            if not ligand_sdf or not os.path.exists(ligand_sdf):
                for fallback in ["ligand.sdf", f"{pdb_id}_ligand_clean.sdf",
                                 f"{pdb_id}_ligand.sdf"]:
                    ligand_sdf = self._find_file(struct_dir, pdb_id, fallback)
                    if ligand_sdf and os.path.exists(ligand_sdf):
                        break
            if not ligand_sdf or not os.path.exists(ligand_sdf):
                log.warning(f"SKIP {pdb_id}: no ligand SDF")
                continue
            resolved.append((pdb_id, protein_pdb, ligand_sdf))

        # CSV-batch mode: collect cached, build manifest of remaining, run once.
        if self.csv_batch_mode:
            cached, todo = [], []
            for pdb_id, protein_pdb, ligand_sdf in resolved:
                out_path = os.path.join(sandbox, pdb_id)
                pred = self._get_prediction(out_path)
                if pred is not None:
                    cached.append((pred, pdb_id))
                else:
                    todo.append((pdb_id, protein_pdb, ligand_sdf))

            log.info(f"CSV-batch mode: {len(cached)} cached, {len(todo)} to run")
            predictions.extend(cached)

            if todo:
                manifest_csv = os.path.join(sandbox, "manifest.csv")
                with open(manifest_csv, "w") as f:
                    f.write("id,input_receptor,input_ligand,input_template\n")
                    for pdb_id, protein_pdb, ligand_sdf in todo:
                        prot_abs = os.path.abspath(protein_pdb)
                        lig_abs = os.path.abspath(ligand_sdf)
                        f.write(f"{pdb_id},{prot_abs},{lig_abs},{prot_abs}\n")
                log.info(f"Wrote manifest with {len(todo)} rows: {manifest_csv}")
                ok = self._run_flowdock_batch(manifest_csv, sandbox)
                if not ok:
                    log.warning("Batch run returned non-zero; collecting whatever landed")

                for pdb_id, _, _ in todo:
                    out_path = os.path.join(sandbox, pdb_id)
                    pred = self._get_prediction(out_path)
                    if pred is not None:
                        predictions.append((pred, pdb_id))
                    else:
                        failures.append((pdb_id, "no_prediction"))

            output_file = os.path.join(self.output_dir, "predictions.csv")
            with open(output_file, 'w') as f:
                for pred, name in predictions:
                    f.write(f"{pred},{name}\n")
            log.info(f"Wrote {len(predictions)} predictions to {output_file}")
            if failures:
                fail_file = os.path.join(self.output_dir, "failures.csv")
                with open(fail_file, 'w') as f:
                    f.write("TargetID,ErrorType\n")
                    for name, err in failures:
                        f.write(f"{name},{err}\n")
                log.info(f"{len(failures)} failures recorded")
            return

        # Original per-target subprocess loop (default)
        for i, (pdb_id, protein_pdb, ligand_sdf) in enumerate(resolved, 1):
            # protein_pdb / ligand_sdf already resolved above

            # Check if prediction already exists
            out_path = os.path.join(sandbox, pdb_id)
            pred = self._get_prediction(out_path)
            if pred is not None:
                log.info(f"[{i}/{len(resolved)}] {pdb_id}: cached pred={pred:.3f}")
                predictions.append((pred, pdb_id))
                continue

            log.info(f"[{i}/{len(resolved)}] Running FlowDock for {pdb_id}...")
            os.makedirs(out_path, exist_ok=True)
            success = self._run_flowdock(pdb_id, protein_pdb, ligand_sdf, out_path)

            if success:
                pred = self._get_prediction(out_path)
                if pred is not None:
                    predictions.append((pred, pdb_id))
                    log.info(f"  {pdb_id}: pred={pred:.3f}")
                else:
                    failures.append((pdb_id, "no_prediction"))
                    log.warning(f"  {pdb_id}: no prediction output")
            else:
                failures.append((pdb_id, "flowdock_failed"))
                log.warning(f"  {pdb_id}: FlowDock execution failed")

        # Write predictions.csv
        output_file = os.path.join(self.output_dir, "predictions.csv")
        with open(output_file, 'w') as f:
            for pred, name in predictions:
                f.write(f"{pred},{name}\n")
        log.info(f"Wrote {len(predictions)} predictions to {output_file}")

        # Write failures.csv if any
        if failures:
            fail_file = os.path.join(self.output_dir, "failures.csv")
            with open(fail_file, 'w') as f:
                f.write("TargetID,ErrorType\n")
                for name, err in failures:
                    f.write(f"{name},{err}\n")
            log.info(f"{len(failures)} failures recorded")
