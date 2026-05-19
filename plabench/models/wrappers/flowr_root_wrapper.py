import csv
import glob
import logging
import os
import statistics
import subprocess
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class FlowrRootWrapper:
    """Wrapper for FLOWR.root affinity prediction.

    Calls `python -m flowr.predict.predict_from_pdb` per target per seed,
    averages the predicted pKd over seeds (high-quality ensemble per
    `forks/flowr_root/scripts/predict_aff.sl`), and writes a standard
    `predictions.csv` (prediction, target_id).
    """

    def __init__(self, config: DictConfig):
        self.cfg = config
        self.output_dir = self.cfg.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.flowr_env = self.cfg.flowr_root_env
        self.flowr_dir = os.path.abspath(self.cfg.flowr_root_dir)
        self.ckpt_path = os.path.abspath(self.cfg.ckpt_path)

        self.arch = self.cfg.arch
        self.pocket_type = self.cfg.pocket_type
        self.pocket_noise = self.cfg.pocket_noise
        self.cut_pocket = bool(self.cfg.cut_pocket)
        self.pocket_cutoff = float(self.cfg.pocket_cutoff)
        self.batch_cost = int(self.cfg.batch_cost)
        self.coord_noise_scale = float(self.cfg.coord_noise_scale)
        self.seeds = list(self.cfg.seeds)
        self.cuda_device_index = int(getattr(self.cfg, "cuda_device_index", 0))
        self.per_seed_timeout = int(getattr(self.cfg, "per_seed_timeout", 600))
        # Per-ligand aggregation over the four affinity heads (paper §5.4 / Fig 5).
        # Default median over (pic50, pki, pkd, pec50); configurable via affinity_keys.
        keys = getattr(self.cfg, "affinity_keys", ["pic50", "pki", "pkd", "pec50"])
        self.affinity_keys = [str(k).lower() for k in keys]

        # Optional prep flags exposed by predict_from_pdb.py — disabled by default
        # to match `predict_aff.sl` template. Enable per ablation experiment.
        self.flag_compute_interactions = bool(getattr(self.cfg, "compute_interactions", False))
        self.flag_fixed_interactions = bool(getattr(self.cfg, "fixed_interactions", False))
        self.flag_add_hs = bool(getattr(self.cfg, "add_hs", False))
        self.flag_add_hs_and_optimize = bool(getattr(self.cfg, "add_hs_and_optimize", False))
        self.flag_kekulize = bool(getattr(self.cfg, "kekulize", False))
        self.flag_protonate_pocket = bool(getattr(self.cfg, "protonate_pocket", False))
        self.flag_use_pdbfixer = bool(getattr(self.cfg, "use_pdbfixer", False))
        self.flag_rotation_alignment = bool(getattr(self.cfg, "rotation_alignment", False))
        # CSV-batch mode: load model once, iterate manifest CSV in a single
        # Python process. Saves the per-(target,seed) ckpt-load overhead vs.
        # spawning N subprocesses. Default off; enable for large datasets
        # like chembl35_full (7091 compounds × 5 seeds).
        self.csv_batch_mode = bool(getattr(self.cfg, "csv_batch_mode", False))
        self.flag_permutation_alignment = bool(getattr(self.cfg, "permutation_alignment", False))
        # Sandbox suffix lets ablation runs keep separate cached SDFs.
        self.sandbox_suffix = str(getattr(self.cfg, "sandbox_suffix", "")).strip()

    def _parse_pdb_ids(self, gt_path):
        pdb_ids = []
        with open(gt_path) as f:
            if gt_path.endswith(".dat"):
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split()
                    if parts:
                        pdb_ids.append(parts[0])
            else:
                for row in csv.DictReader(f):
                    keys = list(row.keys())
                    col = (
                        row.get("Target ID")
                        or row.get("﻿Target ID")
                        or row.get("PDBID")
                        or row.get(keys[0])
                    )
                    if col:
                        pdb_ids.append(col.strip())
        return pdb_ids

    def _find_file(self, base_dir, pdb_id, pattern):
        pat = pattern.replace("{pdb_id}", pdb_id)
        candidate = os.path.join(base_dir, pat)
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.join(base_dir, pdb_id, pat)
        if os.path.exists(candidate):
            return candidate
        if "*" in pat:
            matches = glob.glob(os.path.join(base_dir, pat))
            if matches:
                return matches[0]
        return None

    def _parse_affinity_from_sdf(self, sdf_path):
        """Read predicted affinity props from gen_lig_with_aff.sdf and aggregate.

        Per paper §5.4 / Figure 5, the recommended per-ligand score is the
        median over the four affinity heads (pic50, pki, pkd, pec50).
        NaN heads are dropped; if all four are NaN/missing, returns None.
        """
        try:
            from rdkit import Chem
            import math

            suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
            for mol in suppl:
                if mol is None:
                    continue
                vals = []
                for key in self.affinity_keys:
                    if not mol.HasProp(key):
                        continue
                    try:
                        v = float(mol.GetProp(key))
                    except (ValueError, TypeError):
                        continue
                    if math.isnan(v) or math.isinf(v):
                        continue
                    vals.append(v)
                if not vals:
                    return None
                return statistics.median(vals)
            return None
        except Exception as e:
            log.warning(f"Failed to parse {sdf_path}: {e}")
            return None

    def _run_one_seed(self, pdb_file, ligand_file, save_dir, seed, dataset_tag):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device_index)
        env["PYTHONPATH"] = self.flowr_dir + ":" + env.get("PYTHONPATH", "")

        cmd = [
            self.flowr_env, "-m", "flowr.predict.predict_from_pdb",
            "--pdb_file", os.path.abspath(pdb_file),
            "--ligand_file", os.path.abspath(ligand_file),
            "--dataset", dataset_tag,
            "--gpus", "1",
            "--seed", str(seed),
            "--batch_cost", str(self.batch_cost),
            "--arch", self.arch,
            "--pocket_type", self.pocket_type,
            "--pocket_noise", self.pocket_noise,
            "--pocket_cutoff", str(self.pocket_cutoff),
            "--ckpt_path", self.ckpt_path,
            "--save_dir", os.path.abspath(save_dir),
            "--coord_noise_scale", str(self.coord_noise_scale),
        ]
        if self.cut_pocket:
            cmd.append("--cut_pocket")
        # Optional prep flags (disabled by default to mirror predict_aff.sl)
        if self.flag_compute_interactions:
            cmd.append("--compute_interactions")
        if self.flag_fixed_interactions:
            cmd.append("--fixed_interactions")
        if self.flag_add_hs:
            cmd.append("--add_hs")
        if self.flag_add_hs_and_optimize:
            cmd.append("--add_hs_and_optimize")
        if self.flag_kekulize:
            cmd.append("--kekulize")
        if self.flag_protonate_pocket:
            cmd.append("--protonate_pocket")
        if self.flag_use_pdbfixer:
            cmd.append("--use_pdbfixer")
        if self.flag_rotation_alignment:
            cmd.append("--rotation_alignment")
        if self.flag_permutation_alignment:
            cmd.append("--permutation_alignment")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=self.flowr_dir, env=env, timeout=self.per_seed_timeout,
            )
            if result.returncode != 0:
                log.error(f"flowr_root stderr (seed={seed}): {result.stderr[-500:]}")
                return False
            return True
        except subprocess.TimeoutExpired:
            log.error(f"flowr_root timeout (seed={seed})")
            return False
        except Exception as e:
            log.error(f"flowr_root failed (seed={seed}): {e}")
            return False

    def run(self, input_dir):
        log.info(f"FlowrRoot Wrapper: input_dir={input_dir}")
        struct_dir = os.path.abspath(input_dir)
        gt_path = os.path.abspath(self.cfg.ground_truth_path)
        boltz2_mode = bool(getattr(self.cfg, "boltz2_mode", False))

        pdb_ids = self._parse_pdb_ids(gt_path)
        log.info(f"Found {len(pdb_ids)} PDB IDs in ground truth")

        dataset_name = self.cfg.name
        suffix = f"_{self.sandbox_suffix}" if self.sandbox_suffix else ""
        sandbox = os.path.join("tmp", f"flowr_root_{dataset_name}{suffix}_sandbox")
        os.makedirs(sandbox, exist_ok=True)

        prot_pattern = getattr(self.cfg, "prot_pattern", "{pdb_id}_protein.pdb")
        lig_pattern = getattr(self.cfg, "lig_pattern", "{pdb_id}_ligand.sdf")
        exp_sdf_dir = (
            os.path.abspath(self.cfg.experimental_sdf_dir)
            if hasattr(self.cfg, "experimental_sdf_dir") else None
        )

        predictions = []
        failures = []

        # Resolve all targets first (used for both per-target and CSV-batch).
        resolved = []  # list of (pdb_id, protein_pdb, ligand_sdf)
        for pdb_id in sorted(pdb_ids):
            protein_pdb = self._find_file(struct_dir, pdb_id, prot_pattern)
            if not protein_pdb:
                for fallback in ["protein.pdb", f"{pdb_id}_protein_clean.pdb",
                                 f"{pdb_id}_protein.pdb"]:
                    protein_pdb = self._find_file(struct_dir, pdb_id, fallback)
                    if protein_pdb:
                        break
            if not protein_pdb:
                log.warning(f"SKIP {pdb_id}: no protein PDB")
                continue

            ligand_sdf = None
            if boltz2_mode:
                sdf_pat = lig_pattern.replace("{pdb_id}", pdb_id)
                if not sdf_pat.endswith(".sdf"):
                    sdf_pat = sdf_pat.rsplit(".", 1)[0] + ".sdf"
                ligand_sdf = self._find_file(struct_dir, pdb_id, sdf_pat)
            else:
                if exp_sdf_dir:
                    for suff in [f"{pdb_id}_ligand_clean.sdf",
                                 f"{pdb_id}_ligand.sdf"]:
                        candidate = os.path.join(exp_sdf_dir, pdb_id, suff)
                        if os.path.exists(candidate):
                            ligand_sdf = candidate
                            break
            if not ligand_sdf:
                for fallback in ["ligand.sdf", f"{pdb_id}_ligand_clean.sdf",
                                 f"{pdb_id}_ligand.sdf"]:
                    ligand_sdf = self._find_file(struct_dir, pdb_id, fallback)
                    if ligand_sdf:
                        break
            if not ligand_sdf:
                log.warning(f"SKIP {pdb_id}: no ligand SDF")
                continue
            resolved.append((pdb_id, protein_pdb, ligand_sdf))

        # CSV-batch mode: 1 Python subprocess loads model once, iterates all.
        if self.csv_batch_mode:
            manifest_csv = os.path.join(sandbox, "manifest.csv")
            with open(manifest_csv, "w") as f:
                f.write("id,pdb_path,sdf_path\n")
                for pdb_id, protein_pdb, ligand_sdf in resolved:
                    f.write(
                        f"{pdb_id},{os.path.abspath(protein_pdb)},"
                        f"{os.path.abspath(ligand_sdf)}\n"
                    )
            log.info(f"CSV-batch mode: manifest with {len(resolved)} rows -> {manifest_csv}")

            output_file = os.path.join(self.output_dir, "predictions.csv")
            batch_script = os.path.join(
                "plabench", "models", "flowr_root", "batch_inference.py"
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device_index)
            env["PYTHONPATH"] = self.flowr_dir + ":" + env.get("PYTHONPATH", "")

            cmd = [
                self.flowr_env, batch_script,
                "--manifest_csv", os.path.abspath(manifest_csv),
                "--ckpt_path", self.ckpt_path,
                "--sandbox_dir", os.path.abspath(sandbox),
                "--predictions_csv", os.path.abspath(output_file),
                "--seeds", ",".join(str(s) for s in self.seeds),
                "--coord_noise_scale", str(self.coord_noise_scale),
                "--pocket_cutoff", str(self.pocket_cutoff),
                "--batch_cost", str(self.batch_cost),
                "--arch", self.arch,
                "--pocket_type", self.pocket_type,
                "--pocket_noise", self.pocket_noise,
            ]
            if self.cut_pocket:
                cmd.append("--cut_pocket")
            log.info(f"Launching batch_inference.py: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env)  # inherits stdout/stderr
            if result.returncode != 0:
                log.error(f"batch_inference.py exited {result.returncode}")
            return

        for i, pdb_id in enumerate(sorted(pdb_ids), 1):
            # Find protein PDB
            protein_pdb = self._find_file(struct_dir, pdb_id, prot_pattern)
            if not protein_pdb:
                for fallback in ["protein.pdb", f"{pdb_id}_protein_clean.pdb",
                                 f"{pdb_id}_protein.pdb"]:
                    protein_pdb = self._find_file(struct_dir, pdb_id, fallback)
                    if protein_pdb:
                        break
            if not protein_pdb:
                log.warning(f"SKIP {pdb_id}: no protein PDB")
                continue

            # Find ligand SDF
            ligand_sdf = None
            if boltz2_mode:
                sdf_pat = lig_pattern.replace("{pdb_id}", pdb_id)
                if not sdf_pat.endswith(".sdf"):
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

            if not ligand_sdf:
                for fallback in ["ligand.sdf", f"{pdb_id}_ligand_clean.sdf",
                                 f"{pdb_id}_ligand.sdf"]:
                    ligand_sdf = self._find_file(struct_dir, pdb_id, fallback)
                    if ligand_sdf:
                        break
            if not ligand_sdf:
                log.warning(f"SKIP {pdb_id}: no ligand SDF")
                continue

            target_dir = os.path.join(sandbox, pdb_id)
            os.makedirs(target_dir, exist_ok=True)

            # Run all seeds and aggregate
            seed_preds = []
            for seed in self.seeds:
                seed_dir = os.path.join(target_dir, f"seed_{seed}")
                sdf_path = os.path.join(seed_dir, "gen_lig_with_aff.sdf")

                cached = self._parse_affinity_from_sdf(sdf_path)
                if cached is not None:
                    seed_preds.append(cached)
                    log.info(f"[{i}/{len(pdb_ids)}] {pdb_id} seed={seed}: cached 4-head-median={cached:.3f}")
                    continue

                os.makedirs(seed_dir, exist_ok=True)
                ok = self._run_one_seed(
                    protein_pdb, ligand_sdf, seed_dir, seed,
                    dataset_tag=dataset_name,
                )
                if not ok:
                    log.warning(f"  {pdb_id} seed={seed}: run failed")
                    continue
                pred = self._parse_affinity_from_sdf(sdf_path)
                if pred is not None:
                    seed_preds.append(pred)
                    log.info(f"[{i}/{len(pdb_ids)}] {pdb_id} seed={seed}: 4-head-median={pred:.3f}")
                else:
                    log.warning(f"  {pdb_id} seed={seed}: no valid affinity heads in output SDF")

            if seed_preds:
                final = statistics.median(seed_preds)
                predictions.append((final, pdb_id))
                log.info(f"  {pdb_id}: cross-seed-median={final:.3f} from {len(seed_preds)}/{len(self.seeds)} seeds")
            else:
                failures.append((pdb_id, "all_seeds_failed"))
                log.warning(f"  {pdb_id}: all seeds failed")

        # Write predictions.csv (no header, prediction,name)
        output_file = os.path.join(self.output_dir, "predictions.csv")
        with open(output_file, "w") as f:
            for pred, name in predictions:
                f.write(f"{pred},{name}\n")
        log.info(f"Wrote {len(predictions)} predictions to {output_file}")

        if failures:
            fail_file = os.path.join(self.output_dir, "failures.csv")
            with open(fail_file, "w") as f:
                f.write("TargetID,ErrorType\n")
                for name, err in failures:
                    f.write(f"{name},{err}\n")
            log.info(f"{len(failures)} failures recorded")
