
import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path
from plabench.utils.structure_utils import merge_ligands_pdb, convert_pdb_to_mol2
from plabench.utils.data_utils import prepare_smiles_files

class HaipingWrapper:
    def __init__(self, config):
        self.config = config
        self.mode = config.mode
        self.model_path = os.path.abspath(config.model_path)
        self.aa_dict_path = os.path.abspath(config.aa_dict_path)
        self.output_dir = os.path.abspath(config.output_dir)
        self.device = config.get('device', 'cuda')

        os.makedirs(self.output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                            filename=os.path.join(self.output_dir, 'wrapper.log'),
                            format='%(asctime)s - %(levelname)s - %(message)s')

        self.base_path = Path(__file__).parent.parent / 'haiping'
        self.extract_script = self.base_path / 'extract_pocket.py'
        self.preprocess_script = self.base_path / 'preprocess.py'
        self.inference_script = self.base_path / 'inference.py'
        self.models_dir = config.get('models_source_dir', None)

    def run(self, input_data):
        logging.info(f"Starting HAIPING Wrapper in {self.mode} mode.")

        # 1. Sandbox Setup
        sandbox_dir = os.path.join(self.output_dir, 'sandbox')
        os.makedirs(sandbox_dir, exist_ok=True)

        if self.models_dir and os.path.exists(self.models_dir):
            target_models = os.path.join(sandbox_dir, 'models')
            if not os.path.exists(target_models):
                os.symlink(os.path.abspath(self.models_dir), target_models)

        # 2. Data Preparation (mode-specific input gathering + shared processing)
        if self.mode == 'stage2':
            self._prepare_stage2(input_data, sandbox_dir)
        elif self.mode == 'stage1':
            self._prepare_stage1(input_data, sandbox_dir)
        elif self.mode in ('casf2013_boltz2', 'casf2016_boltz2', 'casf2013_exp', 'casf2016_exp',
                           'csar_hiq51_exp', 'csar_hiq36_exp', 'csar_hiq51_boltz2', 'csar_hiq36_boltz2',
                           'cofold_stage1'):
            self._prepare_casf_csar(sandbox_dir)

        # 3. Preprocessing
        data_dir = sandbox_dir
        pocket_dir = os.path.join(sandbox_dir, 'pocket')
        processed_name = "test_data"
        processed_file = os.path.join(data_dir, 'processed', f"{processed_name}.pt")

        if os.path.exists(processed_file):
            logging.info(f"Preprocessing skipped. Output exists at {processed_file}")
        else:
            logging.info("Running Preprocessing...")
            cmd_prep = [
                sys.executable, str(self.preprocess_script),
                data_dir, pocket_dir, processed_name, self.aa_dict_path
            ]
            self._run_subprocess(cmd_prep, sandbox_dir)

            fail_csv = os.path.join(sandbox_dir, "preprocessing_failures.csv")
            if os.path.exists(fail_csv):
                 shutil.copy(fail_csv, os.path.join(self.output_dir, "failures.csv"))
                 logging.warning(f"Preprocessing failures detected. See {os.path.join(self.output_dir, 'failures.csv')}")

        # 4. Inference
        output_file = os.path.join(self.output_dir, "predictions.csv")

        if os.path.exists(output_file):
             logging.info(f"Inference skipped. Output exists at {output_file}")
        else:
            logging.info("Running Inference...")
            cmd_inf = [
                sys.executable, str(self.inference_script),
                processed_file, self.model_path, output_file, self.device
            ]
            self._run_subprocess(cmd_inf, sandbox_dir)

        logging.info("Pipeline completed.")

    # ----------------------------------------------------------------
    # Shared processing: pocket extraction + .smi writing
    # All modes should use this after gathering mode-specific inputs.
    # ----------------------------------------------------------------
    def _process_targets(self, targets, sandbox_dir):
        """
        Shared processing for all structure-based modes.

        Args:
            targets: list of dicts, each with:
                - target_id: str
                - protein_pdb: str (path)
                - ligand_for_pocket: str (path to ligand file for pocket extraction)
                - smiles: str or None (if provided, writes .smi file)
            sandbox_dir: str
        Returns:
            int: number of successfully processed targets
        """
        pocket_dir = os.path.join(sandbox_dir, 'pocket')
        os.makedirs(pocket_dir, exist_ok=True)

        success = 0
        for t in targets:
            tid = t['target_id']

            # Extract pocket
            pocket_out = os.path.join(pocket_dir, f"{tid}_poc.pdb")
            cmd_extract = [
                sys.executable, str(self.extract_script),
                t['protein_pdb'], t['ligand_for_pocket'],
                "-o", pocket_out
            ]
            try:
                subprocess.run(cmd_extract, check=True)
            except subprocess.CalledProcessError:
                logging.error(f"Pocket extraction failed for {tid}")
                continue

            # Write .smi file if SMILES provided
            if t.get('smiles'):
                smi_file = os.path.join(sandbox_dir, f"{tid}.smi")
                with open(smi_file, 'w') as f:
                    f.write(t['smiles'] + '\n')

            success += 1

        return success

    # ----------------------------------------------------------------
    # Mode-specific input gathering
    # ----------------------------------------------------------------
    def _prepare_stage2(self, input_dir, sandbox_dir):
        """Gather inputs for Stage 2 (CASP16 experimental structures), then shared processing."""
        logging.info("Preparing Stage 2 Data...")

        input_path = Path(input_dir)
        import fnmatch
        pattern = self.config.get('target_pattern', '*')

        targets = []
        for target_dir in input_path.glob("L*"):
            if not target_dir.is_dir(): continue
            target_id = target_dir.name
            if not fnmatch.fnmatch(target_id, pattern):
                continue

            protein_file = target_dir / "protein_aligned.pdb"
            if not protein_file.exists():
                logging.warning(f"Missing protein for {target_id}")
                continue

            ligand_files = sorted(list(target_dir.glob("ligand*.pdb")))
            if not ligand_files:
                logging.warning(f"Missing ligands for {target_id}")
                continue

            # Merge ligands if multiple
            merged_ligand = os.path.join(sandbox_dir, f"{target_id}_ligand.pdb")
            if len(ligand_files) > 1:
                logging.info(f"Merging {len(ligand_files)} ligands for {target_id}")
                merge_ligands_pdb([str(f) for f in ligand_files], merged_ligand)
            else:
                shutil.copy(ligand_files[0], merged_ligand)

            targets.append({
                'target_id': target_id,
                'protein_pdb': str(protein_file),
                'ligand_for_pocket': merged_ligand,
                'smiles': None,  # SMILES copied separately below
            })

        # Shared pocket extraction
        count = self._process_targets(targets, sandbox_dir)
        logging.info(f"Stage 2 pocket extraction: {count}/{len(targets)} targets")

        # Copy SMILES from source dir
        smiles_source = self.config.get('smiles_source_dir')
        if smiles_source and os.path.exists(smiles_source):
            logging.info(f"Copying SMILES from {smiles_source}")
            prepare_smiles_files(smiles_source, sandbox_dir)
        else:
            logging.warning("No SMILES source provided for Stage 2.")

    def _prepare_casf_csar(self, sandbox_dir):
        """Gather inputs for CASF/CSAR/AF3 experiments (Boltz2 or experimental structures)."""
        logging.info(f"Preparing {self.mode} Data...")

        struct_dir = os.path.abspath(self.config.path)
        exp_sdf_dir = os.path.abspath(self.config.experimental_sdf_dir) if self.config.get('experimental_sdf_dir') else None
        smiles_source_dir = self.config.get('smiles_source_dir')

        # Parse ground truth for PDB IDs
        gt_path = os.path.abspath(self.config.ground_truth_path)
        pdb_ids = []
        if os.path.exists(gt_path):
            with open(gt_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    line = line.strip()
                    if ',' in line:
                        parts = [p.strip() for p in line.split(',')]
                    else:
                        parts = line.split()

                    if len(parts) >= 1:
                        pdb_id = parts[0]
                        # Skip header lines (non-PDB ID)
                        if pdb_id.upper() in ('PDBID', 'PDB_ID', 'NUMBER', 'ID', 'NAME',
                                               'TARGET ID', 'TARGET_ID',
                                               'COMPOUND_ID', 'COMPOUND ID'):
                            continue
                        pdb_ids.append(pdb_id)
        logging.info(f"Found {len(pdb_ids)} PDB IDs in ground truth")

        try:
            from rdkit import Chem
        except ImportError:
            logging.error("RDKit not available.")
            return

        targets = []
        for pdb_id in pdb_ids:
            # Protein: prefer _protein_clean.pdb, then _protein.pdb (flat, nested, glob)
            protein_pdb = None
            for suffix in [f"{pdb_id}_protein_clean.pdb", f"{pdb_id}_protein.pdb"]:
                candidate = os.path.join(struct_dir, suffix)
                if os.path.exists(candidate):
                    protein_pdb = candidate
                    break
                candidate = os.path.join(struct_dir, pdb_id, suffix)
                if os.path.exists(candidate):
                    protein_pdb = candidate
                    break
            # Boltz2/AF3 nested naming: {pdb_id}/protein.pdb
            if not protein_pdb:
                candidate = os.path.join(struct_dir, pdb_id, "protein.pdb")
                if os.path.exists(candidate):
                    protein_pdb = candidate
            if not protein_pdb:
                import glob as _glob
                matches = _glob.glob(os.path.join(struct_dir, f"*_{pdb_id}_protein.pdb"))
                protein_pdb = matches[0] if matches else os.path.join(struct_dir, pdb_id, f"{pdb_id}_protein.pdb")
            if not os.path.exists(protein_pdb):
                logging.warning(f"Missing protein for {pdb_id}")
                continue

            # Ligand for pocket extraction: try flat, nested (prefixed), nested (plain), then glob
            ligand_mol2 = os.path.join(struct_dir, f"{pdb_id}_ligand.mol2")
            if not os.path.exists(ligand_mol2):
                ligand_mol2 = os.path.join(struct_dir, pdb_id, f"{pdb_id}_ligand.mol2")
            if not os.path.exists(ligand_mol2):
                # Boltz2/AF3 nested naming: {pdb_id}/ligand.mol2
                candidate = os.path.join(struct_dir, pdb_id, "ligand.mol2")
                if os.path.exists(candidate):
                    ligand_mol2 = candidate
            if not os.path.exists(ligand_mol2):
                import glob as _glob
                matches = _glob.glob(os.path.join(struct_dir, f"*_{pdb_id}_ligand.mol2"))
                ligand_mol2 = matches[0] if matches else ligand_mol2
            if not os.path.exists(ligand_mol2):
                logging.warning(f"Missing ligand MOL2 for {pdb_id}")
                continue

            # SMILES: try .smi file first (CASP16), then experimental SDF (CASF/CSAR)
            smiles = None

            if smiles_source_dir:
                # Try .smi first, then .tsv (CASP16 format: TSV with SMILES in 3rd column).
                # CASP16 keeps the .smi flat next to the directory; the AF3 structure
                # sets keep it inside the complex directory, beside ligand.sdf.
                smi_path = os.path.join(smiles_source_dir, f"{pdb_id}.smi")
                if not os.path.exists(smi_path):
                    smi_path = os.path.join(smiles_source_dir, pdb_id, "ligand.smi")
                if os.path.exists(smi_path):
                    with open(smi_path) as sf:
                        smiles = sf.read().strip().split()[0]
                if not smiles:
                    tsv_path = os.path.join(smiles_source_dir, f"{pdb_id}.tsv")
                    if os.path.exists(tsv_path):
                        with open(tsv_path) as sf:
                            for line in sf:
                                parts = line.strip().split('\t')
                                if len(parts) >= 3 and parts[0] != 'ID':
                                    smiles = parts[2]
                                    break

            if not smiles and exp_sdf_dir:
                sdf_path = os.path.join(exp_sdf_dir, pdb_id, f"{pdb_id}_ligand_clean.sdf")
                if not os.path.exists(sdf_path):
                    sdf_path = os.path.join(exp_sdf_dir, pdb_id, f"{pdb_id}_ligand.sdf")
                if os.path.exists(sdf_path):
                    try:
                        suppl = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=True)
                        mols = [m for m in suppl if m is not None]
                        if not mols:
                            suppl = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=True)
                            mols = [m for m in suppl if m is not None]
                            if mols:
                                try:
                                    mols[0].UpdatePropertyCache(strict=False)
                                except Exception:
                                    pass
                        if mols:
                            smiles = Chem.MolToSmiles(mols[0], isomericSmiles=True, canonical=True)
                    except Exception as e:
                        logging.warning(f"RDKit error for {pdb_id}: {e}")

            if not smiles:
                logging.warning(f"Missing SMILES for {pdb_id}")
                continue


            targets.append({
                'target_id': pdb_id,
                'protein_pdb': protein_pdb,
                'ligand_for_pocket': ligand_mol2,
                'smiles': smiles,
            })

        # Shared pocket extraction + .smi writing
        count = self._process_targets(targets, sandbox_dir)
        logging.info(f"{self.mode} preparation: {count}/{len(pdb_ids)} complexes ready")

    def _prepare_stage1(self, input_dir, sandbox_dir):
        """Stage 1: copies pre-existing .smi and pocket files."""
        logging.info(f"Preparing Stage 1 Data from {input_dir}...")

        src_path = Path(input_dir)
        target_pattern = self.config.get('target_pattern', '*')
        import fnmatch

        smiles_source = self.config.get('smiles_source_dir')
        smiles_found = False

        if smiles_source and os.path.exists(smiles_source):
            logging.info(f"Copying SMILES from {smiles_source} for Stage 1")
            count = prepare_smiles_files(smiles_source, sandbox_dir, pattern=target_pattern)
            if count > 0:
                smiles_found = True

        if not smiles_found:
            src_all_data = src_path / 'all_data'
            search_dir = src_all_data if src_all_data.exists() else src_path
            for f in search_dir.glob("*.smi"):
                if fnmatch.fnmatch(f.stem, target_pattern) or fnmatch.fnmatch(f.name, target_pattern):
                    shutil.copy(f, sandbox_dir)

        pocket_dir = os.path.join(sandbox_dir, 'pocket')
        os.makedirs(pocket_dir, exist_ok=True)

        src_pocket = src_path / 'pocket'
        shared_pocket_name = self.config.get('pocket_source')

        if src_pocket.exists():
            if shared_pocket_name:
                shared_pocket_path = src_pocket / shared_pocket_name
                if shared_pocket_path.exists():
                    logging.info(f"Using shared pocket {shared_pocket_name} for all targets.")
                    for smi_file in Path(sandbox_dir).glob("*.smi"):
                        target_id = smi_file.stem
                        dst = os.path.join(pocket_dir, f"{target_id}_poc.pdb")
                        shutil.copy(shared_pocket_path, dst)
                else:
                    logging.warning(f"Shared pocket {shared_pocket_name} not found in {src_pocket}")
            for f in src_pocket.glob("*_poc.pdb"):
                shutil.copy(f, pocket_dir)

    def _run_subprocess(self, cmd, cwd):
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(cwd) + os.pathsep + env.get('PYTHONPATH', '')
            subprocess.run(cmd, cwd=cwd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed: {' '.join(cmd)}")
            raise e
