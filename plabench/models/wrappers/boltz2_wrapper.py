import os
import subprocess
import logging

log = logging.getLogger(__name__)


class Boltz2Wrapper:
    """PLABench wrapper for Boltz2 affinity prediction.

    Supports two input modes:
      1. CASP16 mode: single protein FASTA + per-target SMILES TSV directory.
         Config fields: sequence_file, smiles_dir
      2. Standardized CSV mode: CSV with compound_id, target_sequence, compound_iso_smiles.
         Config field: standardized_csv

    Delegates to plabench/models/boltz2/inference.py.
    """

    def __init__(self, config):
        self.config = config
        self.output_dir = os.path.abspath(config.output_dir)
        self.conda_env = config.get(
            "conda_env", "/bmlfast/Lyuwei/miniconda3/envs/boltz")
        self.boltz_bin = os.path.join(self.conda_env, "bin", "boltz")
        self.boltz_python = os.path.join(self.conda_env, "bin", "python")
        self.ground_truth_path = os.path.abspath(config.ground_truth_path) if config.get("ground_truth_path") else None
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, input_data):
        log.info(f"Boltz2 Wrapper: output={self.output_dir}")

        pred_file = os.path.join(self.output_dir, "predictions.csv")
        if os.path.exists(pred_file):
            log.info(f"Predictions already exist at {pred_file}, skipping.")
            return

        inference_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "boltz2", "inference.py")

        cmd = [
            self.boltz_python, inference_script,
            "--output_dir", self.output_dir,
            "--boltz_bin", self.boltz_bin,
        ]

        if self.config.get("standardized_csv"):
            cmd.extend(["--standardized_csv", os.path.abspath(self.config.standardized_csv)])
        else:
            cmd.extend([
                "--sequence_file", os.path.abspath(self.config.sequence_file),
                "--smiles_dir", os.path.abspath(self.config.smiles_dir),
            ])

        if self.ground_truth_path:
            cmd.extend(["--ground_truth_path", self.ground_truth_path])

        log.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"Boltz2 inference failed with code {result.returncode}")

        log.info("Boltz2 inference completed.")
