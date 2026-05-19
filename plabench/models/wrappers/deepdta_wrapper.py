
import os
import sys
import logging
import subprocess
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class DeepDTAWrapper:
    def __init__(self, config: DictConfig):
        self.cfg = config
        self.output_dir = self.cfg.output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, input_data):
        """
        Run DeepDTA inference.
        input_data: path to the test CSV file (from dataset config 'path' field).
        """
        log.info(f"Initialized DeepDTA Wrapper")
        log.info(f"Test file: {input_data}")
        log.info(f"Output Directory: {self.output_dir}")

        script_path = "plabench/models/deepdta/inference.py"
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Inference script not found: {script_path}")

        # Model directory (relative path from config)
        model_dir = self.cfg.model_dir
        if not os.path.isabs(model_dir):
            model_dir = os.path.abspath(model_dir)

        # New args for CV support
        fold = self.cfg.get("fold", 1)
        task = self.cfg.get("task", "warm_start")
        dataset_name = self.cfg.get("dataset_name", "unknown")

        output_file = os.path.join(self.output_dir, "predictions.csv")
        device = self.cfg.get("device", "cuda")

        cmd = [
            sys.executable, script_path,
            "--test_file", os.path.abspath(input_data),
            "--model_dir", model_dir,
            "--output_file", os.path.abspath(output_file),
            "--device", device,
            "--dataset_name", dataset_name,
            "--fold", str(fold),
            "--task", task,
        ]

        log.info(f"Executing: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            log.info("DeepDTA inference completed successfully.")
        except subprocess.CalledProcessError as e:
            log.error(f"DeepDTA inference failed: {e}")
            raise e
