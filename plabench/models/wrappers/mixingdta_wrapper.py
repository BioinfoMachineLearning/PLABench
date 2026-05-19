
import os
import sys
import logging
import subprocess
from omegaconf import DictConfig

log = logging.getLogger(__name__)

class MixingDTAWrapper:
    def __init__(self, config: DictConfig):
        self.config = config
        self.output_dir = self.config.output_dir
        
        # Determine python binary
        self.python_bin = config.get("python_bin", "python")
        
        # Verify environment
        try:
            subprocess.run([self.python_bin, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            log.warning(f"Python binary {self.python_bin} not found or not executable.")

    def run(self, input_data):
        log.info(f"Running MixingDTA inference on {input_data}")
        
        # Use relative path from project root
        script_path = "plabench/models/mixingdta/inference.py"
        model_dir = self.config.model_dir
        
        output_file = os.path.join(self.output_dir, "predictions.csv")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract arguments from config
        dataset_name = self.config.get("dataset_name") or self.config.get("name") or "DAVIS"
        task = self.config.get("task", "warm_start")
        fold = self.config.get("fold", 1)
        device = self.config.get("device", "cuda")
        
        # Use self.python_bin (Fixing previous bug where sys.executable was used)
        # model_root: use fork (forks/MixingDTA) by default; fallback to configured model_dir's ancestor
        model_root = self.config.get("model_root") or os.path.abspath("forks/MixingDTA")

        cmd = [
            self.python_bin, "plabench/models/mixingdta/inference.py",
            "--test_file", os.path.abspath(input_data),
            "--model_root", model_root,
            "--output_file", os.path.abspath(output_file),
            "--device", device,
            "--dataset_name", dataset_name,
            "--task", task,
            "--fold", str(fold)
        ]
        
        log.info(f"Command: {' '.join(cmd)}")
        
        try:
            # We must use text=True to capture stdout/stderr as strings
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            log.info(result.stdout)
        except subprocess.CalledProcessError as e:
            log.error(f"Error running MixingDTA: {e}")
            log.error(f"Stdout: {e.stdout}")
            log.error(f"Stderr: {e.stderr}")
            # Raise to handle failure
            raise e
            
        log.info(f"MixingDTA inference completed. Output saved to {output_file}")
