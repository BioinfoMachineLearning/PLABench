
import os
import sys
import logging
import subprocess
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

class LLFWrapper:
    def __init__(self, config: DictConfig):
        self.cfg = config
        self.output_dir = config.output_dir
        self.model_dir = config.model_dir
        # Use get to be safe or access via dict
        if hasattr(config, "dataset") and "path" in config.dataset:
            self.test_file = config.dataset.path
        elif "path" in config:
            self.test_file = config.path
        else:
            # Fallback or error with more info
            log.warning(f"Config keys: {config.keys()}. 'path' not found, will rely on dynamic resolution if possible.")
            self.test_file = None

    def run(self, input_data=None):
        test_file = input_data if input_data else self.test_file
        log.info(f"Running LLF inference on {test_file}")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        output_csv = os.path.join(self.output_dir, "predictions.csv")

        # Construct command
        script_path = "plabench/models/llf/inference.py"
        
        # Python interpreter from config
        # Use specific python path if provided, else current executable
        python_exe = self.cfg.get("python_path", sys.executable)
        
        # Determine device
        device = self.cfg.get("device", "cuda")
        
        # New args for CV support
        # Default fold to 1 if not specified
        fold = self.cfg.get("fold", 1)
        # Default task to 'warm_start' if not specified
        task = self.cfg.get("task", "warm_start")
        # Dataset name (the flattened config contains 'dataset_name' from yaml)
        dataset_name = self.cfg.get("dataset_name") or self.cfg.get("name") or "unknown"

        cmd = [
            python_exe, "plabench/models/llf/inference.py",
            "--test_file", os.path.abspath(test_file),
            "--output_file", os.path.abspath(output_csv),
            "--dataset_name", dataset_name,
            "--fold", str(fold),
            "--task", task,
            "--device", device,
        ]

        # Optional overrides from dataset/model config
        model_path_override = self.cfg.get("model_path_override") or self.cfg.get("model_path")
        if model_path_override:
            cmd.extend(["--model_path", os.path.abspath(model_path_override)])
        max_seq_len = self.cfg.get("max_seq_len")
        if max_seq_len:
            cmd.extend(["--max_seq_len", str(max_seq_len)])
        
        log.info(f"Command: {' '.join(cmd)}")
        
        # Environment setup if needed (e.g. for creating graph on the fly)
        env = os.environ.copy()
        
        # Run inference
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                cwd=os.getcwd() # Run from project root
            )
            log.info(result.stdout)
            if result.stderr:
                log.warning(f"Stderr: {result.stderr}")
                
        except subprocess.CalledProcessError as e:
            log.error(f"LLF inference failed with exit code {e.returncode}")
            log.error(f"Stdout: {e.stdout}")
            log.error(f"Stderr: {e.stderr}")
            raise RuntimeError("LLF inference failed") from e
            
        log.info(f"LLF inference completed. Output saved to {output_csv}")
