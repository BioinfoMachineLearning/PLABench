import os
import sys
import subprocess
import logging

log = logging.getLogger(__name__)


class MFEWrapper:
    """PLABench wrapper for MFE (Multimodal Fusion Encoder).

    Delegates to plabench/models/mfe/inference.py running in the MFE
    conda environment.
    """

    def __init__(self, config):
        self.config = config
        self.output_dir = os.path.abspath(config.output_dir)
        self.checkpoint = os.path.abspath(config.get(
            'checkpoint', 'checkpoints/structure/mfe/best_model.pt'))
        self.mfe_root = os.path.abspath(config.get('mfe_root', 'forks/MFE'))
        self.device = config.get('device', 'cuda')
        self.protbert_model = config.get('protbert_model', 'Rostlab/prot_bert')
        # Note: this wrapper relies on the caller invoking run_benchmark.py with
        # the MFE env's Python (e.g. `/bmlfast/Lyuwei/miniconda3/envs/MFE/bin/python ...`),
        # so we just use `sys.executable`. The `conda_env` config field is no
        # longer required (was a hardcoded fallback to the old user miniforge
        # path that broke when that env was deleted).
        self.target_pattern = config.get('target_pattern', 'L*')
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, input_data):
        log.info(f"MFE Wrapper: input={input_data}, output={self.output_dir}")

        pred_file = os.path.join(self.output_dir, 'predictions.csv')
        if os.path.exists(pred_file):
            log.info(f"Predictions already exist at {pred_file}, skipping.")
            return

        inference_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'mfe', 'inference.py')

        cmd = [
            sys.executable, inference_script,
            '--input_dir', os.path.abspath(input_data),
            '--output_dir', self.output_dir,
            '--checkpoint', self.checkpoint,
            '--mfe_root', self.mfe_root,
            '--protbert_model', self.protbert_model,
            '--device', self.device,
            '--target_pattern', self.target_pattern,
        ]

        # GT-filter convention (agent.md §2.1): pass ground_truth_path so
        # multi_protein datasets only run targets present in GT.
        gt_path = self.config.get('ground_truth_path')
        if gt_path and os.path.exists(gt_path):
            cmd.extend(['--filter_file', os.path.abspath(gt_path)])

        log.info(f"Running: {' '.join(cmd)}")
        env = os.environ.copy()
        env['PYTHONPATH'] = self.mfe_root + os.pathsep + env.get('PYTHONPATH', '')

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"MFE inference failed with code {result.returncode}")

        # Copy sandbox-prep failures.csv to output_dir for visibility (agent.md §3.1).
        # The inference script writes per-target sandbox failures to
        # tmp/mfe_{dataset}_sandbox/failures_sandbox.csv; surface it as the
        # standardized failures.csv next to predictions.csv.
        dataset_name = os.path.basename(os.path.dirname(self.output_dir))
        sandbox_fail = os.path.join(
            'tmp', f'mfe_{dataset_name}_sandbox', 'failures_sandbox.csv'
        )
        if os.path.exists(sandbox_fail):
            try:
                import shutil
                shutil.copy2(sandbox_fail, os.path.join(self.output_dir, 'failures.csv'))
                log.info(f"Copied sandbox failures.csv to {self.output_dir}/failures.csv")
            except Exception as e:
                log.warning(f"Failed to copy failures.csv: {e}")

        log.info("MFE inference completed.")
