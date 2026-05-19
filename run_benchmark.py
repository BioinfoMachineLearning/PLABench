
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import logging
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Registry of available wrappers
# In the future, this could be dynamic or plugin-based
WRAPPER_REGISTRY = {
    "haiping": "plabench.models.wrappers.haiping_wrapper.HaipingWrapper",
    "bapred": "plabench.models.wrappers.bapred_wrapper.BapredWrapper",
    "deepdta": "plabench.models.wrappers.deepdta_wrapper.DeepDTAWrapper",
    "llf": "plabench.models.wrappers.llf_wrapper.LLFWrapper",
    "mixingdta": "plabench.models.wrappers.mixingdta_wrapper.MixingDTAWrapper",
    "flowdock": "plabench.models.wrappers.flowdock_wrapper.FlowDockWrapper",
    "mfe": "plabench.models.wrappers.mfe_wrapper.MFEWrapper",
    "boltz2": "plabench.models.wrappers.boltz2_wrapper.Boltz2Wrapper",
    "flowr_root": "plabench.models.wrappers.flowr_root_wrapper.FlowrRootWrapper",
}

def get_wrapper_class(model_name):
    """
    Dynamically import and return the wrapper class for the given model name.
    """
    if model_name not in WRAPPER_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(WRAPPER_REGISTRY.keys())}")
    
    module_path, class_name = WRAPPER_REGISTRY[model_name].rsplit('.', 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import wrapper {class_name} from {module_path}: {e}")

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point for PLA-Bench.

    Environment setup examples:
    *   **Haiping (graph_rg)**: `/bmlfast/Lyuwei/miniconda3/envs/graph_rg` — 运行命令示例: `conda run -p /bmlfast/Lyuwei/miniconda3/envs/graph_rg python run_benchmark.py model=haiping dataset=haiping_casp16_l3000_stage2`
    *   **BAPred**: `/bmlfast/Lyuwei/miniconda3/envs/BAPred` — 运行命令示例: `conda run -p /bmlfast/Lyuwei/miniconda3/envs/BAPred python run_benchmark.py model=bapred dataset=bapred_casp16_l3000_stage2`

    Usage: python run_benchmark.py model=haiping dataset=haiping_casp16_l3000_stage2
    """
    log.info(f"Starting PLA-Bench Execution")
    log.info(f"Model: {cfg.model.name}")
    log.info(f"Dataset: {cfg.dataset.name}")

    # Ensure logs directory exists
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Add FileHandler to root logger
    log_file = os.path.join(log_dir, f"{cfg.model.name}_{cfg.dataset.name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    log.info(f"Logging to {log_file}")
    
    # 1. Resolve Configuration
    # Merge model and dataset configs into a single config for the wrapper
    # Using OmegaConf.to_container to resolve interpolations if any
    try:
        # Check if configs are properly loaded
        if 'name' not in cfg.model:
            log.error("Model config missing 'name'. Did you specify model=...?")
            return
        if 'name' not in cfg.dataset:
            log.error("Dataset config missing 'name'. Did you specify dataset=...?")
            return

        model_conf = OmegaConf.to_container(cfg.model, resolve=True)
        dataset_conf = OmegaConf.to_container(cfg.dataset, resolve=True)
        
        # Combined config passed to wrapper
        wrapper_config = OmegaConf.create({**model_conf, **dataset_conf})
        # Inject output_dir from Hydra's output directory
        if 'output_dir' not in wrapper_config:
            from hydra.core.hydra_config import HydraConfig
            wrapper_config.output_dir = HydraConfig.get().runtime.output_dir
        
    except Exception as e:
        log.error(f"Configuration resolution failed: {e}")
        return

    # 2. Instantiate Wrapper
    try:
        WrapperClass = get_wrapper_class(cfg.model.name)
        wrapper = WrapperClass(wrapper_config)
        log.info(f"Initialized {WrapperClass.__name__}")
    except Exception as e:
        log.error(f"Failed to initialize wrapper for {cfg.model.name}: {e}")
        return

    # 3. Execution
    try:
        # Input data path from dataset config
        input_data = cfg.dataset.path
        log.info(f"Running inference on {input_data}")
        
        wrapper.run(input_data)
        log.info("Wrapper execution completed successfully.")
        
    except Exception as e:
        log.error(f"Execution failed: {e}")
        # We don't exit(1) to allow cleanup if needed, but here we stop.
        return

    # 4. Evaluation (Optional / Integrated)
    # Wrappers typically generate predictions.csv.
    # We can optionally call evaluate() here if the wrapper doesn't do it,
    # or rely on collect_results.py for batch evaluation.
    # For a single run, printing the result is helpful.
    
    output_dir = wrapper.output_dir if hasattr(wrapper, 'output_dir') else None
    gt_path = cfg.dataset.get("ground_truth_path")
    
    output_file = os.path.join(output_dir, "predictions.csv")
    if output_dir and gt_path and os.path.exists(output_file) and os.path.exists(gt_path):
        log.info("Running Immediate Evaluation...")
        try:
            import pandas as pd
            from plabench.analysis.metrics import calculate_metrics

            preds_df = pd.read_csv(output_file, header=None, names=["prediction", "name"])

            gt_format = cfg.dataset.get("ground_truth_format", "csv")

            if gt_format == "casf_dat":
                # Parse CoreSet.dat: space-delimited, column 0=PDB, column 3=-logKd/Ki
                gt_rows = []
                with open(gt_path, 'r') as f:
                    for line in f:
                        if line.startswith('#') or not line.strip():
                            continue
                        parts = line.split()
                        if len(parts) >= 4:
                            gt_rows.append({"pdb_id": parts[0], "affinity": float(parts[3])})
                gt_df = pd.DataFrame(gt_rows)
                merged = pd.merge(preds_df, gt_df, left_on="name", right_on="pdb_id")
                gt_score_col = "affinity"
            else:
                gt_df = pd.read_csv(gt_path)
                gt_id_col = gt_df.columns[0]
                if "Target ID" in gt_df.columns: gt_id_col = "Target ID"
                elif "\ufeffTarget ID" in gt_df.columns: gt_id_col = "\ufeffTarget ID"

                gt_score_col = None
                for col in gt_df.columns:
                    if "binding_affinity" in col.lower() or "affinity" in col.lower():
                        gt_score_col = col
                        break
                if not gt_score_col: gt_score_col = gt_df.columns[-1]

                merged = pd.merge(preds_df, gt_df, left_on="name", right_on=gt_id_col)
                if len(merged) == 0:
                     preds_df['name_clean'] = preds_df['name'].astype(str).str.replace(r"['\[\]]", "", regex=True)
                     merged = pd.merge(preds_df, gt_df, left_on="name_clean", right_on=gt_id_col)

                # Unit conversion for CASP16 datasets (kcal/mol -> pKd)
                if "l1000" in cfg.dataset.name.lower() or "l3000" in cfg.dataset.name.lower():
                    merged[gt_score_col] = merged[gt_score_col] * -0.733

            if len(merged) > 0:
                 # Drop rows with NaN predictions or ground truth
                 merged = merged.dropna(subset=["prediction", gt_score_col])
                 merged["prediction"] = pd.to_numeric(merged["prediction"], errors="coerce")
                 merged = merged.dropna(subset=["prediction"])
                 if len(merged) > 0:
                     metrics = calculate_metrics(merged["prediction"], merged[gt_score_col])
                     log.info(f"\n--- Results for {cfg.dataset.name} ({len(merged)} matched) ---")
                     for k, v in metrics.items():
                         log.info(f"{k}: {v}")
                 else:
                     log.warning("Evaluation failed: No valid numeric predictions after filtering NaN.")
            else:
                 log.warning("Evaluation failed: No matching targets found between prediction and ground truth.")

        except Exception as e:
            log.warning(f"Immediate evaluation skipped due to error: {e}")
    else:
        log.info("Skipping immediate evaluation (predictions or GT missing).")

if __name__ == "__main__":
    main()
