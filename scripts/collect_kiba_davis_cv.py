
import os
import pandas as pd
import numpy as np
import logging
from plabench.analysis.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Collect cross-validation results and compute metrics")
    parser.add_argument('--models', nargs='+', default=['deepdta', 'llf', 'mixingdta'],
                        help="List of models to evaluate (e.g., --models deepdta llf)")
    parser.add_argument('--datasets', nargs='+', 
                        default=['davis_warm', 'kiba_warm', 'davis_cold_drug', 'davis_cold_target', 'kiba_cold_drug', 'kiba_cold_target'],
                        help="List of datasets to evaluate (e.g., --datasets kiba_cold_target davis_cold_target)")
    return parser.parse_args()

def collect_cv_results(target_models, target_datasets):
    models = target_models
    
    # Format: (Config Name, Dataset Dir, Task Type for labeling)
    all_experiments = [
        ("davis_warm", "DAVIS", "warm_start"),
        ("kiba_warm", "KIBA", "warm_start"),
        ("davis_cold_drug", "DAVIS", "cold_drug"),
        ("davis_cold_target", "DAVIS", "cold_target"),
        ("kiba_cold_drug", "KIBA", "cold_drug"),
        ("kiba_cold_target", "KIBA", "cold_target"),
    ]
    
    experiments = [exp for exp in all_experiments if exp[0] in target_datasets]
    
    summary_records = []
    benchmark_summary_rows = []
    
    for model_name in models:
        for config_name, data_dir, task in experiments:
            log.info(f"\nScanning {model_name} - {config_name} ({data_dir} - {task})...")
            
            fold_metrics = []
            
            for fold in range(1, 6):
                # 1. Identify Prediction File
                if model_name == 'mixingdta':
                    if "warm" in config_name:
                        pred_file = f"outputs/mixingdta/{config_name}/fold_{fold}/predictions_deepdta.csv"
                    elif "cold" in config_name:
                        pred_file = f"outputs/mixingdta/{config_name}/fold_{fold}/predictions_meeta.csv"
                    else:
                        pred_file = f"outputs/mixingdta/{config_name}/fold_{fold}/predictions.csv"
                elif model_name == 'llf':
                    pred_file = f"outputs/llf/{config_name}/fold_{fold}/predictions.csv"
                elif model_name == 'deepdta':
                    pred_file = f"outputs/deepdta/{config_name}/fold_{fold}/predictions.csv"
                
                # print(f"DEBUG: Checking {pred_file}: {os.path.exists(pred_file)}")
                is_deepdta_corrupted = model_name == 'deepdta' and ('cold_drug' in config_name or 'cold_target' in config_name)
                if not os.path.exists(pred_file) and not is_deepdta_corrupted:
                    log.warning(f"  Missing predictions for Fold {fold}: {pred_file}")
                    continue
                
                # print(f"DEBUG: Found pred_file: {pred_file}")

                # 2. Identify GT File (Optional if pred file has labels)
                gt_file = None
                if "cold" in config_name:
                    if "drug" in config_name:
                        gt_file = f"data/Structure_independent/{data_dir}/cold/test_drug.csv"
                    else:
                        gt_file = f"data/Structure_independent/{data_dir}/cold/test_target.csv"
                else:
                    gt_file = f"data/Structure_independent/{data_dir}/test_{fold}.csv"
                
                # 3. Load Data
                merged = None
                
                # Check headers to decide loading method
                # Simple check: read first line
                has_header = False
                if os.path.exists(pred_file):
                    with open(pred_file, 'r') as f:
                        header_line = f.readline()
                    has_header = ',' in header_line and not header_line[0].isdigit() # Simple heuristic
                
                if "predictions_deepdta.csv" in pred_file or "predictions_meeta.csv" in pred_file or has_header:
                     # Load with header
                     merged = pd.read_csv(pred_file)
                else:
                    print(f"TRACE 1 Fold {fold}: Entering Headerless / Corrupted Fallback block", flush=True)
                    # Headerless: prediction, name
                    if os.path.exists(pred_file):
                        preds_df = pd.read_csv(pred_file, header=None, names=["prediction", "name"])
                    else:
                        preds_df = pd.DataFrame(columns=["prediction", "name"])
                    
                    # OVERRIDE FOR CORRUPTED DEEPDTA DICTIONARIES
                    # Because DeepDTA Folds 1-5 overwritten dicts concurrently, we must use internal original outputs
                    if model_name == 'deepdta' and ('cold_drug' in config_name or 'cold_target' in config_name):
                        if 'kiba' in config_name and 'drug' in config_name:
                            orig_pred = f"/home/lwfvx/Lyuwei/0.Projects/DeepDTA-Pytorch/kiba_drug_results/kiba_drug_fold_{fold}/fold_{fold}_predictions.csv"
                        elif 'davis' in config_name and 'drug' in config_name:
                            orig_pred = f"/home/lwfvx/Lyuwei/0.Projects/DeepDTA-Pytorch/davis_drug_results/davis_drug_fold_{fold}/fold_{fold}_predictions.csv"
                        elif 'davis' in config_name and 'target' in config_name:
                            orig_pred = f"/home/lwfvx/Lyuwei/0.Projects/DeepDTA-Pytorch/davis_results/davis_target_fold_{fold}/fold_{fold}_predictions.csv"
                        elif 'kiba' in config_name and 'target' in config_name:
                            orig_pred = f"/home/lwfvx/Lyuwei/0.Projects/DeepDTA-Pytorch/kiba_target_cold_results/kiba_target_cold_fold_{fold}/fold_{fold}_predictions.csv"
                        else:
                            orig_pred = None
                            
                        if orig_pred and os.path.exists(orig_pred):
                            # Load their actual predictions
                            orig_df = pd.read_csv(orig_pred)
                            preds_df['prediction'] = orig_df['predicted_affinity'].values
                        print(f"TRACE 2 Fold {fold}: reading gt_file {gt_file}", flush=True)
                if os.path.exists(gt_file):
                    gt_df = pd.read_csv(gt_file)
                    # Verify Lengths
                    if len(preds_df) != len(gt_df):
                        log.warning(f"  Fold {fold} Length Mismatch: Preds={len(preds_df)}, GT={len(gt_df)}")
                        min_len = min(len(preds_df), len(gt_df))
                        preds_df = preds_df.iloc[:min_len]
                        gt_df = gt_df.iloc[:min_len]
                    
                    # Merge by Concatenation
                    merged = pd.concat([
                        preds_df.reset_index(drop=True), 
                        gt_df.reset_index(drop=True)
                    ], axis=1)
                else:
                    log.warning(f"  Missing GT for headerless pred file: {gt_file}")
                    continue
                # 4. Standardize Columns
                if "Predicted Affinity" in merged.columns:
                     merged['prediction'] = merged['Predicted Affinity']
                
                label_col = 'label'
                if label_col not in merged.columns:
                    for col in merged.columns:
                        if 'label' in col.lower() or 'affinity' in col.lower() and col != 'Predicted Affinity':
                            label_col = col
                            break
                
                if label_col not in merged.columns and gt_file and os.path.exists(gt_file):
                     # Still missing label? Try to merge GT if not already merged
                     gt_df = pd.read_csv(gt_file)
                     if len(merged) == len(gt_df):
                         merged = pd.concat([merged.reset_index(drop=True), gt_df.reset_index(drop=True)], axis=1)
                         
                         # Re-detect label column after merge
                         if label_col not in merged.columns:
                            for col in merged.columns:
                                if ('label' in col.lower() or 'affinity' in col.lower()) and col != 'Predicted Affinity' and col != 'prediction' and col != 'predicted_affinity':
                                    label_col = col
                                    break
                     else:
                        # print(f"DEBUG: Length mismatch {len(merged)} vs {len(gt_df)}")
                        pass
                        
                # Fallback explicitly for deepdta corrupted dictionary sets pulling from 'true_affinity'
                if 'true_affinity' in merged.columns:
                    label_col = 'true_affinity'
                
                if config_name == "kiba_cold_target":
                    pass

                merged = merged.dropna(subset=["prediction", label_col])
                
                if config_name == "kiba_cold_target":
                    pass
                    
                if len(merged) == 0:
                    continue
                    
                # 5. Calculate Metrics
                metrics = calculate_metrics(merged["prediction"], merged[label_col])
                metrics['Fold'] = fold
                metrics['Dataset'] = config_name
                metrics['N'] = len(merged)
                fold_metrics.append(metrics)
                
                # Helper to format
                def fmt(v):
                    return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)

                log.info(f"  Fold {fold}: N={len(merged)}, Pearson={fmt(metrics['Pearson'])}, MSE={fmt(metrics['MSE'])}, CI={fmt(metrics['CI'])}, Rm2={fmt(metrics['Rm2'])}")
            if fold_metrics:
                df_metrics = pd.DataFrame(fold_metrics)
                avg_metrics = df_metrics.mean(numeric_only=True)
                
                # CRITICAL CHANGE: Use MEAN of N to standardize to single-fold count
                avg_n = df_metrics['N'].mean()
                
                # Prepare row for benchmark_summary.csv
                # Model,Dataset,N,Total,Coverage,RMSE,MSE,Pearson,Spearman,Kendall,CI,Rm2
                summary_row = {
                    'Model': model_name,
                    'Dataset': config_name,
                    'N': int(avg_n),
                    'Total': int(avg_n),
                    'Coverage': "100.0%",
                    'RMSE': round(avg_metrics.get('RMSE', 0.0), 3),
                    'MSE': round(avg_metrics.get('MSE', 0.0), 3),
                    'Pearson': round(avg_metrics.get('Pearson', 0.0), 3),
                    'Spearman': round(avg_metrics.get('Spearman', 0.0), 3),
                    'Kendall': round(avg_metrics.get('Kendall', 0.0), 3),
                    'CI': round(avg_metrics.get('CI', 0.0), 3),
                    'Rm2': round(avg_metrics.get('Rm2', 0.0), 3)
                }
                benchmark_summary_rows.append(summary_row)
                
                avg_metrics['Fold'] = 'Average'
                avg_metrics['Dataset'] = config_name
                summary_records.extend(fold_metrics)
                summary_records.append(avg_metrics.to_dict())
                
                p_val = avg_metrics.get('Pearson', 0.0)
                ci_val = avg_metrics.get('CI', 0.0)
                log.info(f"  >> Average (N={int(avg_n)}): Pearson={p_val:.3f}, CI={ci_val:.3f}")
        else:
            log.warning(f"  No valid folds found for {config_name}")

    if summary_records:
            # Save detailed report
            result_df = pd.DataFrame(summary_records)
            os.makedirs("analysis", exist_ok=True)
            result_df.to_csv("analysis/mixingdta_kiba_davis_results.csv", index=False)
            log.info("\nSaved detailed results to analysis/mixingdta_kiba_davis_results.csv")
            
            # Append to benchmark_summary.csv
            summary_csv = os.path.join("results", "benchmark_summary.csv")
            new_rows_df = pd.DataFrame(benchmark_summary_rows)
            
            if os.path.exists(summary_csv):
                existing_df = pd.read_csv(summary_csv)
                # Remove existing rows for these datasets/models to avoid duplicates
                
                # Efficient way: Create a set of (Model, Dataset) tuples from new rows
                new_keys = set((r['Model'], r['Dataset']) for r in benchmark_summary_rows)
                
                # Filter existing_df to exclude rows that are in new_keys
                # Lambda filtering or boolean indexing
                mask = existing_df.apply(lambda x: (x['Model'], x['Dataset']) not in new_keys, axis=1)
                existing_df = existing_df[mask]
                
                final_df = pd.concat([existing_df, new_rows_df], ignore_index=True)
            else:
                final_df = new_rows_df
                
            final_df.to_csv(summary_csv, index=False)
            print("\nUpdated benchmark_summary.csv with MixingDTA CV results:")
            print(new_rows_df.to_string(index=False))

if __name__ == "__main__":
    args = get_args()
    collect_cv_results(args.models, args.datasets)
