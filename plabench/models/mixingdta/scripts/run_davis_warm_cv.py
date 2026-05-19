import os
import subprocess
import pandas as pd
import numpy as np
import sys

# Ensure plabench is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from plabench.analysis.metrics import calculate_metrics

def run_fold(fold, test_file, output_file):
    cmd = [
        "/home/lwfvx/miniforge3/envs/MixingDTA/bin/python", "-m", "plabench.models.mixingdta.inference",
        "--test_file", test_file,
        "--model_root", "/home/lwfvx/Lyuwei/0.Projects/MixingDTA",
        "--output_file", output_file,
        "--device", "cuda",
        "--dataset_name", "davis_warm",
        "--task", "warm_start",
        "--fold", str(fold)
    ]
    print(f"Running Fold {fold}...")
    subprocess.check_call(cmd)

def main():
    folds = [1, 2, 3, 4, 5]
    results = []
    
    base_data = "/home/lwfvx/Lyuwei/0.Projects/PLABench/data/Structure_independent/DAVIS"
    base_output = "/home/lwfvx/Lyuwei/0.Projects/PLABench/outputs/mixingdta/davis_warm"
    
    for fold in folds:
        test_file = os.path.join(base_data, f"test_{fold}.csv")
        output_dir = os.path.join(base_output, f"fold_{fold}")
        # os.makedirs(output_dir, exist_ok=True) # Inference creates dir
        output_file = os.path.join(output_dir, "predictions_deepdta.csv")
        
        # Run inference (Commented out if you just want to recalc metrics, but this is a run script)
        # run_fold(fold, test_file, output_file)
        
        if os.path.exists(output_file):
            print(f"Calculating metrics for Fold {fold}...")
            gt_df = pd.read_csv(test_file)
            pred_df = pd.read_csv(output_file)
            
            y_true = gt_df['label'].values
            y_pred = pred_df['Predicted Affinity'].values
            
            metrics = calculate_metrics(y_pred, y_true)
            metrics['Fold'] = fold
            results.append(metrics)
            
            print(f"Fold {fold} Metrics: {metrics}")
        else:
             print(f"Fold {fold} output not found at {output_file}")
             # Re-run if needed? Or just skip
             pass

    if results:
        df_results = pd.DataFrame(results)
        print("\n--- Final Results ---")
        print(df_results)
        
        # Convert to numeric to handle potential 'NA'
        df_numeric = df_results.drop(columns=['Fold']).apply(pd.to_numeric, errors='coerce')
        print("\n--- Average ---")
        print(df_numeric.mean())
        
        # Save summary
        df_results.to_csv(os.path.join(base_output, "cv_summary.csv"), index=False)

if __name__ == "__main__":
    main()
