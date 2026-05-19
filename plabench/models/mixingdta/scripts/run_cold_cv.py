import os
import subprocess
import argparse
import pandas as pd
import sys

# Ensure plabench is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from plabench.analysis.metrics import calculate_metrics

def run_fold(dataset_name, task, fold, test_file, output_file, device):
    cmd = [
        "/home/lwfvx/miniforge3/envs/MixingDTA/bin/python", "-m", "plabench.models.mixingdta.inference",
        "--test_file", test_file,
        "--model_root", "/home/lwfvx/Lyuwei/0.Projects/MixingDTA",
        "--output_file", output_file,
        "--device", device,
        "--dataset_name", dataset_name,
        "--task", task,
        "--fold", str(fold)
    ]
    print(f"Running Fold {fold} for {dataset_name}...")
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=['davis_cold_drug', 'davis_cold_target', 'kiba_cold_drug', 'kiba_cold_target'])
    args = parser.parse_args()
    
    if 'davis' in args.dataset_name:
        base_data_path = "/home/lwfvx/Lyuwei/0.Projects/MixingDTA/DTA_DataBase/cold/DAVIS"
    elif 'kiba' in args.dataset_name:
        base_data_path = "/home/lwfvx/Lyuwei/0.Projects/MixingDTA/DTA_DataBase/cold/KIBA"
        
    if 'cold_drug' in args.dataset_name:
        test_file = os.path.join(base_data_path, "test_Drug.pkl")
    else:
        test_file = os.path.join(base_data_path, "test_Target.pkl")
        
    results = []
    
    for fold in range(1, 6):
        output_file = f"outputs/mixingdta/{args.dataset_name}/fold_{fold}/predictions_meeta.csv"
        try:
            run_fold(args.dataset_name, "cold_start", fold, test_file, output_file, "cuda")
            
            if os.path.exists(output_file):
                print(f"Calculating metrics for Fold {fold}...")
                df = pd.read_csv(output_file)
                y_true = df['label'].values if 'label' in df.columns else None # Should check test file for GT if not in prediction
                y_pred = df['prediction'].values
                
                # If label not in prediction file, load from test file?
                # For cold start, test files are PKL. inference.py puts 'label' in output CSV.
                if y_true is None:
                    print("Error: No label column in output file.")
                    continue

                metrics = calculate_metrics(y_pred, y_true)
                metrics['Fold'] = fold
                results.append(metrics)
                print(f"Fold {fold} Metrics: {metrics}")
        except Exception as e:
            print(f"Fold {fold} Failed: {e}")
            
    if results:
        df_res = pd.DataFrame(results)
        print("\n--- Summary ---")
        print(df_res)
        
        df_numeric = df_res.drop(columns=['Fold']).apply(pd.to_numeric, errors='coerce')
        print("Average:")
        print(df_numeric.mean())
        
        # Save summary
        output_dir = os.path.dirname(f"outputs/mixingdta/{args.dataset_name}/") # parent dir
        # actually outputs/mixingdta/{dataset_name}
        os.makedirs(f"outputs/mixingdta", exist_ok=True)
        # We want outputs/mixingdta/{dataset_name}_summary.csv
        df_res.to_csv(f"outputs/mixingdta/{args.dataset_name}_summary.csv", index=False)

if __name__ == "__main__":
    main()
