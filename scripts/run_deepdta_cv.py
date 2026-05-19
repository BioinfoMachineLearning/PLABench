
import os
import subprocess

def run_experiment(dataset_name, task, input_dir):
    print(f"\n--- Running Experiment: {dataset_name} ---")
    
    for fold in range(1, 6):
        # Determine input file
        if "warm" in dataset_name:
            # test_{fold}.csv
            test_file = f"data/Structure_independent/{input_dir}/test_{fold}.csv"
        else:
            # Cold start: fixed test file
            # Naming in Structure_independent: cold/test_drug.csv or test_Drug.csv?
            # MixingDTA scripts standardize on csv.
            # Filenames verified previously: test_drug.csv / test_target.csv
            
            suffix = "drug" if "drug" in dataset_name else "target"
            test_file = f"data/Structure_independent/{input_dir}/cold/test_{suffix}.csv"
            
        output_file = f"outputs/deepdta/{dataset_name}/fold_{fold}/predictions.csv"
        
        if not os.path.exists(test_file):
            print(f"Skipping {dataset_name} Fold {fold}: Input file not found ({test_file})")
            continue
            
        # Force overwrite for rerun
        # if os.path.exists(output_file):
        #     print(f"  Fold {fold}: Already exists, skipping.")
        #     continue
            
        cmd = [
            "/home/lwfvx/miniforge3/envs/deepdta/bin/python", "-m", "plabench.models.deepdta.inference",
            "--test_file", test_file,
            "--output_file", output_file,
            "--dataset_name", dataset_name,
            "--fold", str(fold),
            "--task", task,
            "--device", "cuda",
            "--model_dir", "forks/DeepDTA-Pytorch"
        ]
        
        print(f"  Running Fold {fold}...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  Error in Fold {fold}: {e}")

def main():
    # Define tasks
    tasks = [
        ("davis_warm", "warm_start", "DAVIS"),
        ("kiba_warm", "warm_start", "KIBA"),
        ("davis_cold_drug", "cold_drug", "DAVIS"),
        ("davis_cold_target", "cold_target", "DAVIS"),
        ("kiba_cold_drug", "cold_drug", "KIBA"),
        ("kiba_cold_target", "cold_target", "KIBA")
    ]
    
    for dname, task, input_dir in tasks:
        run_experiment(dname, task, input_dir)

if __name__ == "__main__":
    main()
