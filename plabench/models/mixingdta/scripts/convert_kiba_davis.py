
import pickle
import pandas as pd
import os
import glob
import sys

def convert_dataset(dataset_name):
    base_path = os.path.join("data", "Structure_independent", dataset_name)
    pkl_files = glob.glob(os.path.join(base_path, "test_*.pkl"))
    
    if not pkl_files:
        print(f"No .pkl files found for {dataset_name} in {base_path}")
        return

    print(f"Converting {len(pkl_files)} files for {dataset_name}...")
    
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Data structure: [Index, SMILES, TargetID, Sequence, Label]
            # Verify structure of first item
            if len(data) > 0 and len(data[0]) >= 5:
                records = []
                for item in data:
                    records.append({
                        'compound_iso_smiles': item[1],
                        'Target ID': item[2],
                        'target_sequence': item[3],
                        'label': item[4]
                    })
                
                df = pd.DataFrame(records)
                csv_file = pkl_file.replace('.pkl', '.csv')
                df.to_csv(csv_file, index=False)
                print(f"  Converted {pkl_file} -> {csv_file} ({len(df)} records)")
            else:
                print(f"  Skipping {pkl_file}: Unexpected data structure")
                if len(data) > 0:
                    print(f"  Sample item: {data[0]}")
        except Exception as e:
            print(f"  Failed to convert {pkl_file}: {e}")

if __name__ == "__main__":
    convert_dataset("DAVIS")
    convert_dataset("KIBA")
