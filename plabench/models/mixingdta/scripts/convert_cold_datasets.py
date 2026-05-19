
import pickle
import pandas as pd
import os
import glob
import sys

def convert_cold_dataset(dataset_name):
    source_base = f"forks/MixingDTA/DTA_DataBase/cold/{dataset_name}"
    target_base = f"data/Structure_independent/{dataset_name}/cold"
    
    if not os.path.exists(target_base):
        os.makedirs(target_base)
        print(f"Created directory: {target_base}")
        
    # ONLY convert test files for now to avoid overwriting with train data
    pkl_files = glob.glob(os.path.join(source_base, "test_*.pkl"))
    
    if not pkl_files:
        print(f"No .pkl files found for {dataset_name} in {source_base}")
        return

    print(f"Converting {len(pkl_files)} files for {dataset_name}...")
    
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Expected Data structure: [Index, SMILES, TargetID, Sequence, Label]
            # based on convert_kiba_davis.py logic for similar datasets from same source.
            
            if len(data) > 0:
                # Check structure of first item
                first_item = data[0]
                if isinstance(first_item, (list, tuple)) and len(first_item) >= 5:
                    records = []
                    for item in data:
                        records.append({
                            'compound_iso_smiles': item[1],
                            'Target ID': item[2],
                            'target_sequence': item[3],
                            'label': item[4]
                        })
                    
                    df = pd.DataFrame(records)
                    
                    # Determine output filename
                    base_name = os.path.basename(pkl_file)
                    if base_name == 'test_Drug.pkl':
                        out_name = 'test_drug.csv'
                    elif base_name == 'test_Target.pkl':
                         out_name = 'test_target.csv'
                    else:
                        out_name = base_name.replace('.pkl', '.csv').lower()
                        
                    csv_file = os.path.join(target_base, out_name)
                    df.to_csv(csv_file, index=False)
                    print(f"  Converted {pkl_file} -> {csv_file} ({len(df)} records)")
                else:
                    print(f"  Skipping {pkl_file}: Unexpected structure (Type: {type(first_item)}, Len: {len(first_item) if hasattr(first_item, '__len__') else 'N/A'})")
                    print(f"  Sample: {first_item}")
            else:
                print(f"  Skipping {pkl_file}: Empty data")

        except Exception as e:
            print(f"  Failed to convert {pkl_file}: {e}")

if __name__ == "__main__":
    convert_cold_dataset("DAVIS")
    convert_cold_dataset("KIBA")
