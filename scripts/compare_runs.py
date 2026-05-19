
import pandas as pd
import numpy as np
import os
import glob

# Paths
merged_pred_path = "outputs/haiping/casp16_l3000_stage2/merged_ligands_backup/predictions.csv"
single_pred_path = "outputs/haiping/casp16_l3000_stage2/latest/predictions.csv"

# Load Predictions
print(f"Loading Merged: {merged_pred_path}")
merged_df = pd.read_csv(merged_pred_path, header=None, names=["pred_merged", "target_id"])
print(f"Loading Single: {single_pred_path}")
single_df = pd.read_csv(single_pred_path, header=None, names=["pred_single", "target_id"])

# Merge
df = pd.merge(merged_df, single_df, on="target_id")
print(f"Compared {len(df)} common targets.")

# Calculate Diff
df['diff'] = np.abs(df['pred_merged'] - df['pred_single'])

# Identify Multi-Ligand Targets
# Using the list found earlier (from grepping output or re-scanning)
multi_ligand_targets = []
# Scan directories to find multi-ligand targets
base_dir = "data/casp16_data/stage2_input/L3000_prepared"
# Re-run find logic pythonically
for target_folder in glob.glob(os.path.join(base_dir, "*")):
    target_id = os.path.basename(target_folder)
    ligands = glob.glob(os.path.join(target_folder, "ligand_*.pdb"))
    if len(ligands) > 1:
        multi_ligand_targets.append(target_id)
        
print(f"Found {len(multi_ligand_targets)} multi-ligand targets.")

# Split Analysis
multi_df = df[df['target_id'].isin(multi_ligand_targets)]
single_lig_df = df[~df['target_id'].isin(multi_ligand_targets)]

print("\n=== Single-Ligand Targets Analysis ===")
print(f"Count: {len(single_lig_df)}")
max_diff_single = single_lig_df['diff'].max()
mean_diff_single = single_lig_df['diff'].mean()
print(f"Max Diff: {max_diff_single:.6f}")
print(f"Mean Diff: {mean_diff_single:.6f}")
if max_diff_single > 1e-5:
    print("WARNING: Divergence in single-ligand targets!")
    print(single_lig_df.sort_values('diff', ascending=False).head())
else:
    print("SUCCESS: Single-ligand targets are identical.")

print("\n=== Multi-Ligand Targets Analysis ===")
print(f"Count: {len(multi_df)}")
max_diff_multi = multi_df['diff'].max()
mean_diff_multi = multi_df['diff'].mean()
print(f"Max Diff: {max_diff_multi:.6f}")
print(f"Mean Diff: {mean_diff_multi:.6f}")
if mean_diff_multi > 1e-5:
    print("CONFIRMED: Multi-ligand targets show divergence.")
    print(multi_df.sort_values('diff', ascending=False).head())
else:
    print("WARNING: Multi-ligand targets are IDENTICAL? Did modification work?")

