
import os
import shutil
import glob
import logging

logging.basicConfig(level=logging.INFO)

def cleanup_outputs(root_dir="outputs/haiping"):
    datasets = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for dataset in datasets:
        dataset_path = os.path.join(root_dir, dataset)
        # List all subdirs
        subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
        # Filter for timestamp-like directories (start with 202)
        timestamp_dirs = [d for d in subdirs if d.startswith("202")]
        timestamp_dirs.sort() # ISO timestamps sort chronologically
        
        if not timestamp_dirs:
            logging.warning(f"No timestamp dirs found in {dataset}")
            continue
            
        # Keep the LAST one
        latest_run = timestamp_dirs[-1]
        to_delete = timestamp_dirs[:-1]
        
        logging.info(f"Dataset: {dataset}")
        logging.info(f"  Keeping: {latest_run}")
        
        for d in to_delete:
            full_path = os.path.join(dataset_path, d)
            logging.info(f"  Deleting: {d}")
            try:
                shutil.rmtree(full_path)
            except Exception as e:
                logging.error(f"Failed to delete {d}: {e}")
                
        # Handle backups and other folders
        others = [d for d in subdirs if d not in timestamp_dirs]
        for d in others:
            if d == "merged_ligands_backup":
                full_path = os.path.join(dataset_path, d)
                logging.info(f"  Deleting backup: {d}")
                shutil.rmtree(full_path)
            elif d == "latest":
                # 'latest' might be a symlink or dir. 
                # If symlink, we can keep it or delete it.
                # If dir, we probably should keep it? 
                # But if we have timestamp dirs, 'latest' is likely redundant or a link.
                # Let's check if it points to the one we kept.
                full_path = os.path.join(dataset_path, d)
                if os.path.islink(full_path):
                     target = os.readlink(full_path)
                     logging.info(f"  latest points to {target}")
                     # If points to deleted, delete link?
                     # Just delete 'latest' link/dir to be safe, run_benchmark will recreate if needed.
                     # But we are not running benchmark anymore. 
                     # Let's keep 'latest' if it exists, assume it points to valid.
                     pass
                else:
                     # It's a real directory. 
                     # If we have timestamp dirs, this is likely an old run named 'latest' or 
                     # the current run if timestamps failed?
                     # Given the mess, let's look inside.
                     # Safest: If we kept a timestamp dir, we rely on that.
                     pass
            elif d.startswith("work"):
                 # Legacy?
                 pass
                 
if __name__ == "__main__":
    cleanup_outputs()
