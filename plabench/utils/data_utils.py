
import os
import shutil
import logging
from pathlib import Path
import fnmatch

def prepare_smiles_files(source_dir, target_dir, pattern='*'):
    """
    Copies .smi and .tsv files from source_dir to target_dir.
    Renames .tsv files to .smi during copy.
    
    Args:
        source_dir (str/Path): Source directory containing .smi/.tsv files.
        target_dir (str/Path): Destination directory.
        pattern (str): Glob pattern to filter files (default: '*').
        
    Returns:
        int: Number of files copied.
    """
    source_path = Path(source_dir)
    count = 0
    
    if not source_path.exists():
        logging.warning(f"Source directory {source_dir} does not exist.")
        return 0
        
    for ext in ['*.smi', '*.tsv']:
        for f in source_path.glob(ext):
            # Check pattern match on filename or stem
            if fnmatch.fnmatch(f.stem, pattern) or fnmatch.fnmatch(f.name, pattern):
                # Always save as .smi
                dst = os.path.join(target_dir, f.stem + ".smi")
                
                # Check if destination exists to avoid unnecessary copy? 
                # For "resume", we might want to skip copy if exists?
                # But copy is fast. Let's just copy.
                shutil.copy(f, dst)
                count += 1
                
    return count
