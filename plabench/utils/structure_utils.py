import os
import subprocess
from typing import Optional

def convert_pdb_to_mol2(pdb_path: str, mol2_path: str, pymol_cmd: str = "pymol") -> bool:
    """
    Converts a PDB file to MOL2 format using PyMOL.
    
    Args:
        pdb_path: Path to the input PDB file.
        mol2_path: Path to the output MOL2 file.
        pymol_cmd: Command to run PyMOL (default: "pymol").
        
    Returns:
        True if conversion was successful, False otherwise.
    """
    if not os.path.exists(pdb_path):
        print(f"Error: Input file {pdb_path} does not exist.")
        return False
        
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(mol2_path)), exist_ok=True)
    
    # PyMOL command script: load pdb, save mol2, quit
    # We use -c (command line only), -q (quiet)
    # The script loads the pdb and saves it as mol2
    
    # Escape paths for safety (though simplified here)
    pdb_abs = os.path.abspath(pdb_path)
    mol2_abs = os.path.abspath(mol2_path)
    
    script = f"load {pdb_abs}; save {mol2_abs}; quit"
    
    cmd = [pymol_cmd, "-c", "-q", "-d", script]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(mol2_path):
            return True
        else:
            print(f"Error: PyMOL ran but {mol2_path} was not created.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error running PyMOL: {e}")
        print(f"Stderr: {e.stderr.decode('utf-8')}")
        return False
    except FileNotFoundError:
        print(f"Error: PyMOL command '{pymol_cmd}' not found.")
        return False

def merge_ligands_pdb(ligand_files: list, output_file: str) -> bool:
    """
    Merges multiple PDB ligand files into a single PDB file.
    
    Args:
        ligand_files: List of paths to PDB files to merge.
        output_file: Path to the output merged PDB file.
        
    Returns:
        True if successful, False otherwise.
    """
    if not ligand_files:
        return False
        
    try:
        with open(output_file, 'w') as outfile:
            for i, fpath in enumerate(ligand_files):
                if not os.path.exists(fpath):
                    print(f"Warning: Ligand file {fpath} not found, skipping.")
                    continue
                    
                with open(fpath, 'r') as infile:
                    for line in infile:
                        # Skip END/ENDMDL/MASTER/CONECT lines to avoid premature termination in some viewers
                        # but keep ATOM/HETATM/TER
                        if line.startswith(('END', 'MASTER', 'CONECT')):
                            continue
                        outfile.write(line)
                
                # Add a TER between ligands if not the last one
                if i < len(ligand_files) - 1:
                    outfile.write("TER\n")
            
            outfile.write("END\n")
        return True
    except Exception as e:
        print(f"Error merging ligands: {e}")
        return False

