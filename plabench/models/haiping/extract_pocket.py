
import sys
import os
import numpy as np
import argparse

def extract_pocket(protein_pdb, ligand_file, output_pdb, cutoff=10.0):
    """
    Extract pocket residues within cutoff distance from ligand.
    
    Args:
        protein_pdb (str): Path to protein PDB file.
        ligand_file (str): Path to ligand file (PDB, SDF, MOL2).
        output_pdb (str): Path to output pocket PDB file.
        cutoff (float): Distance cutoff in Angstroms.
    """
    # Parse ligand coordinates
    ligand_coords = []

    # Check ligand file type
    if ligand_file.endswith('.sdf') or ligand_file.endswith('.mol'):
        # Read SDF/MOL file
        with open(ligand_file, 'r') as f:
            lines = f.readlines()
            # Find line with V2000 (start of molecule)
            atom_count = 0
            atom_line_start = -1
            for i, line in enumerate(lines):
                if 'V2000' in line or 'V3000' in line:
                    # Previous line contains atom and bond counts
                    parts = line.split()
                    try:
                        atom_count = int(parts[0])
                        atom_line_start = i + 1
                        break
                    except ValueError:
                        continue

            # Read atom coordinates
            if atom_line_start >= 0 and atom_count > 0:
                for i in range(atom_line_start, min(atom_line_start + atom_count, len(lines))):
                    line = lines[i]
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            z = float(parts[2])
                            ligand_coords.append([x, y, z])
                        except (ValueError, IndexError):
                            continue
                            
    elif ligand_file.endswith('.mol2'):
        # Read MOL2 file
        with open(ligand_file, 'r') as f:
            lines = f.readlines()
            in_atom_block = False
            for line in lines:
                if '@<TRIPOS>ATOM' in line:
                    in_atom_block = True
                    continue
                if '@<TRIPOS>BOND' in line:
                    break
                if in_atom_block:
                    parts = line.split()
                    if len(parts) >= 6: # Need at least X, Y, Z (indices 2, 3, 4 typically)
                        try:
                            # format: atom_id atom_name x y z atom_type ...
                            x = float(parts[2])
                            y = float(parts[3])
                            z = float(parts[4])
                            ligand_coords.append([x, y, z])
                        except (ValueError, IndexError):
                            continue
                            
    elif ligand_file.endswith('.pdb'):
        # Read PDB file
        with open(ligand_file, 'r') as f:
            for line in f:
                if line.startswith('HETATM') or line.startswith('ATOM'):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        ligand_coords.append([x, y, z])
                    except (ValueError, IndexError):
                        continue

    if not ligand_coords:
        print(f"Error: No ligand coordinates found in {ligand_file}")
        return False

    ligand_coords = np.array(ligand_coords)

    # Parse protein and extract pocket
    pocket_residues = set()
    protein_lines = []

    try:
        with open(protein_pdb, 'r') as f:
            for line in f:
                protein_lines.append(line)
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])

                        # Calculate minimum distance to ligand
                        atom_coord = np.array([x, y, z])
                        
                        # Vectorized distance calculation is faster but memory intensive for huge proteins vs large ligands
                        # Using loop here for simplicity as protein size is manageable
                        distances = np.linalg.norm(ligand_coords - atom_coord, axis=1)
                        min_dist = np.min(distances)

                        # If within cutoff, add residue to pocket
                        if min_dist <= cutoff:
                            chain = line[21]
                            res_num = line[22:26].strip()
                            # res_insert = line[26] # Insert code
                            res_name = line[17:20].strip()
                            # Use chain + res_num as unique identifier (insensitive to insert code for now unless needed)
                            pocket_residues.add((chain, res_num, res_name))
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        print(f"Error: Protein file {protein_pdb} not found")
        return False

    # Write pocket PDB
    try:
        with open(output_pdb, 'w') as out:
            for line in protein_lines:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    chain = line[21]
                    res_num = line[22:26].strip()
                    res_name = line[17:20].strip()
                    if (chain, res_num, res_name) in pocket_residues:
                        out.write(line)
                elif line.startswith('END'):
                    out.write(line)
                    break
                elif not (line.startswith('ATOM') or line.startswith('HETATM')):
                    # Keep header lines, CRYST1, etc.
                    out.write(line)
    except Exception as e:
        print(f"Error writing output: {e}")
        return False
        
    print(f"Extracted {len(pocket_residues)} unique residues to {output_pdb}")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract binding pocket from protein structure')
    parser.add_argument('protein', help='Input protein PDB file')
    parser.add_argument('ligand', help='Input ligand file (SDF/MOL2/PDB)')
    parser.add_argument('-o', '--output', required=True, help='Output pocket PDB file')
    parser.add_argument('-c', '--cutoff', type=float, default=10.0,
                        help='Distance cutoff in Angstroms (default: 10.0)')

    args = parser.parse_args()

    if not os.path.exists(args.protein):
        print(f"Error: Protein file {args.protein} not found")
        sys.exit(1)

    if not os.path.exists(args.ligand):
        print(f"Error: Ligand file {args.ligand} not found")
        sys.exit(1)

    success = extract_pocket(args.protein, args.ligand, args.output, args.cutoff)
    if not success:
        sys.exit(1)
