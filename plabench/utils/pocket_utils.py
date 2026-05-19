"""
Pocket extraction utility following PDBbind standard:
- Heavy-atom distance ≤ cutoff (default 10Å)
- Retain entire residue if any heavy atom qualifies
- Only keep standard amino acid residues
"""

import os
import numpy as np

STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL',
    # Treat MSE (selenomethionine) as MET
    'MSE',
}


def _is_heavy_atom_pdb(line):
    """Check if a PDB ATOM/HETATM line is a heavy atom (not hydrogen)."""
    element = line[76:78].strip()
    if element:
        return element != 'H'
    # Fallback: parse atom name
    atom_name = line[12:16].strip()
    return not atom_name.startswith('H')


def _parse_ligand_heavy_coords(ligand_file):
    """Parse heavy atom coordinates from ligand file (mol2/pdb/sdf)."""
    coords = []
    ext = os.path.splitext(ligand_file)[1].lower()

    if ext == '.mol2':
        with open(ligand_file, 'r') as f:
            in_atom = False
            for line in f:
                if '@<TRIPOS>ATOM' in line:
                    in_atom = True
                    continue
                if line.startswith('@<TRIPOS>'):
                    if in_atom:
                        break
                    continue
                if in_atom:
                    parts = line.split()
                    if len(parts) >= 6:
                        # atom_type is parts[5], e.g. "C.3", "N.am", "H"
                        atom_type = parts[5].split('.')[0]
                        if atom_type == 'H':
                            continue
                        try:
                            coords.append([float(parts[2]),
                                           float(parts[3]),
                                           float(parts[4])])
                        except ValueError:
                            continue

    elif ext == '.pdb':
        with open(ligand_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    if not _is_heavy_atom_pdb(line):
                        continue
                    try:
                        coords.append([float(line[30:38]),
                                       float(line[38:46]),
                                       float(line[46:54])])
                    except ValueError:
                        continue

    elif ext == '.sdf':
        with open(ligand_file, 'r') as f:
            lines = f.readlines()
        atom_count = 0
        atom_start = -1
        for i, line in enumerate(lines):
            if 'V2000' in line:
                parts = line.split()
                try:
                    atom_count = int(parts[0])
                    atom_start = i + 1
                except (ValueError, IndexError):
                    pass
                break
        if atom_start >= 0:
            for i in range(atom_start, min(atom_start + atom_count, len(lines))):
                parts = lines[i].split()
                if len(parts) >= 4:
                    # element symbol is typically parts[3]
                    elem = parts[3] if len(parts) > 3 else ''
                    if elem == 'H':
                        continue
                    try:
                        coords.append([float(parts[0]),
                                       float(parts[1]),
                                       float(parts[2])])
                    except ValueError:
                        continue

    if not coords:
        raise ValueError(f"No heavy atom coordinates found in {ligand_file}")
    return np.array(coords)


def extract_pocket(protein_pdb, ligand_file, output_pdb, cutoff=10.0):
    """Extract pocket from protein PDB following PDBbind standard.

    Rules:
    - Use heavy atoms only (exclude H) for distance calculation
    - Residue qualifies if any of its heavy atoms is within cutoff of
      any ligand heavy atom
    - Retain the entire qualifying residue (all atoms including H)
    - Only keep standard amino acid residues (+ MSE treated as MET)

    Args:
        protein_pdb: Path to full protein PDB
        ligand_file: Path to ligand file (mol2/pdb/sdf)
        output_pdb: Path to write pocket PDB
        cutoff: Distance cutoff in Angstroms (default 10.0)

    Returns:
        int: Number of pocket residues extracted
    """
    lig_coords = _parse_ligand_heavy_coords(ligand_file)

    # Pass 1: identify qualifying residues
    pocket_residue_keys = set()
    with open(protein_pdb, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            res_name = line[17:20].strip()
            if res_name not in STANDARD_AA:
                continue
            if not _is_heavy_atom_pdb(line):
                continue
            try:
                coord = np.array([float(line[30:38]),
                                  float(line[38:46]),
                                  float(line[46:54])])
            except ValueError:
                continue
            dists = np.linalg.norm(lig_coords - coord, axis=1)
            if np.min(dists) <= cutoff:
                chain = line[21]
                res_seq = line[22:27]  # resSeq + iCode
                pocket_residue_keys.add((chain, res_seq, res_name))

    # Pass 2: write all atoms of qualifying residues
    with open(protein_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM'):
                res_name = line[17:20].strip()
                chain = line[21]
                res_seq = line[22:27]
                if (chain, res_seq, res_name) in pocket_residue_keys:
                    # Rewrite MSE as MET
                    if res_name == 'MSE':
                        line = line[:17] + 'MET' + line[20:]
                    f_out.write(line)
            elif line.startswith('END'):
                f_out.write(line)
                break
        else:
            f_out.write('END\n')

    return len(pocket_residue_keys)
