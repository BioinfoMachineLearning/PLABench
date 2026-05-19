"""
Convert co-folding CIF files (AF3, SeedFold, Protenix, Boltz2, etc.) to
protein.pdb + ligand.mol2 + ligand.sdf for PLABench.

Pipeline per target:
  1. PyMOL: load CIF → save polymer as protein.pdb
  2. PyMOL: auto-detect ligand residue name → save as ligand.mol2
  3. Ligand SDF generation (two strategies):
     a. Primary: gemmi CIF → PDB block → RDKit MolFromPDBBlock → AssignBondOrdersFromTemplate(SMILES)
        Requires --smiles-csv with (target, smiles) mapping.
        Handles SF5/halogen groups via halogen-halogen bond removal + DetermineConnectivity fallback.
     b. Fallback: obabel ligand.mol2 → ligand.sdf (legacy, no --gen3d)
  4. RDKit: verify SDF sanitize

Input:  CSV with columns (target/Target, cif_path/best_cif_path/Top1_CIF_Path)
Output: {output-dir}/{TARGET_ID}/protein.pdb, ligand.mol2, ligand.sdf

Usage:
    python scripts/gen_cofold_structures.py --csv PATH --output-dir PATH [--smiles-csv PATH]
"""

import os
import sys
import csv
import subprocess
import argparse

# Initialize PyMOL in headless mode
import pymol
from pymol import cmd
pymol.finish_launching(['pymol', '-cq'])

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CSV = None  # must be specified
DEFAULT_OUTPUT = None  # must be specified

# Column name for CIF path varies across CSVs
CIF_PATH_COLUMNS = ["cif_path", "best_cif_path", "Top1_CIF_Path", "best_model_path"]

# Column name for target ID varies across CSVs
TARGET_ID_COLUMNS = ["target", "Target"]

# Known ligand residue names by source, tried in order
KNOWN_LIGAND_RESN = ["LIG_B", "LIG1", "LIG", "l01"]


def extract_protein_pdb(cif_path, output_pdb):
    """Extract polymer (protein) from CIF and save as PDB."""
    cmd.delete('all')
    cmd.load(cif_path, 'structure')
    n = cmd.count_atoms('polymer')
    if n == 0:
        raise RuntimeError(f"No polymer atoms found in {cif_path}")
    cmd.save(output_pdb, 'polymer')
    if not os.path.exists(output_pdb) or os.path.getsize(output_pdb) == 0:
        raise RuntimeError(f"Empty protein PDB: {output_pdb}")
    return n


def detect_ligand_resn(cif_path):
    """Try known ligand residue names in order; return the first one with atoms."""
    cmd.delete('all')
    cmd.load(cif_path, 'structure')
    for resn in KNOWN_LIGAND_RESN:
        if cmd.count_atoms(f'resn {resn}') > 0:
            return resn
    raise RuntimeError(
        f"No known ligand residue found in {cif_path} "
        f"(tried: {KNOWN_LIGAND_RESN})"
    )


def extract_ligand_mol2(cif_path, output_mol2):
    """Extract ligand from CIF (auto-detect residue name) and save as MOL2."""
    resn = detect_ligand_resn(cif_path)
    # structure is already loaded by detect_ligand_resn
    sel = f'resn {resn}'
    n = cmd.count_atoms(sel)
    cmd.save(output_mol2, sel)
    if not os.path.exists(output_mol2) or os.path.getsize(output_mol2) == 0:
        raise RuntimeError(f"Empty MOL2: {output_mol2}")
    # Verify ATOM block exists
    with open(output_mol2) as f:
        content = f.read()
    if "@<TRIPOS>ATOM" not in content:
        os.remove(output_mol2)
        raise RuntimeError(f"MOL2 has no ATOM block: {output_mol2}")
    return n, resn


def mol2_to_sdf(mol2_path, sdf_path):
    """Convert MOL2 to SDF using OpenBabel (no --gen3d). Legacy fallback."""
    result = subprocess.run(
        ['obabel', mol2_path, '-O', sdf_path],
        capture_output=True, text=True
    )
    if not os.path.exists(sdf_path) or os.path.getsize(sdf_path) == 0:
        raise RuntimeError(f"obabel conversion failed: {result.stderr.strip()}")


def cif_to_sdf_with_template(cif_path, smiles, sdf_path):
    """Generate SDF from CIF using gemmi + SMILES template for correct bond orders.

    Strategy 1 (primary): gemmi → PDB block → MolFromPDBBlock(proximityBonding) →
        AssignBondOrdersFromTemplate(SMILES)
    Strategy 2 (fallback for SF5 etc.): gemmi → raw atoms → DetermineConnectivity →
        remove halogen-halogen artifacts → AssignBondOrdersFromTemplate
    """
    import gemmi
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise RuntimeError(f"Invalid SMILES: {smiles}")

    # Parse ligand atoms from CIF
    st = gemmi.read_structure(cif_path)
    model = st[0]
    lig_atoms = []
    for chain in model:
        for res in chain:
            if res.name in KNOWN_LIGAND_RESN:
                for atom in res:
                    if not atom.is_hydrogen():
                        lig_atoms.append(atom)

    if len(lig_atoms) != template.GetNumAtoms():
        raise RuntimeError(
            f"Atom count mismatch: CIF has {len(lig_atoms)} heavy atoms, "
            f"SMILES has {template.GetNumAtoms()}")

    # Strategy 1: PDB block route (preserves implicit H correctly)
    pdb_lines = []
    for i, atom in enumerate(lig_atoms):
        name = atom.name
        if len(name) < 4:
            name = f" {name:<3s}"
        line = (f"HETATM{i+1:5d} {name} LIG B   1    "
                f"{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}"
                f"  1.00  0.00          {atom.element.name:>2s}")
        pdb_lines.append(line)
    pdb_lines.append("END")
    pdb_block = "\n".join(pdb_lines)

    try:
        rd_mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False,
                                       proximityBonding=True, removeHs=True)
        if rd_mol and rd_mol.GetNumAtoms() == template.GetNumAtoms():
            fixed = AllChem.AssignBondOrdersFromTemplate(template, rd_mol)
            Chem.SanitizeMol(fixed)
            Chem.MolToMolFile(fixed, sdf_path)
            return fixed.GetNumAtoms(), "gemmi+pdb"
    except Exception:
        pass  # Fall through to strategy 2

    # Strategy 2: PDB block but with halogen-halogen bond removal
    # MolFromPDBBlock with proximityBonding may create wrong halogen bonds
    # (e.g., F-F bonds in SF5 groups). Remove them, then re-apply template.
    try:
        rd_mol = Chem.RWMol(Chem.MolFromPDBBlock(
            pdb_block, sanitize=False, proximityBonding=True, removeHs=True))
        if rd_mol and rd_mol.GetNumAtoms() == template.GetNumAtoms():
            halogens = {"F", "Cl", "Br", "I"}
            bonds_to_remove = []
            for bond in rd_mol.GetBonds():
                s1 = bond.GetBeginAtom().GetSymbol()
                s2 = bond.GetEndAtom().GetSymbol()
                if s1 in halogens and s2 in halogens:
                    bonds_to_remove.append(
                        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            for a, b in bonds_to_remove:
                rd_mol.RemoveBond(a, b)

            fixed = AllChem.AssignBondOrdersFromTemplate(template, rd_mol.GetMol())
            Chem.SanitizeMol(fixed)
            Chem.MolToMolFile(fixed, sdf_path)
            return fixed.GetNumAtoms(), "gemmi+pdb+halfix"
    except Exception:
        pass

    raise RuntimeError("Both gemmi SDF strategies failed")


def verify_sdf(sdf_path):
    """Verify SDF passes RDKit sanitize."""
    from rdkit import Chem
    suppl = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
    mol = next(suppl, None)
    if mol is None:
        raise RuntimeError(f"RDKit sanitize failed for {sdf_path}")
    return mol.GetNumAtoms()


def main():
    parser = argparse.ArgumentParser(description="Convert co-folding CIF to PDB + MOL2 + SDF")
    parser.add_argument('--csv', required=True, help="CSV file with target and CIF path columns")
    parser.add_argument('--output-dir', required=True, help="Output directory")
    parser.add_argument('--smiles-csv', default=None,
                        help="CSV with (target_id, smiles) for template-based SDF generation. "
                             "Auto-detects column names. If not provided, falls back to obabel.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read CSV (auto-detect target ID and CIF path column names)
    targets = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        cif_col = None
        for col in CIF_PATH_COLUMNS:
            if col in reader.fieldnames:
                cif_col = col
                break
        if cif_col is None:
            print(f"ERROR: CSV must have one of {CIF_PATH_COLUMNS} columns, found: {reader.fieldnames}")
            sys.exit(1)
        tid_col = None
        for col in TARGET_ID_COLUMNS:
            if col in reader.fieldnames:
                tid_col = col
                break
        if tid_col is None:
            print(f"ERROR: CSV must have one of {TARGET_ID_COLUMNS} columns, found: {reader.fieldnames}")
            sys.exit(1)
        for row in reader:
            row["cif_path"] = row[cif_col]   # normalize CIF path column
            row["target"] = row[tid_col]      # normalize target ID column
            targets.append(row)

    # Load SMILES mapping if provided
    smiles_map = {}
    if args.smiles_csv:
        with open(args.smiles_csv) as f:
            reader = csv.DictReader(f)
            # Auto-detect SMILES column
            smi_col = None
            for col in ["smiles", "SMILES", "compound_iso_smiles"]:
                if col in reader.fieldnames:
                    smi_col = col
                    break
            # Auto-detect target ID column
            sid_col = None
            for col in TARGET_ID_COLUMNS + ["compound_id", "Target ID"]:
                if col in reader.fieldnames:
                    sid_col = col
                    break
            if smi_col and sid_col:
                for row in reader:
                    smiles_map[row[sid_col]] = row[smi_col]
        print(f"Loaded {len(smiles_map)} SMILES from {args.smiles_csv}")
    use_template = len(smiles_map) > 0

    print(f"Processing {len(targets)} targets from {args.csv}")
    print(f"Output: {args.output_dir}")
    if use_template:
        print(f"SMILES source: {args.smiles_csv} ({len(smiles_map)} entries)")
    else:
        print(f"SMILES source: per-target ligand.smi files")
    print()

    success, fail = 0, 0

    for t in targets:
        tid = t["target"]
        cif_path = t["cif_path"]
        target_dir = os.path.join(args.output_dir, tid)
        os.makedirs(target_dir, exist_ok=True)

        pdb_path = os.path.join(target_dir, "protein.pdb")
        mol2_path = os.path.join(target_dir, "ligand.mol2")
        sdf_path = os.path.join(target_dir, "ligand.sdf")

        if not os.path.exists(cif_path):
            print(f"  [{tid}] SKIP: CIF not found: {cif_path}")
            fail += 1
            continue

        try:
            # 1. Extract protein PDB
            n_prot = extract_protein_pdb(cif_path, pdb_path)

            # 2. Extract ligand MOL2 (auto-detect residue name)
            n_lig, resn = extract_ligand_mol2(cif_path, mol2_path)

            # 3. Generate ligand SDF (requires SMILES for correct bond orders)
            # Resolve SMILES: --smiles-csv > {target_dir}/ligand.smi
            smiles = smiles_map.get(tid)
            if not smiles:
                smi_file = os.path.join(target_dir, "ligand.smi")
                if os.path.exists(smi_file):
                    with open(smi_file) as sf:
                        smiles = sf.read().strip().split()[0]
            if not smiles:
                raise RuntimeError(
                    f"No SMILES found for {tid}. Provide --smiles-csv or "
                    f"place ligand.smi in {target_dir}")

            n_sdf, sdf_method = cif_to_sdf_with_template(
                cif_path, smiles, sdf_path)

            print(f"  [{tid}] OK: protein={n_prot}, ligand={n_lig} (mol2/{resn}), sdf={n_sdf} ({sdf_method})")
            success += 1

        except Exception as e:
            print(f"  [{tid}] FAIL: {e}")
            fail += 1

    print(f"\nSummary: {success} success, {fail} failed out of {len(targets)} targets")


if __name__ == "__main__":
    main()
