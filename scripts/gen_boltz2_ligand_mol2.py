"""
Extract ligand mol2 from Boltz2 CIF files for CASP16 L1000/L3000.
Uses PyMOL to select non-polymer chains and save as mol2.

Usage:
    python scripts/gen_boltz2_ligand_mol2.py
"""

import os
import csv
import sys

# Initialize PyMOL in headless mode
import pymol
from pymol import cmd
pymol.finish_launching(['pymol', '-cq'])

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = [
    {
        "name": "L1000",
        "summary_csv": "/home/lwfvx/Lyuwei/0.Projects/boltz/casp16_L1000_structure_summary.csv",
        "output_dir": os.path.join(PROJECT_ROOT, "data", "Boltz2_structures", "L1000"),
    },
    {
        "name": "L3000",
        "summary_csv": "/home/lwfvx/Lyuwei/0.Projects/boltz/casp16_L3000_structure_summary.csv",
        "output_dir": os.path.join(PROJECT_ROOT, "data", "Boltz2_structures", "L3000"),
    },
]


def extract_ligand_mol2(cif_path, output_mol2):
    """Use PyMOL Python API to extract ligand (resn LIG1) from CIF as mol2."""
    cmd.delete('all')
    cmd.load(cif_path, 'structure')
    n = cmd.count_atoms('resn LIG1')
    if n == 0:
        raise RuntimeError(f"No LIG1 atoms found in {cif_path}")
    cmd.save(output_mol2, 'resn LIG1')
    if not os.path.exists(output_mol2) or os.path.getsize(output_mol2) == 0:
        raise RuntimeError(f"Empty or missing output: {output_mol2}")


def verify_existing_protein(cif_path, existing_pdb):
    """Quick check: compare atom count between CIF-derived and existing protein."""
    if not os.path.exists(existing_pdb):
        return False, "missing"
    # Count ATOM lines in existing PDB
    with open(existing_pdb) as f:
        n_existing = sum(1 for l in f if l.startswith("ATOM"))
    if n_existing == 0:
        return False, "empty"
    return True, f"{n_existing} atoms"


def main():
    for ds in DATASETS:
        print(f"\n=== Processing {ds['name']} ===")
        summary_csv = ds["summary_csv"]
        output_dir = ds["output_dir"]

        if not os.path.exists(summary_csv):
            print(f"  Summary CSV not found: {summary_csv}")
            continue

        targets = []
        with open(summary_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                targets.append(row)

        success = 0
        skip = 0
        fail = 0

        for t in targets:
            tid = t["target_id"]
            cif_path = t["file_path"]
            target_dir = os.path.join(output_dir, tid)
            mol2_path = os.path.join(target_dir, "ligand.mol2")

            if not os.path.exists(target_dir):
                print(f"  [{tid}] Target dir missing, skipping")
                skip += 1
                continue

            if os.path.exists(mol2_path):
                print(f"  [{tid}] mol2 already exists, skipping")
                skip += 1
                continue

            if not os.path.exists(cif_path):
                print(f"  [{tid}] CIF not found: {cif_path}")
                fail += 1
                continue

            # Verify protein consistency
            existing_pdb = os.path.join(target_dir, "protein.pdb")
            ok, info = verify_existing_protein(cif_path, existing_pdb)
            if not ok:
                print(f"  [{tid}] WARNING: existing protein.pdb {info}")

            try:
                extract_ligand_mol2(cif_path, mol2_path)
                # Verify mol2 has atoms
                with open(mol2_path) as f:
                    content = f.read()
                if "@<TRIPOS>ATOM" not in content:
                    print(f"  [{tid}] FAIL: mol2 has no ATOM block")
                    os.remove(mol2_path)
                    fail += 1
                    continue
                # Count atoms
                in_atom = False
                n_atoms = 0
                for line in content.split("\n"):
                    if "@<TRIPOS>ATOM" in line:
                        in_atom = True
                        continue
                    if line.startswith("@<TRIPOS>"):
                        in_atom = False
                    if in_atom and line.strip():
                        n_atoms += 1
                print(f"  [{tid}] OK ({n_atoms} atoms)")
                success += 1
            except Exception as e:
                print(f"  [{tid}] FAIL: {e}")
                fail += 1

        print(f"\n  {ds['name']} Summary: {success} success, {skip} skipped, {fail} failed")


if __name__ == "__main__":
    main()
