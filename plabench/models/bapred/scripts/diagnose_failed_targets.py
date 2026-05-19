#!/usr/bin/env python3
"""
Diagnostic script to investigate why 24 targets failed in BAPred inference.
Tests RDKit parsing of protein PDB and ligand SDF files.
"""

import os
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pandas as pd

# Failed targets list
FAILED_TARGETS = [
    "L3009", "L3019", "L3026", "L3037", "L3039", "L3047", "L3061", "L3064", 
    "L3066", "L3073", "L3076", "L3077", "L3081", "L3095", "L3096", "L3115", 
    "L3126", "L3127", "L3134", "L3151", "L3154", "L3156", "L3159", "L3190"
]

DATA_DIR = Path("data/casp16_data/stage2_input/L3000_prepared")

def check_ligand_sdf(target_id):
    """Check if ligand SDF can be parsed by RDKit."""
    sdf_path = DATA_DIR / target_id / "ligand.sdf"
    
    if not sdf_path.exists():
        return {"status": "MISSING", "error": "SDF file not found"}
    
    try:
        # Try to read with sanitize=True (strict)
        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=False)
        mol = next(iter(suppl))
        
        if mol is None:
            # Try again with sanitize=False
            suppl_no_san = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
            mol_no_san = next(iter(suppl_no_san))
            
            if mol_no_san is None:
                return {"status": "PARSE_FAILED", "error": "RDKit cannot parse SDF (even with sanitize=False)"}
            else:
                return {
                    "status": "SANITIZE_FAILED", 
                    "error": "Molecule parsed but sanitization failed",
                    "num_atoms": mol_no_san.GetNumAtoms(),
                    "num_bonds": mol_no_san.GetNumBonds()
                }
        
        # Successfully parsed
        return {
            "status": "OK",
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "mol_weight": Descriptors.MolWt(mol),
            "num_rings": Descriptors.RingCount(mol)
        }
        
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

def check_protein_pdb(target_id):
    """Check if protein PDB can be parsed."""
    pdb_path = DATA_DIR / target_id / "protein.pdb"
    
    if not pdb_path.exists():
        return {"status": "MISSING", "error": "PDB file not found"}
    
    try:
        # Check file size
        file_size = pdb_path.stat().st_size
        
        # Count atoms in PDB
        with open(pdb_path, 'r') as f:
            atom_lines = [line for line in f if line.startswith('ATOM') or line.startswith('HETATM')]
            num_atoms = len(atom_lines)
        
        # Try to parse with RDKit (optional, as BAPred may use different parser)
        try:
            mol = Chem.MolFromPDBFile(str(pdb_path), sanitize=False, removeHs=False)
            if mol is None:
                rdkit_status = "PARSE_FAILED"
            else:
                rdkit_status = "OK"
        except Exception as e:
            rdkit_status = f"ERROR: {str(e)}"
        
        return {
            "status": "OK",
            "file_size_kb": file_size / 1024,
            "num_atoms": num_atoms,
            "rdkit_parse": rdkit_status
        }
        
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

def main():
    print("=" * 80)
    print("BAPred Failed Targets Diagnostic Report")
    print("=" * 80)
    print(f"\nTesting {len(FAILED_TARGETS)} failed targets...\n")
    
    results = []
    
    for target_id in FAILED_TARGETS:
        print(f"\n{'='*60}")
        print(f"Target: {target_id}")
        print(f"{'='*60}")
        
        # Check ligand SDF
        ligand_result = check_ligand_sdf(target_id)
        print(f"\n📦 Ligand SDF:")
        for key, value in ligand_result.items():
            print(f"  {key}: {value}")
        
        # Check protein PDB
        protein_result = check_protein_pdb(target_id)
        print(f"\n🧬 Protein PDB:")
        for key, value in protein_result.items():
            print(f"  {key}: {value}")
        
        # Aggregate results
        results.append({
            "target_id": target_id,
            "ligand_status": ligand_result.get("status", "UNKNOWN"),
            "ligand_error": ligand_result.get("error", ""),
            "ligand_atoms": ligand_result.get("num_atoms", "N/A"),
            "protein_status": protein_result.get("status", "UNKNOWN"),
            "protein_atoms": protein_result.get("num_atoms", "N/A"),
            "protein_rdkit": protein_result.get("rdkit_parse", "N/A")
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    
    # Ligand status breakdown
    print("\n📦 Ligand SDF Status:")
    print(df['ligand_status'].value_counts())
    
    # Protein status breakdown
    print("\n🧬 Protein PDB Status:")
    print(df['protein_status'].value_counts())
    
    # Detailed table
    print("\n📋 Detailed Results:")
    print(df.to_string(index=False))
    
    # Save to CSV
    output_file = "outputs/failed_targets_diagnostic.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Identify problematic targets
    problematic = df[df['ligand_status'] != 'OK']
    if len(problematic) > 0:
        print(f"\n⚠️  {len(problematic)} targets with ligand parsing issues:")
        print(problematic[['target_id', 'ligand_status', 'ligand_error']].to_string(index=False))
    else:
        print("\n✅ All ligand SDF files can be parsed by RDKit!")
        print("   → Failures are likely due to BAPred-specific issues, not RDKit parsing.")

if __name__ == "__main__":
    main()
