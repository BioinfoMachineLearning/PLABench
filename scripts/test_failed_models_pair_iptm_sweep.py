#!/usr/bin/env python3
"""For each of the 21 failed AF3 cases in af3_chembl35_failures.csv, try all
10 models (seed-{1,2} × sample-{0..4}) in pair_iptm-descending order to find
one whose CIF can be converted to a valid SDF.

Conversion strategies (per model):
  1. gemmi atoms → PDB block → MolFromPDBBlock(proximityBonding) →
     AssignBondOrdersFromTemplate → Sanitize
  2. Same as (1), but remove halogen-halogen bonds first
  3. Skip proximityBonding entirely: build mol from SMILES template, embed a
     conformer, replace its 3D coords with the CIF coords using atom-name match
     fallback to MCS

For each compound, reports the first successful (strategy, model) or marks it
as completely failed.

Outputs:
  - Per-compound summary printed to stdout
  - Final list of compounds that failed across all 10 models
  - Optional CSV at --output: compound_id,winning_model,winning_strategy,winning_pair_iptm
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import tempfile
from typing import List, Optional, Tuple

import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS

KNOWN_LIGAND_RESN = {"LIG", "LIG_B", "UNL", "UNK", "DRG", "INH", "SUB", "X", "HET"}


# ── CIF → ligand atoms ───────────────────────────────────────────────────────
def collect_ligand_atoms(cif_path: str) -> List[gemmi.Atom]:
    st = gemmi.read_structure(cif_path)
    atoms = []
    for chain in st[0]:
        for res in chain:
            if res.name in KNOWN_LIGAND_RESN:
                for atom in res:
                    if not atom.is_hydrogen():
                        atoms.append(atom)
    return atoms


def atoms_to_pdb_block(lig_atoms: List[gemmi.Atom]) -> str:
    """Build a PDB HETATM block from ligand atoms.
    Atom name formatting respects element width (Cl/Br need no leading space).
    """
    lines = []
    for i, atom in enumerate(lig_atoms):
        name = atom.name
        elem = atom.element.name
        if len(elem) == 2:
            # Two-letter element: name fills cols 13-16 left-aligned
            atom_name_field = f"{name:<4s}"
        else:
            # Single-letter element: leading space, then 3-char name
            atom_name_field = f" {name:<3s}"
        line = (f"HETATM{i+1:5d} {atom_name_field} LIG B   1    "
                f"{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}"
                f"  1.00  0.00          {elem:>2s}")
        lines.append(line)
    lines.append("END")
    return "\n".join(lines)


# ── Strategies ───────────────────────────────────────────────────────────────
def strategy_pdb(pdb_block: str, template: Chem.Mol) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """Strategy 1: PDB+proximityBonding+AssignBondOrders."""
    try:
        rd = Chem.MolFromPDBBlock(pdb_block, sanitize=False,
                                  proximityBonding=True, removeHs=True)
        if rd is None:
            return None, "MolFromPDBBlock returned None"
        if rd.GetNumAtoms() != template.GetNumAtoms():
            return None, f"PDB atoms {rd.GetNumAtoms()} != template {template.GetNumAtoms()}"
        fixed = AllChem.AssignBondOrdersFromTemplate(template, rd)
        Chem.SanitizeMol(fixed)
        return fixed, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def strategy_pdb_halfix(pdb_block: str, template: Chem.Mol) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """Strategy 2: same but remove halogen-halogen bonds first."""
    try:
        raw = Chem.MolFromPDBBlock(pdb_block, sanitize=False,
                                   proximityBonding=True, removeHs=True)
        if raw is None:
            return None, "MolFromPDBBlock returned None"
        rd = Chem.RWMol(raw)
        halogens = {"F", "Cl", "Br", "I"}
        removed = 0
        for bond in list(rd.GetBonds()):
            s1 = bond.GetBeginAtom().GetSymbol()
            s2 = bond.GetEndAtom().GetSymbol()
            if s1 in halogens and s2 in halogens:
                rd.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                removed += 1
        fixed = AllChem.AssignBondOrdersFromTemplate(template, rd.GetMol())
        Chem.SanitizeMol(fixed)
        return fixed, f"halfix({removed})"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def strategy_template_coords(lig_atoms: List[gemmi.Atom], template: Chem.Mol) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """Strategy 3: skip proximityBonding entirely.
    Embed a conformer for the SMILES template, then overwrite coords using
    a no-bond PDB block + MCS atom mapping.
    """
    try:
        if template.GetNumAtoms() != len(lig_atoms):
            return None, f"template atoms {template.GetNumAtoms()} != lig atoms {len(lig_atoms)}"
        # Build a "loose" mol from atoms only (no bonds) so MCS can match by element + position
        rw = Chem.RWMol()
        for atom in lig_atoms:
            a = Chem.Atom(atom.element.atomic_number)
            rw.AddAtom(a)
        loose = rw.GetMol()
        conf = Chem.Conformer(len(lig_atoms))
        for i, atom in enumerate(lig_atoms):
            conf.SetAtomPosition(i, (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)))
        loose.AddConformer(conf, assignId=True)

        # Build target template with explicit Hs removed (already)
        tmpl = Chem.Mol(template)
        # Embed a 3D conformer just so we have a placeholder; we'll overwrite
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(tmpl, params) < 0:
            # Try without coords - we'll overwrite anyway
            tmpl_conf = Chem.Conformer(tmpl.GetNumAtoms())
            tmpl.AddConformer(tmpl_conf, assignId=True)

        # Match by element-only MCS to find atom mapping
        mcs = rdFMCS.FindMCS(
            [tmpl, loose],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            ringMatchesRingOnly=False,
            completeRingsOnly=False,
            timeout=15,
        )
        if mcs.canceled or mcs.numAtoms < tmpl.GetNumAtoms():
            return None, f"MCS incomplete: {mcs.numAtoms}/{tmpl.GetNumAtoms()}"
        patt = Chem.MolFromSmarts(mcs.smartsString)
        if patt is None:
            return None, "MCS smarts failed to parse"
        match_tmpl = tmpl.GetSubstructMatch(patt)
        match_loose = loose.GetSubstructMatch(patt)
        if not match_tmpl or not match_loose or len(match_tmpl) != len(match_loose):
            return None, "match indices missing"

        # Copy positions
        loose_conf = loose.GetConformer()
        tmpl_conf = tmpl.GetConformer()
        for ti, li in zip(match_tmpl, match_loose):
            tmpl_conf.SetAtomPosition(ti, loose_conf.GetAtomPosition(li))
        Chem.SanitizeMol(tmpl)
        return tmpl, "tmpl+coords"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def try_all_strategies(cif_path: str, template: Chem.Mol):
    """Yield (strategy_name, mol_or_None, error_or_None)."""
    lig_atoms = collect_ligand_atoms(cif_path)
    if not lig_atoms:
        yield "all", None, f"no ligand atoms in {os.path.basename(cif_path)}"
        return
    if len(lig_atoms) != template.GetNumAtoms():
        yield "all", None, (f"atom count mismatch: cif {len(lig_atoms)} != "
                           f"template {template.GetNumAtoms()}")
        return

    pdb_block = atoms_to_pdb_block(lig_atoms)

    mol, err = strategy_pdb(pdb_block, template)
    yield "gemmi+pdb", mol, err
    if mol is not None:
        return
    mol, err = strategy_pdb_halfix(pdb_block, template)
    yield "gemmi+pdb+halfix", mol, err
    if mol is not None:
        return
    mol, err = strategy_template_coords(lig_atoms, template)
    yield "template+coords", mol, err


def verify_sdf(sdf_path: str) -> Tuple[bool, str]:
    suppl = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=True)
    mols = [m for m in suppl if m is not None]
    if not mols:
        return False, "SDMolSupplier returned no valid mol"
    return True, f"{mols[0].GetNumAtoms()} atoms, smi={Chem.MolToSmiles(mols[0])}"


# ── pair_iptm extraction ─────────────────────────────────────────────────────
def extract_pair_iptm_protein_to_lig(summary_path: str) -> Optional[float]:
    """Read AF3 summary_confidences.json and return chain_pair_iptm of the
    first protein chain (A) → first ligand chain (B). For chembl35 cofold the
    layout is always A=protein, B=ligand."""
    try:
        with open(summary_path) as f:
            d = json.load(f)
        m = d.get("chain_pair_iptm") or d.get("pair_chains_iptm")
        if not m:
            return None
        # 2D list format: chain order is [A, B, ...]; A=0, B=1
        if isinstance(m, list) and len(m) >= 2:
            return float(m[0][1])
        # Dict format
        if isinstance(m, dict):
            outer = m.get("A") or m.get("0") or m.get(0)
            if isinstance(outer, dict):
                v = outer.get("B") or outer.get("1") or outer.get(1)
                return float(v) if v is not None else None
        return None
    except Exception:
        return None


# ── Main loop ────────────────────────────────────────────────────────────────
def parse_cif_path(cif_abspath: str):
    """Return (compound_dir, prefix) so other models can be globbed.
    Example: .../seed-1_sample-2/o96028_003_seed-1_sample-2_model.cif
             -> compound_dir = .../chembl35_full
                prefix = o96028_003
    """
    seed_dir = os.path.dirname(cif_abspath)         # .../seed-1_sample-2
    compound_dir = os.path.dirname(seed_dir)        # .../chembl35_full
    fname = os.path.basename(cif_abspath)
    # Strip _seed-X_sample-Y_model.cif
    m = re.match(r"^(.+?)_seed-\d+_sample-\d+_model\.cif$", fname)
    if not m:
        return compound_dir, None
    return compound_dir, m.group(1)


def find_all_models(compound_dir: str, prefix: str) -> List[Tuple[str, str]]:
    """Return list of (cif_path, summary_path) for all seed/sample combos."""
    results = []
    pattern = os.path.join(compound_dir, "seed-*_sample-*",
                           f"{prefix}_seed-*_sample-*_model.cif")
    for cif in sorted(glob.glob(pattern)):
        seed_dir = os.path.dirname(cif)
        base = os.path.basename(cif).replace("_model.cif", "")
        summary = os.path.join(seed_dir, f"{base}_summary_confidences.json")
        results.append((cif, summary))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/bmlfast/Lyuwei/0.Projects/PLABench/tmp/af3_chembl35_failures.csv")
    parser.add_argument("--output", default=None,
                        help="Optional output CSV (default: print only)")
    parser.add_argument("--keep-sdf-dir", default=None,
                        help="If set, save winning SDFs into this dir")
    args = parser.parse_args()

    if args.keep_sdf_dir:
        os.makedirs(args.keep_sdf_dir, exist_ok=True)

    rows = []
    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} failed cases\n")

    output_records = []
    completely_failed = []

    for row in rows:
        compound_id = row["compound_id"]
        smiles = row["smiles"]
        cif_abspath = row["src_cif_abspath"]

        template = Chem.MolFromSmiles(smiles)
        if template is None:
            print(f"[{compound_id}] SKIP: invalid SMILES")
            completely_failed.append((compound_id, "invalid_smiles"))
            continue

        compound_dir, prefix = parse_cif_path(cif_abspath)
        if not prefix:
            print(f"[{compound_id}] SKIP: cannot parse cif path")
            completely_failed.append((compound_id, "bad_cif_path"))
            continue

        models = find_all_models(compound_dir, prefix)
        if not models:
            print(f"[{compound_id}] SKIP: no models found for prefix={prefix}")
            completely_failed.append((compound_id, "no_models"))
            continue

        # Sort by pair_iptm desc
        scored = []
        for cif, summary in models:
            piptm = extract_pair_iptm_protein_to_lig(summary)
            scored.append((piptm if piptm is not None else -1.0, cif, summary))
        scored.sort(reverse=True, key=lambda x: x[0])

        print(f"[{compound_id}] (prefix={prefix}) {len(scored)} models, trying in pair_iptm-desc order")

        winner = None
        per_model_errors = []
        for piptm, cif, _ in scored:
            model_tag = os.path.basename(cif).replace("_model.cif", "")
            attempt_errs = []
            for strategy, mol, err in try_all_strategies(cif, template):
                if mol is not None:
                    # Try writing + verifying
                    fd, sdf_path = tempfile.mkstemp(suffix=".sdf")
                    os.close(fd)
                    try:
                        Chem.MolToMolFile(mol, sdf_path)
                        ok, msg = verify_sdf(sdf_path)
                        if ok:
                            winner = (strategy, model_tag, piptm, sdf_path)
                            print(f"    ✓ {model_tag} (pair_iptm={piptm:.3f}) -> {strategy} | {msg}")
                            break
                        else:
                            attempt_errs.append(f"{strategy}: verify failed: {msg}")
                            os.remove(sdf_path)
                    except Exception as e:
                        attempt_errs.append(f"{strategy}: write error: {e}")
                        if os.path.exists(sdf_path):
                            os.remove(sdf_path)
                else:
                    attempt_errs.append(f"{strategy}: {err}")
            if winner:
                break
            per_model_errors.append((model_tag, piptm, attempt_errs))

        if winner:
            strategy, model_tag, piptm, sdf_path = winner
            if args.keep_sdf_dir:
                dest = os.path.join(args.keep_sdf_dir, f"{compound_id}.sdf")
                os.replace(sdf_path, dest)
                print(f"    saved -> {dest}")
            else:
                os.remove(sdf_path)
            output_records.append({
                "compound_id": compound_id,
                "winning_model": model_tag,
                "winning_strategy": strategy,
                "pair_iptm": piptm,
            })
        else:
            print(f"  ✗ {compound_id}: all {len(scored)} models failed")
            for tag, p, errs in per_model_errors:
                print(f"      {tag} (pair_iptm={p:.3f})")
                for e in errs:
                    print(f"        - {e}")
            completely_failed.append((compound_id, "all_models_failed"))

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(rows)} cases")
    print(f"  Recovered: {len(output_records)} ({len(output_records)/len(rows)*100:.1f}%)")
    print(f"  Failed   : {len(completely_failed)}")
    if completely_failed:
        print("\nCompletely failed compounds (all 10 models failed):")
        for cid, reason in completely_failed:
            print(f"  - {cid}  ({reason})")

    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["compound_id", "winning_model",
                                              "winning_strategy", "pair_iptm"])
            w.writeheader()
            for r in output_records:
                w.writerow(r)
        print(f"\nSaved winners to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
