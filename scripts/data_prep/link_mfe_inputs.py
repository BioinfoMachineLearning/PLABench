"""Rebuild the eight data/*_prepared directories MFE reads on CASF and CSAR-HiQ.

MFE wants one directory per complex holding exactly `protein.pdb` and
`ligand.mol2`. Every other model reads the corpora in their native layout, so
rather than copying 440 MB of structures a second time, these eight directories
are symlink farms: 1,107 links, no bytes of their own. That is also why they are
not in the Zenodo deposit. Four of them point into CASF-2013, CASF-2016 and the
two CSAR-HiQ sets, none of which anyone may redistribute, and the other four
point at Boltz-2 structures that archive 02 already ships.

Run it after the corpora are in place; it links whatever it finds and says what
it skipped.

    python scripts/data_prep/link_mfe_inputs.py
    python scripts/data_prep/link_mfe_inputs.py --check   # report only

Where the four experimental sources come from is in data/SOURCES.tsv. Counts on
the machine the paper was run on: CASF-2013 195, CASF-2016 285, CSAR-HiQ 36 and
51. The Boltz-2 sets are smaller (180, 275, 36, 49) because folding did not
return a pose for every complex.
"""
from __future__ import annotations

import argparse
import os
import sys

WANTED = ("protein.pdb", "ligand.mol2")

# dest, source, layout. "nested" is <source>/<id>/<id>_protein.pdb, the layout
# CASF and CSAR-HiQ ship. "flat" is <source>/<...>_<id>_protein.pdb, what the
# Boltz-2 conversion writes; the CSAR-HiQ 51 files carry a csar51_0000_ prefix
# there, so the identifier is read as the last underscore-separated field.
JOBS = (
    ("data/casf2013_exp_prepared", "data/CASF-2013/coreset", "nested"),
    ("data/casf2016_exp_prepared", "data/CASF-2016/coreset", "nested"),
    ("data/csar_hiq36_exp_prepared", "data/CSAR-HIQ_36", "nested"),
    ("data/csar_hiq51_exp_prepared", "data/CSAR-HIQ_51", "nested"),
    ("data/casf2013_boltz2_prepared", "data/Boltz2_structures/casf2013", "flat"),
    ("data/casf2016_boltz2_prepared", "data/Boltz2_structures/casf2016", "flat"),
    ("data/csar_hiq36_boltz2_prepared", "data/Boltz2_structures/csar36", "flat"),
    ("data/csar_hiq51_boltz2_prepared", "data/Boltz2_structures/csar51", "flat"),
)


def nested_pairs(source):
    """(id, {protein.pdb: path, ligand.mol2: path}) for <source>/<id>/<id>_*."""
    for entry in sorted(os.listdir(source)):
        complex_dir = os.path.join(source, entry)
        if not os.path.isdir(complex_dir):
            continue
        files = {w: os.path.join(complex_dir, f"{entry}_{w}") for w in WANTED}
        if all(os.path.exists(p) for p in files.values()):
            yield entry, files


def flat_pairs(source):
    """(id, {...}) for <source>/<prefix>_<id>_protein.pdb and its ligand."""
    for name in sorted(os.listdir(source)):
        if not name.endswith("_protein.pdb"):
            continue
        stem = name[: -len("_protein.pdb")]
        target_id = stem.split("_")[-1]
        files = {
            "protein.pdb": os.path.join(source, name),
            "ligand.mol2": os.path.join(source, f"{stem}_ligand.mol2"),
        }
        if all(os.path.exists(p) for p in files.values()):
            yield target_id, files


def link(dest, source, layout, check_only=False):
    if not os.path.isdir(source):
        print(f"skip {dest}: {source} is not on disk")
        return None

    pairs = list(nested_pairs(source) if layout == "nested" else flat_pairs(source))
    if not pairs:
        print(f"skip {dest}: found no complete complex under {source}")
        return 0

    made = 0
    for target_id, files in pairs:
        complex_dir = os.path.join(dest, target_id)
        for wanted, src in files.items():
            dst = os.path.join(complex_dir, wanted)
            # Relative, so the tree survives being moved or mounted elsewhere.
            rel = os.path.relpath(src, complex_dir)
            if os.path.islink(dst) and os.readlink(dst) == rel:
                continue
            made += 1
            if check_only:
                continue
            os.makedirs(complex_dir, exist_ok=True)
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(rel, dst)

    verb = "would write" if check_only else "wrote"
    suffix = f", {verb} {made} link{'s' if made != 1 else ''}" if made else ", already linked"
    print(f"{dest}: {len(pairs)} complexes from {source}{suffix}")
    return len(pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report without writing")
    args = ap.parse_args()

    counts = [link(*job, check_only=args.check) for job in JOBS]
    ready = sum(1 for c in counts if c)
    print(f"\n{ready} of {len(JOBS)} MFE input sets are ready.")
    if ready < len(JOBS):
        print("The rest need their corpus first. See data/SOURCES.tsv.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
