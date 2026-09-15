"""Rewrite the affinity column of the CASP16 sequence-model test sets from the labels.

data/Structure_independent/L{1000,3000}_casp16_test.csv are the inputs the three
sequence models read for CASP16. Their compound_iso_smiles and target_sequence
columns are built once and left alone; only affinity is derived, and it is a unit
conversion of binding_affinity in data/casp16_data/labels/.

The deposited copies were converted with RT ln(10) = 1.3642470 kcal/mol, the value
at 298.15 K. CASP16 built its labels at 300 K, where the constant is 1.372571
(plabench.analysis.units), and everything scored in the paper goes through that
constant: scripts/collect_results.py reads the label files directly and converts
them itself, so no published number moves when this script runs. What it fixes is
the deposited CSV, which a reader would otherwise score against and land 0.6% low
on every error metric.

Idempotent. Run from the repository root:

    python scripts/data_prep/sync_casp16_sequence_affinity.py
    python scripts/data_prep/sync_casp16_sequence_affinity.py --check   # report only
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plabench.analysis.units import dg_to_pkd  # noqa: E402

SERIES = ("L1000", "L3000")
LABELS = "data/casp16_data/labels/{series}_exper_affinity.csv"
TESTSET = "data/Structure_independent/{series}_casp16_test.csv"
# The label files have a UTF-8 BOM, and L1000's leading column name carries it.
LABEL_ENCODING = "utf-8-sig"


def sync(series, check_only=False):
    label_path = LABELS.format(series=series)
    test_path = TESTSET.format(series=series)
    for path in (label_path, test_path):
        if not os.path.exists(path):
            sys.exit(f"Missing {path}. Unpack archive 1 of the Zenodo deposit first.")

    labels = pd.read_csv(label_path, encoding=LABEL_ENCODING)
    pkd = dict(zip(labels["Target ID"].astype(str), dg_to_pkd(labels["binding_affinity"])))

    test = pd.read_csv(test_path)
    missing = set(test["molecule_id"].astype(str)) - set(pkd)
    if missing:
        sys.exit(f"{test_path}: {len(missing)} molecule_id have no label, e.g. {sorted(missing)[:3]}")

    wanted = test["molecule_id"].astype(str).map(pkd)
    delta = (wanted - test["affinity"]).abs().max()
    print(f"{series}: n={len(test)}, largest affinity change {delta:.6f} log units")
    # The CSV is written at repr precision, so a re-read differs from the freshly
    # converted value in the last bit or two. Anything under that is already synced.
    if check_only or delta < 1e-9:
        return delta

    test["affinity"] = wanted
    test.to_csv(test_path, index=False)
    print(f"  wrote {test_path}")
    return delta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report the difference without writing")
    args = ap.parse_args()
    for series in SERIES:
        sync(series, check_only=args.check)


if __name__ == "__main__":
    main()
