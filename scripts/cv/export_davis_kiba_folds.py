"""Write the Davis and KIBA training and validation folds out as CSV.

The five-fold splits reach this repository as pickles, because that is the format
the MixingDTA release ships and the format its loader reads. Only the test folds
were ever converted to CSV, which is enough to reproduce the numbers in the paper
but not enough to retrain anything. This converts the rest.

    python scripts/cv/export_davis_kiba_folds.py            # check, then write
    python scripts/cv/export_davis_kiba_folds.py --check    # check only

A pickled row is [drug_index, smiles, target_id, target_sequence, label] and the
CSV keeps the last four under the header the test files already use. Before
writing anything the script rebuilds every CSV that already exists and compares
it to the file on disk, so a silent change in the row layout fails here rather
than in someone's training run.

Both arms read from data/Structure_independent/. The cold pickles also live in
the MixingDTA checkout, because its cold-start runner reads them from there, so
the script falls back to MIXINGDTA_ROOT (default forks/MixingDTA) for anyone
working from a clone rather than from the Zenodo deposit.
"""
import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "Structure_independent"
MIXINGDTA_ROOT = Path(os.environ.get("MIXINGDTA_ROOT", REPO_ROOT / "forks" / "MixingDTA"))
COLD_SRC = MIXINGDTA_ROOT / "DTA_DataBase" / "cold"

DATASETS = ["DAVIS", "KIBA"]
FOLDS = [1, 2, 3, 4, 5]
HEADER = ["compound_iso_smiles", "Target ID", "target_sequence", "label"]


def source(cold_dir: Path, name: str) -> Path:
    """The staged copy under data/ if it is there, otherwise the MixingDTA checkout."""
    staged = cold_dir / name
    return staged if staged.exists() else COLD_SRC / cold_dir.parent.name / name


def jobs():
    """(source pickle, destination csv) for every fold of both splits.

    Warm and cold are named differently upstream: warm folds are train_1.pkl,
    cold ones are train_1_Drug.pkl, and the cold test fold has no index because
    there is only one. The destinations follow whatever the existing CSVs in
    each directory are called, so the runners keep resolving them.
    """
    for ds in DATASETS:
        warm = DATA_ROOT / ds
        for split in ("train", "valid", "test"):
            for k in FOLDS:
                yield warm / f"{split}_{k}.pkl", warm / f"{split}_{k}.csv"

        cold = warm / "cold"
        for held_out in ("Drug", "Target"):
            for split in ("train", "valid"):
                for k in FOLDS:
                    yield (source(cold, f"{split}_{k}_{held_out}.pkl"),
                           cold / f"{split}_{k}_{held_out.lower()}.csv")
            yield (source(cold, f"test_{held_out}.pkl"),
                   cold / f"test_{held_out.lower()}.csv")


def rows(pkl: Path):
    """The four published columns, dropping the drug index the loader uses internally."""
    with open(pkl, "rb") as fh:
        return [r[1:5] for r in pickle.load(fh)]


def write(dst: Path, table):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(table)


def read(dst: Path):
    with open(dst, newline="") as fh:
        r = csv.reader(fh)
        head = next(r)
        return head, [tuple(row) for row in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the CSVs that exist and write nothing")
    args = ap.parse_args()

    checked, written, missing, bad = 0, 0, [], []

    for src, dst in jobs():
        if not src.exists():
            missing.append(src)
            continue
        table = rows(src)

        if dst.exists():
            head, have = read(dst)
            # str() is what csv.writer would emit, including for the float label,
            # so this compares what a rewrite would produce rather than reparsing.
            want = [tuple(str(c) for c in row) for row in table]
            if head != HEADER or have != want:
                bad.append(dst)
            checked += 1
            continue

        if not args.check:
            write(dst, table)
            written += 1
            print(f"  {dst.relative_to(REPO_ROOT)}  {len(table)} rows")

    for p in missing:
        print(f"  MISSING {p}", file=sys.stderr)
    for p in bad:
        print(f"  MISMATCH {p.relative_to(REPO_ROOT)}", file=sys.stderr)

    print(f"{checked} existing CSVs verified, {written} written, "
          f"{len(bad)} mismatched, {len(missing)} sources missing")
    return 1 if bad or missing else 0


if __name__ == "__main__":
    sys.exit(main())
