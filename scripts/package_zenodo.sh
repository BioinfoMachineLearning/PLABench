#!/usr/bin/env bash
# Assemble the Zenodo deposit (doi:10.5281/zenodo.22716174) described in the paper's
# Data Availability statement. One tarball per item, written to $OUT.
#
# Run from the repository root:
#     bash scripts/package_zenodo.sh --dry-run    # print the tree of every archive, write nothing
#     bash scripts/package_zenodo.sh              # build the tarballs
#
# Every path inside an archive is repository-relative, so `tar xzf` at the repository
# root puts each file back where the benchmark expects it. Nothing is renamed on the
# way in.
#
# What goes in is decided by provenance, not by convenience. PLABench-generated
# artifacts ship; corpora other people built are linked instead, the same split
# checkpoints/ already makes between MANIFEST.tsv and THIRD_PARTY.tsv. data/SOURCES.tsv
# records the call for every path under data/ and rides along in archive 01. Three
# consequences worth knowing before you read the pack list:
#
#   PDBbind v2020 forbids redistribution without written permission, so the deposit
#   carries the 4,465 / 497 partition as compound_id + split and nothing else. Rejoin
#   it against huggingface.co/datasets/photonmz/pdbbindpp-2020 to get the rows back.
#
#   CSAR-HiQ was never released under terms that allow redistribution, so the
#   structures are not shipped. The standardised PDB-code/SMILES/pKd tables are. The
#   CSAR portal is gone; bindingmoad.org/Home/download still serves the CSAR-NRC HiQ
#   set and its update, and Binding MOAD is a static archive now, so the safer route
#   for these 87 complexes is to pull them from the RCSB by PDB code.
#
#   Davis and KIBA ship as the fold CSVs only. The .pkl copies the MixingDTA
#   cold-start runner reads are the authors' release, row-identical, and linked in
#   SOURCES.tsv.
#
# PDBBIND_SPLITS points at the partition, which lives outside the repository.
set -euo pipefail

OUT="${OUT:-zenodo_staging}"
PDBBIND_SPLITS="${PDBBIND_SPLITS:-/bmlfast/Lyuwei/1.Datasets/pdbbind_refined_91}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

[ -f run_benchmark.py ] || { echo "run from the repository root" >&2; exit 1; }

# The two files below are derived, small, and belong under data/ rather than in a
# scratch directory, so the script regenerates them in place on every run.
SPLIT_CSV=data/pdbbind2020/refined_split.csv
AF3_NOTICE=data/AF3_structures/NOTICE_ALPHAFOLD3.txt

# A run belongs in the deposit only if both halves still have a live Hydra config,
# the same rule scripts/collect_results.py applies to results/. Keeps predictions
# from archived experiments out.
live_predictions() {
    find outputs -name predictions.csv -print0 | while IFS= read -r -d '' p; do
        # outputs/<model>/<dataset>/...; a path with fewer components is a stray
        # file rather than a run, and fails the config test below anyway
        IFS=/ read -r _ m d _ <<< "$p"
        if [ -f "configs/model/$m.yaml" ] && [ -f "configs/dataset/$d.yaml" ]; then
            printf '%s\n' "$p"
        fi
    done
}

# compound_id,target_sequence,compound_iso_smiles,affinity,status -> compound_id,split.
# The partition is ours to publish; the four columns it was cut from are PDBbind's.
pdbbind_partition() {
    echo "compound_id,split" > "$SPLIT_CSV"
    for half in train val; do
        local src="$PDBBIND_SPLITS/pdbbind_${half}_full.csv"
        [ -f "$src" ] || { echo "  MISSING $src" >&2; continue; }
        tail -n +2 "$src" | cut -d, -f1 | sed "s/\$/,$half/" >> "$SPLIT_CSV"
    done
}

# Two notes on why these two functions are written the way they are. `head` in a
# pipeline closes the pipe on the writer, and under `set -o pipefail` the SIGPIPE
# comes back as a failure that kills the run, so line limiting is done with sed.
# And a bare `[ cond ] && printf` as the last statement of a function returns 1
# when the condition is false, which `set -e` also treats as fatal, so every test
# here is a full if.

# The names directly under $1, at most $2 of them, indented by $3. A subdirectory
# is summarised by what it holds one level down, which is enough to tell a reader
# whether they are looking at per-complex structure folders or at checkpoints.
peek() {
    local dir="$1" limit="$2" pad="$3"
    local -a entries
    mapfile -t entries < <(find -L "$dir" -mindepth 1 -maxdepth 1 | sort)
    local i e inner
    for (( i = 0; i < ${#entries[@]} && i < limit; i++ )); do
        e="${entries[i]}"
        if [ ! -d "$e" ]; then
            printf '%s%s\n' "$pad" "$(basename "$e")"
            continue
        fi
        inner=$(find -L "$e" -maxdepth 1 -type f -printf '%f\n' | sort | sed -n '1,4p' | tr '\n' ' ')
        if [ -z "$inner" ]; then
            inner=$(find -L "$e" -mindepth 1 -maxdepth 1 -type d -printf '%f/\n' | sort | sed -n '1,4p' | tr '\n' ' ')
        fi
        printf '%s%-38s %s\n' "$pad" "$(basename "$e")/" "$inner"
    done
    if [ "${#entries[@]}" -gt "$limit" ]; then
        printf '%s... and %s more\n' "$pad" "$(( ${#entries[@]} - limit ))"
    fi
}

# One line per file, or one line per directory with its shape spelled out, so a
# reader can see what a member actually contains without unpacking it.
show() {
    local m="$1"
    if [ ! -d "$m" ]; then
        printf '   %-62s %s\n' "$m" "$(du -hL "$m" | cut -f1)"
        return 0
    fi
    local files subdirs
    files=$(find -L "$m" -type f | wc -l)
    subdirs=$(find -L "$m" -mindepth 1 -maxdepth 1 -type d | wc -l)
    printf '   %-62s %s  (%s files' "$m/" "$(du -shL "$m" | cut -f1)" "$files"
    if [ "$subdirs" -gt 0 ]; then printf ' in %s subdirectories' "$subdirs"; fi
    printf ')\n'
    peek "$m" 6 '       '
    return 0
}

pack() {  # pack <archive> <member>...
    local name="$1"; shift
    local present=() missing=()
    for m in "$@"; do
        if [ -e "$m" ] || [ -L "$m" ]; then present+=("$m"); else missing+=("$m"); fi
    done
    [ ${#missing[@]} -eq 0 ] || printf '  MISSING %s\n' "${missing[@]}"
    [ ${#present[@]} -gt 0 ] || { echo "  nothing to pack, skipping $name"; return; }
    echo "== $name  [$(du -shcL "${present[@]}" 2>/dev/null | tail -1 | cut -f1) uncompressed]"
    for m in "${present[@]}"; do show "$m"; done
    [ "$DRY" = 1 ] && return
    tar -czhf "$OUT/$name" "${present[@]}"
}

[ "$DRY" = 1 ] || mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# (1) benchmark inputs and split partitions
# ---------------------------------------------------------------------------
pdbbind_partition

pack 01_benchmark_inputs.tar.gz \
    data/SOURCES.tsv \
    "$SPLIT_CSV" \
    data/Structure_independent/DAVIS/*.csv \
    data/Structure_independent/DAVIS/cold/*.csv \
    data/Structure_independent/KIBA/*.csv \
    data/Structure_independent/KIBA/cold/*.csv \
    data/Structure_independent/L1000_casp16_test.csv \
    data/Structure_independent/L3000_casp16_test.csv \
    data/Structure_independent/CSAR-HIQ_36_standardized_test.csv \
    data/Structure_independent/CSAR-HIQ_51_standardized_test.csv \
    data/chembl35/chembl35_filtered_input.csv \
    data/chembl35/chembl35_full_input.csv \
    data/chembl35/chembl35_filter_ledger.csv \
    data/chembl35/chembl35_removed_rows.csv \
    data/chembl35/target_class_map.csv \
    data/chembl35/Data_S1_target_aaseq.csv

# ---------------------------------------------------------------------------
# (2) structural models: the AlphaFold3 inputs the benchmark actually scored.
# chembl35_full is the monomer arm, chembl35_multimer the chain-count comparison.
# The other AF3_structures/ subdirectories are superseded or exploratory passes.
# The Model Parameters Terms of Use permit distributing Output and require clear
# notice of any modification, which is what the NOTICE file is for.
# ---------------------------------------------------------------------------
cat > "$AF3_NOTICE" <<'EOF'
The structures in this archive are AlphaFold 3 Output, predicted with AlphaFold 3
version 3.0.1 (commit a8ecdb2). Their use is subject to the AlphaFold 3 Output
Terms of Use:

    https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md

They are for non-commercial use. They may not be used to train machine learning
models for biomolecular structure prediction.

Modification notice, as those Terms require: this is not raw AlphaFold 3 output.
Each prediction's top-ranked sample was converted from mmCIF into a separated
protein.pdb plus ligand.sdf and ligand.mol2 by
scripts/data_prep/gen_cofold_structures.py. Eleven ChEMBL35 ligands whose
top-ranked sample would not convert were recovered from a lower-ranked sample by
scripts/data_prep/rescue_af3_conversion_failures.py. Neither step altered any
coordinate.

No AlphaFold 3 model parameters are included here or anywhere in PLABench.
Request those from Google DeepMind directly: https://forms.gle/svvpY4u2jsHEwWYS6
EOF

pack 02_af3_structures.tar.gz \
    "$AF3_NOTICE" \
    data/AF3_structures/chembl35_full \
    data/AF3_structures/chembl35_multimer \
    data/AF3_structures/L1000 \
    data/AF3_structures/L3000

# ---------------------------------------------------------------------------
# (3) retrained checkpoints. checkpoints/structure holds third-party weights that
# are linked, not redistributed, except MFE, which PLABench trained itself.
# ---------------------------------------------------------------------------
pack 03_checkpoints.tar.gz \
    checkpoints/sequence \
    checkpoints/structure/mfe \
    checkpoints/MANIFEST.tsv \
    checkpoints/THIRD_PARTY.tsv \
    checkpoints/README.md

# ---------------------------------------------------------------------------
# (4) leakage tables, per-model predictions and the scored metrics
# ---------------------------------------------------------------------------
PRED_LIST=$(mktemp)
live_predictions | sort > "$PRED_LIST"
echo "== 04_predictions_and_metrics.tar.gz"
echo "   outputs/<model>/<dataset>/.../predictions.csv   $(du -shc $(tr '\n' ' ' < "$PRED_LIST") 2>/dev/null | tail -1 | cut -f1)  ($(wc -l < "$PRED_LIST") files from live configs, $(cut -d/ -f2 "$PRED_LIST" | sort -u | wc -l) models)"
for m in results/benchmark_summary.csv results/per_target_evaluation.csv \
         results/weighted_summary.csv results/benchmark_failures.csv \
         results/chembl35 results/casp16 results/chembl35_leakage \
         results/casp16_leakage results/tables results/README.md; do
    if [ -e "$m" ]; then show "$m"; fi
done
if [ "$DRY" = 0 ]; then
    tar -czhf "$OUT/04_predictions_and_metrics.tar.gz" \
        -T "$PRED_LIST" \
        results/benchmark_summary.csv results/per_target_evaluation.csv \
        results/weighted_summary.csv results/benchmark_failures.csv \
        results/chembl35 results/casp16 results/chembl35_leakage \
        results/casp16_leakage results/tables results/README.md
fi
rm -f "$PRED_LIST"

if [ "$DRY" = 1 ]; then
    echo
    echo "dry run, nothing written"
    exit 0
fi

# ---------------------------------------------------------------------------
# What the Zenodo record itself shows. Four tarballs with no description are hard
# to use, so the landing page gets a README and the checksums to verify against.
# ---------------------------------------------------------------------------
cat > "$OUT/README.txt" <<'EOF'
PLABench: benchmark inputs, structures, checkpoints and predictions

Companion deposit for "Leakage-controlled benchmarking reveals generalization
limits of deep learning for protein-ligand binding affinity prediction".
Code, and the instructions for using any of this, are at
https://github.com/BioinfoMachineLearning/PLABench

  01_benchmark_inputs.tar.gz        Split partitions, the filtered and full
                                    ChEMBL35 sets with their removal ledger, the
                                    CASP16 targets in pKd, and SOURCES.tsv
  02_af3_structures.tar.gz          AlphaFold 3 structures for ChEMBL35 and CASP16
  03_checkpoints.tar.gz             The weights PLABench trained, plus the
                                    third-party weight inventory
  04_predictions_and_metrics.tar.gz Every per-model prediction, the leakage
                                    tables and the scored metrics

Every path inside is relative to the repository root, so unpack from there:

    git clone --recurse-submodules https://github.com/BioinfoMachineLearning/PLABench.git
    cd PLABench
    for f in /path/to/0*.tar.gz; do tar xzf "$f"; done
    sha256sum -c /path/to/SHA256SUMS.txt   # run from the directory holding the tarballs

WHAT IS NOT HERE, AND WHY

PLABench artifacts are deposited; corpora other people built are linked instead.
data/SOURCES.tsv in archive 01 gives the origin and the license of every path,
and three cases change what you get:

  PDBbind v2020 forbids redistribution without written permission. Archive 01
  carries the 4,465 / 497 refined partition as compound_id,split and nothing
  else. Rejoin it against
  https://huggingface.co/datasets/photonmz/pdbbindpp-2020

  CSAR-HiQ was never released under terms that allow redistribution, so the 36
  and 51 structure sets are NOT here. Their standardised PDB-code / SMILES / pKd
  tables are, which is enough to rescore once you have the complexes. Fetch the
  87 entries from https://www.rcsb.org/ by PDB code.

  Davis and KIBA are here as the fold CSVs. The .pkl copies the MixingDTA
  cold-start runner reads are the authors' release and are row-identical; get
  them from https://github.com/rokieplayer20/MixingDTA

LICENSES

The ChEMBL35 sets inherit ChEMBL's CC-BY-SA 3.0 and share-alike propagates. The
Davis and KIBA folds and the target class map are CC-BY 4.0. The AlphaFold 3
structures are covered by the AlphaFold 3 Output Terms of Use, non-commercially;
see NOTICE_ALPHAFOLD3.txt in archive 02 for the modification notice those terms
require. Third-party checkpoints keep the terms of their own releases, recorded
per file in checkpoints/THIRD_PARTY.tsv. The code is MIT.
EOF

( cd "$OUT" && sha256sum ./*.tar.gz > SHA256SUMS.txt )

echo
ls -lh "$OUT"
