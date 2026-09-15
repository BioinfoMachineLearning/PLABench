#!/usr/bin/env bash
# Assemble the Zenodo deposit (doi:10.5281/zenodo.22716174) described in the paper's
# Data Availability statement. One tarball per item, written to $OUT.
#
# Run from the repository root:
#     bash scripts/package_zenodo.sh --dry-run    # print the tree of every archive, write nothing
#     bash scripts/package_zenodo.sh              # build the tarballs
#
# Every path inside an archive is repository-relative, so `tar xf` at the repository
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
#   structures are not shipped. The standardized PDB-code/SMILES/pKd tables are. The
#   CSAR portal is gone; bindingmoad.org/Home/download still serves the CSAR-NRC HiQ
#   set and its update, and Binding MOAD is a static archive now, so the safer route
#   for these 87 complexes is to pull them from the RCSB by PDB code.
#
#   Davis and KIBA ship as the pickles MixingDTA released, which is every fold of
#   both arms and not just the test ones. They stay pickles because the target
#   sequence repeats on every row and pickle memoizes it: 28 MB against 621 MB for
#   the same folds as CSV. scripts/cv/export_davis_kiba_folds.py converts them.
#
# PDBBIND_SPLITS points at the partition, which lives outside the repository.
set -euo pipefail

OUT="${OUT:-zenodo_staging}"
PDBBIND_SPLITS="${PDBBIND_SPLITS:-/bmlfast/Lyuwei/1.Datasets/pdbbind_refined_91}"
# ONLY=01 rebuilds one archive and leaves the others as they are on disk. A tarball
# is not byte-reproducible, so rebuilding all four after a one-archive change would
# churn three checksums and force three re-uploads for nothing. The README and
# SHA256SUMS are always rewritten from whatever ends up in $OUT.
ONLY="${ONLY:-}"
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
# is summarized by what it holds one level down, which is enough to tell a reader
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

want() {  # want <archive>: in scope for this run?
    [ -z "$ONLY" ] || [ "${1#"$ONLY"}" != "$1" ]
}

pack() {  # pack <archive> <member>...
    local name="$1"; shift
    local present=() missing=()
    if ! want "$name"; then
        echo "== $name  [kept, ONLY=$ONLY]"
        return
    fi
    for m in "$@"; do
        if [ -e "$m" ] || [ -L "$m" ]; then present+=("$m"); else missing+=("$m"); fi
    done
    [ ${#missing[@]} -eq 0 ] || printf '  MISSING %s\n' "${missing[@]}"
    [ ${#present[@]} -gt 0 ] || { echo "  nothing to pack, skipping $name"; return; }
    echo "== $name  [$(du -shcL "${present[@]}" 2>/dev/null | tail -1 | cut -f1) uncompressed]"
    for m in "${present[@]}"; do show "$m"; done
    [ "$DRY" = 1 ] && return
    compress_to "$OUT/$name" "${present[@]}"
}

# The archive name picks the compressor. Archives 01 and 02 are text, CSV and
# PDB/SDF/MOL2, where xz's window catches redundancy across files that gzip's
# 32 KB cannot: 93 MB becomes 12 and 636 MB becomes 283. Archive 03 is model
# weights, near-random float32, where xz buys 8% for ten minutes of CPU and is
# not worth the churn. tar reads either without being told which.
compress_to() {
    local out="$1"; shift
    case "$out" in
        *.tar.xz) XZ_OPT="-9 -T0" tar -cJhf "$out" "$@" ;;
        *)        tar -czhf "$out" "$@" ;;
    esac
}

[ "$DRY" = 1 ] || mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# (1) benchmark inputs and split partitions
# ---------------------------------------------------------------------------
pdbbind_partition

pack 01_benchmark_inputs.tar.xz \
    data/SOURCES.tsv \
    "$SPLIT_CSV" \
    data/Structure_independent/DAVIS/*.csv \
    data/Structure_independent/DAVIS/*.pkl \
    data/Structure_independent/DAVIS/cold/*.csv \
    data/Structure_independent/DAVIS/cold/*.pkl \
    data/Structure_independent/KIBA/*.csv \
    data/Structure_independent/KIBA/*.pkl \
    data/Structure_independent/KIBA/cold/*.csv \
    data/Structure_independent/KIBA/cold/*.pkl \
    data/Structure_independent/L1000_casp16_test.csv \
    data/Structure_independent/L3000_casp16_test.csv \
    data/Structure_independent/CASF-2013_standardized_test.csv \
    data/Structure_independent/CASF-2016_standardized_test.csv \
    data/Structure_independent/CSAR-HIQ_36_standardized_test.csv \
    data/Structure_independent/CSAR-HIQ_51_standardized_test.csv \
    data/casp16_data/labels \
    data/casp16_data/smiles \
    data/casp16_data/stage1_input \
    data/chembl35/chembl35_filtered_input.csv \
    data/chembl35/chembl35_full_input.csv \
    data/chembl35/chembl35_filter_ledger.csv \
    data/chembl35/chembl35_removed_rows.csv \
    data/chembl35/target_class_map.csv \
    data/chembl35/Data_S1_target_aaseq.csv

# ---------------------------------------------------------------------------
# (2) the predicted input structures the benchmark scored, from both folding
# tools. chembl35_full is the AF3 monomer arm, chembl35_multimer the chain-count
# comparison. The other AF3_structures/ subdirectories are superseded or
# exploratory passes. The Boltz2_ directories are the lower rungs of the pose
# ladder behind Figure 3, PLABench-generated like everything else here; the
# tplpocket set is the template-guided rung. Their CASF and CSAR halves are ours
# to publish, but the matching experimental complexes are not (see below), so
# those configs still need the corpus fetched from its licensor.
#
# The AF3 Model Parameters Terms of Use permit distributing Output and require
# clear notice of any modification, which is what the NOTICE file is for.
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

pack 02_af3_structures.tar.xz \
    "$AF3_NOTICE" \
    data/AF3_structures/chembl35_full \
    data/AF3_structures/chembl35_multimer \
    data/AF3_structures/L1000 \
    data/AF3_structures/L3000 \
    data/Boltz2_structures/L1000 \
    data/Boltz2_structures/L3000 \
    data/Boltz2_structures/casf2013 \
    data/Boltz2_structures/casf2016 \
    data/Boltz2_structures/csar36 \
    data/Boltz2_structures/csar51 \
    data/Boltz2_tplpocket_structures/L3000

# ---------------------------------------------------------------------------
# (3) retrained checkpoints. Both halves of checkpoints/ hold third-party weights
# that are linked, not redistributed: under structure/ everything but MFE, and
# under sequence/ the MixingDTA Davis and KIBA warm folds, which are the authors'
# Google Drive release and carry no license. So the members below are spelled out
# one PLABench-trained subtree at a time rather than as checkpoints/sequence. It
# costs a line per model, and in exchange a directory nobody vetted cannot ride
# along, and --dry-run shows what actually ships. MANIFEST.tsv is the authority
# for which is which.
# ---------------------------------------------------------------------------
pack 03_checkpoints.tar.gz \
    checkpoints/sequence/deepdta \
    checkpoints/sequence/llf \
    checkpoints/sequence/mixingdta/pdbbind \
    checkpoints/sequence/mixingdta/davis/cold_drug \
    checkpoints/sequence/mixingdta/davis/cold_target \
    checkpoints/sequence/mixingdta/kiba/cold_drug \
    checkpoints/sequence/mixingdta/kiba/cold_target \
    checkpoints/structure/mfe \
    checkpoints/MANIFEST.tsv \
    checkpoints/THIRD_PARTY.tsv \
    checkpoints/README.md

# ---------------------------------------------------------------------------
# (4) leakage tables, per-model predictions and the scored metrics
# ---------------------------------------------------------------------------
if ! want 04_predictions_and_metrics.tar.gz; then
    echo "== 04_predictions_and_metrics.tar.gz  [kept, ONLY=$ONLY]"
else
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
fi

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

  01_benchmark_inputs.tar.xz        Split partitions, the filtered and full
                                    ChEMBL35 sets with their removal ledger, the
                                    CASP16 labels, SMILES and stage-1 inputs, the
                                    standardized CASF and CSAR-HiQ affinity
                                    tables, every Davis and KIBA fold, and
                                    SOURCES.tsv
  02_af3_structures.tar.xz          The predicted structures the benchmark
                                    scored: AlphaFold 3 for ChEMBL35 and CASP16,
                                    Boltz-2 for CASP16, CASF and CSAR-HiQ, plus
                                    the template-guided CASP16 rung
  03_checkpoints.tar.gz             The weights PLABench trained, plus the
                                    third-party weight inventory
  04_predictions_and_metrics.tar.gz Every per-model prediction, the leakage
                                    tables and the scored metrics

Every path inside is relative to the repository root, so unpack from there:

    git clone --recurse-submodules https://github.com/BioinfoMachineLearning/PLABench.git
    cd PLABench
    for f in /path/to/0*.tar.*; do tar xf "$f"; done
    sha256sum -c /path/to/SHA256SUMS.txt   # run from the directory holding the tarballs

DAVIS AND KIBA

Complete: train, validation and test, for the warm-start arm and for both
cold-start arms. Everything but the test folds is in the pickle format the
MixingDTA authors released, because the target sequence repeats on every row and
pickle stores it once, which is 28 MB against 621 MB for the same folds as CSV.
To get CSVs with the same four columns as the test files, run, from the
repository root:

    python scripts/cv/export_davis_kiba_folds.py

WHAT IS NOT HERE, AND WHY

PLABench artifacts are deposited; corpora other people built are linked instead.
data/SOURCES.tsv in archive 01 gives the origin and the license of every path.
These are the cases that change what you get:

  PDBbind v2020 forbids redistribution without written permission. Archive 01
  carries the 4,465 / 497 refined partition as compound_id,split and nothing
  else. Rejoin it against
  https://huggingface.co/datasets/photonmz/pdbbindpp-2020

  CSAR-HiQ was never released under terms that allow redistribution, so the 36
  and 51 structure sets are NOT here. Their standardized PDB-code / SMILES / pKd
  tables are, which is enough to rescore once you have the complexes. Fetch the
  87 entries from https://www.rcsb.org/ by PDB code.

  CASF-2013 and CASF-2016 come from the CASF authors under their own terms, so
  the core sets are not here either. Request them at
  http://www.pdbbind.org.cn/casf.php. The Boltz-2 poses for both, and the
  standardized affinity tables, are in archives 02 and 01.

  The experimental CASP16 stage-2 complexes are the organizers' release, not
  ours, so data/casp16_data/stage2_input is not here. Nothing in the paper needs
  it: the CASP16 metrics are scored from the labels in archive 01, and
  configs/manifests/ in the repository records which targets each stage-2 run
  covered, so scripts/collect_results.py reports the same 93 of 93 without it.
  The six stage-2 configs themselves cannot be re-run until you have the
  complexes from https://predictioncenter.org/casp16/.

  The CASP16 stage-1 submissions of the other predictor groups
  (data/casp16_data/official_submissions) are the Prediction Center's to
  publish and no script here reads them. They are at the same URL.

  The eight data/*_prepared directories hold no bytes of their own. MFE wants
  one directory per complex with protein.pdb and ligand.mol2 inside, so they are
  2,214 symlinks into CASF, CSAR-HiQ and the Boltz-2 poses. Rebuild them once
  the corpora are in place:

      python scripts/data_prep/link_mfe_inputs.py

  The four Boltz-2 ones need only archive 02 and work straight after unpacking.

LICENSES

The ChEMBL35 sets inherit ChEMBL's CC-BY-SA 3.0 and share-alike propagates. The
Davis and KIBA folds and the target class map are CC-BY 4.0. The AlphaFold 3
structures are covered by the AlphaFold 3 Output Terms of Use, non-commercially;
see NOTICE_ALPHAFOLD3.txt in archive 02 for the modification notice those terms
require. Third-party checkpoints keep the terms of their own releases, recorded
per file in checkpoints/THIRD_PARTY.tsv. The code is MIT.
EOF

( cd "$OUT" && sha256sum ./*.tar.gz ./*.tar.xz 2>/dev/null | sort -k2 > SHA256SUMS.txt )

echo
ls -lh "$OUT"
