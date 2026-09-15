#!/usr/bin/env bash
set -euo pipefail

# Fetch the third-party checkpoints PLABench depends on from their official
# sources and verify them against the SHA256 values recorded in
# checkpoints/THIRD_PARTY.tsv. Nothing here is re-hosted by PLABench.
#
# Usage: bash scripts/download_third_party.sh [model ...] [--check-only]
#   model in: flowr_root flowdock bapred haiping mixingdta_warm esm3 boltz2
#   (default: all). Files that already exist with the right hash are skipped.
#   ESM3 is gated on Hugging Face: export HF_TOKEN after accepting the license.
#   Google Drive downloads need `gdown` (pip install gdown).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
CKPT="${REPO_ROOT}/checkpoints"
DL="${CKPT}/.download"
mkdir -p "${DL}"

CHECK_ONLY=0
targets=()
for arg in "$@"; do
    case "$arg" in
        --check-only) CHECK_ONLY=1 ;;
        -h|--help) sed -n '4,13p' "$0"; exit 0 ;;
        *) targets+=("$arg") ;;
    esac
done
[[ ${#targets[@]} -eq 0 ]] && targets=(flowr_root flowdock bapred haiping mixingdta_warm esm3 boltz2)

# Expected hashes (also listed in checkpoints/THIRD_PARTY.tsv).
SHA_FLOWR_ROOT=8d19cb7881214c156b07c42d4f26c686e235e99caf8bd378cec15bb15fb826ab
SHA_FLOWDOCK=2b598d53cde76dd36ae147e3e4ae7e171a0d4071468652a92263a9de317c48a7
MD5_FLOWDOCK_ARCHIVE=a19bbf4d49f20af5f94303c9370f9dee
SHA_BAPRED=927038d73a1ad405c15d3e2c351780e15b471f3629d02e52dfd68d524bc322ae
SHA_HAIPING=1f5e62f9eecae97f074d6a647a6899317acf43c77739e5c0bfa988c73d8d31e5
SHA_MIXINGDTA_TAR=500200bfff33a9ab65fe706c3851f0d9ce3dcd9b36b7690ead52ba85881aef78
SHA_ESM3=5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9
SHA_BOLTZ_CONF=090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1
SHA_BOLTZ_AFF=dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e
SHA_BOLTZ_MOLS=39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7

GDRIVE_FLOWR_ROOT_ID=1n_UHnC7lttJbHKtA4wwi58EeKhMIACLP
GDRIVE_MIXINGDTA_ID=13atZOJOXkvy03lH8VfvgSAhYu9gUNQ-i
URL_FLOWDOCK_ARCHIVE="https://zenodo.org/records/15066450/files/flowdock_checkpoints.tar.gz?download=1"
URL_BAPRED="https://github.com/eightmm/BA-Pred/raw/4c2ce6168eb4a265cba2f77673232a5ec9a4f88f/bapred/weight/BAPred.pth"
URL_HAIPING="https://media.githubusercontent.com/media/haiping1010/haiping_methods/d787e31f88941ea4cded9d7d67255a86d7236738/work3_VS_n_general_competation_gtrans_RG_cutoff0.8_redo/full_model_out1800.model"
URL_ESM3="https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1/resolve/main/data/weights/esm3_sm_open_v1.pth"

status=0
sha_ok() { [[ -f "$1" ]] && [[ "$(sha256sum -- "$1" | cut -d' ' -f1)" == "$2" ]]; }
verify() {  # verify <path> <sha256> <label>
    if sha_ok "$1" "$2"; then printf 'OK       %s\n' "$3"; return 0; fi
    if [[ -f "$1" ]]; then printf 'BAD HASH %s (%s)\n' "$3" "$1" >&2; else printf 'MISSING  %s (%s)\n' "$3" "$1" >&2; fi
    status=1; return 1
}
need_gdown() { command -v gdown >/dev/null 2>&1 || { printf 'gdown is required for %s (pip install gdown)\n' "$1" >&2; status=1; return 1; }; }
fetch() { curl --fail --location --retry 3 --output "$1.part" "$2" && mv -- "$1.part" "$1"; }

for t in "${targets[@]}"; do
    case "$t" in
    flowr_root)
        dest="${CKPT}/structure/flowr_root/flowr_root_v2.1.ckpt"
        if ! sha_ok "$dest" "$SHA_FLOWR_ROOT" && [[ $CHECK_ONLY -eq 0 ]] && need_gdown flowr_root; then
            mkdir -p "$(dirname "$dest")"; gdown --id "$GDRIVE_FLOWR_ROOT_ID" -O "$dest"
        fi
        verify "$dest" "$SHA_FLOWR_ROOT" "FLOWR.ROOT v2.1 (Google Drive, MIT)" || true ;;
    flowdock)
        dest="${CKPT}/structure/flowdock/esmfold_prior_paper_weights-EMA.ckpt"
        if ! sha_ok "$dest" "$SHA_FLOWDOCK" && [[ $CHECK_ONLY -eq 0 ]]; then
            archive="${DL}/flowdock_checkpoints.tar.gz"
            [[ -f "$archive" ]] || fetch "$archive" "$URL_FLOWDOCK_ARCHIVE"
            if [[ "$(md5sum -- "$archive" | cut -d' ' -f1)" != "$MD5_FLOWDOCK_ARCHIVE" ]]; then
                printf 'FlowDock archive md5 mismatch: %s\n' "$archive" >&2; status=1
            else
                mkdir -p "$(dirname "$dest")"
                tar -xzf "$archive" --to-stdout checkpoints/esmfold_prior_paper_weights-EMA.ckpt > "$dest.part" && mv -- "$dest.part" "$dest"
            fi
        fi
        verify "$dest" "$SHA_FLOWDOCK" "FlowDock esmfold_prior_paper_weights-EMA (Zenodo 15066450, MIT)" || true ;;
    bapred)
        dest="${CKPT}/structure/bapred/BAPred.pth"
        if ! sha_ok "$dest" "$SHA_BAPRED" && [[ $CHECK_ONLY -eq 0 ]]; then mkdir -p "$(dirname "$dest")"; fetch "$dest" "$URL_BAPRED"; fi
        verify "$dest" "$SHA_BAPRED" "BA-Pred BAPred.pth (GitHub, Apache-2.0)" || true ;;
    haiping)
        dest="${CKPT}/structure/haiping/full_model_out1800.model"
        if ! sha_ok "$dest" "$SHA_HAIPING" && [[ $CHECK_ONLY -eq 0 ]]; then mkdir -p "$(dirname "$dest")"; fetch "$dest" "$URL_HAIPING"; fi
        verify "$dest" "$SHA_HAIPING" "Graph_RG/Haiping full_model_out1800.model (GitHub LFS, Apache-2.0)" || true ;;
    mixingdta_warm)
        # The MixingDTA authors' Davis/KIBA MEETA weights (70 files) ship as one
        # tarball on their Google Drive; PLABench stores them per fold/case.
        root="${CKPT}/sequence/mixingdta"
        tarball="${DL}/mixingdta_DeepDTA.tar.gz"
        missing=0
        for ds in davis kiba; do for fold in 1 2 3 4 5; do for case in results_none results_all_pair results_drug results_protein results_drug_and_protein results_reversed result_integration; do
            [[ -f "${root}/${ds}/warm/fold_${fold}/${case}.pth" ]] || missing=1
        done; done; done
        if [[ $missing -eq 1 && $CHECK_ONLY -eq 0 ]] && need_gdown mixingdta_warm; then
            sha_ok "$tarball" "$SHA_MIXINGDTA_TAR" || gdown --id "$GDRIVE_MIXINGDTA_ID" -O "$tarball"
            if sha_ok "$tarball" "$SHA_MIXINGDTA_TAR"; then
                tmp="$(mktemp -d "${DL}/mixingdta.XXXXXX")"; tar -xzf "$tarball" -C "$tmp"
                find "$tmp" -name '*_fold_valid_best_checkpoint.pth' | while read -r f; do
                    case_name="$(basename "$(dirname "$(dirname "$f")")")"; ds="$(basename "$(dirname "$f")")"; fold="${f##*/}"; fold="${fold%%_*}"
                    dest="${root}/${ds,,}/warm/fold_${fold}/${case_name}.pth"; mkdir -p "$(dirname "$dest")"; cp -p -- "$f" "$dest"
                done
                rm -rf -- "$tmp"
            else
                printf 'MixingDTA tarball hash mismatch: %s\n' "$tarball" >&2; status=1
            fi
        fi
        n=$(find "${root}" -path '*/warm/fold_*/*.pth' 2>/dev/null | wc -l)
        if [[ "$n" -eq 70 ]]; then
            if sha_ok "$tarball" "$SHA_MIXINGDTA_TAR"; then printf 'OK       MixingDTA Davis/KIBA warm: 70 files from verified tarball\n'
            else printf 'PRESENT  MixingDTA Davis/KIBA warm: 70 files (tarball not on disk, per-file hashes not re-checked)\n'; fi
        else printf 'MISSING  MixingDTA Davis/KIBA warm: %s/70 files\n' "$n" >&2; status=1; fi ;;
    esm3)
        dest="${CKPT}/ESM3/esm3_sm_open_v1.pth"
        if ! sha_ok "$dest" "$SHA_ESM3" && [[ $CHECK_ONLY -eq 0 ]]; then
            if [[ -z "${HF_TOKEN:-}" ]]; then printf 'ESM3 is gated: accept the license on Hugging Face and export HF_TOKEN\n' >&2; status=1
            else mkdir -p "$(dirname "$dest")"; curl --fail --location --retry 3 -H "Authorization: Bearer ${HF_TOKEN}" --output "$dest.part" "$URL_ESM3" && mv -- "$dest.part" "$dest"; fi
        fi
        verify "$dest" "$SHA_ESM3" "ESM3 esm3_sm_open_v1.pth (Hugging Face, non-commercial)" || true
        # esm resolves data/weights/ relative to the working directory when INFRA_PROVIDER is set.
        [[ -e "${REPO_ROOT}/data/weights" ]] || { mkdir -p "${REPO_ROOT}/data"; ln -s ../checkpoints/ESM3 "${REPO_ROOT}/data/weights"; } ;;
    boltz2)
        cache="${BOLTZ_CACHE:-$HOME/.boltz}"
        verify "${cache}/boltz2_conf.ckpt" "$SHA_BOLTZ_CONF" "Boltz-2 boltz2_conf.ckpt (Hugging Face, MIT)" || printf '         boltz downloads this itself on the first `boltz predict`\n'
        verify "${cache}/boltz2_aff.ckpt" "$SHA_BOLTZ_AFF" "Boltz-2 boltz2_aff.ckpt" || true
        verify "${cache}/mols.tar" "$SHA_BOLTZ_MOLS" "Boltz-2 mols.tar" || true ;;
    *) printf 'Unknown target: %s\n' "$t" >&2; status=2 ;;
    esac
done

if [[ $status -eq 0 ]]; then
    printf 'All requested third-party checkpoints verified. Recreate fork links with: bash scripts/download_checkpoints.sh --links-only\n'
fi
exit "$status"
