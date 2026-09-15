#!/usr/bin/env bash
set -euo pipefail

# Download the published checkpoint bundle (PLABench-trained weights) and recreate
# compatibility links. Third-party weights are not in the bundle: they are linked
# when present and otherwise reported with their official source (THIRD_PARTY.tsv).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
CHECKPOINT_DIR="${REPO_ROOT}/checkpoints"
# Archive 3 of the Zenodo deposit, doi:10.5281/zenodo.22716174. Override any of
# the three to install from a local copy or a mirror; PLABENCH_CHECKPOINT_SHA256=""
# turns the checksum test off.
ARCHIVE_NAME="${PLABENCH_CHECKPOINT_ARCHIVE_NAME:-03_checkpoints.tar.gz}"
ARCHIVE_URL="${PLABENCH_CHECKPOINT_URL:-https://zenodo.org/records/22716174/files/${ARCHIVE_NAME}?download=1}"
ARCHIVE_SHA256="${PLABENCH_CHECKPOINT_SHA256-fd6b9a3c4087a42e6e3f1883662e03c42d823ce79cdf6b13863c384400a34b30}"
FORCE=0
CHECK_ONLY=0
LINKS_ONLY=0

usage() { printf 'Usage: %s [--force] [--check-only] [--links-only]\n  --links-only  skip the archive entirely and only (re)create the compatibility links\n' "$0"; }
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --check-only) CHECK_ONLY=1 ;;
        --links-only) LINKS_ONLY=1; CHECK_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$arg" >&2; usage >&2; exit 2 ;;
    esac
done

mkdir -p "${CHECKPOINT_DIR}/.download"
archive_path="${CHECKPOINT_DIR}/.download/${ARCHIVE_NAME}"
if [[ "${LINKS_ONLY}" -eq 0 && ! -f "${archive_path}" ]]; then
    if [[ "${CHECK_ONLY}" -eq 1 ]]; then
        printf 'Checkpoint archive is not downloaded: %s\n' "${archive_path}" >&2
        exit 1
    fi
    command -v curl >/dev/null 2>&1 || { printf 'curl is required.\n' >&2; exit 2; }
    printf 'Downloading checkpoints from %s\n' "${ARCHIVE_URL}"
    curl --fail --location --retry 3 --output "${archive_path}.part" "${ARCHIVE_URL}"
    mv -- "${archive_path}.part" "${archive_path}"
fi

if [[ "${LINKS_ONLY}" -eq 0 && -n "${ARCHIVE_SHA256}" ]]; then
    printf '%s  %s\n' "${ARCHIVE_SHA256}" "${archive_path}" | sha256sum --check --status - || {
        printf 'Checkpoint archive SHA256 mismatch: %s\n' "${archive_path}" >&2
        exit 1
    }
fi

if [[ "${CHECK_ONLY}" -eq 0 ]]; then
    # The archive has a top-level checkpoints/ directory.
    tar --extract --gzip --file "${archive_path}" --directory "${REPO_ROOT}"
fi

links_file="${CHECKPOINT_DIR}/LINKS.tsv"
[[ -f "${links_file}" ]] || { printf 'Missing %s\n' "${links_file}" >&2; exit 1; }
missing=0
link_one() {
    local source_rel="$1" target_rel="$2"
    local source_abs="${CHECKPOINT_DIR}/${source_rel}"
    local target_abs="${REPO_ROOT}/${target_rel}"
    if [[ ! -e "${source_abs}" ]]; then
        printf 'Missing checkpoint listed in LINKS.tsv: %s\n' "${source_abs}" >&2
        missing=1
        return
    fi
    mkdir -p "$(dirname -- "${target_abs}")"
    if [[ -e "${target_abs}" && ! -L "${target_abs}" ]]; then
        # Upstream forks ship some weights as regular files (BA-Pred, Haiping);
        # an identical file needs no link.
        if cmp -s -- "${source_abs}" "${target_abs}"; then
            return
        fi
        if [[ "${FORCE}" -ne 1 ]]; then
            printf 'Refusing to replace regular file (use --force): %s\n' "${target_abs}" >&2
            missing=1
            return
        fi
    fi
    [[ -L "${target_abs}" || -e "${target_abs}" ]] && rm -- "${target_abs}"
    # Absolute targets keep fork subprocesses independent of their working cwd.
    ln --symbolic --no-dereference "${source_abs}" "${target_abs}"
}
# Third-party files are not in the archive: link them when present, otherwise
# print the official source and continue without failing.
link_optional() {
    local source_rel="$1" target_rel="$2" hint="$3"
    if [[ -e "${CHECKPOINT_DIR}/${source_rel}" ]]; then
        link_one "${source_rel}" "${target_rel}"
    else
        printf 'Third-party checkpoint not installed: checkpoints/%s\n  see checkpoints/THIRD_PARTY.tsv%s\n' \
            "${source_rel}" "${hint:+ ($hint)}" >&2
    fi
}
while IFS=$'\t' read -r source_rel target_rel; do
    [[ -z "${source_rel}" || "${source_rel}" == \#* ]] && continue
    # '-' means the model config reads the checkpoints/ path directly.
    [[ -z "${target_rel}" || "${target_rel}" == "-" ]] && continue
    link_one "${source_rel}" "${target_rel}"
done < "${links_file}"

third_party_file="${CHECKPOINT_DIR}/THIRD_PARTY.tsv"
if [[ -f "${third_party_file}" ]]; then
    while IFS=$'\t' read -r model source_path target_rel url _rest; do
        [[ -z "${model}" || "${model}" == \#* ]] && continue
        [[ "${source_path}" != checkpoints/* ]] && continue
        # Rows with placeholder or descriptive link columns are handled elsewhere.
        [[ -z "${target_rel}" || "${target_rel}" == "-" || "${target_rel}" == *" "* ]] && continue
        link_optional "${source_path#checkpoints/}" "${target_rel}" "${model}: ${url}"
    done < "${third_party_file}"
fi

# The sequence CV loaders retain their upstream directory names. The archive
# uses a compact canonical layout while these loops recreate every five-fold
# warm/cold compatibility path.
for dataset in DAVIS KIBA; do
    dataset_lc="${dataset,,}"
    for fold in 1 2 3 4 5; do
        link_one "sequence/deepdta/${dataset_lc}/warm/fold_${fold}.pt" \
            "forks/DeepDTA-Pytorch/${dataset_lc}_warm_results/${dataset_lc}_warm_fold_${fold}/$([[ ${dataset} == DAVIS ]] && echo best_model_fold_${fold}.pt || echo deepdta-warm-fold_${fold}-prk12-ldk8.pt)"
        link_one "sequence/deepdta/${dataset_lc}/cold_drug/fold_${fold}.pt" \
            "forks/DeepDTA-Pytorch/${dataset_lc}_drug_results/${dataset_lc}_drug_fold_${fold}/deepdta-fold_${fold}-prk12-ldk8.pt"
        if [[ "${dataset}" == DAVIS ]]; then
            deep_target_dir="davis_results/davis_target_fold_${fold}"
            deep_target_name="deepdta-fold_${fold}-prk12-ldk8.pt"
        else
            deep_target_dir="kiba_target_cold_results/kiba_target_cold_fold_${fold}"
            deep_target_name="best_model_fold_${fold}.pt"
        fi
        link_one "sequence/deepdta/${dataset_lc}/cold_target/fold_${fold}.pt" \
            "forks/DeepDTA-Pytorch/${deep_target_dir}/${deep_target_name}"
        # Cold-start DeepDTA on Davis also loads fold-local vocabulary sidecars.
        # The KIBA cold folds were trained without them (the loader falls back
        # to its built-in vocabulary), so only Davis has sidecars to link.
        if [[ "${dataset}" == DAVIS ]]; then
            for vocab in protein_dict-prk12-ldk8.json ligand_dict-prk12-ldk8.json; do
                link_one "sequence/deepdta/davis/cold_drug/${vocab}" "forks/DeepDTA-Pytorch/davis_drug_results/davis_drug_fold_${fold}/${vocab}"
                link_one "sequence/deepdta/davis/cold_target/${vocab}" "forks/DeepDTA-Pytorch/davis_results/davis_target_fold_${fold}/${vocab}"
            done
        fi

        link_one "sequence/llf/${dataset_lc}/warm/fold_${fold}.pth" \
            "forks/LLF/models/${dataset_lc}_warm/fold_${fold}/best_model.pth"
        link_one "sequence/llf/${dataset_lc}/cold_drug/fold_${fold}.model" \
            "forks/LLF/models/${dataset_lc}_drug_cv/best_model_fold_${fold}.model"
        link_one "sequence/llf/${dataset_lc}/cold_target/fold_${fold}.pth" \
            "forks/LLF/models/${dataset_lc}_target_cv/fold_${fold}/best_model.pth"

        # MixingDTA warm-start weights are the authors' Google Drive release
        # (third-party, see THIRD_PARTY.tsv); they are optional here.
        mix_hint="MixingDTA authors' Davis/KIBA weights"
        for case in results_none results_all_pair results_drug results_protein results_drug_and_protein results_reversed; do
            link_optional "sequence/mixingdta/${dataset_lc}/warm/fold_${fold}/${case}.pth" \
                "forks/MixingDTA/model_weights/${case}/${dataset}/${fold}_fold_valid_best_checkpoint.pth" "${mix_hint}"
        done
        link_optional "sequence/mixingdta/${dataset_lc}/warm/fold_${fold}/result_integration.pth" \
            "forks/MixingDTA/model_weights/result_integration/${dataset}/${fold}_fold_valid_best_checkpoint.pth" "${mix_hint}"
        cold_drug_dir="results_cold_drug_case4_protein"
        cold_target_dir="results_cold_target_case3_drug"
        link_one "sequence/mixingdta/${dataset_lc}/cold_drug/fold_${fold}.pth" \
            "forks/MixingDTA/MEETA/${cold_drug_dir}/${dataset}/${fold}_fold_valid_best_checkpoint.pth"
        link_one "sequence/mixingdta/${dataset_lc}/cold_target/fold_${fold}.pth" \
            "forks/MixingDTA/MEETA/${cold_target_dir}/${dataset}/${fold}_fold_valid_best_checkpoint.pth"
    done
done

if [[ "${missing}" -ne 0 ]]; then
    printf 'Checkpoint installation is incomplete.\n' >&2
    exit 1
fi
printf 'Checkpoint installation complete under %s\n' "${CHECKPOINT_DIR}"
