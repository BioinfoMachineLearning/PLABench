# Model forks

Every model is a git submodule under `forks/`. `.gitmodules` points each one at
the `plabench` branch of the fork under github.com/lyuweiorg (to be transferred
to BioinfoMachineLearning; only the URLs in `.gitmodules` change then), which
is the pinned upstream commit plus the benchmark patches listed below. Model weights are not
in the forks: `scripts/download_checkpoints.sh --links-only` links them in from
`checkpoints/` (see `checkpoints/MANIFEST.tsv`).

| Fork | Upstream | Pinned upstream commit | `plabench` changes |
|---|---|---|---|
| `BA-Pred` | eightmm/BA-Pred | 4c2ce61 (2025-10-27) | `bapred/data/data.py`: retry `MolFromPDBBlock` with `sanitize=False` when sanitization fails. |
| `boltz` | jwohlwend/boltz | 832486d (2025-09-01, the commit the benchmark env was installed from) | `data/parse/mmcif.py`: correct residue indexing across alignment gaps/mismatches. `data/parse/schema.py`: template chains without a sequence (ligands) are skipped instead of raising. `layers/pairformer.py`, `modules/trunkv2.py`, `modules/confidencev2.py`: in-place pair conditioning and optional chunked z-transition in the confidence stack (memory only, numerically identical). |
| `DeepDTA-Pytorch` | KSUN63/DeepDTA-Pytorch | 78c1ffb (2024-12-03) | `.gitignore` only. Training scripts for the PDBbind and Davis/KIBA models live outside the fork (see below). |
| `flowr_root` | jule-c/flowr_root | 9ff49e8 (2026-04-15) | none. |
| `haiping_methods` | haiping1010/haiping_methods | d787e31 (2024-10-24) | `.gitignore` only. |
| `LLF` | Koreaj9u7n/LLF | 88b334f (2024-06-07) | `.gitignore` only. |
| `MFE` | Sultans0fSwing/MFE | f807cb7 (2024-03-30) | `process.py`: set `atom_coords_batch` on the protein `Data` object. |
| `MixingDTA` | rokieplayer20/MixingDTA | 73492fc (2025-04-04) | `MEETA/config_pdbbind2020_refined.py`: configuration of the PDBbind 2020 refined (4465/497) MEETA ensemble used by PLABench. |
| `MULTICOM_ligand` | BioinfoMachineLearning/MULTICOM_ligand | 99f3da2 (2025-09-13) | vendored `forks/FlowDock/src`: fixed seed + deterministic sampling, per-sample output directories in csv mode, UNK residue (index 20) in template tables, skip atoms with unknown elements. |

## Publishing the forks

The nine forks live under github.com/lyuweiorg with the `plabench` branch
pushed (2026-09-13). Each submodule has a `plabench` remote pointing there:

```bash
for d in BA-Pred boltz DeepDTA-Pytorch flowr_root haiping_methods LLF MFE MixingDTA MULTICOM_ligand; do
    git -C forks/$d push plabench plabench:plabench
done
```

Fresh clones work with `git clone --recurse-submodules` or
`git submodule update --init --recursive`. To move the forks into the
BioinfoMachineLearning organization, transfer each repository on GitHub and
replace `lyuweiorg` with `BioinfoMachineLearning` in `.gitmodules`, then run
`git submodule sync`.

## Patches

`patches/` carries the same changes as standalone git patches, 33 KB for all
nine forks. They exist so that a copy of this repository without the submodules
is still enough to rebuild `forks/`, which is what the archived code snapshot
relies on. To reconstruct one fork from its upstream:

```bash
git clone https://github.com/jwohlwend/boltz.git forks/boltz
git -C forks/boltz checkout 832486d
git -C forks/boltz am ../../forks/patches/boltz.patch
```

The upstream repository and the commit to check out are the second and third
columns of the table above. `flowr_root` has no patch because it is unchanged
from the pinned commit. Run `bash scripts/export_fork_patches.sh` to regenerate
after touching any fork; it reads the commits out of that same table.

## Environments

Each model has its own conda environment (paths are set in
`configs/model/*.yaml`). The `boltz` environment installs `forks/boltz` in
editable mode; the other structure models run their fork sources through
`models_source_dir`.
