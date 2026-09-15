# Environments

Nine models, nine conda environments, plus a tenth for the analysis scripts.
The nine genuinely conflict: Graph_RG is on Python 3.8 and PyTorch 1.5, Boltz-2
on Python 3.11 and PyTorch 2.8, so there is no single environment that runs the
whole benchmark. Each model YAML is a `conda env export --no-builds` of the
environment its model was benchmarked in, with the machine-local `prefix:`
stripped.

Creating one is a single command, the same command for every file here:

```bash
conda env create -f environments/boltz2.yaml
```

Every file declares `name: plabench-<model>`, so they land side by side as
`plabench-bapred`, `plabench-boltz2`, `plabench-deepdta` and so on. `mamba env
create -f ...` takes the same arguments and is considerably faster on the four
large ones (`flowdock`, `flowr_root`, `mixingdta`, `deepdta`).

The versions inside each file are also what the runtime table in the top-level
README quotes, so the two cannot drift apart.

## Wheels that are not on PyPI

`bapred.yaml`, `llf.yaml` and `mfe.yaml` start their `pip:` section with an
`--extra-index-url` and a `--find-links` line. PyTorch, torchvision, DGL and the
four PyG extensions are compiled per torch/CUDA pair and published on their own
index, not on PyPI. `conda env export` writes down the local version label it
sees, `+cu113` or `+pt113cu117`, but it has no field for the index the wheel
came from, so those lines are added back by hand. Do not drop them: without
them pip cannot resolve the pin and the environment will not build. The three
files are otherwise verbatim exports like the rest.

## The analysis environment

`analysis.yaml` is the odd one out. It runs no model, so it is not an export of
a benchmark run but a hand-written minimum, pinned to the versions the published
tables and figures were produced with. Everything under `scripts/` runs in it:
the results collector, the table and figure generators, and the leakage audits.

```bash
conda env create -f environments/analysis.yaml
conda activate plabench-analysis
python scripts/collect_results.py
python scripts/weighted_summary.py
```

It carries `torch` only because two leakage scripts compute Tanimoto similarity
on the GPU. They fall back to the CPU and give the same numbers, more slowly.

## How each model is launched

Two patterns, and which one a model uses decides where you need the environment.

Five models are spawned as a subprocess. `run_benchmark.py` itself can then live
in any environment that has `hydra-core` and `omegaconf`; point the variable at
the environment you created and the wrapper does the rest.

| Model | Variable | What it wants |
| --- | --- | --- |
| Boltz-2 | `BOLTZ2_ENV` | conda prefix |
| FlowDock | `FLOWDOCK_PYTHON` | interpreter path |
| FLOWR.ROOT | `FLOWR_ROOT_PYTHON` | interpreter path |
| LLF | `LLF_PYTHON` | interpreter path |
| MixingDTA | `MIXINGDTA_PYTHON` | interpreter path |

```bash
export BOLTZ2_ENV="$(conda info --base)/envs/plabench-boltz2"
export FLOWR_ROOT_PYTHON="$(conda info --base)/envs/plabench-flowr_root/bin/python"
```

The other four (LCDD-team, DeepDTA, Graph_RG and MFE) run in the interpreter
that started `run_benchmark.py`. There is no variable for them; you activate the
environment and launch the benchmark from inside it:

```bash
conda activate plabench-haiping
python run_benchmark.py model=haiping dataset=haiping_casp16_l3000_stage2
```

Because of that, these four need `run_benchmark.py`'s own dependencies
(`hydra-core` 1.3.2, `omegaconf` 2.3.0, pandas) in the same environment.
`bapred.yaml`, `haiping.yaml` and `mfe.yaml` already carry them.

`configs/model/mfe.yaml` still has a `conda_env` field. It is inert, because the
MFE wrapper reads `sys.executable`, and `MFE_ENV` does nothing.

## Two steps a conda export cannot carry

**Boltz-2 runs from the patched fork, not from PyPI.** `boltz2.yaml` pins
`boltz==2.2.0` from PyPI because that is what the export saw, but the benchmark
was run against `forks/boltz`, which is upstream 832486d plus the patches listed
in `forks/README.md`. Overwrite it after creating the environment:

```bash
conda activate plabench-boltz2
pip install -e forks/boltz
```

**DeepDTA's export has no `hydra-core`.** `deepdta.yaml` is the environment
`DEEPDTA_PYTHON` points at, and the cross-validation runner drives it as a
subprocess, so it never needed the driver's dependencies. Add them if you want
to run the non-CV DeepDTA datasets, which go through the in-process path:

```bash
conda activate plabench-deepdta
pip install hydra-core==1.3.2 omegaconf==2.3.0
```

## GLIBCXX_3.4.29 not found

On a host whose system `libstdc++` predates the one the wheels were built
against, importing pandas or scipy dies with `version GLIBCXX_3.4.29 not found
(required by .../_ckdtree.cpython-38-x86_64-linux-gnu.so)`. The environment has
a new enough copy; the dynamic loader just reaches `/lib64` first. Preload it:

```bash
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
```

On our hosts `deepdta` and `mfe` need it and the other eight do not, but that is
a property of the machine, not of the YAML. Check with
`python -c "import pandas, scipy.spatial"` after creating an environment.

## AlphaFold3

You do not need to install AlphaFold3 to run the benchmark. AF3 is not one of
the nine models; it is the upstream tool that built the input structures, and
every structure the paper scores is already in the Zenodo deposit as
`protein.pdb` + `ligand.sdf` + `ligand.mol2`. Unpack archive 02 into `data/` and
the structure-based configs will find what they need. That is why there is no
`environments/alphafold3.yaml`: AF3 is a container, not a conda environment, so
a YAML that looked like one would only mislead.

Installing it is only worth the trouble if you want to **add targets of your
own**. In that case, build the container from
[the official guide](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md),
request the model parameters from
[Google DeepMind](https://forms.gle/svvpY4u2jsHEwWYS6) (nobody may redistribute
them, this repository included), and fetch the genetic databases with
`./fetch_databases.sh <DB_DIR>`, about 630 GB uncompressed. The structures here
came from version 3.0.1 at commit a8ecdb2, run from a Singularity image because
the GPU cluster had no Docker daemon.

Then repeat the two steps that turn AF3 output into benchmark input:

```bash
# CIF -> protein.pdb + ligand.sdf + ligand.mol2, one directory per complex
python scripts/data_prep/gen_cofold_structures.py \
    --csv <cif_paths.csv> --output-dir data/AF3_structures/<your_set>

# only if some complexes failed above: sweep the lower-ranked samples
python scripts/data_prep/rescue_af3_conversion_failures.py --help
```

Each prediction gives ten ranked samples and the first script keeps the
top-ranked one. The second exists because that sample occasionally will not
convert; it walks the other nine in `pair_iptm` order and recovered 11 of the
released ChEMBL35 ligands that way. Point a new `configs/dataset/*.yaml` at the
output directory and the run works like any other.
