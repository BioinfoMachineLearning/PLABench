# Configs

`run_benchmark.py` is a Hydra application. A run is one model config crossed
with one dataset config:

```bash
python run_benchmark.py model=<model> dataset=<dataset>
```

`config.yaml` sets the defaults (`model: haiping`, `dataset:
casp16_l1000_stage1`) and sends output to
`outputs/${model.name}/${dataset.name}/latest`. Any field can be overridden on
the command line, for example `++model.device=cpu`.

## `model/`

One file per model: `bapred`, `boltz2`, `deepdta`, `flowdock`, `flowr_root`,
`haiping`, `llf`, `mfe`, `mixingdta`. Each names the checkpoint, the model
source under `forks/`, and the interpreter or conda prefix to run it with.

The interpreter paths are the ones used for the published runs, written as
`${oc.env:VAR,default}` so an environment variable overrides them without
editing the file. The top-level README lists the variables.

## `dataset/`

68 configs, one per combination of input source, benchmark and evaluation mode
that the paper reports. Runs that did not make the paper live in
`archive/configs/dataset/`.

### Naming

A config name is built from up to four tokens:

    [<model_family>_]<input_source>_<benchmark>[_<stage>]

`<model_family>` appears only when a model needs its own input specification for
the same underlying data. `mfe_`, `haiping_` and `flowr_root_` point at
pre-prepared copies or add fields those models require, for example
`smiles_source_dir`. `sequence_` marks the shared config for the three
sequence-only models, which read a CSV of SMILES and target sequences rather
than structures. A name with no model prefix is the generic structure config
that BA-Pred and FlowDock consume.

`<input_source>` names the receptor and pose handed to the model:

| Token | Input |
| --- | --- |
| `experimental`, `_exp` | Deposited crystal complex |
| `af3` | AlphaFold3 top-1 predicted complex |
| `boltz2` | Boltz-2 unguided co-folded complex |
| `tplpocket` | Boltz-2 co-folded with the experimental receptor as template and a pocket constraint |
| absent | The benchmark's own default input |

Watch the token order on the CASF and CSAR configs: `boltz2_casf2013` is the
Boltz-2 **model** scoring CASF-2013, while `casf2013_boltz2` is any model
scoring **Boltz-2-predicted structures** of CASF-2013. Boltz-2 predicts affinity
from sequence and does not read an external structure, so it needs its own
config wherever the other models take a structure argument.

`<benchmark>` is `casp16_l1000` / `casp16_l3000`, `chembl35_full`,
`chembl35_multimer`, `casf2013` / `casf2016`, `csar_hiq36` / `csar_hiq51`, or
the Davis / KIBA cross-validation splits.

`<stage>` is the CASP16 round: `stage1` means the model predicted the complex
itself, `stage2` means the organizers supplied the poses.

### The two ChEMBL35 configs are different experiments

They are not duplicates. Both are needed.

`af3_chembl35_full` is the main ChEMBL35 benchmark: 84 targets, 7650
protein-ligand pairs, one AlphaFold3 monomer per target. Every ChEMBL35 number
in the main results table comes from this config. Ground truth is
`data/chembl35/chembl35_full_input.csv`.

`af3_chembl35_multimer` is the multimer chain-count ablation behind the ion
channel paragraph in the protein-family section. It covers two targets only, the
homo-pentameric nicotinic acetylcholine receptor alpha7 (UniProt P36544) and the
homo-tetrameric hERG channel (UniProt Q12809), and re-folds each complex at two
chain counts under template guidance. The `condition` column holds the arm:
`k1` is the single-chain monomer, `k5` the alpha7 pentamer, `k4` the hERG
tetramer. That gives 39 matched monomer-versus-assembly pairs, 19 for alpha7 and
20 for hERG, which are the `n` values quoted in the paper. Ground truth travels
with the structures in `data/AF3_structures/chembl35_multimer/ground_truth.csv`.

The two configs answer different questions, so they are scored separately and
their numbers are not comparable: one measures accuracy across a broad target
panel, the other measures whether adding the missing subunits changes anything
on two targets where every model fails.

### `tplpocket` means template plus pocket

`tplpocket_casp16_l3000_stage1` is the second rung of the four-tier pose-quality
ladder in the input-pose-accuracy figure. Boltz-2 re-folded each CASP16 L3000
complex while given the experimental receptor as a structural template **and** a
forced pocket constraint, which pulls the median ligand RMSD down from 6.96 A
for unguided co-folding to 3.72 A. It fills the gap between a free prediction
and the near-native AlphaFold3 pose.

The full ladder, low to high pose quality:

| Rung | Config prefix | Median ligand RMSD |
| --- | --- | --- |
| Boltz-2 unguided co-fold | `boltz2_` | 6.96 A |
| Boltz-2 template + pocket | `tplpocket_` | 3.72 A |
| AlphaFold3 top-1 | `af3_` | 0.76 A |
| Deposited crystal | `experimental_` | 0 A |

`tplpocket` is a contraction of "template + pocket". Each of the four model
families has its own copy of the config, so all four `*tplpocket*` files
describe the same 93 structures under `data/Boltz2_tplpocket_structures/L3000`.
