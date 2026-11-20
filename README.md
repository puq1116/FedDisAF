# FedDisAF

Feature-space decomposition, Gaussian-perturbed class prototypes, cosine-weighted
prototype fusion, and classifier-based prototype supervision for federated crop
disease and pest classification.

This archive contains **FedDisAF and seven base-method + FedDisAF combinations**:
FedAvg, FedNova, FedProx, SCAFFOLD, FedALA, FedFTG, and FedPop. Selecting a base
method always runs its FedDisAF combination. Independent baselines and FedFed
comparison experiments are outside this archive.

## Installation

Use Python 3.10 or later. Run from the repository root:

```bash
python -m pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yaml
conda activate feddisaf
```

The torch/torchvision versions are taken from the supplied environment export.
Install the appropriate PyTorch CUDA build for your machine when necessary.
An optional editable installation, `python -m pip install -e .`, provides the
`feddisaf` command. No data, model weights, or training outputs are bundled.

## Prepare data

Provide explicit training and test splits using the same class vocabulary.
`--data-dir` points to the parent of `PlantVillage/` and `ip102_v1.1/`.

## Run a combination

```bash
python main.py --config configs/plantvillage.yaml --algorithm FedAvg --data-dir ./data
python main.py --config configs/plantvillage.yaml --algorithm FedProx --data-dir ./data
python main.py --config configs/ip102.yaml --algorithm SCAFFOLD --data-dir ./data
```

Supported `--algorithm` values:

| Value | Executed combination |
|---|---|
| `FedAvg` | FedAvg + FedDisAF |
| `FedNova` | FedNova + FedDisAF |
| `FedProx` | FedProx + FedDisAF |
| `SCAFFOLD` | SCAFFOLD + FedDisAF |
| `FedALA` | FedALA + FedDisAF |
| `FedFTG` | FedFTG + FedDisAF |
| `FedPop` | FedPop + FedDisAF |
| `all` | All seven combinations, sequentially |

For example:

```bash
python main.py --config configs/plantvillage.yaml --algorithm all --data-dir ./data
python main.py --config configs/plantvillage.yaml --algorithm FedProx --local-epochs 3 --partition-alpha 0.1
python main.py --config configs/ip102.yaml --algorithm FedPop --print-config
python main.py --help
```

CLI options override YAML. Each YAML file defaults to `FedAvg+FedDisAF`, ten
clients, five participants, batch size 32, one local epoch, and pretrained
ResNet-18. Round budgets are 1000 for PlantVillage and 200 for IP102. `all`
reuses one data partition and resets training RNGs before each combination.
`--no-pretrained --device cpu` supports local checks without downloading weights.

The original model-weight attention setting is retained as
`--feddisaf-num-heads` (source default: 8). It affects the FedALA/FedFTG/FedPop
model-aggregation branches. Setting it to 0 keeps FedDisAF enabled and still uses
cosine prototype fusion. It is not a switch to a baseline or a FedFed mode.

## Independent runs and reporting

Launch a separate process for each seed of one combination:

```bash
python scripts/run_seeds.py --seeds 1 2 3 --config configs/plantvillage.yaml --algorithm FedAvg --data-dir ./data
```

The default summary averages the final five rounds within each run and reports
the mean and sample standard deviation across the run-level means.
Use the generated `aggregate.json` for cross-seed results. Single-run
`within_run_round_std_*` fields describe round fluctuations, not cross-seed SD.
Accuracy and recall in the aggregate are both percentages.

Each run saves configuration, full-precision histories, metrics, a partition
manifest, and optionally the best/final model.

## Method integration

```text
CLI base-method selection
          |
          v
    WithFedDisAF combination
      |                 |
      |                 +--> method-specific model-state aggregation
      |
      +--> backbone -> private/shared features
      +--> local prediction + class-averaged prototype CE
      +--> noisy local class prototypes -> FedDisAFPlugin -> global library
```

The shared [FedDisAFPlugin](feddisaf/plugin.py) handles prototype supervision and
library fusion. Each file in `feddisaf/combinations/` provides one integrated
training path and its necessary internal aggregation helpers. 

## Layout

```text
feddisaf/
  plugin.py             shared FedDisAF prototype pathway
  models.py             feature decomposition and local classifier
  combinations/         seven WithFedDisAF implementations and registry
  legacy_attention.py   source model-weight attention dependency
  training.py           round loop, evaluation, and outputs
  cli.py                combinations-only configuration interface
  data.py               source data loading and allocation
  logger.py             per-round logging
configs/                PlantVillage and IP102 examples
scripts/                independent-seed execution and aggregation
tests/                  numerical, scope, and integration checks
docs/                   method, data, and provenance notes
main.py                 training entry point
```

Run checks with `python -m unittest discover -s tests -v`. 

