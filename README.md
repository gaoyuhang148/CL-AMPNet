# CL-AMPNet

CL-AMPNet is an antimicrobial peptide prediction model combining ProtT5 sequence features and ESM-2-predicted residue contacts. This repository provides the datasets and code for training, evaluation and ablation experiments.

## Installation

Use Linux with an NVIDIA GPU. From the repository directory:

```bash
conda create -n clamp_env python=3.11 -y
conda activate clamp_env
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

ProtT5 and ESM-2 weights are downloaded automatically during the first run. Internet access is required for this download.

## Project structure

```text
CL-AMPNet/
├── data/
│   ├── main/
│   │   ├── train.csv
│   │   ├── APD3_test.csv
│   │   ├── ...
│   │   └── YADAMP_test.csv
│   └── xu/
│       ├── train.csv
│       └── XUAMP_test.csv
├── src/
│   ├── features.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   └── ...
├── clamp_ablation/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── ...
├── run_experiment.py
├── run_training.py
├── evaluate.py
├── run_ablation.py
├── training_config.json
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

The datasets are included. CSV columns are `sequence` and `label` (`1`: AMP; `0`: non-AMP).

## Training and evaluation

### Main model

Train on `data/main/train.csv` and evaluate on APD3, dbAMP, DRAMP, LAMP and YADAMP.

```bash
python -u run_experiment.py --protocol main --stage all --device cuda --resume
```

### XUAMP benchmark

Train and evaluate using the original splits in `data/xu/`.

```bash
python -u run_experiment.py --protocol xu --stage all --device cuda --resume
```

Each command extracts features, trains five-fold models for seeds 42–46, and evaluates them at threshold 0.5. Settings are in `training_config.json`. `--resume` reuses completed folds and cached results.

### Evaluate saved models

After training, run evaluation separately with:

```bash
python -u run_experiment.py --protocol main --stage test --device cuda --resume
python -u run_experiment.py --protocol xu --stage test --device cuda --resume
```

## Ablation experiments

After completing the XUAMP benchmark run:

```bash
python -u run_ablation.py --stage all --device cuda --resume
```

This reuses the full model and trains the three variants: `no_contrastive`, `sequence_only` and `graph_only`.

## Results

- Main model: `results/main/paper_tables.md`
- XUAMP benchmark: `results/xu/paper_tables.md`
- Ablation experiments: `results/xu_ablation/paper_tables.md`

Checkpoints are saved under `runs/`. Prediction files and per-seed metrics are saved under the corresponding `results/` directory. Feature files are cached in `data/features/`.
