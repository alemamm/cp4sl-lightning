______________________________________________________________________

<div align="center">

# Self-Supervised Structure Learning for Cyber-Physical Systems

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

What it does

## Installation

#### Conda

```bash
cd cp4sl-lightning

# create conda environment and install dependencies
conda env create -f environment.yaml -n cp4sl

# activate conda environment
conda activate cp4sl
```

## How to run

### Scripted experiments

To run five seeded experiments for each respective configuration run:

Static
```bash
bash scripts/schedule_kuramoto_static.sh
```

Dynamic
```bash
bash scripts/schedule_kuramoto_dynamic.sh
```

Correlation
```bash
bash scripts/schedule_kuramoto_correlation_adj.sh
```

True
```bash
bash scripts/schedule_kuramoto_true_adj.sh
```

Full
```bash
bash scripts/schedule_kuramoto_full_graph.sh
```

Features
```bash
bash scripts/schedule_kuramoto_features_as_embeddings.sh
```

### Running single experiments

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
