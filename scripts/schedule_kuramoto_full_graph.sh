#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule_kuramoto_full_graph.sh

python src/train.py experiment=kuramoto_full_graph_seed_0
python src/train.py experiment=kuramoto_full_graph_seed_1
python src/train.py experiment=kuramoto_full_graph_seed_2
python src/train.py experiment=kuramoto_full_graph_seed_3
python src/train.py experiment=kuramoto_full_graph_seed_4

# If you want to save the results in a csv file instead of using wandb, add " logger=csv" to the commands above
