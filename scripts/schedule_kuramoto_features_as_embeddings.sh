#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule_kuramoto_features_as_embeddings.sh

python src/train.py experiment=kuramoto_features_embeddings_seed_0
python src/train.py experiment=kuramoto_features_embeddings_seed_1
python src/train.py experiment=kuramoto_features_embeddings_seed_2
python src/train.py experiment=kuramoto_features_embeddings_seed_3
python src/train.py experiment=kuramoto_features_embeddings_seed_4

# If you want to save the results in a csv file instead of using wandb, add " logger=csv" to the commands above
