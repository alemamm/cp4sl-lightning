#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule_swat.sh

python src/train.py experiment=swat_static
python src/train.py experiment=swat_dynamic

# If you want to save the results in a csv file instead of using wandb, add " logger=csv" to the commands above
