#!/bin/bash
# Add the project root to PYTHONPATH
export PYTHONPATH=.:$PYTHONPATH

python src/model_scripts/test_source_val.py \
        --input_data_dir "data/source_validation" \
        --model_checkpoint_path "models/source_validation/results/best_model" \
        --split "dev" \
        --gpu 1

python src/model_scripts/test_source_val.py \
        --input_data_dir "data/source_validation" \
        --model_checkpoint_path "models/source_validation/results/best_model" \
        --split "test" \
        --gpu 1