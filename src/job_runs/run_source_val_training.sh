#!/bin/bash

# Specify GPU
export CUDA_VISIBLE_DEVICES=2

# Add the project root to PYTHONPATH
export PYTHONPATH=.:$PYTHONPATH

# Run the script
python src/model_scripts/train_source_val.py \
        --input_dir "data/source_validation" \
        --output_dir "models/source_validation" \
        --seed 42