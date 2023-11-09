#!/bin/bash
# Add the project root to PYTHONPATH
export PYTHONPATH=.:$PYTHONPATH

python src/model_scripts/test_source_val.py \
        --input_data_dir "data/source_validation" \
        --model_checkpoint_path "models/source_validation/results/best_model" \
        --chatgpt_output_file_path "models/source_validation/chatgpt/dev_gpt-3.5-turbo-0301_responses.jsonl" \
        --split "dev" \
        --metrics_output_dir "src/metrics/" \
        --gpu 1

python src/model_scripts/test_source_val.py \
        --input_data_dir "data/source_validation" \
        --chatgpt_output_file_path "models/source_validation/chatgpt/test_gpt-3.5-turbo-0301_responses.jsonl" \
        --model_checkpoint_path "models/source_validation/results/best_model" \
        --split "test" \
        --metrics_output_dir "src/metrics/"
        --gpu 1