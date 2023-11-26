#!/bin/bash
# Add the project root to PYTHONPATH
export PYTHONPATH=.:$PYTHONPATH

python src/model_scripts/test_source_val.py \
        --input_data_dir "data/source_validation" \
        --model_checkpoint_path "models/source_validation/longformer_source_val_best_model" \
        --chatgpt_output_file_path "models/source_validation/chatgpt/dev_gpt-3.5-turbo-0301_responses.jsonl" \
        --llama_output_file_path "models/source_validation/llama/dev_sv_llama_13b_responses.jsonl" \
        --split "dev" \
        --metrics_output_dir "src/metrics/" \
        --gpu 0

python src/model_scripts/test_source_val.py \
        --input_data_dir "data/source_validation" \
        --model_checkpoint_path "models/source_validation/longformer_source_val_best_model" \
        --chatgpt_output_file_path "models/source_validation/chatgpt/test_gpt-3.5-turbo-0301_responses.jsonl" \
        --llama_output_file_path "models/source_validation/llama/test_sv_llama_13b_responses.jsonl" \
        --split "test" \
        --metrics_output_dir "src/metrics/" \
        --gpu 0