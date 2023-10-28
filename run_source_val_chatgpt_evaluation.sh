#!/bin/bash
# Add the project root to PYTHONPATH
export PYTHONPATH=.:$PYTHONPATH

python src/model_scripts/test_chatgpt_on_source_val.py \
        --input_dir "data/source_validation/llm_prompt_format/" \
        --output_dir "models/source_validation/chatgpt/" \
        --split "test" 