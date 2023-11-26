#!/bin/bash
# Add the project root to PYTHONPATH
export PYTHONPATH=.:$PYTHONPATH

python src/model_scripts/test_chatgpt_on_cdae.py \
        --input_dir "data/cross_doc_role_extraction/llm_prompt_format/" \
        --output_dir "models/cdae/chatgpt/" \
        --split "dev" 

python src/model_scripts/test_chatgpt_on_cdae.py \
        --input_dir "data/cross_doc_role_extraction/llm_prompt_format/" \
        --output_dir "models/cdae/chatgpt/" \
        --split "test" 