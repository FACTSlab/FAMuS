#!/bin/bash
# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

FAMUS_RELEASE_DIR="$SCRIPT_DIR/../../data/source_validation/"

echo "Running the LLM Prompt format preprocessing pipeline for Source Validation"
python "$SCRIPT_DIR/preprocess_source_val_to_llm_prompt.py" \
        --input_dir "$FAMUS_RELEASE_DIR" \
        --output_dir "$SCRIPT_DIR/../../data/source_validation/llm_prompt_format"