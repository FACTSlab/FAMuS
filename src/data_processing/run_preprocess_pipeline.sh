#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

FAMUS_RELEASE_DIR="$SCRIPT_DIR/../../data/cross_doc_role_extraction/"

echo "Running the IterX format preprocessing pipeline"
python "$SCRIPT_DIR/preprocess_release_format_to_iterx_format.py" \
        --input_dir "$FAMUS_RELEASE_DIR" \
        --output_dir "$SCRIPT_DIR/../../data/cross_doc_role_extraction/iterx_format"

echo "Running the QA format preprocessing pipeline"
python "$SCRIPT_DIR/preprocess_release_format_to_qa_format.py" \
        --input_dir "$FAMUS_RELEASE_DIR" \
        --output_dir "$SCRIPT_DIR/../../data/cross_doc_role_extraction/qa_format"

echo "Running the LLM prompt format preprocessing pipeline"
python "$SCRIPT_DIR/preprocess_release_format_to_llm_prompt_format.py" \
        --input_dir "$FAMUS_RELEASE_DIR" \
        --output_dir "$SCRIPT_DIR/../../data/cross_doc_role_extraction/llm_prompt_format"