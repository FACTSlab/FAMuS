#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

FAMUS_RELEASE_DIR="$SCRIPT_DIR/../../data/cross_doc_role_extraction/"

echo "Running the IterX format preprocessing pipeline (mixed spans)"
python "$SCRIPT_DIR/preprocess_release_format_to_iterx_format.py" \
        --input_dir "$FAMUS_RELEASE_DIR" \
        --output_dir "$SCRIPT_DIR/../../data/cross_doc_role_extraction/iterx_format"

echo "Running the IterX format preprocessing pipeline (gold spans)"
python "$SCRIPT_DIR/preprocess_iterx_format_to_gold_spans.py" \
        --input_dir "$SCRIPT_DIR/../../data/cross_doc_role_extraction/iterx_format"

echo "Running the IterX format preprocessing pipeline (predicted spans)"
python "$SCRIPT_DIR/preprocess_iterx_format_to_predicted_spans.py" \
        --input_release_dir "$FAMUS_RELEASE_DIR" \
        --input_iterx_format_dir "$SCRIPT_DIR/../../data/cross_doc_role_extraction/iterx_format"

echo "Running the QA format preprocessing pipeline"
python "$SCRIPT_DIR/preprocess_release_format_to_qa_format.py" \
        --input_dir "$FAMUS_RELEASE_DIR" \
        --output_dir "$SCRIPT_DIR/../../data/cross_doc_role_extraction/qa_format"

echo "Running the LLM prompt format preprocessing pipeline"
python "$SCRIPT_DIR/preprocess_release_format_to_llm_prompt_format.py" \
        --input_dir "$FAMUS_RELEASE_DIR" \
        --output_dir "$SCRIPT_DIR/../../data/cross_doc_role_extraction/llm_prompt_format"

echo "Run the Coref Clusters for all Cross-Doc-Role Extraction data"
python "$SCRIPT_DIR/create_coref_clusters_for_data.py" \
        --data_dir "$FAMUS_RELEASE_DIR" \
        --output_dir "$FAMUS_RELEASE_DIR/coref_clusters" \
        --gpu 2