FAMUS_RELEASE_DIR="/data/svashishtha/FAMuS/data/cross_doc_role_extraction/"

echo "Running the IterX format preprocessing pipeline"
python preprocess_release_format_to_iterx_format.py \
        --input_dir $FAMUS_RELEASE_DIR \
        --output_dir "/data/svashishtha/FAMuS/data/cross_doc_role_extraction/iterx_format"

echo "Running the QA format preprocessing pipeline"
python preprocess_release_format_to_qa_format.py \
        --input_dir $FAMUS_RELEASE_DIR \
        --output_dir "/data/svashishtha/FAMuS/data/cross_doc_role_extraction/qa_format"

echo "Running the LLM prompt format preprocessing pipeline"
python preprocess_release_format_to_llm_prompt_format.py \
        --input_dir $FAMUS_RELEASE_DIR \
        --output_dir "/data/svashishtha/FAMuS/data/cross_doc_role_extraction/llm_prompt_format"