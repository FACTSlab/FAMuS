# Add the project root to PYTHONPATH
export PYTHONPATH=.:$PYTHONPATH

# Run the script
# Report QA
python src/model_scripts/test_cross_doc_role_extraction_qa.py \
        --input_data_dir "data/cross_doc_role_extraction/qa_format/report_data/" \
        --coref_cluster_data_path "data/cross_doc_role_extraction/coref_clusters/" \
        --model_checkpoint "models/cdae/best_model_report_qa/" \
        --split "dev" \
        --gpu 1

python src/model_scripts/test_cross_doc_role_extraction_qa.py \
        --input_data_dir "data/cross_doc_role_extraction/qa_format/report_data/" \
        --coref_cluster_data_path "data/cross_doc_role_extraction/coref_clusters/" \
        --model_checkpoint "models/cdae/best_model_report_qa/" \
        --split "test" \
        --gpu 1

# Source QA
python src/model_scripts/test_cross_doc_role_extraction_qa.py \
        --input_data_dir "data/cross_doc_role_extraction/qa_format/source_data/" \
        --coref_cluster_data_path "data/cross_doc_role_extraction/coref_clusters/" \
        --model_checkpoint "models/cdae/best_model_source_qa/" \
        --split "dev" \
        --gpu 1

python src/model_scripts/test_cross_doc_role_extraction_qa.py \
        --input_data_dir "data/cross_doc_role_extraction/qa_format/source_data/" \
        --coref_cluster_data_path "data/cross_doc_role_extraction/coref_clusters/" \
        --model_checkpoint "models/cdae/best_model_source_qa/" \
        --split "test" \
        --gpu 1
