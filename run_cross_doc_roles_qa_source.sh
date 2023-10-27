#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --time=24:0:0
#SBATCH --gpus=1

eval "$(conda shell.bash hook)"
conda activate iterx

cd /brtx/601-nvme1/svashis3/FAMuS

# Add the project root to PYTHONPATH
export PYTHONPATH=.:$PYTHONPATH

# Run the script
python src/model_scripts/train_cross_doc_role_extraction_qa.py \
        --input_data_dir "data/cross_doc_role_extraction/qa_format/source_data/" \
        --model_output_dir "models/cross_doc_role_extraction/qa_source/" \
        --epochs 15 \
        --save_total_limit 1 \
        --num_optuna_trials 5 \
        --experiment_name "finetuned-source-qa-optuna"