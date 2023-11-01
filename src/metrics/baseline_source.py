"""
This is just to get the baseline score of the source exraction 
by only extracting from the report (perfectly extracting from report)
"""
import sys
import os
import json
import numpy as np
import argparse
# /home/amartin/famus/FAMuS/src/model_scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.model_scripts.test_cross_doc_role_extraction_qa import (
                                            compute_metrics, ListDataset,
                                            instance2answerAndcluster)

from src.metrics.cross_doc_role_metrics import (
                                tp_fp_fn_tn_role_agreement_multiple_gold,
                                tp_fp_fn_tn_role_agreement_single_gold,
                                agreement_score,
                                exact_match_score)




def baseline_source(data):
    """
    Baselines source extraction using report data
    """
    score = compute_metrics(data,
                            data,
                            unique_id_to_source_coref_clusters,
                            report_or_source="report",
                            eval_metric_fn=tp_fp_fn_tn_role_agreement_multiple_gold,
                            exact_match = False
                            )
        


def main():
    parser = argparse.ArgumentParser(description='Baseline source extraction')
    parser.add_argument("--data_dir", type=str, default="data/cross_doc_role_extraction", help="directory of data") 
    parser.add_argument("--dataset", type=str, default="test", help="dataset to analyze") # train, dev, test
    args = parser.parse_args()

    # load data
    if args.dataset == "all":
        data = []
        for dataset in ["train", "dev", "test"]:
            with open(os.path.join(args.data_dir, dataset + ".jsonl"), "r") as f:
                for line in f:
                    data.append(json.loads(line))

    else:
        with open(os.path.join(args.data_dir, args.dataset + ".jsonl"), "r") as f:
            data = [json.loads(line) for line in f]

    data_list = ListDataset(data)

    # get baseline score


    

    
if __name__ == "__main__":
    main()
