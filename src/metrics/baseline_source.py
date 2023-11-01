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


def format_as_prediction(data):
    """
    Formats the gold report data as a predicted source data
    """
    



def baseline_source(data, unique_id_to_source_coref_clusters, report_or_source="report"):
    """
    Baselines source extraction using report data
    """
    score = compute_metrics(data,
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

    report_as_results = format_as_prediction(data)

    data_list = ListDataset(data)

    # /home/amartin/famus/FAMuS/data/cross_doc_role_extraction/coref_clusters/instance_id_to_coref_clusters.json
    with open(os.path.join(args.data_dir, "coref_clusters/instance_id_to_coref_clusters.json"), "r") as f:
        unique_id_to_source_coref_clusters = json.load(f)

    baseline_source(data_list, unique_id_to_source_coref_clusters, report_or_source="source")

    # get baseline score


    

    
if __name__ == "__main__":
    main()
