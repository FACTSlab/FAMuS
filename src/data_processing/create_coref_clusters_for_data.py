from datasets import load_dataset
import os
import argparse
from fastcoref import FCoref
from data_utils import loadJsonl
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", 
                        type=int, 
                        default=0)
    # data path
    parser.add_argument("--data_dir",
                        type=str,
                        default="../../data/cross_doc_role_extraction/")
    
    parser.add_argument("--output_dir",
                        type=str,
                        default="../../data/cross_doc_role_extraction/coref_clusters")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    coref_model = FCoref(device=f'cuda:{args.gpu}')
    # Load the data
    train_data = loadJsonl(os.path.join(args.data_dir, "train.jsonl"))
    dev_data = loadJsonl(os.path.join(args.data_dir, "dev.jsonl"))
    test_data = loadJsonl(os.path.join(args.data_dir, "test.jsonl"))
    # Get the unique contexts
    unique_context_tuples = []
    for instance in train_data + dev_data + test_data:
        unique_context_tuples.append((instance['instance_id'],
                                      {'report_doctext': instance['report_dict']['doctext'],
                                       'source_doctext': instance['source_dict']['doctext']})),
                                    
    print(f"Unique contexts for coref: {len(unique_context_tuples)}")
    unique_id_to_source_coref_clusters = {}
    ########################################################
    ### Report Coref Clusters
    ########################################################
    # Get the coref chains for report
    coref_preds_report = coref_model.predict(
                            texts=[doctext_dict['report_doctext'] for _, doctext_dict in 
                                    unique_context_tuples]
                                )
    # For report
    for unique_id, coref_pred_report in zip([instance_id for instance_id, _ in unique_context_tuples], 
                                             coref_preds_report):
        unique_id_to_source_coref_clusters[unique_id] = {'report': coref_pred_report.get_clusters()}
    ########################################################
    ### Source Coref Clusters
    ########################################################
    # Get the coref chains for source
    coref_preds_source = coref_model.predict(
                texts=[doctext_dict['source_doctext'] for _, doctext_dict in
                                    unique_context_tuples]
                    )
    for unique_id, coref_pred_source in zip([instance_id for instance_id, _ in unique_context_tuples],
                                             coref_preds_source):
        unique_id_to_source_coref_clusters[unique_id]['source'] = coref_pred_source.get_clusters()

    ########################################################
    ## Save the coref clusters
    ########################################################
    with open(os.path.join(args.output_dir, "instance_id_to_source_coref_clusters.json"), "w") as f:
        json.dump(unique_id_to_source_coref_clusters, f)


if __name__ == "__main__":
    main()
