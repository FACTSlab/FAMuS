from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import os
from datasets import load_dataset
import argparse
from transformers import pipeline
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from fastcoref import FCoref
from src.metrics.cross_doc_role_metrics import (
                              tp_fp_fn_tn_role_agreement_multiple_gold,
                              tp_fp_fn_tn_role_agreement_single_gold,
                              agreement_score,
                              exact_match_score)

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list
    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]
    

def instance2answerAndcluster(instance,
                              unique_id_to_source_coref_clusters,
                              report_or_source="report"):
    """
    Given a QA instance, find the cluster that contains the answer span
    using the coref clusters from the source document.
    """
    qa_id = instance['id']
    famus_id = qa_id.split("-Role-")[0]
    coref_clusters = unique_id_to_source_coref_clusters[famus_id][report_or_source]
    actual_answer = instance['answers']['text']
    if actual_answer == []:
        actual_answer = ""
    else:
        actual_answer = actual_answer[0]

    matched_cluster = []    
    for cluster in coref_clusters:
        for entity_span in cluster:
            if entity_span == actual_answer:
                matched_cluster = set(cluster)

    if matched_cluster == []:
        matched_cluster = set(instance['answers']['text'])

    return actual_answer, matched_cluster


def compute_metrics(dataset,
                    results,
                    unique_id_to_source_coref_clusters,
                    report_or_source="report",
                    eval_metric_fn=tp_fp_fn_tn_role_agreement_multiple_gold,
                    exact_match = False
                    ):
    """
    Given a dataset and results, compute the metrics

    Args:
        dataset: a dataset object
        results: a list of dictionaries with keys: id, answer, score
        unique_id_to_source_coref_clusters: a dictionary mapping from
                                            famus_id to coref clusters
        report_or_source: whether to use the report or source coref clusters
        eval_metric_fn: a function that takes in a gold answer and a predicted
                        answer and returns the tp, fp, fn, tn scores
        exact_match: whether to use exact match or agreement score in the
                        eval_metric_fn
    """
    if exact_match:
        agreement_fn = exact_match_score
    else:
        agreement_fn = agreement_score

    # we fetch the first answer from the top k answers (this is a temporary
    # way of getting the answer, we will change this later)
    results_based_on_threshold = [result[0]['answer'] for result in results]
    tps, fps, fns, tns = 0, 0, 0, 0
    for idx, prediction in tqdm(enumerate(results_based_on_threshold)):
        instance = dataset[idx]
        gold_answer, gold_cluster = instance2answerAndcluster(instance,
                                                              unique_id_to_source_coref_clusters,
                                                              report_or_source=report_or_source)
        if "multiple_gold" in eval_metric_fn.__name__:
            gold_answer = gold_cluster

        c_tp, c_fp, c_fn, c_tn = eval_metric_fn(gold_answer, 
                                                prediction,
                                                agreement_fn=agreement_fn)

        tps += c_tp
        fps += c_fp
        fns += c_fn
        tns += c_tn

    # round to 2 decimal places
    output_string_metrics = f"Precision: {tps*100/(tps+fps):.2f}, Recall: {tps*100/(tps+fns):.2f}, F1: {2*tps*100/(2*tps+fps+fns):.2f}, Accuracy: {(tps+tns)*100/(tps+tns+fps+fns):.2f}"
    output_string_tp_fp_fn_tn = f"TP: {tps:.2f}, FP: {fps:.2f}, FN: {fns:.2f}, TN: {tns:.2f}"

    return output_string_metrics + "\n"  + output_string_tp_fp_fn_tn

    
def parse_args():

    args = argparse.ArgumentParser()
    
    args.add_argument("--input_data_dir",
                        type=str,
                        default="../../data/cross_doc_role_extraction/qa_format/report_data")
    
    args.add_argument("--coref_cluster_data_path",
                        type=str,
                        default="../../data/cross_doc_role_extraction/coref_clusters/")

    args.add_argument("--model_checkpoint", 
                      type=str, 
                      default="../../models/cross_doc_role_extraction/qa_report/best_model/")
    
    args.add_argument("--split",
                        type=str,
                        default="dev",
                        help="which split to evaluate on: dev or test")
    
    args.add_argument("--gpu",
                        type=int,
                        default=1)
    
    
    return args.parse_args()

def main():
    args = parse_args()
    # based on input dir, figure out if its report or source data
    if "report" in args.input_data_dir:
        report_or_source = "report"
    elif "source" in args.input_data_dir:
        report_or_source = "source"
    # Load the dataset
    datasets = load_dataset("json", data_files={'train': os.path.join(args.input_data_dir, 
                                                                      "train.json"),
                                             'validation': os.path.join(args.input_data_dir,
                                                                         "dev.json"),
                                            'test': os.path.join(args.input_data_dir, 
                                                                 "test.json")})
    if args.split == "dev":
        dataset = datasets["validation"]
    elif args.split == "test":
        dataset = datasets["test"]

    # Evaluate on the dataset
    data_list = ListDataset(dataset)

    # If results already exist, skip
    if not os.path.exists(os.path.join(args.model_checkpoint,
                                    f"results_{args.split }.json")):
        # Load the model
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        

        # Invoke the pipeline
        question_answerer = pipeline("question-answering", 
                                    model=model,
                                    tokenizer=tokenizer,
                                    device=args.gpu)
        

        results = []
        for out in tqdm(question_answerer(data_list, 
                                        doc_stride=256, 
                                        max_answer_len = 64,
                                        max_seq_len = 2048,
                                        handle_impossible_answer=True,
                                        top_k=5)):
            results.append(out)
        
        # Save the results
        with open(os.path.join(args.model_checkpoint, 
                                f"results_{args.split }.json"), "w") as f:
            json.dump(results, f)

    else:
        print(f"{args.split} results already exist on this model. Skipping inference...")

    ########################################################
    # Compute Metrics and Output to File
    ########################################################
    with open(os.path.join(args.coref_cluster_data_path,
                           "instance_id_to_coref_clusters.json") , "r") as f:
        unique_id_to_source_coref_clusters = json.load(f)

    results_from_file = json.load(open(os.path.join(args.model_checkpoint, 
                            f"results_{args.split }.json")))
    
    metrics_file_string = ""
    metrics_file_string += "Exact match without Coref\n"
    metrics_file_string += compute_metrics(dataset,
                    results_from_file,
                    unique_id_to_source_coref_clusters,
                    report_or_source=report_or_source,
                    eval_metric_fn=tp_fp_fn_tn_role_agreement_single_gold,
                    exact_match=True
                    )
    metrics_file_string += "\n\nExact match with Coref\n"
    metrics_file_string += compute_metrics(dataset,
                    results_from_file,
                    unique_id_to_source_coref_clusters,
                    report_or_source=report_or_source,
                    eval_metric_fn=tp_fp_fn_tn_role_agreement_multiple_gold,
                    exact_match=True
                    )
    metrics_file_string += "\n\nAgreement without Coref\n"
    metrics_file_string += compute_metrics(dataset,
                    results_from_file,
                    unique_id_to_source_coref_clusters,
                    report_or_source=report_or_source,
                    eval_metric_fn=tp_fp_fn_tn_role_agreement_single_gold,
                    exact_match=False
                    )
    metrics_file_string += "\n\nAgreement with Coref\n"
    metrics_file_string += compute_metrics(dataset,
                    results_from_file,
                    unique_id_to_source_coref_clusters,
                    report_or_source=report_or_source,
                    eval_metric_fn=tp_fp_fn_tn_role_agreement_multiple_gold,
                    exact_match=False
                    )
    with open(os.path.join(args.model_checkpoint,
                            f"metrics_{args.split}.txt"), "w") as f:
        f.write(metrics_file_string)
    

if __name__ == "__main__":
    main()