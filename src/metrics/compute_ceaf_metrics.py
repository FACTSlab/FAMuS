import sys
sys.path.append("/data/svashishtha/iterx/src/")
import json
from iterx.metrics.famus.iterx_famus import IterXFAMuSMetric
import numpy as np
from tqdm import tqdm
import argparse
import os
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_compute_ceafe_rme_scores(gold_file, predictions,
                                   ignore_no_template_doc = False ,
                                   sanitize_special_chars= False,):
    # Exact Match
    iterx_famus = IterXFAMuSMetric({gold_file: gold_file},
                               scorer_type = 'phi-3',
                               ignore_no_template_doc = ignore_no_template_doc,
                               sanitize_special_chars = sanitize_special_chars)
    iterx_famus(predictions, 
            gold_file,
            normalize_role = False)
    
    exact_match_dict = iterx_famus.get_metric(reset=True)
    metrics_string = ""
    metrics_string += f"{exact_match_dict['iterx_famus_slot_p']*100:.2f} & "
    metrics_string += f"{exact_match_dict['iterx_famus_slot_r']*100:.2f} & "
    metrics_string += f"{exact_match_dict['iterx_famus_slot_f1']*100:.2f} & "

    # Soft Match
    iterx_famus = IterXFAMuSMetric({gold_file: gold_file},
                                   scorer_type = 'phi-3-levenshtein',
                                   ignore_no_template_doc = ignore_no_template_doc,
                                   sanitize_special_chars = sanitize_special_chars)
    iterx_famus(predictions,
                gold_file,
                normalize_role = False)
    soft_match_dict = iterx_famus.get_metric(reset=True)

    metrics_string += f"{soft_match_dict['iterx_famus_slot_p']*100:.2f} & "
    metrics_string += f"{soft_match_dict['iterx_famus_slot_r']*100:.2f} & "
    metrics_string += f"{soft_match_dict['iterx_famus_slot_f1']*100:.2f} & "
    print(metrics_string)

    return metrics_string 


def aggregate_predictions_for_QA(gold_ids, predictions):
    result = {}
    for gold_id, role_fillers in zip(gold_ids, predictions):
        # Split the gold_id into famus_id and Role
        famus_id, _, role = gold_id.rpartition('-Role-')

        # Get the incident_type from the famus_id
        incident_type = famus_id.split('-frame-', 1)[-1]

        # Initialize the dictionary for this famus_id if it doesn't exist
        if famus_id not in result:
            result[famus_id] = [{'incident_type': incident_type}]

        # Add the prediction to the appropriate role
        if role_fillers:  # Ignore entries with empty predictions
            result[famus_id][0][role] = role_fillers

    # Remove incident_type from the dicts that have no other keys
    for famus_id in result:
        if len(result[famus_id][0]) == 1:  # Only 'incident_type' is present
            result[famus_id] = []

    return result


def get_results_based_on_threshold_qa(results,
                                   threshold = 0.02):
    """
    Given a list of QA results, 
    return a list of predictions for each role
    based on the threshold
    """
    role_predictions = []
    for role_instance in results:
        current_instance_predictions = []
        for span_pred in role_instance:
            if span_pred['score'] >= threshold:
                answer = span_pred['answer']
                if answer:
                    current_instance_predictions.append([span_pred['answer']])
        role_predictions.append(current_instance_predictions)

    return role_predictions


def get_highest_score_results_qa(results,
                            ):
    """
    Given a list of QA results, 
    return the span with highest score for each role
    """
    role_predictions = []
    for role_instance in results:
        current_instance_predictions = []
        span_pred = role_instance[0]
        answer = span_pred['answer']
        if answer:
            current_instance_predictions.append([span_pred['answer']])
        role_predictions.append(current_instance_predictions)

    return role_predictions

def chatgpt_response_to_iterx_format(chatgpt_predictions):
    """
    Convert a chatgpt response to iterx format
    """
    result = {}
    for prediction in chatgpt_predictions:
        famus_id = prediction['instance_id']
        incident_type = famus_id.split('-frame-', 1)[-1]
        try:
            response_dict = json.loads(prediction['response'])
        except:
            response_dict = {}

        # Initialize the dictionary for this famus_id if it doesn't exist
        if famus_id not in result:
            result[famus_id] = [{'incident_type': incident_type}]

        for role, fillers in response_dict.items():
            # Add the prediction to the appropriate role
            if fillers:  # Ignore entries with empty predictions
                result[famus_id][0][role] = [[filler] for filler in fillers]

    # Remove incident_type from the dicts that have no other keys
    for famus_id in result:
        if len(result[famus_id][0]) == 1:  # Only 'incident_type' is present
            result[famus_id] = []

    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",
                        type=str,
                        default="test",
                        required=True)
    parser.add_argument("--output_dir", 
                        type=str, 
                        required=True)
    # boolean to check whether to use coref or not
    parser.add_argument("--use_coref",
                        type=str2bool,
                        default=False,
                        required=True)
    args = parser.parse_args()
    return args

def return_cdae_iterx_data_filename(split, 
                                    context, 
                                    spans,
                                    use_coref = False):
    if use_coref:
        return f"/data/svashishtha/FAMuS/data/cross_doc_role_extraction/iterx_format/{context}_data/{spans}_spans/{split}_gold_with_silver_coref.jsonl"
    else:
        return f"/data/svashishtha/FAMuS/data/cross_doc_role_extraction/iterx_format/{context}_data/{spans}_spans/{split}.jsonl"

def return_cdae_qa_predictions(split,
                               context):
    with open(f"../../models/cdae/best_model_{context}_qa/results_{split}.json") as f:
        results = json.load(f)
    with open(f"../../data/cross_doc_role_extraction/qa_format/{context}_data/{split}.json") as f:
        gold = [json.loads(line) for line in f.readlines()]

    predictions_qa = aggregate_predictions_for_QA([x['id'] for x in gold],
                                                get_highest_score_results_qa(results))
    
    return predictions_qa


def return_cdae_iterx_predictions(split,
                                  context,
                                  spans):
    predictions_file = f"/data/svashishtha/FAMuS/models/cdae/famus_model_{context}_data_{spans}_spans/{split}_predictions.jsonl"
    with open(predictions_file) as f:
        predictions_data = [json.loads(line) for line in f]
    predictions = {doc_id: templates for doc_pred_dict in predictions_data 
                        for doc_id, templates in doc_pred_dict.items()}
    return predictions


def return_cdae_chatgpt_predictions(split,
                                    context):
    with open(f"/data/svashishtha/FAMuS/models/cdae/chatgpt/{split}_{context}_gpt-3.5-turbo-0301_responses.jsonl") as f:
        chatgpt_predictions = [json.loads(line) for line in f]
    predictions_in_format = chatgpt_response_to_iterx_format(chatgpt_predictions)
    return predictions_in_format

def return_cdae_llama_predictions(split,
                                  context):
    with open(f"/data/svashishtha/FAMuS/models/cdae/llama/{split}_{context}_llama_13b_responses.jsonl") as f:
        llama_predictions = [json.loads(line) for line in f]
    predictions_in_format = chatgpt_response_to_iterx_format(llama_predictions)
    return predictions_in_format

def main():
    args = parse_args()
    # coref or not
    use_coref = args.use_coref
    print(f"Using coref: {use_coref}")

    os.makedirs(args.output_dir, exist_ok=True)
    split = args.split
    metrics_string = "(CEAF_RME_phi-3) P, R, F1, (CEAF_RME_phi-a) P, R, F1:\n"
    ##########################
    ## Iter-X models
    ##########################
    ##### Gold Spans
    ## Report
    spans = 'gold'
    context = 'report'
    metrics_string += "#################### Iter-X ####################\n"
    metrics_string += "Gold Spans (Report) \n"
    metrics_string += print_compute_ceafe_rme_scores(
                                return_cdae_iterx_data_filename(split, context,"mixed", use_coref=use_coref),
                                return_cdae_iterx_predictions(split, context, spans))
    ## Source
    context = 'source'
    metrics_string += "\nGold Spans (Source) \n"
    metrics_string += print_compute_ceafe_rme_scores(
                                return_cdae_iterx_data_filename(split, context,"mixed", use_coref=use_coref),
                                return_cdae_iterx_predictions(split, context, spans))
    ##### Predicted Spans
    ## Report
    spans = 'predicted'
    context = 'report'
    metrics_string += "\nPredicted Spans (Report) \n"
    metrics_string += print_compute_ceafe_rme_scores(
                                return_cdae_iterx_data_filename(split, context,"mixed", use_coref=use_coref),
                                return_cdae_iterx_predictions(split, context, spans))
    ## Source
    context = 'source'
    metrics_string += "\nPredicted Spans (Source) \n"
    metrics_string += print_compute_ceafe_rme_scores(
                                return_cdae_iterx_data_filename(split, context,"mixed", use_coref=use_coref),
                                return_cdae_iterx_predictions(split, context, spans))
    ##### Mixed Spans
    ## Report
    spans = 'mixed'
    context = 'report'
    metrics_string += "\nMixed Spans (Report) \n"
    metrics_string += print_compute_ceafe_rme_scores(
                                return_cdae_iterx_data_filename(split, context,"mixed", use_coref=use_coref),
                                return_cdae_iterx_predictions(split, context, spans))
    ## Source
    context = 'source'
    metrics_string += "\nMixed Spans (Source) \n"
    metrics_string += print_compute_ceafe_rme_scores(
                                return_cdae_iterx_data_filename(split, context,"mixed", use_coref=use_coref),
                                return_cdae_iterx_predictions(split, context, spans))
    ##########################
    ## QA models
    ##########################
    context = 'report'
    metrics_string += "\n#################### QA-models ####################\n"
    metrics_string += "Report \n"
    metrics_string += print_compute_ceafe_rme_scores(
                                return_cdae_iterx_data_filename(split, context,"mixed", use_coref=use_coref),
                                return_cdae_qa_predictions(split, context))
    context = 'source'
    metrics_string += "\nSource \n"
    metrics_string += print_compute_ceafe_rme_scores(
                                return_cdae_iterx_data_filename(split, context,"mixed", use_coref=use_coref),
                                return_cdae_qa_predictions(split, context))
    ##########################
    ## ChatGPT models
    ##########################
    context = 'report'
    metrics_string += "\n#################### ChatGPT ####################\n"
    metrics_string += "Report \n"
    metrics_string += print_compute_ceafe_rme_scores(
                                return_cdae_iterx_data_filename(split, context,"mixed", use_coref=use_coref),    
                                return_cdae_chatgpt_predictions(split, context))
    context = 'source'
    metrics_string += "\nSource \n"
    metrics_string += print_compute_ceafe_rme_scores(
                                return_cdae_iterx_data_filename(split, context,"mixed", use_coref=use_coref),
                                return_cdae_chatgpt_predictions(split, context))

    ##########################
    ## Llama models
    ##########################
    context = 'report'
    metrics_string += "\n#################### Llama ####################\n"
    metrics_string += "Report \n"
    metrics_string += print_compute_ceafe_rme_scores(
                                return_cdae_iterx_data_filename(split, context,"mixed", use_coref=use_coref),    
                                return_cdae_llama_predictions(split, context))

    context = 'source'
    metrics_string += "\nSource \n"
    metrics_string += print_compute_ceafe_rme_scores(
                                return_cdae_iterx_data_filename(split, context,"mixed", use_coref=use_coref),
                                return_cdae_llama_predictions(split, context))
    
    with open(f"{args.output_dir}/{split}_caefe_results_coref_{use_coref}.txt", 'w') as f:
        f.write(metrics_string)

if __name__ == "__main__":
    main()
    
    
        
        

    