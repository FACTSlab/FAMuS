import sys
sys.path.append('../../')

import os
import json
from test_source_val import parse_llm_response_for_source_val, is_lemma_in_source
from train_source_val import instances2TextAndLabels
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import numpy as np
import seaborn as sns
import pandas as pd


def decide_bin(length, bins):
    for bin_key in bins.keys():
        if "-" in bin_key:
            lower_bound, upper_bound = map(int, bin_key.split("-"))
            if lower_bound <= length < upper_bound:
                return bin_key
        elif "<" in bin_key:
            upper_bound = int(bin_key[1:])
            if length < upper_bound:
                return bin_key
        elif "+" in bin_key:
            lower_bound = int(bin_key[:-1])
            if length >= lower_bound:
                return bin_key
    return None
    
def bin_data_with_indices(source_val_data, source_lengths, predictions):
    # Define percentile values
    percentile_values = [10, 25, 50, 75, 90]
    # Calculate the bin edges based on the percentiles of the source lengths
    bin_edges = np.percentile(source_lengths, percentile_values)
    # Add the maximum source length to the bin edges to include 90+
    bin_edges = list(bin_edges) + [max(source_lengths) + 1]
    # Initialize bins as empty dictionaries
    bins = {
        f"<{int(bin_edges[0])}": {"indices": [], "instances": [], "predictions": []},
    }
    bins.update({
        f"{int(bin_edges[i])}-{int(bin_edges[i+1])}": {"indices": [], "instances": [], "predictions": []} for i in range(len(bin_edges)-2)
    })
    bins[f"{int(bin_edges[-2])}+"] = {"indices": [], "instances": [], "predictions": []}


    # Assign each instance to a bin along with its index and prediction
    # Assign each instance to a bin along with its index and prediction
    for index, instance, length, prediction in zip(range(len(source_val_data)), source_val_data, source_lengths, predictions):
        bin_key = decide_bin(length, bins)
        if bin_key is not None:
            bins[bin_key]["indices"].append(index)
            bins[bin_key]["instances"].append(instance)
            bins[bin_key]["predictions"].append(prediction)

    return bins

def source_val_instance_to_source_length(instance):
    return len(instance['source_dict']['doctext'].split())

def source_val_instances_predictions_to_true_and_predicted_labels(
                                            test_instances,
                                            model_predictions,
                                            model='longformer'):
    """
    Args:
        test_instances: list of test instances
        model_predictions: list of model predictions (format depends on model)
        model: one of ['longformer', 'lemma', 'chatgpt', 'llama']

    This function extracts the true and predicted labels from 
    model predictions and instances. For the lemma model,
    the model_predictions are the instances themselves (we just
    need to check if the lemma is in the source). For the longformer
    """
    if model == 'longformer':
        prediction_labels = [int(x['label'].split('_')[-1]) for x in model_predictions]
    elif model == 'lemma':
        prediction_labels = [int(is_lemma_in_source(instance)) for instance in model_predictions]
    elif model == 'chatgpt' or model == 'llama-2-13b':
        prediction_labels = [parse_llm_response_for_source_val(response_dct['response']) 
                                    for response_dct in model_predictions]
    else:
        raise ValueError(f"Model {model} not supported")  
    
    _, true_labels = instances2TextAndLabels(test_instances)

    return true_labels, prediction_labels

def compute_source_val_metric(true_labels, predicted_labels,
                              metric = 'accuracy'):
    """
    Args:
        true_labels: list of true labels
        predicted_labels: list of predicted labels
        metric: one of ['accuracy', 'precision', 'recall', 'f1']
    """
    if metric == 'accuracy':
        # round to 2 decimal places
        return float(f"{accuracy_score(true_labels, predicted_labels)*100:.2f}")
    elif metric == 'precision':
        return float(f"{precision_score(true_labels, predicted_labels, average='binary')*100:.2f}")
    elif metric == 'recall':
        return float(f"{recall_score(true_labels, predicted_labels,average='binary')*100:.2f}")
    elif metric == 'f1':
        return float(f"{f1_score(true_labels, predicted_labels,average='binary')*100:.2f}")
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", 
                        type=str, 
                        default='../../data/source_validation')
    parser.add_argument("--model_path", 
                        type=str, 
                        default='../../models/source_validation')

    return parser.parse_args()


def main():
    args = parse_args()

    source_val_data_path = args.data_path
    source_val_model_path = args.model_path
    longformer_model_path = os.path.join(source_val_model_path, "results", "best_model")
    chatgpt_model_path = os.path.join(source_val_model_path, "chatgpt")
    llama_model_path = os.path.join(source_val_model_path, "llama")

    ## Load data
    with open(os.path.join(source_val_data_path, "dev.jsonl"), 'r') as f:
        source_val_data = [json.loads(line) for line in f.readlines()]
    source_lengths = [source_val_instance_to_source_length(instance) for instance in source_val_data]
    
    ## Load Model Predictions
    with open(os.path.join(longformer_model_path, "results_dev.json"), 'r') as f:
        longformer_predictions = json.load(f)
    with open(os.path.join(chatgpt_model_path, "dev_gpt-3.5-turbo-0301_responses.jsonl"), 'r') as f:
        chatgpt_predictions = [json.loads(line) for line in f.readlines()]

    with open(os.path.join(llama_model_path, "dev_sv_llama_13b_responses.jsonl"), 'r') as f:
        llama_predictions = [json.loads(line) for line in f.readlines()]

    ## Create bins based on Source Length
    longformer_bins_with_indices = bin_data_with_indices(source_val_data, 
                                          source_lengths, 
                                            longformer_predictions)
    lemma_model_bins_with_indices = bin_data_with_indices(source_val_data, 
                                            source_lengths, 
                                            source_val_data)
    chatgpt_model_bins_with_indices = bin_data_with_indices(source_val_data,
                                                source_lengths,
                                            chatgpt_predictions)

    llama_model_bins_with_indices = bin_data_with_indices(source_val_data,
                                                source_lengths,
                                            llama_predictions)  

    ## Compute metrics across bins
    models = ['longformer', 'lemma', 'chatgpt', 'llama-2-13b']
    # Initialize the dictionary
    metric_values_dict = {'accuracy': {model: [] for model in models},
                          'precision': {model: [] for model in models},
                          'recall': {model: [] for model in models},
                          'f1': {model: [] for model in models}}
    for metric, metric_values in metric_values_dict.items():
        for model in models:
            if model == 'longformer':
                bins_with_indices = longformer_bins_with_indices
            elif model == 'lemma':
                bins_with_indices = lemma_model_bins_with_indices
            elif model == 'chatgpt':
                bins_with_indices = chatgpt_model_bins_with_indices
            elif model == 'llama-2-13b':
                bins_with_indices = llama_model_bins_with_indices

            # Compute the metrics for each bin
            for bin_key in bins_with_indices:
                true_labels, predicted_labels = source_val_instances_predictions_to_true_and_predicted_labels(
                                                    bins_with_indices[bin_key]['instances'],
                                                    bins_with_indices[bin_key]['predictions'],
                                                    model=model)
                metric_value = compute_source_val_metric(true_labels, predicted_labels, metric=metric)
                metric_values[model].append(metric_value)  # Store the metric value in the dictionary   

    ## Plot the metrics
    bins = list(bins_with_indices.keys())
    metric = 'f1'
    metric_values = metric_values_dict[metric]
    # Create a DataFrame from the metric_values dictionary
    df = pd.DataFrame(metric_values, index=bins)

    # "Melt" the dataset to "long-form" representation
    df = df.reset_index().melt('index', var_name='model',  value_name=metric)

    # Create a barplot
    plt.figure(figsize=(10,6))
    barplot = sns.barplot(x='index', y=metric, hue='model', data=df, palette='Set2')

    # Add labels
    plt.xlabel("Source Length (in tokens)", fontsize=16)
    plt.ylabel(metric.upper(), fontsize=16)
    plt.title("SV Performance (Dev Set)", fontsize=18)
    plt.xticks(fontsize=15)

    # Move the legend outside of the plot area, at the bottom
    plt.legend(bbox_to_anchor=(0.5, -0.2), 
                loc='upper center', 
                ncol=len(df['model'].unique()),
                fontsize=14,)

    # Adding data labels
    for p in barplot.patches:
        height = p.get_height()
        # Only annotate bars with height greater than zero
        if height > 0:
            # Show accuracy values to two decimal places
            barplot.annotate(int(height),
                            (p.get_x() + p.get_width() / 2., height*0.90),
                            ha = 'center', va = 'center', 
                            color='ivory', fontsize=15)

    plt.tight_layout()
    # Save the figure before plt.show(), with desired resolution and format.
    plt.savefig('source_val_performance_by_source_length.png', format='png', dpi=300)
    plt.show()
    plt.close()

if __name__=="__main__":
    main()
        

