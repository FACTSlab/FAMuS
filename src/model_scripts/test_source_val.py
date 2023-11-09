from train_source_val import instances2TextAndLabels
from transformers import LongformerTokenizer
from transformers import LongformerForSequenceClassification
import random
from transformers import pipeline
from tqdm import tqdm
import os
import json
import pandas as pd
import argparse
from torch.utils.data import Dataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from bs4 import BeautifulSoup
from nltk.stem import 	WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list
    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

def is_lemma_in_source(instance):
    """
    check if trigger word (or its lemma) is in the source
    """
    trigger_word = instance['report_dict']['frame-trigger-span'][0]
    trigger_lemma = wordnet_lemmatizer.lemmatize(trigger_word)


    source_text_lemmatized_tokens = [wordnet_lemmatizer.lemmatize(token)
                                for token in instance['source_dict']['doctext'].split()]
    
    source_text_lemmatized = ' '.join(source_text_lemmatized_tokens)
    if (trigger_word in source_text_lemmatized_tokens) or (
        trigger_lemma in source_text_lemmatized_tokens):
        return True
    else:
        return False
    

def parse_llm_response_for_source_val(response,
                                      verbose = False):
    soup = BeautifulSoup(response, 'html.parser')
    # check if 'valid_source' tag is present, if yes extract the text
    if soup.find('valid_source'):
        prediction = 1 if 'yes' in soup.find('valid_source').text.strip().lower() else 0
    else:
        if verbose:
            print(f"No valid_source tag found in response: {response}\n")
        prediction = 0

    return prediction


def parse_args():
    # input dir
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_data_dir", 
                        type=str, 
                        default="../../data/source_validation/"
                        )
    
    parser.add_argument("--model_checkpoint_path",
                        type=str, 
                        default="../../models/source_validation/results/best_model/"
                        )
    
    parser.add_argument("--chatgpt_output_file_path",
                        type=str,
                        required = True,
                        help="Path to the chatgpt output file")
    
    parser.add_argument("--split", 
                        type=str, 
                        default="test",
                        help="train, dev, or test"
                        )

    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="gpu id"
                        )
    
    parser.add_argument("--metrics_output_dir",
                        type=str,
                        required=True,
                        help = "Path to the metrics output directory")
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args() 

    ########################
    # Load dataset
    ########################
    with open(os.path.join(args.input_data_dir, f"{args.split}.jsonl"), 'r') as f:
        test_instances = [json.loads(line) for line in f.readlines()]

    test_texts, test_labels = instances2TextAndLabels(test_instances)
    print(f"{args.split} size: {len(test_texts)}")
    print(f"Ratio of positive labels in {args.split}: {sum(test_labels)/len(test_texts)}")
    ########################
    # Load Model
    ########################
    model = LongformerForSequenceClassification.from_pretrained(args.model_checkpoint_path)
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

    classifier = pipeline("text-classification", 
                model=model, 
                tokenizer=tokenizer,
                padding =True,
                truncation=True,
                device=args.gpu)
    ########################
    # Run Inference
    ########################
    test_dataset_texts = ListDataset(test_texts)
    results = []
    # check if results already exist
    if os.path.exists(os.path.join(args.model_checkpoint_path, f"results_{args.split}.json")):
        print(f"Inferences already exist at {os.path.join(args.model_checkpoint_path, f'results_{args.split}.json')}")
    else:
        print("Running inference on the dataset")
        for out in tqdm(classifier(test_dataset_texts)):
            results.append(out)
        # Export results
        with open(os.path.join(args.model_checkpoint_path, f"results_{args.split}.json"), 'w') as f:
            f.write(json.dumps(results, indent=4))
        print(f"Results exported to: {os.path.join(args.model_checkpoint_path, f'results_{args.split}.json')}")
    # Lonfgormer predictions
    with open(os.path.join(args.model_checkpoint_path, f"results_{args.split}.json")) as f:
        model_results = json.loads(f.read())
    test_longformer_predictions = [int(x['label'].split('_')[-1]) for x in model_results]
    ########################
    # Lemma Model predictions
    ########################
    ## For each instance, get predictions based on whether trigger word (or its lemma) is in the source
    test_lemma_predictions = [int(is_lemma_in_source(instance)) for instance in tqdm(test_instances)]

    ########################
    # ChatGPT predictions
    ########################
    # check if chatgpt output file exists
    chatgpt_predictions = []
    if os.path.exists(args.chatgpt_output_file_path):
        with open(args.chatgpt_output_file_path) as f:
            chatgpt_responses = [json.loads(line) for line in f.readlines()]
        for response_dct in chatgpt_responses:
            chatgpt_predictions.append(parse_llm_response_for_source_val(response_dct['response']))
    # Write metrics
    ########################
    with open(os.path.join(args.metrics_output_dir, f"metrics_source_val_{args.split}.json"), 'w') as f:
        f.write("###########################\n")
        f.write(f"Longformer Model: {args.model_checkpoint_path}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Accuracy, Precision, Recall, F1:\n")
        a = f"{accuracy_score(test_labels, test_longformer_predictions)*100:.2f}"
        p = f"{precision_score(test_labels, test_longformer_predictions, average='binary')*100:.2f}"
        r = f"{recall_score(test_labels, test_longformer_predictions, average='binary')*100:.2f}"
        f1 = f"{f1_score(test_labels, test_longformer_predictions, average='binary')*100:.2f}"
        f.write(f"{a} & {p} & {r} & {f1}\n")
        f.write("\n###########################\n")
        f.write("Lemma Model Metrics:\n")
        lemma_a = f"{accuracy_score(test_labels, test_lemma_predictions)*100:.2f}"
        lemma_p = f"{precision_score(test_labels, test_lemma_predictions, average='binary')*100:.2f}"
        lemma_r = f"{recall_score(test_labels, test_lemma_predictions, average='binary')*100:.2f}"
        lemma_f1 = f"{f1_score(test_labels, test_lemma_predictions, average='binary')*100:.2f}"
        f.write(f"{lemma_a} & {lemma_p} & {lemma_r} & {lemma_f1}\n")
        f.write("\n###########################\n")
        if chatgpt_predictions:
            f.write("ChatGPT Model Metrics:\n")
            chatgpt_a = f"{accuracy_score(test_labels, chatgpt_predictions)*100:.2f}"
            chatgpt_p = f"{precision_score(test_labels, chatgpt_predictions, average='binary')*100:.2f}"
            chatgpt_r = f"{recall_score(test_labels, chatgpt_predictions, average='binary')*100:.2f}"
            chatgpt_f1 = f"{f1_score(test_labels, chatgpt_predictions, average='binary')*100:.2f}"
            f.write(f"{chatgpt_a} & {chatgpt_p} & {chatgpt_r} & {chatgpt_f1}\n")
    print(f"Metrics exported to: {os.path.join(args.metrics_output_dir, f'metrics_source_val_{args.split}.json')}")


if __name__=="__main__":
    main()
