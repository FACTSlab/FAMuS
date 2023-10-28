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

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list
    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

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
    print("Running inference on the dataset")
    for out in tqdm(classifier(test_dataset_texts)):
        results.append(out)
    # Export results
    with open(os.path.join(args.model_checkpoint_path, f"results_{args.split}.json"), 'w') as f:
        f.write(json.dumps(results, indent=4))
    print(f"Results exported to: {os.path.join(args.model_checkpoint_path, f'results_{args.split}.json')}")
    ########################
    # Write metrics
    ########################
    with open(os.path.join(args.model_checkpoint_path, f"results_{args.split}.json")) as f:
        model_results = json.loads(f.read())
    test_longformer_predictions = [int(x['label'].split('_')[-1]) for x in model_results]
    with open(os.path.join(args.model_checkpoint_path, f"metrics_{args.split}.json"), 'w') as f:
        f.write(f"Accuracy: {accuracy_score(test_labels, test_longformer_predictions)*100:.2f}\n")
        f.write(f"Precision: {precision_score(test_labels, test_longformer_predictions, average='binary')*100:.2f}\n")
        f.write(f"Recall: {recall_score(test_labels, test_longformer_predictions, average='binary')*100:.2f}\n")
        f.write(f"F1: {f1_score(test_labels, test_longformer_predictions, average='binary')*100:.2f}\n")
    print(f"Metrics exported to: {os.path.join(args.model_checkpoint_path, f'metrics_{args.split}.json')}")


if __name__=="__main__":
    main()
