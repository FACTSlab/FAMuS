import json
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import argparse
import os
import matplotlib.pyplot as plt
# Set global font size
plt.rcParams['font.size'] = 16

def encode(tokenizer: PreTrainedTokenizerFast,
           query: str, passage: str, title: str = '-') -> BatchEncoding:
    return tokenizer(query,
                     text_pair='{}: {}'.format(title, passage),
                     max_length=512,
                     padding=True,
                     truncation=True,
                     return_tensors='pt')


def fetch_similarity(report_text, 
                     source_text,
                     similarity_model=None,
                     tokenizer=None,
                     gpu=1):
    with torch.no_grad():
        encoding = encode(tokenizer, report_text, source_text).to(f'cuda:{gpu}')
        output: SequenceClassifierOutput = similarity_model(**encoding)
        return output.logits[0][0].item()

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', 
                        type=str, 
                        default='../../data/source_validation/')
    parser.add_argument('--gpu', 
                        type=int, 
                        default=2)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    with open(os.path.join(args.input_dir, "train.jsonl")) as f:
        train_source_val = [json.loads(line) for line in f.readlines()]

    with open(os.path.join(args.input_dir, "dev.jsonl")) as f:
        dev_source_val = [json.loads(line) for line in f.readlines()]

    tokenizer = AutoTokenizer.from_pretrained('intfloat/simlm-msmarco-reranker')
    model = AutoModelForSequenceClassification.from_pretrained('intfloat/simlm-msmarco-reranker')
    model.eval()
    model = model.to(f'cuda:{args.gpu}')

    pos_instances = [instance for instance in train_source_val + dev_source_val
                 if instance['valid_source']]
    neg_instances = [instance for instance in train_source_val + dev_source_val
                        if not instance['valid_source']] 
        
    pos_sim_scores =   []
    for instance in tqdm(pos_instances):
        pos_sim_scores.append(fetch_similarity(instance['report_dict']['doctext'],
                                                instance['source_dict']['doctext'],
                                                model, 
                                                tokenizer, gpu=args.gpu))
    plt.hist(pos_sim_scores, bins=100)
    plt.xlabel('Similarity Score between Report and Source')
    plt.ylabel('Frequency of Docs')
    plt.xlim(-9, 4)
    plt.title('SV (+) Instances (Train + Dev)')
    # Save the figure before plt.show(), with desired resolution and format.
    plt.savefig('histogram_simLM_positive_examples.png', format='png', dpi=300)
    plt.show()
    plt.close()

    neg_sim_scores =   []
    for instance in tqdm(neg_instances):
        neg_sim_scores.append(fetch_similarity(instance['report_dict']['doctext'],
                                                instance['source_dict']['doctext'],
                                                model, 
                                                tokenizer, gpu=args.gpu))
    plt.hist(neg_sim_scores, bins=100)
    plt.xlabel('Similarity Score between Report and Source')
    plt.ylabel('Frequency of Docs')
    plt.xlim(-9, 4)
    plt.title('SV (-) Instances (Train + Dev)')
    # Save the figure before plt.show(), with desired resolution and format.
    plt.savefig('histogram_simLM_negative_examples.png', format='png', dpi=300)
    plt.show()
    plt.close()

    ## distrubution of gold neg examples
    gold_neg_instances = [instance for instance in train_source_val + dev_source_val
                        if not instance['valid_source'] and not instance['bool_generated']]
    
    neg_gold_sim_scores =   []
    for instance in tqdm(gold_neg_instances):
        neg_gold_sim_scores.append(fetch_similarity(instance['report_dict']['doctext'],
                                                instance['source_dict']['doctext'],
                                                model, 
                                                tokenizer, gpu=args.gpu))
        
    # Plot the distribution of similarity scores for negative instances
    plt.hist(neg_gold_sim_scores, bins=100)
    plt.xlabel('Similarity Score between Report and Source')
    plt.ylabel('Frequency of Docs')
    plt.xlim(-9, 4)
    plt.title('Gold SV (-): (Train + Dev)')
    # Save the figure before plt.show(), with desired resolution and format.
    plt.savefig('histogram_simLM_gold_negative_examples.png', format='png', dpi=300)
    plt.show()
    plt.close()

    ## distrubution of generated neg examples
    gen_neg_instances = [instance for instance in train_source_val + dev_source_val
                        if not instance['valid_source'] and instance['bool_generated']]
    neg_gen_sim_scores =   []
    for instance in tqdm(gen_neg_instances):
        neg_gen_sim_scores.append(fetch_similarity(instance['report_dict']['doctext'],
                                                instance['source_dict']['doctext'],
                                                model, 
                                                tokenizer, gpu=args.gpu))
        
    # Plot the distribution of similarity scores for negative instances
    plt.hist(neg_gen_sim_scores, bins=100)
    plt.xlabel('Similarity Score between Report and Source')
    plt.ylabel('Frequency of Docs')
    plt.xlim(-9, 4)
    plt.title('Silver SV (-): (Train + Dev)')
    # Save the figure before plt.show(), with desired resolution and format.
    plt.savefig('histogram_simLM_silver_negative_examples.png', format='png', dpi=300)
    plt.show()
    plt.close()

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot for positive similarity scores
    axs[0, 0].hist(pos_sim_scores, bins=100)
    axs[0, 0].set_xlabel('Similarity Score between Report and Source',  
                            fontsize=16)
    axs[0, 0].set_ylabel('Frequency of Docs',  
                            fontsize=16)
    axs[0, 0].set_xlim(-9, 4)
    axs[0, 0].set_title('SV (+) Instances (Train + Dev)',  
                            fontsize=20)

    # Plot for negative similarity scores
    axs[0, 1].hist(neg_sim_scores, bins=100)
    axs[0, 1].set_xlabel('Similarity Score between Report and Source',  
                            fontsize=16)
    axs[0, 1].set_ylabel('Frequency of Docs',  
                            fontsize=16)
    axs[0, 1].set_xlim(-9, 4)
    axs[0, 1].set_title('SV (-) Instances (Train + Dev)',  
                            fontsize=20)

    # Plot for gold negative similarity scores
    axs[1, 0].hist(neg_gold_sim_scores, bins=100)
    axs[1, 0].set_xlabel('Similarity Score between Report and Source',  
                            fontsize=16)
    axs[1, 0].set_ylabel('Frequency of Docs',  
                            fontsize=16)
    axs[1, 0].set_xlim(-9, 4)
    axs[1, 0].set_title('Gold SV (-): (Train + Dev)',  
                            fontsize=20)

    # Plot for generated negative similarity scores
    axs[1, 1].hist(neg_gen_sim_scores, bins=100)
    axs[1, 1].set_xlabel('Similarity Score between Report and Source',  
                            fontsize=16)
    axs[1, 1].set_ylabel('Frequency of Docs',  
                            fontsize=16)
    axs[1, 1].set_xlim(-9, 4)
    axs[1, 1].set_title('Silver SV (-): (Train + Dev)',  
                            fontsize=20)

    # Improve layout
    plt.tight_layout()

    # Save the figure before plt.show(), with desired resolution and format.
    plt.savefig('histogram_simLM_all_examples.png', format='png', dpi=300)
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
