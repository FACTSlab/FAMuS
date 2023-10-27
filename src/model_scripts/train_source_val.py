import pandas as pd
import json
import random
from tqdm import tqdm
import torch
import os
from collections import defaultdict
from transformers import LongformerTokenizer
from transformers import (LongformerForSequenceClassification, Trainer, TrainingArguments,
                        AutoModelForSequenceClassification)
import evaluate
from src.data_processing.data_utils import famusInstance2ModifiedReportwithTrigger
import numpy as np
import argparse

accuracy = evaluate.load("accuracy")

class SourceValDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

 
def instance2ConcatenatedReportSource(instance,
                                    sep_token = '</s>'):
    """
    given an instance from source validation release version,
    get a concatenation of report (with trigger tagged) and
    source text. A <SEP> separates report and source document.
    """
    
    report_with_trigger_dct = famusInstance2ModifiedReportwithTrigger(instance, trigger_tag='event')
    source_text = instance['source_dict']['doctext']

    return report_with_trigger_dct['doctext'] + f" {sep_token} " + source_text

def instances2TextAndLabels(instance_list):
    """
    extract text and labels from a list of instances
    for the source validation task
    """
    texts = [instance2ConcatenatedReportSource(instance) for instance in instance_list]
    labels = [int(instance['valid_source']) for instance in instance_list]
    return texts, labels

def parse_args():
    # input dir
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", 
                        type=str, 
                        default="../../data/source_validation/"
                        )
    parser.add_argument("--output_dir",
                        type=str,
                        default="../../models/source_validation/longformer/"
                        )
    
    # seed
    parser.add_argument("--seed",
                        type=int,
                        default=42
                        )
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    MODEL_DIR = args.output_dir
    ########################
    # Load dataset
    ########################
    with open(os.path.join(args.input_dir, "train.jsonl"), 'r') as f:
        train_instances = [json.loads(line) for line in f.readlines()]

    with open(os.path.join(args.input_dir, "dev.jsonl"), 'r') as f:
        val_instances = [json.loads(line) for line in f.readlines()]

    ########################
    # Prepare Train/Dev/Test
    ########################
    ## Shuffle train, dev instances
    random.seed(args.seed)
    random.shuffle(train_instances)
    random.shuffle(val_instances)

    train_texts, train_labels = instances2TextAndLabels(train_instances)
    val_texts, val_labels = instances2TextAndLabels(val_instances)

    
    print(f"train size: {len(train_texts)}")
    print(f"train positives: {sum(train_labels)/len(train_texts)}")
    print(f"val size: {len(val_texts)}")
    print(f"val positives: {sum(val_labels)/len(val_texts)}")

    ########################
    # Run a small sample
    ########################
    # train_texts = train_texts[:20]
    # train_labels = train_labels[:20]
    # val_texts = val_texts[:5]
    # val_labels = val_labels[:5]

    ########################
    # Prepare Pytorch Dataset
    ########################
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    # test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = SourceValDataset(train_encodings, train_labels)
    val_dataset = SourceValDataset(val_encodings, val_labels)
    # test_dataset = SourceValDataset(test_encodings, test_labels)

    ########################
    # Trainer
    ########################
    # Make model output directory
    model_output_dir = os.path.join(MODEL_DIR, "results")
    model_log_dir = os.path.join(MODEL_DIR, "logs")

    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(model_log_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,          # output directory
        num_train_epochs=10,              # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=2,   # batch size for evaluation
        learning_rate=1.25e-06,
        warmup_steps=400,                # number of warmup steps for learning rate scheduler
        weight_decay=0.00,               # strength of weight decay
        logging_dir=model_log_dir,            # directory for storing logs
        save_total_limit = 1,
        save_strategy="epoch",
        logging_strategy = "epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        )

    model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", gradient_checkpointing=True)

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained("allenai/longformer-base-4096", 
                                                                 return_dict=True,
                                                                 gradient_checkpointing=True)

    trainer = Trainer(
        model_init=model_init,                 # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
        compute_metrics=compute_metrics            
    )

    def optuna_hp_space(trial):
        return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        }

    best_trial = trainer.hyperparameter_search(
                direction="maximize",
                backend="optuna",
                hp_space=optuna_hp_space,
                n_trials=5,
            )

    # Get the best hyperparameters
    best_hyperparameters = best_trial.hyperparameters

    # Update training_args with the best hyperparameters
    training_args.learning_rate = best_hyperparameters["learning_rate"]
    training_args.weight_decay = best_hyperparameters["weight_decay"]

    # Reinitialize the Trainer with the new training_args
    trainer = Trainer(
            model_init=model_init,                 
            args=training_args,                  
            train_dataset=train_dataset,         
            eval_dataset=val_dataset,            
            compute_metrics=compute_metrics            
        )

    # Now, train the model with the optimized hyperparameters
    trainer.train()

    # Save the best model
    trainer.save_model(output_dir=os.path.join(model_output_dir, 'best_model'))
    

if __name__=="__main__":
    main()
