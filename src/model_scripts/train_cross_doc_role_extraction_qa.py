import transformers
import torch
from datasets import load_dataset
import argparse
import os
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator
import optuna

MODEL_CHECKPOINT = "allenai/longformer-base-4096"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def prepare_train_features(examples,
                           tokenizer=TOKENIZER,
                           max_length = 2048,
                           doc_stride = 256):
    """
    Args:nvidi
        examples: dataset examples
        tokenizer: tokenizer
        max_length: The maximum length of a feature (question and context)
        doc_stride: The authorized overlap between two part of the context 
                    when splitting it is needed.
    """
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # For this notebook to work with any kind of models, we need to account for the 
    # special case where the model expects padding on the left (in which case we 
    # switch the order of the question and the context):
    pad_on_right = tokenizer.padding_side == "right"

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )


    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def parse_args():
    # add argument for data path
    # add argument for model path
    # add argument for output path
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_data_dir", 
                        type=str, default="../../data/cross_doc_role_extraction/qa_format/report_data/",
                        help="Path to the data where train.json, dev.json and test.json are present")
    
    parser.add_argument("--model_output_dir",
                        type=str, 
                        default="../../models/cross_doc_role_extraction/qa_report/",
                        help="Path to the model output directory")
    
    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of epochs to train the model for")
    
    parser.add_argument("--learning_rate",
                        type=float,
                        default=3.859688337814384e-05,
                        help="Learning rate for the model")
    
    parser.add_argument("--weight_decay",
                        type=float,
                        default=5.89501379340659e-05,
                        help="Weight decay for the model")
    
    parser.add_argument("--save_total_limit",
                        type=int,
                        default=1,
                        help="Number of checkpoints to save")
    
    parser.add_argument("--num_optuna_trials",
                       type=int,
                       defualt = 5,
                       help="Number of optuna trials to run")

    parser.add_argument("--experiment_name",
                        type=str,
                        default="finetuned-report-qa-optuna",
                        help="Name of the experiment name under which the model will be saved")
    
    args = parser.parse_args()


    return args



def main():
    args = parse_args() 
    # This flag is the difference between SQUAD v1 or 2 
    # (if you're using another dataset, it indicates if impossible
    # answers are allowed or not).
    squad_v2 = True
    batch_size = 1

    datasets = load_dataset("json", data_files={
        'train': os.path.join(args.input_data_dir, "train.json"),
        'validation': os.path.join(args.input_data_dir, "dev.json"),
        'test': os.path.join(args.input_data_dir, "test.json")},
            )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # The following assertion ensures that our tokenizer is a fast tokenizers
    # (backed by Rust) from the 🤗 Tokenizers library. 
    # Those fast tokenizers are available for almost all models, 
    # and we will need some of the special features they have for our preprocessing.
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    tokenized_datasets = datasets.map(prepare_train_features, 
                                      batched=True, 
                                      remove_columns=datasets["train"].column_names)
    
    model_output_dir = os.path.join(args.model_output_dir, args.experiment_name)

    def model_init():
        return AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT,
                                                            #  return_dict=True,
                                                            #  gradient_checkpointing=True
                                                             )
    
    def objective(trial: optuna.Trial):
        # Hyperparameters to tune and their ranges
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        
        trial_output_dir = os.path.join(model_output_dir, f"trial-{trial.number}")
        os.makedirs(trial_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir = trial_output_dir,
            logging_dir = os.path.join(trial_output_dir, "logs"),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=args.epochs,
            save_strategy="epoch",
            logging_strategy = "epoch",
            evaluation_strategy="epoch",
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            push_to_hub=False,
        )

        model = model_init()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=default_data_collator,
            tokenizer=tokenizer,
        )

        # Train the model and get the results
        result = trainer.train()

        # Save the step number of the best model as a user attribute of the trial
        best_model_checkpoint = trainer.state.best_model_checkpoint
        best_model_step = best_model_checkpoint.split('-')[-1]
        trial.set_user_attr('best_model_step', best_model_step)

        # Save the best model to the desired directory
        best_model_dir = os.path.join(model_output_dir, 'best_model')
        os.makedirs(best_model_dir, exist_ok=True)
        model = AutoModelForQuestionAnswering.from_pretrained(best_model_checkpoint)
        model.save_pretrained(best_model_dir)

        return result.training_loss

    # Create the study and run the optimization
    study = optuna.create_study(direction="minimize", storage=f'sqlite:///{args.experiment_name}.db')
    study.optimize(objective, n_trials=args.num_optuna_trials)


if __name__ == "__main__":
    main()