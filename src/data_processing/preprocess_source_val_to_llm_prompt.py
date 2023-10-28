import os
import json
import pandas as pd
from typing import Dict, List
import tiktoken
import argparse
from data_utils import famusInstance2ModifiedReportwithTrigger


def length_of_prompt(prompt: str,
                        tokenizer):
    """
    Returns the length of the prompt.

    Args:
        prompt: The prompt.
        tokenizer: The tokenizer.

    Returns:
        The length of the prompt.
    """
    return len(tokenizer.encode(prompt))

def create_4_shot_prompt_from_instance(instance,
                                       max_prompt_tokens = 3900):
    """
    Creates a prompt from a source val instance 
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")    
    report_with_trigger_dct = famusInstance2ModifiedReportwithTrigger(instance,
                                                                       trigger_tag='event')
    report = report_with_trigger_dct['doctext']
    source = instance['source_dict']['doctext']

    prompt_prefix = f"""
    In this task, you are given a report document marked up an XML tag 'report'.
    The report describes an event denoted with an XML tag 'event'.
    You are also given a source document marked up an XML tag 'source'.

    Your task is to determine whether the 'source' document contains the \
    'event' described in the 'report' or not. This is equivalent to determining \
    whether the source is a valid reference for the tagged event
    in the report.

    Steps to follow to arrive at the answer:
    1. Summarize the 'event' described in the 'report' in one line.
    2. Check if the 'source' document describes the summarized 'event' or not.
    If the 'source' document describes the summarized 'event', then 
    in one line explain how the 'source' document describes the 'event' and answer 'yes'.
    If the 'source' document does not describe the summarized 'event', then
    in one line explain how the 'source' document does not describe the 'event' and answer 'no'.

    The answer 'Yes' or 'No' should be in a separate line at the end inside the 
    <valid_source> tag. Below are some examples.

    <report> Jon <event> picked </event> up the gun. </report>
    <source> Jon enjoyed hunting. One day, he grabbed his gun and went to the forest. </source>
    <answer>
    The report focuses on the event of Jon picking up the gun. 
    The source describes Jon grabbing his gun which is the same event tagged in the report.
    <valid_source> Yes <valid_source> 
    </answer>

    <report> Jon  <event> picked </event> up Janice. </report>
    <source> Jon enjoyed driving a lot. One day, he picked up Daniel from a store. </source>
    <answer>
    The report focuses on the event of Jon picking up Janice.
    The source describes Jon picking up Daniel which is not the same event tagged in the report.
    <valid_source> No <valid_source> 
    </answer>

    <report> <event> Riots </event>  erupted in various parts of the city after the violent speech. </report>
    <source> Various violent acts were seen in the city after the minister's controversial hate speech. </source>
    <answer>
    The report focuses on the event of riots erupting in various parts of the city.
    The source describes various violent acts in the city which is the same event tagged in the report.
    <valid_source> Yes <valid_source> 
    </answer>

    <report>  Osama Bin Laden was <event> killed </event> in Abbottabad, Pakistan on May 2, 2011 </report>
    <source> Osama bin Mohammed bin Awad bin Laden was a Saudi Arabian-born militant and founder of the pan-Islamic militant organization Al-Qaeda. </source>
    <answer>
    The report focuses on the killing of Osama Bin Laden.
    The source does not mention anything about the killing of Osama Bin Laden.
    <valid_source> No <valid_source> 
    </answer>
    """

    prompt_suffix = f"""
    <report> {report} </report>
    <source> {source} </source>
    <answer>
    """

    # check length of prompt prefix
    prompt_prefix_length = length_of_prompt(prompt_prefix, tokenizer)

    # check length of suffix
    prompt_suffix_length = length_of_prompt(prompt_suffix, tokenizer)

    # total length of prompt
    prompt_length = prompt_prefix_length + prompt_suffix_length

    # if total length of prompt is greater than max_prompt_tokens, then truncate 
    # source and recompute the prompt_suffix
    if prompt_length > max_prompt_tokens:
        # get the length of the source
        source_length = length_of_prompt(source, tokenizer)

        # truncate the source
        source = source[:source_length - (prompt_length - max_prompt_tokens)]

        # recompute the prompt_suffix
        prompt_suffix = f"""
        <report> {report} </report>
        <source> {source} </source>
        <answer>
        """
    prompt = prompt_prefix + prompt_suffix
    return prompt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', 
                        type=str, 
                        required=True,
                        default = "../../data/source_validation/",
                        help='Path to the dir containing the famus release files: train, dev, test')
    
    parser.add_argument('--output_dir', type=str, 
                        required=True,
                        default = "../../data/source_validation/llm_prompt_format/",
                        help='Path to the output dir where the llm prompt format files will be written')
    
    args = parser.parse_args()


    return args


def main():
    args = parse_args()
    # load the train, dev, test instances
    # train
    with open(os.path.join(args.input_dir, "train.jsonl"), 'r') as f:
        train_instances = [json.loads(line) for line in f.readlines()]
    # dev
    with open(os.path.join(args.input_dir, "dev.jsonl"), 'r') as f:
        dev_instances = [json.loads(line) for line in f.readlines()]
    # test
    with open(os.path.join(args.input_dir, "test.jsonl"), 'r') as f:
        test_instances = [json.loads(line) for line in f.readlines()]
    # create 4-shot prompts for train, dev, test
    # train
    train_prompts = []
    for instance in train_instances:
        current_prompt_dict = {'instance_id': instance['instance_id'],
                               'llm_prompt': create_4_shot_prompt_from_instance(instance),
                               'valid_source': instance['valid_source']
                               }
        train_prompts.append(current_prompt_dict)
    # dev
    dev_prompts = []
    for instance in dev_instances:
        current_prompt_dict = {'instance_id': instance['instance_id'],
                               'llm_prompt': create_4_shot_prompt_from_instance(instance),
                               'valid_source': instance['valid_source']
                               }
        dev_prompts.append(current_prompt_dict)
    # test
    test_prompts = []
    for instance in test_instances:
        current_prompt_dict = {'instance_id': instance['instance_id'],
                               'llm_prompt': create_4_shot_prompt_from_instance(instance),
                               'valid_source': instance['valid_source']
                               }
        test_prompts.append(current_prompt_dict)
    # create output dir if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    # write the prompts to files
    # train
    with open(os.path.join(args.output_dir, "train.jsonl"), 'w') as f:
        for prompt in train_prompts:
            f.write(json.dumps(prompt) + '\n')
    # dev
    with open(os.path.join(args.output_dir, "dev.jsonl"), 'w') as f:
        for prompt in dev_prompts:
            f.write(json.dumps(prompt) + '\n')
    # test
    with open(os.path.join(args.output_dir, "test.jsonl"), 'w') as f:
        for prompt in test_prompts:
            f.write(json.dumps(prompt) + '\n')
    print(f"Train prompts: {len(train_prompts)}")
    print(f"Dev prompts: {len(dev_prompts)}")
    print(f"Test prompts: {len(test_prompts)}")
    print(f"Exported llm prompt format data to {args.output_dir}")

if __name__ == "__main__":
    main()