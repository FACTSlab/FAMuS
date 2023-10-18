import sys
sys.path.append("/data/svashishtha/rams2/src/")

import json
import pandas as pd
import argparse
import os
import stanza
import tiktoken

from tqdm import tqdm
from typing import List, Dict
from framenet_utils import frame_to_llm_prompt_info_dct
from data_utils import famusInstance2ModifiedReportwithTrigger


EMBEDDING_ENCODING = 'cl100k_base'

SOURCE_SYSTEM_PROMPT = """
You are a system that generates high quality cross-document role annotations based on Framenet ontology.
The following inputs are given to you:
1. Event Type <event_type>: A Frame name from the FrameNet ontology (eg: Hiring, Arrest, etc.)
2. Event Definition <event_definition>: Definition of the event type along with an optional example. 
3. Roles <event_roles>: All roles (or participants) of the event type (or frame) followed with an optional example.
4. Report Document <report_document>: A report document with a tagged event '<event>' of the given event type.
5. Source Document <source_document>: A document from which the roles are to be extracted.

Your job is to extract all the roles of the tagged Report <event> from the <source_document>. The ouput should be in a JSON string
where each key represents the role name as provided in the <event_roles> and its corresponding value should be a contiguous text span
from the <source_document>. Note that if no text span is found for a role, the value should be an empty string. 
Your text spans should strictly come from the <source_document>. DO NOT use spans from Event Definition, Roles, or Report Document sections.
"""

REPORT_SYSTEM_PROMPT = """
You are a system that generates high quality document role annotations based on Framenet ontology.
The following inputs are given to you:
1. Event Type <event_type>: A Frame name from the FrameNet ontology (eg: Hiring, Arrest, etc.)
2. Event Definition <event_definition>: Definition of the event type along with an optional example. 
3. Roles <event_roles>: All roles (or participants) of the event type (or frame) followed with an optional example.
4. Report Document <report_document>: A report document with a tagged event '<event>' of the given event type.

Your job is to extract all the roles of the tagged Report <event> from the <report_document>. The ouput should be in a JSON string
where each key represents the role name as provided in the <event_roles> and its corresponding value should be a
list of contiguous text spans from the <report_document> that are valid for that role. 
Note that if no text span is found for a role, the value should be an empty list. 
Your text spans should strictly come from the <report_document>. DO NOT use spans from Event Definition or Roles sections.
"""

ONE_SHOT_DOC = {
    "report_document": "He was <event> hired </event> as a Research Scientist by Microsoft.",
    "source_document": "John Smith is a recent graduate of the University of Washington. He interned at Microsoft Research in Seattle, Washington. After 6 rounds of interviewing, he was hired as a Research Scientist by Microsoft to work on their new chatbot."}

TWO_SHOT_DOC = {
    "report_document": "In 2023, the governor granted <event> clemency </event> to the prisoner who had been wrongly convicted.",
    "source_document": "In 2023, Governor Rick granted clemency to John Doe, who had been wrongly convicted of murder in 1993. Doe was exonerated in 2012 after DNA testing proved that he was innocent. He spent 19 years in prison before being released."}

REPORT_ONE_SHOT_ASSISTANT_PROMPT = """
{
  "Employee": ["He"],
  "Employer": ["Microsoft"],
  "Task": [],
  "Position": ["as a Research Scientist"],
  "Field": []
}
"""

REPORT_TWO_SHOT_ASSISTANT_PROMPT = """
{
  "Offender": ["the prisoner"],
  "Crime": [],
  "Executive_authority": ["The governor"],
  "Time": ["2023"],
  "Place": []
}
"""

SOURCE_ONE_SHOT_ASSISTANT_PROMPT = """
{
  "Employee": ["John Smith"],
  "Employer": ["Microsoft"],
  "Task": ["to work on their new chatbot"],
  "Position": ["as a Research Scientist"],
  "Field": []
}
"""

SOURCE_TWO_SHOT_ASSISTANT_PROMPT = """
{
  "Offender": ["John Doe"],
  "Crime": ["murder"],
  "Executive_authority": ["Governor Rick"],
  "Time": ["2023"],
  "Place": []
}
"""

def generate_user_prompt_for_cross_doc_source_roles(frame: str, 
                                            tagged_report_document: str, 
                                            source_document: str,
                                            stanza_nlp = None,
                                            tokenizer = None,
                                            ignore_prompt_length: bool = False,
                                            previous_prompt_length: int = 0,
                                            prompt_response_length: int = 128,
                                            max_prompt_length: int = 4000):
    """Given a frame, report and source doc, create a user prompt for the LLM
    to extract cross-doc roles
    ------------
    Parameters:
    ------------
        frame: The frame name
        tagged_report_document: The report document with a tagged event.
        source_document: The source document.
        stanza_nlp: The stanza pipeline.
        tokenizer: The tokenizer used to decide truncation for source document.
        ignore_prompt_length: Whether to ignore the prompt length.
        previous_prompt_length: The length of the previous prompt.
        prompt_response_length: The length of the prompt response.
        max_prompt_length: The maximum length of the prompt.
    
    """
    event_llm_info_dct = frame_to_llm_prompt_info_dct(frame, stanza_nlp)
    
    prefix_event_role_report_prompt = f"""<event_type> {event_llm_info_dct['event_type']} </event_type> \
        <event_definition> {event_llm_info_dct['event_definition']} </event_definition>\
        <event_roles> {event_llm_info_dct['event_roles']} </event_roles>\
        <report_document> {tagged_report_document} </report_document>
    """

    # If the prompt length is ignored, return the prompt
    if ignore_prompt_length:
        source_only_prompt=f"""<source_document> {source_document} </source_document>
        """
        return prefix_event_role_report_prompt + source_only_prompt
    

    # Truncate document to fit the max token size
    length_of_prompt_without_source_document = previous_prompt_length + \
                                        len(tokenizer.encode(prefix_event_role_report_prompt)) 
    
    remaining_length =  max_prompt_length - (length_of_prompt_without_source_document + prompt_response_length)

    # Truncated Document 
    source_document_truncated = tokenizer.decode(tokenizer.encode(source_document)[:remaining_length])
    source_only_prompt=f"""<source_document> {source_document_truncated} </source_document>
    """
    return prefix_event_role_report_prompt + source_only_prompt



def generate_user_prompt_for_cross_doc_report_roles(frame: str, 
                                                    tagged_report_document: str, 
                                                    stanza_nlp = None,
                                                    ):
    """Given a frame, report and source doc, create a user prompt for the LLM
    to extract cross-doc roles
    ------------
    Parameters:
    ------------
        frame: The frame name
        tagged_report_document: The report document with a tagged event.
        stanza_nlp: The stanza pipeline.
    """
    event_llm_info_dct = frame_to_llm_prompt_info_dct(frame, stanza_nlp)
    
    report_prompt = f"""<event_type> {event_llm_info_dct['event_type']} </event_type> \
        <event_definition> {event_llm_info_dct['event_definition']} </event_definition>\
        <event_roles> {event_llm_info_dct['event_roles']} </event_roles>\
        <report_document> {tagged_report_document} </report_document>
    """
    return report_prompt


def combine_multiple_prompts_for_chatgpt(system_prompt: str,
                                         few_shot_prompts: List[Dict],
                                         suffix_prompt: str):
    """
    Combines the system prompt, few shot prompts and suffix prompt.

    Args:

        system_prompt: The system prompt.
        few_shot_prompts: The few shot prompts. This is a list of dicts
                            with keys 'user' and 'assistant' which contain
                            the user and assistant prompts respectively.

        suffix_prompt: The suffix prompt

    Returns:
        The combined prompt.

    """
    # Create a base message dictionary that will be used to create the messages
    messages=[{"role": "system", "content": system_prompt}]

    # Add the few shot prompts
    for few_shot_prompt in few_shot_prompts:
        messages.append({"role": "user", "content": few_shot_prompt['user']})
        messages.append({"role": "assistant", "content": few_shot_prompt['assistant']})

    # Add the suffix prompt
    messages.append({"role": "user", "content": suffix_prompt})

    return messages


def length_of_few_shot_prompts(few_shot_prompts: List[Dict[str, str]],
                               tokenizer):
    """
    Returns the length of the few shot prompts.

    Args:
        few_shot_prompts: The few shot prompts.
        tokenizer: The tokenizer.

    Returns:
        The length of the few shot prompts.
    """
    length = 0
    for prompt in few_shot_prompts:
        length += len(tokenizer.encode(prompt['user']))
        length += len(tokenizer.encode(prompt['assistant']))
    return length


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


def famusInstance_to_LLMPromptInstance_for_Source(instance,
                                     tokenizer = None,
                                     stanza_nlp = None,
                                     few_shot_prompts = None,
                                     max_output_prompt_length = 128,
                                     max_input_prompt_length = 4000,
                                    ):
    """Convert a single Source FAMuS instance to a LLM Prompt Instance
    """
    report_with_trigger_dict = famusInstance2ModifiedReportwithTrigger(instance,
                                                                    trigger_tag="event")
    report_context_with_trigger = report_with_trigger_dict['doctext']

    # Each FAMuS instance has exactly one template
    gold_frame = instance['frame']

    # Compute the previous_prompt_length
    previous_prompt_length =  length_of_prompt(SOURCE_SYSTEM_PROMPT, tokenizer) + \
                              length_of_few_shot_prompts(few_shot_prompts, 
                                                         tokenizer) 
    
    # Generate the final query prompt
    source_llm_prompt = generate_user_prompt_for_cross_doc_source_roles(gold_frame, 
                                                                report_context_with_trigger,
                                                                instance['source_dict']['doctext'],
                                                                stanza_nlp=stanza_nlp,
                                                                tokenizer = tokenizer,
                                                                ignore_prompt_length=False,
                                                                previous_prompt_length= previous_prompt_length,
                                                                prompt_response_length=max_output_prompt_length,
                                                                max_prompt_length=max_input_prompt_length)
    
    messages_prompt = combine_multiple_prompts_for_chatgpt(SOURCE_SYSTEM_PROMPT,
                                                            few_shot_prompts,
                                                              source_llm_prompt)

    return messages_prompt


def famusInstances_to_LLMPromptInstances_for_Source(instances,
                                     tokenizer = None,
                                     stanza_nlp = None,
                                     few_shot_prompts = None,
                                     max_output_prompt_length = 128,
                                     max_input_prompt_length = 4000,):
    """Convert a list of Source FAMuS instances to a list of LLM Prompt Instances
    """
    llm_prompt_instances = []
    for instance in tqdm(instances):
        current_instance = {'instance_id': instance['instance_id'],
                            'instance_id_raw_lome_predictor': instance['instance_id_raw_lome_predictor'],
                            'llm_prompt': famusInstance_to_LLMPromptInstance_for_Source(instance,
                                                                    tokenizer = tokenizer,
                                                                    stanza_nlp = stanza_nlp,
                                                                    few_shot_prompts = few_shot_prompts,
                                                                    max_output_prompt_length = max_output_prompt_length,
                                                                    max_input_prompt_length = max_input_prompt_length,
                                                                    ),
                            'gold_roles': json.dumps(famus_role_spans_to_llm_response_dict(instance['source_dict']['role_annotations']))}
        
        llm_prompt_instances.append(current_instance)

    return llm_prompt_instances


def famusInstance_to_LLMPromptInstance_for_Report(instance,
                                     stanza_nlp = None,
                                     few_shot_prompts = None,
                                    ):
    """Convert a single Report FAMuS instance to a LLM Prompt Instance
    """
    report_with_trigger_dict = famusInstance2ModifiedReportwithTrigger(instance,
                                                                    trigger_tag="event")
    report_context_with_trigger = report_with_trigger_dict['doctext']

    gold_frame = instance['frame']

    # Generate the final query prompt
    report_llm_prompt = generate_user_prompt_for_cross_doc_report_roles(gold_frame, 
                                                                report_context_with_trigger,
                                                                stanza_nlp=stanza_nlp,
                                                                )
    
    messages_prompt = combine_multiple_prompts_for_chatgpt(REPORT_SYSTEM_PROMPT,
                                                            few_shot_prompts,
                                                              report_llm_prompt)

    return messages_prompt


def famusInstances_to_LLMPromptInstances_for_Report(instances,
                                        stanza_nlp = None,
                                        few_shot_prompts = None,
                                        ):
    """Convert a list of Report FAMuS instances to a list of LLM Prompt Instances
    """
    llm_prompt_instances = []
    for instance in tqdm(instances):
        current_instance = {'instance_id': instance['instance_id'],
                            'instance_id_raw_lome_predictor': instance['instance_id_raw_lome_predictor'],
                            'llm_prompt': famusInstance_to_LLMPromptInstance_for_Report(instance,
                                                                    stanza_nlp = stanza_nlp,
                                                                    few_shot_prompts = few_shot_prompts,
                                                                    ),
                            'gold_roles': json.dumps(famus_role_spans_to_llm_response_dict(instance['report_dict']['role_annotations']))}
        
        llm_prompt_instances.append(current_instance)

    return llm_prompt_instances



def famus_role_spans_to_llm_response_dict(role_annotations):
    """
    Convert iterx template to llm response dict format
    """
    response_dict = {}
    for role_name, role_spans in role_annotations.items():
        if role_name == "role-spans-indices-in-all-spans":
            continue
        
        # fetch textual spans for each role
        if role_spans == []:
            response_dict[role_name] = []
        else:
            for span in role_spans:
                # it is assumed that there is only one span per coref cluster
                coref_single_value = span[0]
                span_text = coref_single_value[0]
                if role_name in response_dict:
                    response_dict[role_name].append(span_text)
                else:
                    response_dict[role_name] = [span_text]

    return response_dict


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', 
                        type=str, 
                        required=True,
                        default = "/data/svashishtha/FAMuS/data/cross_doc_role_extraction",
                        help='Path to the dir containing the famus release files: train, dev, test')
    
    parser.add_argument('--output_dir', type=str, 
                        required=True,
                        default = "/data/svashishtha/FAMuS/data/cross_doc_role_extraction/prompt_format",
                        help='Path to the output dir where the prompt format files will be written')
    
    args = parser.parse_args()


    return args


def main():
    args = parse_args()

    stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize', use_gpu=False)
    chat_gpt_tokenizer = tiktoken.get_encoding('cl100k_base')
    ##########################################################
    #### Report Few-shot Prompts ####
    ##########################################################
    report_one_shot_user_prompt = generate_user_prompt_for_cross_doc_report_roles('Hiring',
                                                               ONE_SHOT_DOC['report_document'],
                                                                stanza_nlp = stanza_nlp,
                                                                )

    report_two_shot_user_prompt = generate_user_prompt_for_cross_doc_report_roles('Clemency',
                                                                TWO_SHOT_DOC['report_document'],
                                                                stanza_nlp = stanza_nlp,
                                                                )
    

    report_fixed_two_shot_prompt = [{'user': report_one_shot_user_prompt,
                                    'assistant': REPORT_ONE_SHOT_ASSISTANT_PROMPT},
                                    {'user': report_two_shot_user_prompt,
                                    'assistant': REPORT_TWO_SHOT_ASSISTANT_PROMPT}]
    ##########################################################
    #### Source Few-shot Prompts ####
    ##########################################################
    source_one_shot_user_prompt = generate_user_prompt_for_cross_doc_source_roles('Hiring',
                                                               ONE_SHOT_DOC['report_document'],
                                                                ONE_SHOT_DOC['source_document'],
                                                                stanza_nlp = stanza_nlp,
                                                                ignore_prompt_length = True,
                                                                )

    source_two_shot_user_prompt = generate_user_prompt_for_cross_doc_source_roles('Clemency',
                                                                TWO_SHOT_DOC['report_document'],
                                                                TWO_SHOT_DOC['source_document'],
                                                                stanza_nlp = stanza_nlp,
                                                                ignore_prompt_length = True,)
    

    source_fixed_two_shot_prompts = [{'user': source_one_shot_user_prompt,
                                    'assistant': SOURCE_ONE_SHOT_ASSISTANT_PROMPT},
                                    {'user': source_two_shot_user_prompt,
                                    'assistant': SOURCE_TWO_SHOT_ASSISTANT_PROMPT}]

    ##################################################################
    ####### Load train, dev, test files ########
    ##################################################################
    # print(f"Loading train, dev, test files from {args.input_dir}...")
    with open(os.path.join(args.input_dir, "train.jsonl")) as f:
        train = [json.loads(line) for line in f]

    with open(os.path.join(args.input_dir, "dev.jsonl")) as f:
        dev = [json.loads(line) for line in f]

    with open(os.path.join(args.input_dir, "test.jsonl")) as f:
        test = [json.loads(line) for line in f]

    ##########################################################
    ####### Convert to LLM instances -- Report ######
    ##########################################################
    print(f"Converting report docs to LLM instances...")

    train_report_prompt_instances = famusInstances_to_LLMPromptInstances_for_Report(
                                                        train,
                                                        stanza_nlp = stanza_nlp,
                                                        few_shot_prompts=report_fixed_two_shot_prompt)
    
    dev_report_prompt_instances = famusInstances_to_LLMPromptInstances_for_Report(
                                                        dev,
                                                        stanza_nlp = stanza_nlp,
                                                        few_shot_prompts=report_fixed_two_shot_prompt)
    
    test_report_prompt_instances = famusInstances_to_LLMPromptInstances_for_Report(
                                                        test,
                                                        stanza_nlp = stanza_nlp,
                                                        few_shot_prompts=report_fixed_two_shot_prompt)  
    
    ## Export to jsonl files
    report_export_path = os.path.join(args.output_dir, "report_data")
    os.makedirs(report_export_path, exist_ok=True)
    pd.DataFrame(train_report_prompt_instances).to_json(os.path.join(report_export_path, "train.json"),
                                                lines=True, orient="records")
    
    pd.DataFrame(dev_report_prompt_instances).to_json(os.path.join(report_export_path, "dev.json"),
                                                lines=True, orient="records")
    
    pd.DataFrame(test_report_prompt_instances).to_json(os.path.join(report_export_path, "test.json"),
                                                lines=True, orient="records")
    
    print(f"Exported report data to {report_export_path}")

    ##########################################################
    ####### Convert to LLM instances -- Source ######
    ##########################################################
    print(f"Converting sources docs to LLM instances...")
    
    train_source_prompt_instances = famusInstances_to_LLMPromptInstances_for_Source(
                                                        train,
                                                        tokenizer = chat_gpt_tokenizer,
                                                        stanza_nlp = stanza_nlp,
                                                        few_shot_prompts=source_fixed_two_shot_prompts,
                                                        max_output_prompt_length = 128,
                                                        max_input_prompt_length = 4064)
       
    dev_source_prompt_instances = famusInstances_to_LLMPromptInstances_for_Source(
                                                        dev,
                                                        tokenizer = chat_gpt_tokenizer,
                                                        stanza_nlp = stanza_nlp,
                                                        few_shot_prompts=source_fixed_two_shot_prompts,
                                                        max_output_prompt_length = 128,
                                                        max_input_prompt_length = 4064)
    
    test_source_prompt_instances = famusInstances_to_LLMPromptInstances_for_Source(
                                                        test,
                                                        tokenizer = chat_gpt_tokenizer,
                                                        stanza_nlp = stanza_nlp,
                                                        few_shot_prompts=source_fixed_two_shot_prompts,
                                                        max_output_prompt_length = 128,
                                                        max_input_prompt_length = 4064)
    
    ### Export to jsonl files
    source_export_path = os.path.join(args.output_dir, "source_data")
    os.makedirs(source_export_path, exist_ok=True)
    pd.DataFrame(train_source_prompt_instances).to_json(os.path.join(source_export_path, "train.json"),
                                                lines=True, orient="records")
    
    pd.DataFrame(dev_source_prompt_instances).to_json(os.path.join(source_export_path, "dev.json"),
                                                lines=True, orient="records")
    
    pd.DataFrame(test_source_prompt_instances).to_json(os.path.join(source_export_path, "test.json"),
                                                lines=True, orient="records")
    
    print(f"Exported source data to {source_export_path}")


if __name__ == "__main__":
    main()