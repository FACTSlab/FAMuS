import json
import pandas as pd
from tqdm import tqdm
import argparse
import os
from data_utils import famusInstance2ModifiedReportwithTrigger


def famusInstance2QAInstances(instance,
                            report_or_source = "source"):
    """Convert a single FAMuS instance to a list of QA instances
    Each role of the frame creates a new QA instance
    """
    squad_instances = []
    report_context_with_trigger = famusInstance2ModifiedReportwithTrigger(instance, 
                                                                trigger_tag="event")['doctext']
    report_trigger_text = instance['report_dict']['frame-trigger-span'][0]
    gold_frame = instance['frame']

    # Each FAMuS instance has exactly one template
    for role_name, role_spans in instance[f"{report_or_source}_dict"]['role_annotations'].items():
        curr_squad_instance = {}
        curr_squad_instance['id'] = instance['instance_id'] + "-Role-" + role_name
        # SKip keys that are not roles
        if role_name == 'role-spans-indices-in-all-spans':
            continue
        else:
        # Create a question for each role based on Report or Source
            if report_or_source == "report":
                question = f'''Event: {gold_frame}, Role: {role_name}, Trigger: {report_trigger_text}'''
            elif report_or_source == "source":
                question = f'''Event: {gold_frame}, Role: {role_name}, Report: {report_context_with_trigger}'''
                
            curr_squad_instance['question'] = question
            curr_squad_instance['context'] = instance[f"{report_or_source}_dict"]['doctext']

            if role_spans == []:
                answers = {'text': [] ,
                        'answer_start': []}
            else:
                role_span_tupl = role_spans[0][0]
                answers = {'text': [role_span_tupl[0]],
                        'answer_start': [role_span_tupl[1]]}
                
            curr_squad_instance['answers'] = answers

        squad_instances.append(curr_squad_instance)

    return squad_instances


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', 
                        type=str, 
                        required=True,
                        default = "/data/svashishtha/FAMuS/data/cross_doc_role_extraction/",
                        help='Path to the dir containing the famus release files: train, dev, test')
    
    parser.add_argument('--output_dir', type=str, 
                        required=True,
                        default = "/data/svashishtha/FAMuS/data/cross_doc_role_extraction/qa_format/",
                        help='Path to the output dir where the qa format files will be written')
    
    args = parser.parse_args()


    return args


def main():
    args = parse_args()
    ##################################################################
    ####### Load train, dev, test files ########
    ##################################################################
    print(f"Loading train, dev, test files from {args.input_dir}...")
    with open(os.path.join(args.input_dir, "train.jsonl")) as f:
        train = [json.loads(line) for line in f]

    with open(os.path.join(args.input_dir, "dev.jsonl")) as f:
        dev = [json.loads(line) for line in f]

    with open(os.path.join(args.input_dir, "test.jsonl")) as f:
        test = [json.loads(line) for line in f]

    
    ##########################################################
    ####### Convert to QA instances -- Report ######
    ##########################################################
    print(f"Converting report docs to QA instances...")
    train_QA_instances = []
    for instance in tqdm(train):
        train_QA_instances += famusInstance2QAInstances(instance,
                                                           report_or_source = "report")
        
    dev_QA_instances = []
    for instance in tqdm(dev):
        dev_QA_instances += famusInstance2QAInstances(instance,
                                                         report_or_source = "report")
        
    test_QA_instances = []
    for instance in tqdm(test):
        test_QA_instances += famusInstance2QAInstances(instance,
                                                          report_or_source = "report")
        
    ### Export to jsonl files
    report_export_path = os.path.join(args.output_dir, "report_data")
    os.makedirs(report_export_path, exist_ok=True)
    pd.DataFrame(train_QA_instances).to_json(os.path.join(report_export_path, "train.json"),
                                                lines=True, orient="records")
    pd.DataFrame(dev_QA_instances).to_json(os.path.join(report_export_path, "dev.json"),
                                                lines=True, orient="records")
    pd.DataFrame(test_QA_instances).to_json(os.path.join(report_export_path, "test.json"),
                                                lines=True, orient="records")
    ##########################################################
    print(f"Train QA instances: {len(train_QA_instances)}")
    print(f"Dev QA instances: {len(dev_QA_instances)}")
    print(f"Test QA instances: {len(test_QA_instances)}")
    ##########################################################

    print(f"Exported report data to {report_export_path}")
    
    ##########################################################
    ####### Convert to QA instances -- Source ######
    ##########################################################
    print(f"Converting source docs to QA instances...")
    train_QA_instances = []
    for instance in tqdm(train):
        train_QA_instances += famusInstance2QAInstances(instance,
                                                           report_or_source = "source")
        
    dev_QA_instances = []
    for instance in tqdm(dev):
        dev_QA_instances += famusInstance2QAInstances(instance,
                                                         report_or_source = "source")
        
    test_QA_instances = []
    for instance in tqdm(test):
        test_QA_instances += famusInstance2QAInstances(instance,
                                                          report_or_source = "source")
        

    ### Export to jsonl files
    source_export_path = os.path.join(args.output_dir, "source_data")
    os.makedirs(source_export_path, exist_ok=True)
    pd.DataFrame(train_QA_instances).to_json(os.path.join(source_export_path, "train.json"),
                                                lines=True, orient="records")
    pd.DataFrame(dev_QA_instances).to_json(os.path.join(source_export_path, "dev.json"),
                                                lines=True, orient="records")
    pd.DataFrame(test_QA_instances).to_json(os.path.join(source_export_path, "test.json"),
                                                lines=True, orient="records")
    
    print(f"Exported source data to {source_export_path}")

    ##########################################################
    print(f"Train QA instances: {len(train_QA_instances)}")
    print(f"Dev QA instances: {len(dev_QA_instances)}")
    print(f"Test QA instances: {len(test_QA_instances)}")
    ##########################################################


if __name__ == "__main__":
    main()