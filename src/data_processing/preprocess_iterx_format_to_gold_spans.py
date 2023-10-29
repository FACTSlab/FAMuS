import os
import json
import argparse
from tqdm import tqdm
from preprocess_release_format_to_iterx_format import exportList2Jsonl, loadJsonl

def mixedSpanIterxInstance2GoldSpanIterxInstance(iterx_instance):
    """
    Given a mixed span iterx instance, return a gold span iterx instance
    (be it report or source)
    Mixed-spans: SpanF + NER Entities from Stanza + Manual Annotated Spans by Workers
    The keys that change are: 
        - 'all-spans' (only keep the gold spans)
        - 'templates' (the indexes of the spans in all-spans change)
        - 'all-span-sets' (based on the new all-spans)
        - 'spans-to-spanset' (based on the new all-spans)
    """
    import copy
    new_dict = {}
    # fields that remain the same
    copy_fields = ['docid', 'doctext', 'doctext-tok',
                   'tok2char', 'char2tok']
    # Copy the fields that remain the same
    for field in copy_fields:
        new_dict[field] = iterx_instance[field]
    # We assume there is only one template in the mixed span iterx instance
    # This is true for all instances in the FAMuS dataset
    indexes_of_gold_spans = iterx_instance['templates'][0]['template-spans']
    trigger_span_inside_context = iterx_instance['all-spans'][0]
    gold_spans = [curr_span for idx, curr_span in enumerate(iterx_instance['all-spans'])
                        if idx in indexes_of_gold_spans]
    # We ALWAYS append the trigger span (as per the mixed-span context) to the beginning of the gold spans
    gold_spans = [trigger_span_inside_context] + gold_spans
    # Sort the span set based on the start_char_idx (following original iterx format)
    gold_spans = sorted(gold_spans, key=lambda x: x[1] +  x[2]/1000)
    # Create a dict map of gold spans:
    span_tuple_to_gold_span_idx = {tuple(span): idx for idx, span in enumerate(gold_spans)}
    # Create a new list of template spans
    new_template_spans = []
    for old_idx in indexes_of_gold_spans:
        tupl_span = tuple(iterx_instance['all-spans'][old_idx])
        new_idx = span_tuple_to_gold_span_idx[tupl_span]
        new_template_spans.append(new_idx)
    # deepcopy the old template
    new_template = copy.deepcopy(iterx_instance['templates'][0])
    new_template['template-spans'] = sorted(new_template_spans)
    
    new_dict = {'docid': iterx_instance['docid'],
                'doctext': iterx_instance['doctext'],
                'doctext-tok': iterx_instance['doctext-tok'],
                'all-spans': gold_spans,
                'templates': [new_template],
                'tok2char': iterx_instance['tok2char'],
                'char2tok': iterx_instance['char2tok'],
                'all-span-sets': [[span] for span in gold_spans],
                'spans-to-spanset': [i for i in range(len(gold_spans))]}
    
    return new_dict


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', 
                        type=str, 
                        required=True,
                        default = "data/cross_doc_role_extraction/iterx_format",
                        help='Path to the input mixed span iterx format files')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    ## Load the mixed spans data
    # Report Data
    train_iterx_report = loadJsonl(os.path.join(args.input_dir, "report_data", 
                                                "mixed_spans", "train.jsonl"))
    dev_iterx_report = loadJsonl(os.path.join(args.input_dir, "report_data", 
                                              "mixed_spans","dev.jsonl"))
    test_iterx_report = loadJsonl(os.path.join(args.input_dir, "report_data",
                                                "mixed_spans","test.jsonl"))
    # Source data
    train_iterx_source = loadJsonl(os.path.join(args.input_dir, "source_data",
                                                 "mixed_spans", "train.jsonl"))
    dev_iterx_source = loadJsonl(os.path.join(args.input_dir, "source_data", 
                                              "mixed_spans","dev.jsonl"))
    test_iterx_source = loadJsonl(os.path.join(args.input_dir, "source_data", 
                                               "mixed_spans","test.jsonl"))
    ##########################################
    # Report Data
    ###### Gold Spans ######
    # Gold spans: 
    # Only the spans annotated by the workers
    ##########################################
    report_export_path_gold_spans = os.path.join(args.input_dir,
                                        "report_data",
                                        "gold_spans")
    os.makedirs(report_export_path_gold_spans, exist_ok=True)
    train_iterx_report_gold_spans = [mixedSpanIterxInstance2GoldSpanIterxInstance(instance) 
                                        for instance in tqdm(train_iterx_report)]
    dev_iterx_report_gold_spans = [mixedSpanIterxInstance2GoldSpanIterxInstance(instance)
                                        for instance in tqdm(dev_iterx_report)]
    test_iterx_report_gold_spans = [mixedSpanIterxInstance2GoldSpanIterxInstance(instance)
                                        for instance in tqdm(test_iterx_report)]
    exportList2Jsonl(train_iterx_report_gold_spans, 
                     os.path.join(report_export_path_gold_spans, "train.jsonl"))
    exportList2Jsonl(dev_iterx_report_gold_spans,
                        os.path.join(report_export_path_gold_spans, "dev.jsonl"))
    exportList2Jsonl(test_iterx_report_gold_spans,
                        os.path.join(report_export_path_gold_spans, "test.jsonl"))
    print(f"Report Data (gold spans) Files exported to: {report_export_path_gold_spans}")
    #########################
    ###### Gold Spans ######
    # Source Data
    # Gold spans: Only the spans annotated by the workers
    #########################
    source_export_path_gold_spans = os.path.join(args.input_dir,
                                        "source_data",
                                        "gold_spans")
    os.makedirs(source_export_path_gold_spans, exist_ok=True)
    train_iterx_source_gold_spans = [mixedSpanIterxInstance2GoldSpanIterxInstance(instance)
                                        for instance in tqdm(train_iterx_source)]
    dev_iterx_source_gold_spans = [mixedSpanIterxInstance2GoldSpanIterxInstance(instance)
                                        for instance in tqdm(dev_iterx_source)]
    test_iterx_source_gold_spans = [mixedSpanIterxInstance2GoldSpanIterxInstance(instance)
                                        for instance in tqdm(test_iterx_source)]
    exportList2Jsonl(train_iterx_source_gold_spans,
                        os.path.join(source_export_path_gold_spans, "train.jsonl"))
    exportList2Jsonl(dev_iterx_source_gold_spans,
                        os.path.join(source_export_path_gold_spans, "dev.jsonl"))
    exportList2Jsonl(test_iterx_source_gold_spans,
                        os.path.join(source_export_path_gold_spans, "test.jsonl"))
    print(f"Source Data Files (gold spans) exported to: {source_export_path_gold_spans}")

if __name__ == "__main__":
    main()
