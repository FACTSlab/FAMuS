import spacy_alignments as tokenizations
import argparse
import json
import os
from tqdm import tqdm
from typing import Dict
from data_utils import (famusInstance2ModifiedReportwithTrigger, 
                        loadJsonl,
                        exportList2Jsonl)

def _modify_template_spans(frame_name: str,
                            template: Dict,
                            charoffset: int,
                            tokenoffset: int,
                            spans_to_idx_map: Dict):
    import copy
    """Modify the template spans to account for the concatenated report and source text.
    
    Args:
        frame_name: the frame name of the template
        template: the template dict
        charoffset: the offset to be added to the char indices
        tokenoffset: the offset to be added to the token indices
        spans_to_idx_map: a dict mapping a span to its index in all-spans
    """
    new_template_dict = {}
    span_index_list = []
    new_template_dict['incident_type'] = frame_name
    # each role can have multiple spans
    for role_name, role_spans in template.items():
        if role_name == 'role-spans-indices-in-all-spans':
            continue
        copied_role_spans = copy.deepcopy(role_spans)
        # non-emty role spans
        if role_spans != []:
            # we modify the indices for each span in the role
            # print(copied_role_spans)
            for current_span in copied_role_spans:
                current_span[1] += charoffset
                current_span[2] += charoffset
                current_span[3] += tokenoffset
                current_span[4] += tokenoffset
                span_index_list.append(spans_to_idx_map[tuple(current_span[1:5])])
            # Each span in Iter-X is supposed to be a list of coref mentions
            # so we wrap each span in a list, because we have only one coref 
            # mention per span
            coref_copied_role_spans = []
            for current_span in copied_role_spans:
                coref_copied_role_spans.append([current_span])
        # empty role spans
        else:
            coref_copied_role_spans = []
        new_template_dict[role_name] = coref_copied_role_spans

    # find the indices of the spans in the all-spans list
    span_index_list_unique = sorted(list(set(span_index_list)))
    new_template_dict['template-spans'] = span_index_list_unique 

    return [new_template_dict]


def famusInstance2IterXSourceInstance(instance,
                                      trigger_tag='event'):
    """
    Given a FAMUS instance, return an IterX instance with the following fields:
    'doctext': report text with trigger + source text
    'all-spans': trigger span + all spans from the source text
    'templates': templates from the source text with correct span indices
    'doctext-tok': report text with trigger + source text tokenized
    'tok2char': 
    'char2tok', 
    'all-span-sets',
    'spans-to-spanset'
    """
    passage_with_trigger_dict = famusInstance2ModifiedReportwithTrigger(instance,
                                                                trigger_tag=trigger_tag)
    source_data = instance["source_dict"]
    # We concatenate the report text with the Source text
    reportAndSource_text = passage_with_trigger_dict['doctext'] + " " + source_data['doctext']
    reportAndSource_tokens = passage_with_trigger_dict['doctext-tok'] + source_data['doctext-tok']

    tok2char, char2tok = tokenizations.get_alignments(reportAndSource_tokens, 
                                                      [char for char in reportAndSource_text])
    
    ############################ Fix Indices for all-spans ##################################
    # Shift all-spans Chars and token indices by the length of the report text
    modified_spans = []
    # the 1 is added to account for the space between the report and source text
    charoffset = len(passage_with_trigger_dict['doctext']) + 1
    tokenoffset = len(passage_with_trigger_dict['doctext-tok'])

    # Add the trigger span as the first span
    modified_spans.append(passage_with_trigger_dict['frame-trigger-span'])

    # Add the rest of the spans using the offsets
    for span_string, ch_start, ch_end, tok_start, tok_end, extra_string_val in source_data['all-spans']:
        modified_spans.append([span_string, 
                               ch_start + charoffset, 
                               ch_end + charoffset,
                               tok_start + tokenoffset, 
                               tok_end + tokenoffset, 
                               extra_string_val])
        
    #####################  Get Span-to-Idx Dictionary To be used by Templates ####################
    # Sort the list based on the start_char_idx and then the end_char_idx
    modified_spans = sorted(modified_spans, key=lambda x: x[1] +  x[2]/1000)

    # add spans_to_idx_map which maps a span to its index in all-spans
    # this is used to fill template-spans field inside the templates field of each instance
    spans_to_idx_map = {}
    for span_idx, span in enumerate(modified_spans):
        # the key is a tuple of the start char index, end char index, start token index, end token index
        spans_to_idx_map[tuple((span[1:5]))] = span_idx
    
    IterxSourceInstance = {'docid': instance['instance_id'],
                           'doctext': reportAndSource_text,
                           'doctext-tok': reportAndSource_tokens,
                           'all-spans': modified_spans,
                           'templates': _modify_template_spans(instance['frame'],
                                                               source_data['role_annotations'],
                                                                charoffset,
                                                                tokenoffset,
                                                                spans_to_idx_map),
                           'tok2char': tok2char,
                           'char2tok': char2tok,
                           'all-span-sets': [[span] for span in modified_spans],
                           'spans-to-spanset': [i for i in range(len(modified_spans))]
                          }
    
    return IterxSourceInstance


def famusInstance2IterXReportInstance(instance, trigger_tag='event'):
    """
    Given a FAMUS instance, return an IterX instance with the following fields:
    'doctext': report text with trigger + source text
    'all-spans': trigger span + all spans from the source text
    'templates': templates from the source text with correct span indices
    'doctext-tok': report text with trigger + source text tokenized
    'tok2char': 
    'char2tok', 
    'all-span-sets',
    'spans-to-spanset'
    """
    # For report, we fetch passage_with_trigger_dict only to get the trigger characters and tokens
    # we don't need the trigger spans as they would the the first token and chars in the report
    passage_with_trigger_dict = famusInstance2ModifiedReportwithTrigger(instance,
                                                                trigger_tag=trigger_tag)
    
    report_data = instance['report_dict']
    trigger_text_in_report = passage_with_trigger_dict['frame-trigger-span'][0]
    trigger_token_start_idx = passage_with_trigger_dict['frame-trigger-span'][3]
    trigger_token_end_idx = passage_with_trigger_dict['frame-trigger-span'][4]
    trigger_tokens_in_report = passage_with_trigger_dict['doctext-tok'][trigger_token_start_idx:trigger_token_end_idx+1]

    # We concatenate the report text with the Source text
    triggerAndReport_text = trigger_text_in_report + " " + report_data['doctext']
    triggerAndReport_tokens = trigger_tokens_in_report + report_data['doctext-tok']

    tok2char, char2tok = tokenizations.get_alignments(triggerAndReport_tokens, 
                                                      [char for char in triggerAndReport_text])
    
    ############################ Fix Indices for all-spans ##################################
    # Shift all-spans Chars and token indices by the length of the report text
    modified_spans = []
    # the 1 is added to account for the space between the report and source text
    charoffset = len(trigger_text_in_report) + 1
    tokenoffset = len(trigger_tokens_in_report)

    # Add the trigger span as the first span
    # -1 is added because the end index is inclusive
    trigger_span = [trigger_text_in_report, 0 , 
                    len(trigger_text_in_report)-1, 
                    0, len(trigger_tokens_in_report)-1, 
                    '']
    
    modified_spans.append(trigger_span)

    # Add the rest of the spans using the offsets
    for span_string, ch_start, ch_end, tok_start, tok_end, extra_string_val in report_data['all-spans']:
        modified_spans.append([span_string, 
                               ch_start + charoffset, 
                               ch_end + charoffset,
                               tok_start + tokenoffset, 
                               tok_end + tokenoffset, 
                               extra_string_val])
        
    #####################  Get Span-to-Idx Dictionary To be used by Templates ####################
    # Sort the list based on the start_char_idx and then the end_char_idx
    modified_spans = sorted(modified_spans, key=lambda x: x[1] +  x[2]/1000)

    # add spans_to_idx_map which maps a span to its index in all-spans
    # this is used to fill template-spans field inside the templates field of each instance
    spans_to_idx_map = {}
    for span_idx, span in enumerate(modified_spans):
        # the key is a tuple of the start char index, end char index, start token index, end token index
        spans_to_idx_map[tuple((span[1:5]))] = span_idx
    
    IterxReportInstance = {'docid': instance['instance_id'],
                           'doctext': triggerAndReport_text,
                           'doctext-tok': triggerAndReport_tokens,
                           'all-spans': modified_spans,
                           'templates': _modify_template_spans(instance['frame'],
                                                               report_data['role_annotations'],
                                                                charoffset,
                                                                tokenoffset,
                                                                spans_to_idx_map),
                           'tok2char': tok2char,
                           'char2tok': char2tok,
                           'all-span-sets': [[span] for span in modified_spans],
                           'spans-to-spanset': [i for i in range(len(modified_spans))]
                          }
    
    return IterxReportInstance


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', 
                        type=str, 
                        required=True,
                        default = "data/cross_doc_role_extraction/",
                        help='Path to the input release format files')
    
    parser.add_argument('--output_dir', type=str, 
                        required=True,
                        default = "data/cross_doc_role_extraction/iterx_format",
                        help='Path to the output iterx format files')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    print("Reading the release format files...")
    train = loadJsonl(os.path.join(args.input_dir, "train.jsonl"))
    dev = loadJsonl(os.path.join(args.input_dir, "dev.jsonl"))
    test = loadJsonl(os.path.join(args.input_dir, "test.jsonl"))
    ####################################################################
    ########## Report Text Data ##########################################
    ####################################################################
    # convert the train, dev, test instances to iterx format
    #########################
    ###### Mixed Spans ######
    # Mixed spans: 
    #   SpanF + \
    #   NER Entities from Stanza + \
    #   Manual Annotated Spans by Workers
    #########################
    print("Converting to iterx format for report...")
    train_iterx_report = [famusInstance2IterXReportInstance(instance) for instance in tqdm(train)]
    dev_iterx_report = [famusInstance2IterXReportInstance(instance) for instance in tqdm(dev)]
    test_iterx_report = [famusInstance2IterXReportInstance(instance) for instance in tqdm(test)]

    report_export_path_mixed_spans = os.path.join(args.output_dir, 
                                      "report_data", 
                                      "mixed_spans")    
    os.makedirs(report_export_path_mixed_spans, exist_ok=True)
    # Export the iterx format instances
    exportList2Jsonl(train_iterx_report, os.path.join(report_export_path_mixed_spans,
                                                       "train.jsonl"))
    exportList2Jsonl(dev_iterx_report, os.path.join(report_export_path_mixed_spans,
                                                     "dev.jsonl"))
    exportList2Jsonl(test_iterx_report, os.path.join(report_export_path_mixed_spans, 
                                                     "test.jsonl")) 
    print(f"Report Data (mixed spans) Files exported to: {report_export_path_mixed_spans}")
    
    ####################################################################
    ########## Source Text Data ########################################
    ####################################################################
    # convert the train, dev, test instances to iterx format
    print("Converting to iterx format for source...")
    train_iterx_source = [famusInstance2IterXSourceInstance(instance) for instance in tqdm(train)]
    dev_iterx_source = [famusInstance2IterXSourceInstance(instance) for instance in tqdm(dev)]
    test_iterx_source = [famusInstance2IterXSourceInstance(instance) for instance in tqdm(test)]

    source_export_path_mixed_spans = os.path.join(args.output_dir, 
                                      "source_data",
                                      "mixed_spans")
    os.makedirs(source_export_path_mixed_spans, exist_ok=True)
    # Export the iterx format instances
    exportList2Jsonl(train_iterx_source, os.path.join(source_export_path_mixed_spans,
                                                         "train.jsonl"))
    exportList2Jsonl(dev_iterx_source, os.path.join(source_export_path_mixed_spans,
                                                         "dev.jsonl"))
    exportList2Jsonl(test_iterx_source, os.path.join(source_export_path_mixed_spans,
                                                            "test.jsonl"))
    
    print(f"Source Data Files (mixed spans) exported to: {source_export_path_mixed_spans}")

if __name__ == "__main__":
    main()
    