import argparse
import os
from preprocess_release_format_to_iterx_format import exportList2Jsonl, loadJsonl
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_release_dir', 
                        type=str, 
                        required=True,
                        default = "data/cross_doc_role_extraction/",
                        help='Path to the input famus release dir')
    
    parser.add_argument('--input_iterx_format_dir', 
                        type=str, 
                        required=True,
                        default = "data/cross_doc_role_extraction/iterx_format",
                        help='Path to the iterx format dir')
    
    return parser.parse_args()


def main():

    args = parse_arguments()
    ##########################################
    ## Load FAMuS data
    ##########################################
    famus_train = loadJsonl(os.path.join(args.input_release_dir, "train.jsonl"))
    famus_dev = loadJsonl(os.path.join(args.input_release_dir, "dev.jsonl"))
    famus_test = loadJsonl(os.path.join(args.input_release_dir, "test.jsonl"))

    
    famus_id_2_all_spans_manual_auto = {}
    for instance in tqdm(famus_train + famus_dev + famus_test):
        famus_id_2_all_spans_manual_auto[instance["instance_id"]] = {
                'report': instance['report_dict']['all-spans-to-manual-auto'],
                'source': instance['source_dict']['all-spans-to-manual-auto']}

    ##########################################
    ## Load the mixed spans data - Report Data
    ##########################################
    dev_iterx_report = loadJsonl(os.path.join(args.input_iterx_format_dir, "report_data", 
                                              "mixed_spans","dev.jsonl"))
    test_iterx_report = loadJsonl(os.path.join(args.input_iterx_format_dir, "report_data",
                                                "mixed_spans","test.jsonl"))
    
    # the first span in Iterx is always the trigger span
    for instance in dev_iterx_report + test_iterx_report:
        manual_or_auto_list = famus_id_2_all_spans_manual_auto[instance['docid']]['report']
        assert len(instance['all-spans'][1:]) == len(manual_or_auto_list)
        # we add 1 to the span index because the first span is the trigger span
        # 0 is added by default since the first span is always the trigger span
        span_idx_of_auto = [0] + [all_span_idx+1 for all_span_idx, string in enumerate(manual_or_auto_list) if string == 'auto']

        instance['all-pred-spans'] = [span for span_idx, span in enumerate(instance['all-spans']) 
                                        if span_idx in span_idx_of_auto]
        instance['all-pred-span-sets'] = [[span] for span in instance['all-pred-spans']]
        instance["pred-spans-to-spanset"] = [i for i in range(len(instance['all-pred-spans']))]

    # Export the report data
    report_export_path_predicted_spans = os.path.join(args.input_iterx_format_dir,
                                                    "report_data",
                                                    "predicted_spans")
    os.makedirs(report_export_path_predicted_spans, exist_ok=True)
    exportList2Jsonl(dev_iterx_report, os.path.join(args.input_iterx_format_dir, "report_data",
                                                        "predicted_spans", "dev.jsonl"))
    exportList2Jsonl(test_iterx_report, os.path.join(args.input_iterx_format_dir, "report_data",
                                                        "predicted_spans", "test.jsonl"))
    
    print(f"Report Data (predicted spans) Files exported to: {report_export_path_predicted_spans}")

    ##########################################
    ## Load the mixed spans data - Source Data
    ##########################################
    dev_iterx_source = loadJsonl(os.path.join(args.input_iterx_format_dir, "source_data",
                                                "mixed_spans","dev.jsonl"))
    test_iterx_source = loadJsonl(os.path.join(args.input_iterx_format_dir, "source_data",
                                                    "mixed_spans","test.jsonl"))
    
    # the first span in Iterx is always the trigger span
    for instance in dev_iterx_source + test_iterx_source:
        manual_or_auto_list = famus_id_2_all_spans_manual_auto[instance['docid']]['source']
        assert len(instance['all-spans'][1:]) == len(manual_or_auto_list)
        # we add 1 to the span index because the first span is the trigger span
        # 0 is added by default since the first span is always the trigger span
        span_idx_of_auto = [0] + [all_span_idx+1 for all_span_idx, string in enumerate(manual_or_auto_list) if string == 'auto']

        instance['all-pred-spans'] = [span for span_idx, span in enumerate(instance['all-spans']) 
                                        if span_idx in span_idx_of_auto]
        instance['all-pred-span-sets'] = [[span] for span in instance['all-pred-spans']]
        instance["pred-spans-to-spanset"] = [i for i in range(len(instance['all-pred-spans']))]

    # Export the source data
    source_export_path_predicted_spans = os.path.join(args.input_iterx_format_dir,
                                                    "source_data",
                                                    "predicted_spans")
    os.makedirs(source_export_path_predicted_spans, exist_ok=True)
    exportList2Jsonl(dev_iterx_source, os.path.join(args.input_iterx_format_dir, "source_data",
                                                        "predicted_spans", "dev.jsonl"))
    exportList2Jsonl(test_iterx_source, os.path.join(args.input_iterx_format_dir, "source_data",
                                                        "predicted_spans", "test.jsonl"))
    print(f"Source Data (predicted spans) Files exported to: {source_export_path_predicted_spans}")

if __name__ == "__main__":
    main()
