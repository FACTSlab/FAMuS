# This script converts the Iter-x format data into a new format
# which prefixes the role labels with the frame label.
import os
import json
import argparse
from data_utils import loadJsonl, exportList2Jsonl
from tqdm import tqdm


def convert_iterxInstance2roleByFrameIterxInstance(instance):
    """
    Given an iterx instance, change the templates to have 
    role names prefixed by their corresponding frame names.
    """
    import copy
    # deep copy the instance
    instance_new = copy.deepcopy(instance)
    # each FAMuS Iterx instance has exactly one template
    template = instance_new['templates'][0]
    frame = template['incident_type']
    new_template = {}
    for role, role_fillers in template.items():
        if role == 'incident_type' or role == 'template-spans':
            new_template[role] = role_fillers
        else:
            new_role_name = f"{frame}.{role}"
            # change the last field in the role filler
            # to have the new role name
            if role_fillers:
                for entity in role_fillers:
                    # each entity is a list of its coref mentions, but we only one
                    coref_mention = entity[0]
                    coref_mention[-1] = new_role_name
                new_template[new_role_name] = role_fillers
            else:
                new_template[new_role_name] = role_fillers
    instance_new['templates'] = [new_template]

    return instance_new
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', 
                        type=str, 
                        required=True,
                        default = "data/cross_doc_role_extraction/iterx_format",
                        help='Path to the input data iterx format dir')
    
    parser.add_argument('--output_dir', 
                        type=str, 
                        required=True,
                        default = "data/cross_doc_role_extraction/iterx_format_with_prefixed_roles",
                        help='Path to the input data iterx format dir')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    ##########################################
    # Report Data
    ##########################################
    ##########################
    ###### Gold Spans ########
    ##########################
    report_gold_train_iterx_instances = loadJsonl(os.path.join(args.input_dir, "report_data", "gold_spans", "train.jsonl"))
    report_gold_dev_iterx_instances = loadJsonl(os.path.join(args.input_dir, "report_data", "gold_spans", "dev.jsonl"))
    report_gold_test_iterx_instances = loadJsonl(os.path.join(args.input_dir, "report_data", "gold_spans", "test.jsonl"))
    report_gold_train_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(report_gold_train_iterx_instances)]
    report_gold_dev_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(report_gold_dev_iterx_instances)]
    report_gold_test_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(report_gold_test_iterx_instances)]
    ###### Export ########
    report_gold_dir = os.path.join(args.output_dir, "report_data", "gold_spans")
    os.makedirs(report_gold_dir, exist_ok=True)
    exportList2Jsonl(report_gold_train_iterx_instances_with_prefixed_roles,
                        os.path.join(report_gold_dir, "train.jsonl"))
    exportList2Jsonl(report_gold_dev_iterx_instances_with_prefixed_roles,
                        os.path.join(report_gold_dir, "dev.jsonl"))
    exportList2Jsonl(report_gold_test_iterx_instances_with_prefixed_roles,
                        os.path.join(report_gold_dir, "test.jsonl"))
    print(f"Report Data (gold spans) Files exported to: {report_gold_dir}")
    ##########################
    ###### Mixed Spans ########
    ##########################
    report_mixed_train_iterx_instances = loadJsonl(os.path.join(args.input_dir, "report_data", "mixed_spans", "train.jsonl"))
    report_mixed_dev_iterx_instances = loadJsonl(os.path.join(args.input_dir, "report_data", "mixed_spans", "dev.jsonl"))
    report_mixed_test_iterx_instances = loadJsonl(os.path.join(args.input_dir, "report_data", "mixed_spans", "test.jsonl"))
    report_mixed_train_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(report_mixed_train_iterx_instances)]
    report_mixed_dev_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(report_mixed_dev_iterx_instances)]
    report_mixed_test_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(report_mixed_test_iterx_instances)]
    ###### Export ########
    report_mixed_dir = os.path.join(args.output_dir, "report_data", "mixed_spans")
    os.makedirs(report_mixed_dir, exist_ok=True)
    exportList2Jsonl(report_mixed_train_iterx_instances_with_prefixed_roles,
                        os.path.join(report_mixed_dir, "train.jsonl"))
    exportList2Jsonl(report_mixed_dev_iterx_instances_with_prefixed_roles,
                        os.path.join(report_mixed_dir, "dev.jsonl"))
    exportList2Jsonl(report_mixed_test_iterx_instances_with_prefixed_roles,
                        os.path.join(report_mixed_dir, "test.jsonl"))
    print(f"Report Data (mixed spans) Files exported to: {report_mixed_dir}")
    ##########################
    ###### Predicted Spans ########
    ##########################
    report_predicted_dev_iterx_instances = loadJsonl(os.path.join(args.input_dir, "report_data", "predicted_spans", "dev.jsonl"))
    report_predicted_test_iterx_instances = loadJsonl(os.path.join(args.input_dir, "report_data", "predicted_spans", "test.jsonl"))
    report_predicted_dev_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(report_predicted_dev_iterx_instances)]
    report_predicted_test_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(report_predicted_test_iterx_instances)]
    ###### Export ########
    report_predicted_dir = os.path.join(args.output_dir, "report_data", "predicted_spans")
    os.makedirs(report_predicted_dir, exist_ok=True)
    exportList2Jsonl(report_predicted_dev_iterx_instances_with_prefixed_roles,
                        os.path.join(report_predicted_dir, "dev.jsonl"))
    exportList2Jsonl(report_predicted_test_iterx_instances_with_prefixed_roles,
                        os.path.join(report_predicted_dir, "test.jsonl"))
    print(f"Report Data (predicted spans) Files exported to: {report_predicted_dir}")

    ##########################################
    # Source Data
    ##########################################
    ##########################
    ###### Gold Spans ########
    ##########################
    source_gold_train_iterx_instances = loadJsonl(os.path.join(args.input_dir, "source_data", "gold_spans", "train.jsonl"))
    source_gold_dev_iterx_instances = loadJsonl(os.path.join(args.input_dir, "source_data", "gold_spans", "dev.jsonl"))
    source_gold_test_iterx_instances = loadJsonl(os.path.join(args.input_dir, "source_data", "gold_spans", "test.jsonl"))
    source_gold_train_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(source_gold_train_iterx_instances)]
    source_gold_dev_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(source_gold_dev_iterx_instances)] 
    source_gold_test_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(source_gold_test_iterx_instances)]
    ###### Export ########
    source_gold_dir = os.path.join(args.output_dir, "source_data", "gold_spans")
    os.makedirs(source_gold_dir, exist_ok=True)
    exportList2Jsonl(source_gold_train_iterx_instances_with_prefixed_roles,
                        os.path.join(source_gold_dir, "train.jsonl"))
    exportList2Jsonl(source_gold_dev_iterx_instances_with_prefixed_roles,
                        os.path.join(source_gold_dir, "dev.jsonl"))
    exportList2Jsonl(source_gold_test_iterx_instances_with_prefixed_roles,
                        os.path.join(source_gold_dir, "test.jsonl"))
    print(f"Source Data (gold spans) Files exported to: {source_gold_dir}")
    ##########################
    ###### Mixed Spans ########
    ##########################
    source_mixed_train_iterx_instances = loadJsonl(os.path.join(args.input_dir, "source_data", "mixed_spans", "train.jsonl"))
    source_mixed_dev_iterx_instances = loadJsonl(os.path.join(args.input_dir, "source_data", "mixed_spans", "dev.jsonl"))
    source_mixed_test_iterx_instances = loadJsonl(os.path.join(args.input_dir, "source_data", "mixed_spans", "test.jsonl"))
    source_mixed_train_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(source_mixed_train_iterx_instances)]
    source_mixed_dev_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(source_mixed_dev_iterx_instances)]
    source_mixed_test_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(source_mixed_test_iterx_instances)]
    ###### Export ########
    source_mixed_dir = os.path.join(args.output_dir, "source_data", "mixed_spans")
    os.makedirs(source_mixed_dir, exist_ok=True)
    exportList2Jsonl(source_mixed_train_iterx_instances_with_prefixed_roles,
                        os.path.join(source_mixed_dir, "train.jsonl"))
    exportList2Jsonl(source_mixed_dev_iterx_instances_with_prefixed_roles,
                        os.path.join(source_mixed_dir, "dev.jsonl"))
    exportList2Jsonl(source_mixed_test_iterx_instances_with_prefixed_roles,
                        os.path.join(source_mixed_dir, "test.jsonl"))
    print(f"Source Data (mixed spans) Files exported to: {source_mixed_dir}")
    ##########################
    ###### Predicted Spans ########
    ##########################
    source_predicted_dev_iterx_instances = loadJsonl(os.path.join(args.input_dir, "source_data", "predicted_spans", "dev.jsonl"))
    source_predicted_test_iterx_instances = loadJsonl(os.path.join(args.input_dir, "source_data", "predicted_spans", "test.jsonl"))
    source_predicted_dev_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(source_predicted_dev_iterx_instances)]
    source_predicted_test_iterx_instances_with_prefixed_roles = [convert_iterxInstance2roleByFrameIterxInstance(instance)
                                                            for instance in tqdm(source_predicted_test_iterx_instances)]
    ###### Export ########
    source_predicted_dir = os.path.join(args.output_dir, "source_data", "predicted_spans")
    os.makedirs(source_predicted_dir, exist_ok=True)
    exportList2Jsonl(source_predicted_dev_iterx_instances_with_prefixed_roles,
                        os.path.join(source_predicted_dir, "dev.jsonl"))
    exportList2Jsonl(source_predicted_test_iterx_instances_with_prefixed_roles,
                        os.path.join(source_predicted_dir, "test.jsonl"))
    print(f"Source Data (predicted spans) Files exported to: {source_predicted_dir}")


if __name__ == "__main__":  
    main()










