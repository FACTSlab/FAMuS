import os 
import sys 
import json 
from tqdm import tqdm
import argparse

def count_roles(data: list):
    """
    Counts the number of roles in the data
    """
    report_seen = set()
    source_seen = set()
    possible_roles = set()
    for doc in tqdm(data):
        report_annotations = doc['report_dict']['role_annotations']
        source_annotaitons = doc['source_dict']['role_annotations']
        for role in report_annotations:
            if role == 'role-spans-indices-in-all-spans':
                continue
            elif role == 'Time' or role == 'Place':
                if report_annotations[role] != []:
                    report_seen.add(role)
                if source_annotaitons[role] != []:
                    source_seen.add(role)
                possible_roles.add(role)
            else: 
                role_name = doc['frame'] + '::' + role
                if report_annotations[role] != []:
                    report_seen.add(role_name)
                if source_annotaitons[role] != []:
                    source_seen.add(role_name)
                possible_roles.add(role_name)

    return report_seen, source_seen, possible_roles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/cross_doc_role_extraction", help="directory of data") 
    parser.add_argument("--dataset", type=str, default="train", help="dataset to analyze") # train, dev, test
    parser.add_argument("--output_dir", type=str, default="src/data_processing/statistics/json_files", help="directory to save output")
    parser.add_argument("--verbose", action="store_true", help="print out statistics")
    args = parser.parse_args()

    # load data
    if args.dataset == "all":
        data = []
        for dataset in ["train", "dev", "test"]:
            with open(os.path.join(args.data_dir, dataset + ".jsonl"), "r") as f:
                for line in f:
                    data.append(json.loads(line))

    else:
        with open(os.path.join(args.data_dir, args.dataset + ".jsonl"), "r") as f:
            data = [json.loads(line) for line in f]

    report_seen, source_seen, possible_roles = count_roles(data)
    print(len(report_seen), len(source_seen), len(possible_roles))    



if __name__ == '__main__':
    main()