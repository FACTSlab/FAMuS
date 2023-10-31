import os 
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm



def token_counts(data: list):
    """
    Counts the average number of tokens in the report/source documents

    Returns:
        report tuple
            - average number of tokens in report
            - min report size
            - max report size
        source tuple
            - average number of tokens in source
            - min source size
            - max source size
    """
    report_token_counts = []
    source_token_counts = []
    min_report_size = sys.maxsize
    min_source_size = sys.maxsize
    max_report_size = -1
    max_source_size = -1

    # report token counts is length of report_dict[doctext-tok]
    # source token counts is length of source_dict[doctext-tok]
    for doc in tqdm(data):
        report_len = len(doc["report_dict"]["doctext-tok"])
        source_len = len(doc["source_dict"]["doctext-tok"])
        if report_len < min_report_size:
            min_report_size = report_len
        elif report_len > max_report_size:
            max_report_size = report_len

        if source_len < min_source_size:
            min_source_size = source_len
        elif source_len > max_source_size:
            max_source_size = source_len

        report_token_counts.append(report_len)
        source_token_counts.append(source_len)
    
    return (np.mean(report_token_counts), min_report_size, max_report_size), \
            (np.mean(source_token_counts), min_source_size, max_source_size)

def role_counts(data: list, num_docs=5):
    """
    Counts the average number/proportion of filled roles in report/source

    Returns:
        a dictionary with role data: 
        {
            frame: {
                average: average number of filled roles across all documents for the frame
                counts: {
                    role: average number of filled roles across all documents for the role
                }
            }
        }
    """
    role_info = {} #has structure {frame: average: int, counts:{role: count}}}
    for doc in tqdm(data):
        if doc['frame'] not in role_info:
            role_info[doc['frame']] = {'total': 0, 'average': 0, 'counts': {}}

        # get number of filled roles
        filled_roles = {} # has structure {role: 1/0}, 1 if filled, 0 if not
        for role in doc['report_dict']['role_annotations']:
            if role == 'role-spans-indices-in-all-spans':
                continue
            if doc['report_dict']['role_annotations'][role] != []:
                filled_roles[role] = 1
            else:
                filled_roles[role] = 0

        # update counts
        for role in filled_roles:
            if role not in role_info[doc['frame']]['counts']:
                role_info[doc['frame']]['counts'][role] = 0
            role_info[doc['frame']]['counts'][role] += filled_roles[role]

        # update average
        role_info[doc['frame']]['total'] += sum(filled_roles.values())

    # divide each count by 5 (the number of docs) to get the average 
    for frame in role_info:
        role_info[frame]['average'] = role_info[frame]['total'] / num_docs
        for role in role_info[frame]['counts']:
            role_info[frame]['counts'][role] = role_info[frame]['counts'][role] / num_docs
    
    return role_info

def normalize_roles(role_info: dict):
    """
    Normalize averages by number of roles in the frame

    Returns a dict of same structure receieved
    """
    normalized_roles = {}
    for frame in role_info:
        normalized_roles[frame] = {'average': 0, 'counts': {}}
        for role in role_info[frame]['counts']:
            normalized_roles[frame]['counts'][role] = role_info[frame]['counts'][role]
        normalized_roles[frame]['average'] = role_info[frame]['average'] / len(role_info[frame]['counts'])
    return normalized_roles


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/cross_doc_role_extraction", help="directory of data") 
    parser.add_argument("--dataset", type=str, default="all", help="dataset to analyze") # all, train, dev, test
    parser.add_argument("--output_dir", type=str, default="src/data_processing/statistics", help="directory to save output")
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

    if args.verbose: print("Number of documents: {}".format(len(data)))

    # get average number of tokens in report/source
    report_data, source_data = token_counts(data)

    if args.verbose: 
        print("Average number of tokens in report: {}".format(report_data[0]))
        print("Min Report Tokens: {}".format(report_data[1]))
        print("Max Report Tokens: {}".format(report_data[2]))
        print("Average number of tokens in source: {}".format(source_data[0]))
        print("Min Source Tokens: {}".format(source_data[1]))
        print("Max Source Tokens: {}".format(source_data[2]))

    # get average number/proportion of filled roles in report/source
    num_docs = 5
    if args.dataset == "train": 
        num_docs = 3
    elif args.dataset == "dev":
        num_docs = 1
    elif args.dataset == "test":
        num_docs = 1

    role_info = role_counts(data, num_docs=num_docs)

    if args.verbose:
        all_doc_average = 0
        for frame in role_info:
            all_doc_average += role_info[frame]['average']
        print("Average number of filled roles across all frames: {}".format(all_doc_average / len(role_info)))

    sorted_roles = sorted(role_info.items(), key=lambda x: x[1]['average'], reverse=True)

    # if args.verbose:
    #     print("Top 3 frames with the most filled roles:")
    #     for frame in sorted_roles[:3]:
    #         print(frame[0], frame[1]['average'])

    #     print("Top 3 frames with the least filled roles:")
    #     for frame in sorted_roles[-3:]:
    #         print(frame[0], frame[1]['average'])

    norm_roles = normalize_roles(role_info)

    sorted_norm_roles = sorted(norm_roles.items(), key=lambda x: x[1]['average'], reverse=True)

    if args.verbose:
        print("Top 10 frames with the most normalized filled roles:")
        for frame in sorted_norm_roles[:10]:
            print(frame[0], frame[1]['average'])

        print("Top 10 frames with the least normalized filled roles:")
        for frame in sorted_norm_roles[-10:]:
            print(frame[0], frame[1]['average'])

    # write sorted norm roles to a json file
    json_name = "sorted_norm_roles" + "_" + args.dataset + ".json"
    with open(os.path.join(args.output_dir, json_name), "w") as f:
        json.dump(sorted_norm_roles, f, indent=4)


if __name__ == "__main__":
    main()