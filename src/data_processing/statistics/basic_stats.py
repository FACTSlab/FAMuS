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
                    report{role: count}
                    source{}
                    combined{}
                }
            }
        }
    """
    role_info = {} #has structure {frame: average: int, counts:{role: count}}}
    for doc in tqdm(data):
        if doc['frame'] not in role_info:
            role_info[doc['frame']] = {'combined-total': 0, 
                                        'role-total': 0, 
                                        'source-total': 0,
                                        'combined-average': 0, 
                                        'role-average': 0,
                                        'source-average': 0,
                                        'counts': 
                                        {'report': {}, 'source': {}, 'combined': {}}}
            
        # get number of filled roles for report 
        # both have structure {role: 1/0}, 1 if filled, 0 if not
        filled_report_roles = {}  
        filled_source_roles = {}
        filled_roles_combined = {}

        for role in doc['report_dict']['role_annotations']:
            if role == 'role-spans-indices-in-all-spans':
                continue
            if doc['report_dict']['role_annotations'][role] != []:
                filled_report_roles[role] = 1
            else:
                filled_report_roles[role] = 0

        for role in doc['source_dict']['role_annotations']:
            if role == 'role-spans-indices-in-all-spans':
                continue
            if doc['source_dict']['role_annotations'][role] != []:
                filled_source_roles[role] = 1
            else:
                filled_source_roles[role] = 0

        # get number of filled roles for combined report and source
        for role in filled_report_roles:
            if filled_report_roles[role] == 1 or filled_source_roles[role] == 1:
                filled_roles_combined[role] = 1
            else:
                filled_roles_combined[role] = 0

        # update counts
        for role in filled_report_roles:
            if role not in role_info[doc['frame']]['counts']['report']:
                role_info[doc['frame']]['counts']['report'][role] = 0
            role_info[doc['frame']]['counts']['report'][role] += filled_report_roles[role]

        for role in filled_source_roles:
            if role not in role_info[doc['frame']]['counts']['source']:
                role_info[doc['frame']]['counts']['source'][role] = 0
            role_info[doc['frame']]['counts']['source'][role] += filled_source_roles[role]

        for role in filled_roles_combined:
            if role not in role_info[doc['frame']]['counts']['combined']:
                role_info[doc['frame']]['counts']['combined'][role] = 0
            role_info[doc['frame']]['counts']['combined'][role] += filled_roles_combined[role]

        # update average
        role_info[doc['frame']]['combined-total'] += sum(filled_roles_combined.values())
        role_info[doc['frame']]['role-total'] += sum(filled_report_roles.values())
        role_info[doc['frame']]['source-total'] += sum(filled_source_roles.values())


    # divide each count by 5 (the number of docs) to get the average 
    for frame in role_info:
        role_info[frame]['combined-average'] = role_info[frame]['combined-total'] / num_docs
        role_info[frame]['role-average'] = role_info[frame]['role-total'] / num_docs
        role_info[frame]['source-average'] = role_info[frame]['source-total'] / num_docs
        for split in role_info[frame]['counts']:
            for role in role_info[frame]['counts'][split]:
                role_info[frame]['counts'][split][role] /= num_docs
    return role_info

def argument_counts(data: list, num_docs=5):
    """
    Counts the average number/proportion of filled roles in report/source

    Returns:
        a dictionary with role data: 
        {
            frame: {
                average: average number of filled roles across all documents for the frame
                counts: {
                    report{role: count}
                    source{}
                    combined{}
                }
            }
        }
    """
    arg_info = {} #has structure {frame: average: int, counts:{role: count}}}
    for doc in tqdm(data):
        if doc['frame'] not in arg_info:
            arg_info[doc['frame']] = {'combined-total': 0, 
                                        'role-total': 0, 
                                        'source-total': 0,
                                        'counts': 
                                        {'report': {}, 'source': {}, 'combined': {}}}
            
        # get number of filled roles for report 
        # both have structure {role: #}, len(list) is number of arguments for that role
        filled_report_roles = {}  
        filled_source_roles = {}
        filled_roles_combined = {}

        for role in doc['report_dict']['role_annotations']:
            if role == 'role-spans-indices-in-all-spans':
                continue
            if doc['report_dict']['role_annotations'][role] != []:
                filled_report_roles[role] = len(doc['report_dict']['role_annotations'][role])
            else:
                filled_report_roles[role] = 0

        for role in doc['source_dict']['role_annotations']:
            if role == 'role-spans-indices-in-all-spans':
                continue
            if doc['source_dict']['role_annotations'][role] != []:
                filled_source_roles[role] = len(doc['source_dict']['role_annotations'][role])
            else:
                filled_source_roles[role] = 0

        # get number of filled roles for combined report and source
        for role in filled_report_roles:
            if filled_report_roles[role] > 0 or filled_source_roles[role] > 0:
                filled_roles_combined[role] = filled_report_roles[role] + filled_source_roles[role]
            else:
                filled_roles_combined[role] = 0

        # update counts
        for role in filled_report_roles:
            if role not in arg_info[doc['frame']]['counts']['report']:
                arg_info[doc['frame']]['counts']['report'][role] = 0
            arg_info[doc['frame']]['counts']['report'][role] += filled_report_roles[role]

        for role in filled_source_roles:
            if role not in arg_info[doc['frame']]['counts']['source']:
                arg_info[doc['frame']]['counts']['source'][role] = 0
            arg_info[doc['frame']]['counts']['source'][role] += filled_source_roles[role]

        for role in filled_roles_combined:
            if role not in arg_info[doc['frame']]['counts']['combined']:
                arg_info[doc['frame']]['counts']['combined'][role] = 0
            arg_info[doc['frame']]['counts']['combined'][role] += filled_roles_combined[role]

        # update average
        arg_info[doc['frame']]['combined-total'] += sum(filled_roles_combined.values())
        arg_info[doc['frame']]['role-total'] += sum(filled_report_roles.values())
        arg_info[doc['frame']]['source-total'] += sum(filled_source_roles.values())

    for frame in arg_info:
        arg_info[frame]['combined-average'] = arg_info[frame]['combined-total'] / num_docs
        arg_info[frame]['role-average'] = arg_info[frame]['role-total'] / num_docs
        arg_info[frame]['source-average'] = arg_info[frame]['source-total'] / num_docs
        for split in arg_info[frame]['counts']:
            for role in arg_info[frame]['counts'][split]:
                arg_info[frame]['counts'][split][role] /= num_docs
    return arg_info


def normalize_roles(role_info: dict):
    """
    Normalize averages by number of roles in the frame

    Returns a dict of same structure receieved
    """
    normalized_roles = {}
    for frame in role_info:
        normalized_roles[frame] = {'combined-average': 0, 
                                    'role-average': 0,
                                    'source-average': 0, 
                                    'counts': {}}
        for role in role_info[frame]['counts']:
            normalized_roles[frame]['counts'][role] = role_info[frame]['counts'][role]
        normalized_roles[frame]['combined-average'] = role_info[frame]['combined-average'] / len(role_info[frame]['counts']['combined'])
        normalized_roles[frame]['role-average'] = role_info[frame]['role-average'] / len(role_info[frame]['counts']['report'])
        normalized_roles[frame]['source-average'] = role_info[frame]['source-average'] / len(role_info[frame]['counts']['source'])
    return normalized_roles


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/cross_doc_role_extraction", help="directory of data") 
    parser.add_argument("--dataset", type=str, default="all", help="dataset to analyze") # all, train, dev, test
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
        all_doc_average_combined = 0
        all_doc_average_role = 0
        all_doc_average_source = 0
        for frame in role_info:
            all_doc_average_combined += role_info[frame]['combined-average']
            all_doc_average_role += role_info[frame]['role-average']
            all_doc_average_source += role_info[frame]['source-average']
        print("Average number of filled roles across all frames combined: {}".format(all_doc_average_combined / len(role_info)))
        print("Average number of filled roles across all frames for report: {}".format(all_doc_average_role / len(role_info)))
        print("Average number of filled roles across all frames for source: {}".format(all_doc_average_source / len(role_info)))

    sorted_roles = sorted(role_info.items(), key=lambda x: x[1]['combined-average'], reverse=True)

    # if args.verbose:
    #     print("Top 3 frames with the most filled roles:")
    #     for frame in sorted_roles[:3]:
    #         print(frame[0], frame[1]['average'])

    #     print("Top 3 frames with the least filled roles:")
    #     for frame in sorted_roles[-3:]:
    #         print(frame[0], frame[1]['average'])

    norm_roles = normalize_roles(role_info)

    sorted_norm_roles = sorted(norm_roles.items(), key=lambda x: x[1]['combined-average'], reverse=True)

    if args.verbose:
        print("Top 10 frames with the most normalized filled roles:")
        for frame in sorted_norm_roles[:10]:
            print(frame[0], frame[1]['combined-average'])

        print("Top 10 frames with the least normalized filled roles:")
        for frame in sorted_norm_roles[-10:]:
            print(frame[0], frame[1]['combined-average'])

    # write to a json file
    json_name = "sorted_roles_" + args.dataset + ".json"
    with open(os.path.join(args.output_dir, json_name), "w") as f:
        json.dump(sorted_roles, f, indent=4)

    json_name = "sorted_norm_roles_" + args.dataset + ".json"
    with open(os.path.join(args.output_dir, json_name), "w") as f:
        json.dump(sorted_norm_roles, f, indent=4)


    # get average number of arguments in report/source. There can be multiple args per role
    arg_info = argument_counts(data, num_docs=num_docs)

    if args.verbose:
        all_doc_total_combined = 0
        all_doc_total_role = 0
        all_doc_total_source = 0
        for frame in arg_info:
            all_doc_total_combined += arg_info[frame]['combined-average']
            all_doc_total_role += arg_info[frame]['role-average']
            all_doc_total_source += arg_info[frame]['source-average']
        print("Total number of filled arguments across all frames combined: {}".format(all_doc_total_combined / len(arg_info)))
        print("Total number of filled arguments across all frames for report: {}".format(all_doc_total_role / len(arg_info)))
        print("Total number of filled arguments across all frames for source: {}".format(all_doc_total_source / len(arg_info)))


    sorted_args = sorted(arg_info.items(), key=lambda x: x[1]['combined-total'], reverse=True)

    # write sorted norm roles to a json file
    json_name = "sorted_args" + "_" + args.dataset + ".json"
    with open(os.path.join(args.output_dir, json_name), "w") as f:
        json.dump(sorted_args, f, indent=4)

    



if __name__ == "__main__":
    main()