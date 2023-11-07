import os 
import json
import argparse
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def distance(span1: list, span2: list):
    """
    Computes the distance (number of tokens) between two roles. 
    Distance is the middle of the span to the middle of the span

    Input: 
        span1: list [start, end]
        span2: list [start, end]
    """
    return abs(((span1[0] + span1[1]) / 2) - ((span2[0] + span2[1]) / 2))

def find_first_last_role(roles: list):
    """
    Finds the first and last role in a list of roles

    Input:
        roles: list of roles, where each role is a list [start, end]

    Returns:
        first role: list [start, end]
        last role: list [start, end]
    """
    first_role = roles[0]
    last_role = roles[0]
    for role in roles:
        if role[0] < first_role[0]:
            first_role = role
        if role[0] > last_role[0]:
            last_role = role
    return first_role, last_role

def check_repeated_role_filler(data: list):
    """
    Checks the data to see how often role fillers are repeated in report/source
    """
    report_count = 0
    source_count = 0
    for doc in tqdm(data):
        report_info = doc["report_dict"]["role_annotations"]
        source_info = doc["source_dict"]["role_annotations"]
        report_roles = []
        for frame in report_info:
            if frame == 'role-spans-indices-in-all-spans':
                continue
            if report_info[frame] != []:
                for role in report_info[frame]:
                    report_roles.append(role[3:5])

        source_roles = []
        for frame in source_info:
            if frame == 'role-spans-indices-in-all-spans':
                continue
            if source_info[frame] != []:
                for role in source_info[frame]:
                    source_roles.append(role[3:5])

        if len(report_roles) != 0 and len(source_roles) != 0:
            first_report_role, last_report_role = find_first_last_role(report_roles)
            if distance(first_report_role, last_report_role) == 0:
                if len(report_roles) > 1:
                    print('repeated role in report')
                    print(doc['instance_id'])
                    report_count += 1

            first_source_role, last_source_role = find_first_last_role(source_roles)
            if distance(first_source_role, last_source_role) == 0:
                if len(source_roles) > 1:
                    print('repeated role in source')
                    print(doc['instance_id'])
                    source_count += 1

    return report_count, source_count
            
def max_distances(data: list, verbose=False):
    """
    Computes the distance between the first role and last role in report/source
    """
    report_distances = []
    source_distances = []
    for doc in tqdm(data):
        report_info = doc["report_dict"]["role_annotations"]
        source_info = doc["source_dict"]["role_annotations"]
        report_roles = []
        for frame in report_info:
            if frame == 'role-spans-indices-in-all-spans':
                continue
            if report_info[frame] != []:
                for role in report_info[frame]:
                    report_roles.append(role[3:5])

        source_roles = []
        for frame in source_info:
            if frame == 'role-spans-indices-in-all-spans':
                continue
            if source_info[frame] != []:
                for role in source_info[frame]:
                    source_roles.append(role[3:5])

        if len(report_roles) == 0 or len(source_roles) == 0:
            if verbose: 
                if len(report_roles) == 0: print('no report roles')
                if len(source_roles) == 0: print('no source roles')
                print(doc['instance_id'])
        else:
            # compute distances between first and last role in report
            if len(report_roles) == 1:
                report_distances.append(0)
            else:
                first_report_role, last_report_role = find_first_last_role(report_roles)
                report_distances.append(distance(first_report_role, last_report_role))

            # compute distances between first and last role in source
            if len(source_roles) == 1:
                source_distances.append(0)
            else:
                first_source_role, last_source_role = find_first_last_role(source_roles)
                source_distances.append(distance(first_source_role, last_source_role))
        
    return report_distances, source_distances

def trigger_distance(data: list, verbose=False):
    """
    Computes the distance between the arguments and the trigger
    (only for the report document)

    Returns: 
        distances (list): list of distances between the arguments and the trigger
        avg_distance (float): average distance between the arguments and the trigger
    """
    report_distances = []
    for doc in tqdm(data):
        report_info = doc["report_dict"]["role_annotations"]
        trigger_span = doc["report_dict"]["frame-trigger-span"][3:5]
        report_roles = []
        role_distances = []
        for frame in report_info:
            if frame == 'role-spans-indices-in-all-spans':
                continue
            if report_info[frame] != []:
                for role in report_info[frame]:
                    report_roles.append(role[3:5])

        if len(report_roles) == 0:
            if verbose: 
                print('no report roles')
                print(doc['instance_id'])
        else:
            # compute distances between first and last role in report
            for role in report_roles:
                role_distances.append(distance(role, trigger_span))
            report_distances.append(np.mean(role_distances))

    if len(report_distances) != 0:
        avg_distance = np.mean(report_distances)
        return report_distances, avg_distance
    else:
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/cross_doc_role_extraction", help="directory of data") 
    parser.add_argument("--dataset", type=str, default="all", help="dataset to analyze") # all, train, dev, test
    parser.add_argument("--output_dir", type=str, default="src/data_processing/statistics/assets", help="directory to save output")
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

    # compute statistics
    if args.verbose: print(len(data))
    report_distances, source_distances = max_distances(data)

    report_count, source_count = check_repeated_role_filler(data)
    if args.verbose: print(report_count, source_count)

    trigger_distances, avg_trig_distance = trigger_distance(data)
    if args.verbose: print(avg_trig_distance)

    # plt histogram
    report_fig_name = 'report_distances_' + args.dataset + '.png'
    source_fig_name = 'source_distances_' + args.dataset + '.png'
    plt.hist(report_distances, bins=100)
    plt.title("Distance Between First and Last Role in Report")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(args.output_dir, report_fig_name))
    plt.clf()

    plt.hist(source_distances, bins=100)
    plt.title("Distance Between First and Last Role in Source")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(args.output_dir, source_fig_name))
    plt.clf()

    trigger_distance_fig_name = 'trigger_distances_' + args.dataset + '.png'
    plt.hist(trigger_distances, bins=100)
    plt.title("Average Distances Between Arguments and Trigger in Report")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(args.output_dir, trigger_distance_fig_name))
    plt.clf()


    # print statistics
    if args.verbose:
        print("Report distances:")
        print("Mean:", np.mean(report_distances))
        print("Median:", np.median(report_distances))
        print("Max:", np.max(report_distances))
        print("Min:", np.min(report_distances))
        print("Standard deviation:", np.std(report_distances))
        print("Variance:", np.var(report_distances))
        print("Source distances:")
        print("Mean:", np.mean(source_distances))
        print("Median:", np.median(source_distances))
        print("Max:", np.max(source_distances))
        print("Min:", np.min(source_distances))
        print("Standard deviation:", np.std(source_distances))
        print("Variance:", np.var(source_distances))

    # print stats in MD formatting 
    # Mean : `number` \ 
    if args.verbose:
        print("Report distances:")
        print("Mean: `", np.mean(report_distances), '`\\')
        print("Median: `", np.median(report_distances), '`\\')
        print("Max: `", np.max(report_distances), '`\\')
        print("Min: `", np.min(report_distances), '`\\')
        print("Standard deviation: `", np.std(report_distances), '`\\')
        print("Variance: `", np.var(report_distances), '`\\')
        print("Source distances:")
        print("Mean: `", np.mean(source_distances), '`\\')
        print("Median: `", np.median(source_distances), '`\\')
        print("Max: `", np.max(source_distances), '`\\')
        print("Min: `", np.min(source_distances), '`\\')
        print("Standard deviation: `", np.std(source_distances), '`\\')
        print("Variance: `", np.var(source_distances), '`\\')




    


if __name__ == "__main__":
    main()


