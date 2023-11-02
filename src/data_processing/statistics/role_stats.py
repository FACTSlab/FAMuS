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

    train_data = []
    dev_data = []
    test_data = []
    with open(os.path.join(args.data_dir, "train.jsonl"), "r") as f:
        train_data = [json.loads(line) for line in f]
    with open(os.path.join(args.data_dir, "dev.jsonl"), "r") as f:
        dev_data = [json.loads(line) for line in f]
    with open(os.path.join(args.data_dir, "test.jsonl"), "r") as f:
        test_data = [json.loads(line) for line in f]

    # count number of roles
    # filled_roles, total_roles = count_roles(data)
    # if args.verbose: print("Total number of filled roles: {} out of {}".format(filled_roles, total_roles))

    # count number of roles in train, dev, test
    train_filled_roles_report, train_filled_roles_source, train_total_roles = count_roles(train_data)
    dev_filled_roles, dev_filled_roles_source, dev_total_roles = count_roles(dev_data)
    test_filled_roles, test_filled_roles_source, test_total_roles = count_roles(test_data)

    # what is in test that isn't in the other sets
    test_not_in_train = test_total_roles - train_total_roles
    test_not_in_dev = test_total_roles - dev_total_roles
    test_not_in_train_or_dev = test_not_in_train - dev_total_roles

    train_not_in_test = train_total_roles - test_total_roles
    train_not_in_dev = train_total_roles - dev_total_roles

    dev_not_in_test = dev_total_roles - test_total_roles
    dev_not_in_train = dev_total_roles - train_total_roles

    # print set differenes 
    if args.verbose:
        print("Test not in train: {}".format(test_not_in_train))
        print("Test not in dev: {}".format(test_not_in_dev))
        print("Test not in train or dev: {}".format(test_not_in_train_or_dev))
        print("Train not in test: {}".format(train_not_in_test))
        print("Train not in dev: {}".format(train_not_in_dev))
        print("Dev not in test: {}".format(dev_not_in_test))
        print("Dev not in train: {}".format(dev_not_in_train))



if __name__ == '__main__':
    main()


# # /home/amartin/famus/FAMuS/src/data_processing/statistics/sorted_norm_roles.json
# norm_roles = json.load(open('/home/amartin/famus/FAMuS/src/data_processing/statistics/json_files/sorted_norm_roles_all.json', 'r'))


# filled_role_total = 0
# total_roles = 0
# frames_missing_roles = {}
# for frame in norm_roles: 
#     roles = frame[1]['counts']
#     for role in roles:
#         if role == 'Time' or role == 'Place':
#             continue
#         if roles[role] > 0:
#             filled_role_total += 1
#         else:
#             if frame[0] not in frames_missing_roles:
#                 frames_missing_roles[frame[0]] = []
#             frames_missing_roles[frame[0]].append(role)


#         total_roles += 1


# print("Total number of filled roles: {} out of {}".format(filled_role_total, total_roles))

# # sort missing roles by length of missing role list, greatest to least
# frames_missing_roles = sorted(frames_missing_roles.items(), key=lambda x: len(x[1]), reverse=True)


# # print(json.dumps(frames_missing_roles, indent=4))
# with open('/home/amartin/famus/FAMuS/src/data_processing/statistics/frames_missing_roles.json', 'w') as f:
#     json.dump(frames_missing_roles, f, indent=4)

