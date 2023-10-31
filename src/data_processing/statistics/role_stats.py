import os 
import sys 
import json 



# /home/amartin/famus/FAMuS/src/data_processing/statistics/sorted_norm_roles.json
norm_roles = json.load(open('/home/amartin/famus/FAMuS/src/data_processing/statistics/sorted_norm_roles.json', 'r'))


filled_role_total = 0
total_roles = 0
frames_missing_roles = {}
for frame in norm_roles: 
    roles = frame[1]['counts']
    for role in roles:
        if roles[role] > 0:
            filled_role_total += 1
        else:
            if frame[0] not in frames_missing_roles:
                frames_missing_roles[frame[0]] = []
            frames_missing_roles[frame[0]].append(role)


        total_roles += 1


print("Total number of filled roles: {} out of {}".format(filled_role_total, total_roles))

# sort missing roles by length of missing role list, greatest to least
frames_missing_roles = sorted(frames_missing_roles.items(), key=lambda x: len(x[1]), reverse=True)


# print(json.dumps(frames_missing_roles, indent=4))
with open('/home/amartin/famus/FAMuS/src/data_processing/statistics/frames_missing_roles.json', 'w') as f:
    json.dump(frames_missing_roles, f, indent=4)

