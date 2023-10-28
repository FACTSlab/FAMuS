# Dataset Statistics 
This folder contains the code to generate statistics for the dataset. 


## Basic Statistics
The basic statistics for the dataset are generated using the `basic_statistics.py` script. The script collects the statistics:
1. Average number of tokens in the report and source text.
2. The average number of filled roles across all frames. 
3. The average number of filled roles for each individual frame.

### Token Counts 
Average Report Tokens: `58.95529411764706` \
Min Report Tokens: `10` \
Max Report Tokens: `526` \
Average Source Token: `1163.9843137254902` \
Min Source Tokens: `154` \
Max Source Tokens: `5553` \

### Role Counts
For json outputs of the specific frame/role data see `sorted_norm_roles.json`
Average Roles Across Frames: `3.015686274509804`

Top 10 Frames with most roles (normalized by number of roles in a frame):
1. `Subsisting`-> `1.0`
2. `Transportation_status`-> `1.0`
3. `Chaos`-> `0.95`
4. `Vehicle_landing`-> `0.95`
5. `Visiting`-> `0.95`
6. `Coming_to_be`-> `0.9333333333333332`
7. `Cause_to_make_progress`-> `0.9`
8. `Detonate_explosive`-> `0.9`
9. `Dodging`-> `0.9`
10. `Event`-> `0.8666666666666667`

Top 10 Frames with least roles (normalized by number of roles in a frame):
1. `Killing`-> `0.39999999999999997`
2. `Change_position_on_a_scale`-> `0.38`
3. `Cause_to_be_sharp`-> `0.36`
4. `Motion`-> `0.35555555555555557`
5. `Cure`-> `0.35`
6. `Change_of_leadership`-> `0.3333333333333333`
7. `Proliferating_in_number`-> `0.275`
8. `Cause_motion`-> `0.2727272727272727`
9. `Apply_heat`-> `0.2285714285714286`
10. `Education_teaching`-> `0.21538461538461537`

## Distance Statistics
### Report 
![Alt text](assets/report_distances.png "Report Distances")
![Alt text](asset/sreport_distances_100bucket.png "Report Distances 100 Buckets")

Mean: `18.857704402515722` \
Median: `14.0` \
Max: `135` \
Min: `0` \
Standard deviation: `17.569476741474855` \
Variance: `308.68651296922593` \

### Source
![Alt text](assets/source_distances.png "Source Distances")
![Alt text](assets/source_distances_100bucket.png "Source Distances 100 Buckets")

Mean: `203.41981132075472` \
Median: `64.5` \
Max: `3311` \
Min: `0` \
Standard deviation: `386.984062919` \
Variance: `149756.66495342352` 