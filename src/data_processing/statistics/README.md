# Dataset Statistics 
This folder contains the code to generate statistics for the dataset. 


## Basic Statistics
The basic statistics for the dataset are generated using the `basic_statistics.py` script. The script collects the statistics:
1. Average number of tokens in the report and source text.
2. The average number of filled roles across all frames. 
3. The average number of filled roles for each individual frame.

### Whole Dataset: 
#### Token Counts 
Average Report Tokens: `58.95529411764706` \
Min Report Tokens: `10` \
Max Report Tokens: `526` \
Average Source Token: `1163.9843137254902` \
Min Source Tokens: `154` \
Max Source Tokens: `5553` \

#### Role Counts
For json outputs of the specific frame/role data see `sorted_norm_roles.json`
Average Roles Across Frames: 
- Combined: Intersection: `2.8149019607843155` Union: `3.6807843137254865`
- Report: `3.015686274509804`
- Source: `3.4799999999999973`

Average Arguments Across Frames: 
- Combined: Intersection: `29.474509803921567` Union: `34.10588235294118`
- Report: `15.48235294117647`
- Source: `18.623529411764707`


Top 10 Frames with most roles (combined):
1. `Subsisting`-> `1.0`
2. `Transportation_status`-> `1.0`
3. `Vehicle_landing`-> `0.95`
4. `Cause_to_make_progress`-> `0.9`
5. `Visiting`-> `0.9`
6. `Chaos`-> `0.85`
7. `Detonate_explosive`-> `0.85`
8. `Escaping`-> `0.85`
9. `Being_located`-> `0.8`
10. `Kidnapping`-> `0.8`

Top 10 Frames with least roles (normalized by number of roles in a frame):
1. `Render_nonfunctional`-> `0.36`
2. `Translating`-> `0.35`
3. `Motion`-> `0.3333333333333333`
4. `Cause_to_be_included`-> `0.31428571428571433`
5. `Cure`-> `0.3`
6. `Change_of_leadership`-> `0.2888888888888889`
7. `Proliferating_in_number`-> `0.275`
8. `Cause_motion`-> `0.23636363636363636`
9. `Education_teaching`-> `0.21538461538461537`
10. `Apply_heat`-> `0.17142857142857143`

### Train Dataset: 
Total Documents: 765 
#### Token Counts
Average Report Tokens: `59.209150326797385` \
Min Report Tokens: `10` \
Max Report Tokens: `526` \
Average Source Token: `1087.7660130718955` \
Min Source Tokens: `154` \
Max Source Tokens: `5553` \

#### Role Counts
For json outputs of the specific frame/role data see `sorted_norm_roles_train.json`\
Average Roles Across Frames: 
- Combined: Intersection: `2.7607843137254915` Union: `3.613071895424835`
- Report: `2.943790849673201`
- Source: `3.4300653594771213`

Average Arguments Across Frames: 
- Combined: Intersection: `17.372549019607842` Union: `20.12549019607843`
- Report: `9.101960784313725`
- Source: `11.023529411764706`


Top 10 Frames with most roles (normalized by number of roles in a frame):
1. `Subsisting`-> `1.0`
2. `Transportation_status`-> `1.0`
3. `Visiting`-> `1.0`
4. `Detonate_explosive`-> `0.9166666666666666`
5. `Vehicle_landing`-> `0.9166666666666666`
6. `Activity_stop`-> `0.8888888888888888`
7. `Abusing`-> `0.8333333333333334`
8. `Besieging`-> `0.8333333333333334`
9. `Cause_to_make_progress`-> `0.8333333333333334`
10. `Chaos`-> `0.8333333333333334`

Top 10 Frames with least roles (normalized by number of roles in a frame):
1. `Proliferating_in_number`-> `0.3333333333333333`
2. `Rotting`-> `0.3333333333333333`
3. `Theft`-> `0.3333333333333333`
4. `Motion`-> `0.2962962962962963`
5. `Adjusting`-> `0.26666666666666666`
6. `Change_of_leadership`-> `0.2592592592592593`
7. `Cure`-> `0.25`
8. `Cause_motion`-> `0.2424242424242424`
9. `Education_teaching`-> `0.20512820512820512`
10. `Apply_heat`-> `0.14285714285714285`

### Dev Dataset:
Total Documents: 255

#### Token Counts
Average Report Tokens: `60.062745098039215` \
Min Report Tokens: `11` \
Max Report Tokens: `208` \
Average Source Token: `1511.1490196078432` \
Min Source Tokens: `177` \
Max Source Tokens: `5389` \

#### Role Counts
For json outputs of the specific frame/role data see `sorted_norm_roles_dev.json`\
Average Roles Across Frames: 
- Combined: Intersection: `3.215686274509804` Union: `3.996078431372549`
- Report: `3.392156862745098`
- Source: `3.819607843137255`

Average Arguments Across Frames: 
- Combined: Intersection: `6.788235294117647` Union: `7.63921568627451`
- Report: `3.466666666666667`
- Source: `4.172549019607843`

Top 10 Frames with most roles (normalized by number of roles in a frame):
1. `Activity_start`-> `1.0`
2. `Attack`-> `1.0`
3. `Being_located`-> `1.0`
4. `Cause_change_of_strength`-> `1.0`
5. `Cause_to_make_progress`-> `1.0`
6. `Change_of_consistency`-> `1.0`
7. `Chaos`-> `1.0`
8. `Come_down_with`-> `1.0`
9. `Coming_to_be`-> `1.0`
10. `Commerce_buy`-> `1.0`

Top 10 Frames with least roles (normalized by number of roles in a frame):
1. `Text_creation`-> `0.25`
2. `Translating`-> `0.25`
3. `Cause_to_be_dry`-> `0.2`
4. `Forging`-> `0.2`
5. `Inhibit_movement`-> `0.16666666666666666`
6. `Examination`-> `0.14285714285714285`
7. `Proliferating_in_number`-> `0.125`
8. `Cause_motion`-> `0.09090909090909091`
9. `Exercising`-> `0.0`
10. `Killing`-> `0.0`

### Test Dataset:
Total Documents: 255

#### Token Counts
Average Report Tokens: `57.08627450980392` \
Min Report Tokens: `17` \
Max Report Tokens: `181` \
Average Source Token: `1045.4745098039216` \
Min Source Tokens: `173` \
Max Source Tokens: `4856` \

#### Role Counts
For json outputs of the specific frame/role data see `sorted_norm_roles_test.json`
Average Roles Across Frames: 
- Combined: Intersection: `2.5764705882352943` Union: `3.5686274509803924`
- Report: `2.854901960784314`
- Source: `3.2901960784313724`

Average Arguments Across Frames: 
- Combined: Intersection: `5.313725490196078` Union: `6.341176470588235`
- Report: `2.9137254901960783`
- Source: `3.4274509803921567`

Top 10 Frames with most roles (normalized by number of roles in a frame):
1. `Cause_to_make_progress`-> `1.0`
2. `Escaping`-> `1.0`
3. `Operating_a_system`-> `1.0`
4. `Presence`-> `1.0`
5. `State_of_entity`-> `1.0`
6. `Subsisting`-> `1.0`
7. `Transportation_status`-> `1.0`
8. `Undergoing`-> `1.0`
9. `Vehicle_landing`-> `1.0`
10. `Visiting`-> `1.0`

Top 10 Frames with least roles (normalized by number of roles in a frame):
1. `Forging`-> `0.2`
2. `Grinding`-> `0.2`
3. `Manipulation`-> `0.2`
4. `Render_nonfunctional`-> `0.2`
5. `Submitting_documents`-> `0.2`
6. `Arrest`-> `0.16666666666666666`
7. `Theft`-> `0.16666666666666666`
8. `Education_teaching`-> `0.15384615384615385`
9. `Apply_heat`-> `0.14285714285714285`
10. `Cause_to_be_included`-> `0.14285714285714285`

## Distance Statistics
### All Data
#### Report 
![Alt text](assets/report_distances_all.png "Report Distances 100 Buckets")

Mean: `18.857704402515722` \
Median: `14.0` \
Max: `135` \
Min: `0` \
Standard deviation: `17.569476741474855` \
Variance: `308.68651296922593` 

#### Source
![Alt text](assets/source_distances_all.png "Source Distances 100 Buckets")

Mean: `203.41981132075472` \
Median: `64.5` \
Max: `3311` \
Min: `0` \
Standard deviation: `386.984062919` \
Variance: `149756.66495342352` 

### Train Data
#### Report
![Alt text](assets/report_distances_train.png "Source Distances 100 Buckets")

Mean: `17.926701570680628` \
Median: `13.0` \
Max: `135` \
Min: `0` \
Standard deviation: `17.69993844784599` \
Variance: `313.2878210575368`

#### Source
![Alt text](assets/source_distances_train.png "Source Distances 100 Buckets")

Mean: `190.5091623036649` \
Median: `64.0` \
Max: `3311` \
Min: `0` \
Standard deviation: `361.44022362814894` \
Variance: `130639.03525636632`

### Dev Data
#### Report
![Alt text](assets/report_distances_dev.png "Source Distances 100 Buckets")

Mean: `22.42292490118577` \
Median: `18.0` \
Max: `90` \
Min: `0` \
Standard deviation: `18.635228128824515` \
Variance: `347.2717274133325`

#### Source
![Alt text](assets/source_distances_dev.png "Source Distances 100 Buckets")

Mean: `291.0395256916996` \
Median: `108.0` \
Max: `3043` \
Min: `0` \
Standard deviation: `508.7967655511856` \
Variance: `258874.14863534816`
### Test Data
#### Report
![Alt text](assets/report_distances_test.png "Source Distances 100 Buckets")

Mean: `17.32941176470588` \
Median: `12.0` \
Max: `86` \
Min: `0` \
Standard deviation: `16.268377772238196` \
Variance: `264.6601153402538`

#### Source
![Alt text](assets/source_distances_test.png "Source Distances 100 Buckets")

Mean: `154.33333333333334` \
Median: `40.0` \
Max: `2254` \
Min: `0` \
Standard deviation: `297.6219912459471` \
Variance: `88578.84967320261`