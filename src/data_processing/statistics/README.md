# Dataset Statistics 
This folder contains the code to generate statistics for the dataset. 


## Basic Statistics
The basic statistics for the dataset are generated using the `basic_statistics.py` script. The script collects the statistics:
1. Average number of tokens in the report and source text.
2. The average number of filled roles across all frames. 
3. The average number of filled roles for each individual frame.

### Whole Dataset: 
#### Token Counts 
Average Report Tokens: `58.91699604743083` \
Min Report Tokens: `10` \
Max Report Tokens: `526` \
Average Source Token: `1164.8679841897233` \
Min Source Tokens: `154` \
Max Source Tokens: `5553` \

#### Role Counts
For json outputs of the specific frame/role data see `sorted_norm_roles.json`
Average Roles Across Frames: 
- Combined: `3.7090909090909063`
- Report: `3.035573122529645`
- Source: `3.508300395256914`

Average Arguments Across Frames: 
- Combined: `6.890118577075101`
- Report: `3.124901185770753`
- Source: `3.7652173913043474`


Top 10 Frames with most roles (combined):
1. `Attack`-> `1.0`
2. `Subsisting`-> `1.0`
3. `Transportation_status`-> `1.0`
4. `Vehicle_landing`-> `1.0`
5. `Breaking_out_captive`-> `0.96`
6. `Enforcing`-> `0.96`
7. `Piracy`-> `0.96`
8. `Activity_pause`-> `0.95`
9. `Attempt`-> `0.95`
10. `Chaos`-> `0.95`

Top 10 Frames with least roles (normalized by number of roles in a frame):
1. `Render_nonfunctional`-> `0.48`
2. `Cure`-> `0.475`
3. `Change_of_leadership`-> `0.4666666666666667`
4. `Change_position_on_a_scale`-> `0.45999999999999996`
5. `Motion`-> `0.4444444444444444`
6. `Cause_motion`-> `0.4`
7. `Cause_to_be_sharp`-> `0.4`
8. `Education_teaching`-> `0.38461538461538464`
9. `Proliferating_in_number`-> `0.375`
10. `Apply_heat`-> `0.2857142857142857`

### Train Dataset: 
Total Documents: 765 
#### Token Counts
Average Report Tokens: `59.201581027667984` \
Min Report Tokens: `10` \
Max Report Tokens: `526` \
Average Source Token: `1084.258234519104` \
Min Source Tokens: `154` \
Max Source Tokens: `5553` \

#### Role Counts
For json outputs of the specific frame/role data see `sorted_norm_roles_train.json`\
Average Roles Across Frames: 
- Combined: `3.6455862977602096`
- Report: `2.971014492753622`
- Source: `3.451910408432145`

Average Arguments Across Frames: 
- Combined: `6.776021080368908`
- Report: `3.071146245059288`
- Source: `3.704874835309617`


Top 10 Frames with most roles (normalized by number of roles in a frame):
1. `Abusing`-> `1.0`
2. `Activity_pause`-> `1.0`
3. `Activity_stop`-> `1.0`
4. `Attack`-> `1.0`
5. `Being_born`-> `1.0`
6. `Besieging`-> `1.0`
7. `Coming_to_be`-> `1.0`
8. `Quitting_a_place`-> `1.0`
9. `Receiving`-> `1.0`
10. `Subsisting`-> `1.0`

Top 10 Frames with least roles (normalized by number of roles in a frame):
1. `Cause_motion`-> `0.42424242424242425`
2. `Proliferating_in_number`-> `0.4166666666666667`
3. `Education_teaching`-> `0.41025641025641024`
4. `Motion`-> `0.4074074074074074`
5. `Cause_to_be_sharp`-> `0.4`
6. `Change_position_on_a_scale`-> `0.4`
7. `Grinding`-> `0.4`
8. `Theft`-> `0.3888888888888889`
9. `Apply_heat`-> `0.33333333333333337`
10. `Damaging`-> `0.33333333333333337`

### Dev Dataset:
Total Documents: 255

#### Token Counts
Average Report Tokens: `59.58498023715415` \
Min Report Tokens: `11` \
Max Report Tokens: `208` \
Average Source Token: `1510.7351778656127` \
Min Source Tokens: `177` \
Max Source Tokens: `5389` \

#### Role Counts
For json outputs of the specific frame/role data see `sorted_norm_roles_dev.json`\
Average Roles Across Frames: 
- Combined: `4.063241106719367`
- Report: `3.4545454545454546`
- Source: `3.8932806324110674`

Average Arguments Across Frames: 
- Combined: `7.822134387351778`
- Report: `3.5454545454545454`
- Source: `4.276679841897233`

Top 10 Frames with most roles (normalized by number of roles in a frame):
1. `Achieving_first`-> `1.0`
2. `Activity_pause`-> `1.0`
3. `Activity_ready_state`-> `1.0`
4. `Activity_resume`-> `1.0`
5. `Activity_start`-> `1.0`
6. `Activity_stop`-> `1.0`
7. `Arraignment`-> `1.0`
8. `Arrest`-> `1.0`
9. `Attack`-> `1.0`
10. `Attempt`-> `1.0`

Top 10 Frames with least roles (normalized by number of roles in a frame):
1. `Exercising`-> `0.3333333333333333`
2. `Apply_heat`-> `0.2857142857142857`
3. `Examination`-> `0.2857142857142857`
4. `Cause_motion`-> `0.2727272727272727`
5. `Abusing`-> `0.25`
6. `Besieging`-> `0.25`
7. `Hit_target`-> `0.25`
8. `Cause_to_be_dry`-> `0.2`
9. `Inhibit_movement`-> `0.16666666666666666`
10. `Proliferating_in_number`-> `0.125`

### Test Dataset:
Total Documents: 255

#### Token Counts
Average Report Tokens: `57.39525691699605` \
Min Report Tokens: `17` \
Max Report Tokens: `181` \
Average Source Token: `1060.8300395256917` \
Min Source Tokens: `173` \
Max Source Tokens: `4856` \

#### Role Counts
For json outputs of the specific frame/role data see `sorted_norm_roles_test.json`
Average Roles Across Frames: 
- Combined: `3.5454545454545454`
- Report: `2.8102766798418974`
- Source: `3.292490118577075`

Average Arguments Across Frames: 
- Combined: `6.300395256916996`
- Report: `2.8656126482213438`
- Source: `3.4347826086956523`

Top 10 Frames with most roles (normalized by number of roles in a frame):
1. `Abusing`-> `1.0`
2. `Activity_start`-> `1.0`
3. `Attack`-> `1.0`
4. `Attempt`-> `1.0`
5. `Attending`-> `1.0`
6. `Being_born`-> `1.0`
7. `Besieging`-> `1.0`
8. `Breaking_out_captive`-> `1.0`
9. `Causation`-> `1.0`
10. `Cause_change_of_strength`-> `1.0`

Top 10 Frames with least roles (normalized by number of roles in a frame):
1. `Get_a_job`-> `0.3333333333333333`
2. `Placing`-> `0.3333333333333333`
3. `Process_completed_state`-> `0.3333333333333333`
4. `Rotting`-> `0.3333333333333333`
5. `Emotions_success_or_failure`-> `0.2857142857142857`
6. `Fleeing`-> `0.2857142857142857`
7. `Education_teaching`-> `0.23076923076923078`
8. `Motion`-> `0.2222222222222222`
9. `Manipulation`-> `0.2`
10. `Apply_heat`-> `0.14285714285714285`

## Distance Statistics
### All Data
#### Report 
![Alt text](assets/report_distances_all.png "Report Distances 100 Buckets")

Mean: `18.68066561014263` \
Median: `14.0` \
Max: `135` \
Min: `0` \
Standard deviation: `17.650379654107788` \
Variance: `311.5359019341422` 

#### Source
![Alt text](assets/source_distances_all.png "Source Distances 100 Buckets")

Mean: `207.85182250396196` \
Median: `68.0` \
Max: `3311` \
Min: `0` \
Standard deviation: `390.41874314720144` \
Variance: `152426.79500064044` 

### Train Data
#### Report
![Alt text](assets/report_distances_train.png "Source Distances 100 Buckets")

Mean: `17.886543535620053` \
Median: `13.0` \
Max: `135` \
Min: `0` \
Standard deviation: `17.618660754619327` \
Variance: `310.41720678636324`

#### Source
![Alt text](assets/source_distances_train.png "Source Distances 100 Buckets")

Mean: `190.72295514511873` \
Median: `64.0` \
Max: `3311` \
Min: `0` \
Standard deviation: `353.50339669985965` \
Variance: `124964.65147833836`

### Dev Data
#### Report
![Alt text](assets/report_distances_dev.png "Source Distances 100 Buckets")

Mean: `23.07171314741036` \
Median: `19.0` \
Max: `90` \
Min: `0` \
Standard deviation: `18.700055304192738` \
Variance: `349.69206837986695`

#### Source
![Alt text](assets/source_distances_dev.png "Source Distances 100 Buckets")

Mean: `307.44621513944224` \
Median: `118.0` \
Max: `3043` \
Min: `0` \
Standard deviation: `529.0742347837113` \
Variance: `279919.5459119697`
### Test Data
#### Report
![Alt text](assets/report_distances_test.png "Source Distances 100 Buckets")

Mean: `16.703557312252965` \
Median: `12.0` \
Max: `86` \
Min: `0` \
Standard deviation: `15.877839828585289` \
Variance: `252.10579762220934`

#### Source
![Alt text](assets/source_distances_test.png "Source Distances 100 Buckets")

Mean: `160.36363636363637` \
Median: `44.0` \
Max: `2254` \
Min: `0` \
Standard deviation: `308.61785818889507` \
Variance: `95244.98239310096`