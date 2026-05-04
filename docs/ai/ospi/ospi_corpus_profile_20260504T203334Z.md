# OSPI Corpus Profile

Generated: 2026-05-04T20:37:22Z
Corpus path: `research/ospi_data`

## Safety Scope

This was a read-only profiling pass. No ingestion occurred. No database writes occurred. No corpus files were modified. No OSPI scripts or `download_ospi.py` were run.

## Inventory

| file | shape | records | bytes | sha256 | mtime_utc | parse_error |
| --- | --- | --- | --- | --- | --- | --- |
| assessment.json | array | 155528 | 146852924 | eff989bffa224de7d220716a81f928efb5ee37cfed06a89ec641f9a95a052ae3 | 2026-03-04T06:04:56+00:00 |  |
| attendance.json | array | 75204 | 52051015 | 1a01d33b42a7dba748064f5536701b9153b283f4ad9d30f31cf913fe3ad76aa0 | 2026-03-04T06:04:56+00:00 |  |
| discipline.json | array | 74033 | 79338788 | 79b2530af55fb89b32a4a8fdefca2f35247d9d604b80546881bb3613478e5194 | 2026-03-04T06:04:56+00:00 |  |
| enrollment.json | array | 4562 | 6530509 | 91700ea5e014686258eec47ce09321cbae0f0d091150b2e5dc16f166e1bfa881 | 2026-03-04T06:04:56+00:00 |  |
| graduation.json | array | 9670 | 9010709 | 7df76f589fa707216f5ff3375448741c6c47998ff236d8b343b24a36ac980f13 | 2026-03-04T06:04:56+00:00 |  |
| growth.json | array | 50665 | 42919067 | 17207359cc4001ccc4bd43008ab29a17b5232e0248eb125b30fe7812acc7c3d8 | 2026-03-04T06:04:56+00:00 |  |
| sqss.json | array | 109011 | 77841214 | a991c6c0767b6b2d450be5f3e66d25f4f723121497a08c956cd97b2fdbab8a24 | 2026-03-04T06:04:56+00:00 |  |
| teacher_demographics.json | array | 4745 | 4404885 | 8308483c658af073ee721bdc5596f5613b29feaf54d8a017c7699d38f3e97df2 | 2026-03-04T06:04:56+00:00 |  |
| teacher_experience.json | array | 4602 | 3314081 | 79a0037878611517f18f44bc2ee2ef56e7f3dba1f1bff7f6e4e9da0cd9190dcf | 2026-03-04T06:04:56+00:00 |  |
| wakids.json | array | 210202 | 163941382 | d727602ff1f85276fe4d1e9ebb942f545e255f08d48c71c4d4bfb35c3a5bd93e | 2026-03-04T06:04:56+00:00 |  |

Total bytes: 586,204,574
Reference bytes: 586,204,574
Byte delta: 0
Total records: 698,222
Reference records: 698,222
Record delta: 0

Expected files present: assessment.json, attendance.json, discipline.json, enrollment.json, graduation.json, growth.json, sqss.json, teacher_demographics.json, teacher_experience.json, wakids.json
Expected files missing: none
Unexpected JSON files: none

## Record Counts

| dataset | records |
| --- | --- |
| assessment.json | 155,528 |
| attendance.json | 75,204 |
| discipline.json | 74,033 |
| enrollment.json | 4,562 |
| graduation.json | 9,670 |
| growth.json | 50,665 |
| sqss.json | 109,011 |
| teacher_demographics.json | 4,745 |
| teacher_experience.json | 4,602 |
| wakids.json | 210,202 |

## Schema Summaries

| dataset | top_fields | nested_paths | sample fields | common patterns |
| --- | --- | --- | --- | --- |
| assessment.json | 46 | 0 | county, dataasof, districtcode, districtname, districtorganizationid, esdname, esdorganizationid, gradelevel, organizationlevel, schoolname, schoolyear, studentgroup, studentgrouptype, testadministration | schoolyear, organizationlevel, districtcode, districtname, schoolcode, schoolname |
| attendance.json | 27 | 0 | county, dataasof, districtcode, districtname, districtorganizationid, esdname, gradelevel, organizationlevel, schoolname, schoolyear, studentgroup, studentgrouptype, currentschooltype, schoolcode | schoolyear, organizationlevel, districtcode, districtname, schoolcode, schoolname |
| discipline.json | 29 | 0 | county, dateextracted, disciplinedatnotes, disciplinedenominator, disciplinenumerator, disciplinerate, districtcode, districtname, districtorganizationid, esdname, esdorganizationid, gradelevel, organizationlevel, rateexcluded10daysormore | schoolyear, organizationlevel, districtcode, districtname, schoolcode, schoolname |
| enrollment.json | 49 | 0 | all_students, american_indian_alaskan_native, asian, black_african_american, county, dataasof, districtcode, districtname, districtorganizationid, english_language_learners, esdname, esdorganizationid, female, gender_x | schoolyear, organizationlevel, districtcode, districtname, schoolcode, schoolname |
| graduation.json | 35 | 0 | cohort, county, dataasof, districtcode, districtname, districtorganizationid, esdname, esdorganizationid, organizationlevel, schoolname, schoolyear, studentgroup, studentgrouptype, schoolcode | schoolyear, organizationlevel, districtcode, districtname, schoolcode, schoolname |
| growth.json | 27 | 0 | county, dataasof, districtcode, districtname, districtorganizationid, esdname, esdorganizationid, gradelevel, organizationlevel, schoolname, schoolyear, studentgroup, studentgrouptype, subject | schoolyear, organizationlevel, districtcode, districtname, schoolcode, schoolname |
| sqss.json | 50 | 0 | county, dataasof, districtcode, districtname, districtorganizationid, esdname, gradelevel, organizationlevel, schoolname, schoolyear, studentgroup, studentgrouptype, currentschooltype, schoolcode | schoolyear, organizationlevel, districtcode, districtname, schoolcode, schoolname |
| teacher_demographics.json | 27 | 0 | avgyearsexperience, county, dataasof, demographiccategory, demographiccategoryid, demographiccategorytype, esdname, esdorganizationid, iseevalidated, leacode, leaname, leaorganizationid, ma_count, ma_percent | schoolyear, organizationlevel, schoolcode, schoolname |
| teacher_experience.json | 21 | 0 | county, dataasof, esdname, esdorganizationid, experiencebin, iseevalidated, leacode, leaname, leaorganizationid, organizationid, organizationlevel, organizationlevelid, organizationname, rowid | schoolyear, organizationlevel, schoolcode, schoolname |
| wakids.json | 23 | 0 | county, dataasof, districtname, districtorganizationid, domain, esdname, esdorganizationid, measure, organizationid, organizationlevel, schoolname, schoolyear, studentgroup, studentgrouptype | schoolyear, organizationlevel, districtname, schoolname |

### assessment.json

| field | coverage | types | distinct_cap | overflow | examples |
| --- | --- | --- | --- | --- | --- |
| county | 100.0% | str:155528 | 1 | no | 'King' |
| dataasof | 100.0% | str:155528 | 5 | no | '2022-09-09T00:00:00.000'; '9/8/2023'; '2023-10-27 15:04:41.483' |
| districtcode | 100.0% | str:155528 | 1 | no | '17415' |
| districtname | 100.0% | str:155528 | 1 | no | 'Kent School District' |
| districtorganizationid | 100.0% | str:155528 | 1 | no | '100117' |
| esdname | 100.0% | str:155528 | 1 | no | 'Puget Sound Educational Service District 121' |
| esdorganizationid | 100.0% | str:155528 | 1 | no | '100006' |
| gradelevel | 100.0% | str:155528 | 14 | no | 'All Grades'; 'KG'; '01' |
| organizationlevel | 100.0% | str:155528 | 2 | no | 'School'; 'District' |
| schoolname | 100.0% | str:155528 | 48 | no | 'Grass Lake Elementary School'; 'Horizon Elementary School'; 'Individualized Graduation & Degree Program' |
| schoolyear | 100.0% | str:155528 | 10 | no | '2020-21'; '2018-19'; '2017-18' |
| studentgroup | 100.0% | str:155528 | 30 | no | 'Non-Low Income'; 'Female'; 'Male' |
| studentgrouptype | 100.0% | str:155528 | 11 | no | 'FRL'; 'Gender'; 'homeless' |
| testadministration | 100.0% | str:155528 | 8 | no | 'SBAC'; 'ELPA'; 'WCAS' |
| testsubject | 100.0% | str:155528 | 7 | no | 'Math'; 'ELPA'; 'ELA' |
| schoolorganizationid | 93.54% | str:145487 | 48 | no | '101591'; '101605'; '105491' |
| currentschooltype | 92.94% | str:144551 | 4 | no | 'P'; 'R'; 'A' |
| schoolcode | 92.94% | str:144551 | 47 | no | '3708'; '4345'; '5275' |
| percentmetstandard | 73.69% | str:114615 | 935 | no | '67.7%'; '<10%'; '85.5%' |
| suppression | 73.69% | str:114615 | 23 | no | 'None'; '<10%'; 'N<10' |
| percentlevel1 | 57.53% | str:89476 | 5000 | yes | '0.0977443609022'; '0.0560747663551'; '0.03738317757' |
| percentlevel2 | 57.53% | str:89476 | 5000 | yes | '0.2030075187969'; '0.0654205607476'; '0.1588785046728' |
| percentlevel3 | 57.53% | str:89476 | 5000 | yes | '0.2932330827067'; '0.2757009345794'; '0.214953271028' |
| percentlevel4 | 57.53% | str:89476 | 5000 | yes | '0.3834586466165'; '0.5794392523364'; '0.5747663551401' |
| percent_no_score | 48.76% | str:75836 | 5000 | yes | '0.0225563909774'; '0.0233644859813'; '0.0140186915887' |
| count_of_students_expected | 45.62% | str:70949 | 1978 | no | '133'; '214'; '54' |
| percentmettestedonly | 39.57% | str:61542 | 5000 | yes | '0.6923076923076'; '0.8755980861244'; '0.8009478672985' |
| count_of_students_expected_to_test_including_previously_passed | 38.77% | str:60299 | 1786 | no | '133'; '214'; '54' |
| countmetstandard | 30.0% | str:46659 | 1243 | no | '90'; '183'; '169' |
| dat | 26.31% | str:40913 | 106 | no | 'N<10'; 'None'; 'Cross Student Group - N<10' |
| percent_consistent_grade_level_knowledge_and_above | 17.47% | str:27166 | 1690 | no | 'N<10'; '39.3%'; '35.7%' |
| percent_foundational_grade | 17.34% | str:26964 | 1611 | no | 'N<10'; '52.60%'; '34.70%' |
| test_administration_group | 17.34% | str:26964 | 3 | no | 'AIM'; 'SBAC'; 'WCAS' |
| count_consistent_grade_level_knowledge_and_above | 12.3% | str:19137 | 614 | no | 'NULL'; '11'; '5' |
| percent_consistent_grade | 8.84% | str:13747 | 779 | no | '15.1%'; 'N<10'; '27.4%' |
| percent_consistent_tested_only | 8.77% | str:13640 | 2351 | no | 'NULL'; '0.3928571428571'; '0.3571428571428' |
| percentnoscore | 8.77% | str:13640 | 1353 | no | 'NULL'; '0.0000000000000'; '0.0109170305676' |
| percentparticipation | 8.77% | str:13640 | 1353 | no | 'NULL'; '1.0000000000000'; '0.9890829694323' |
| percent_participation | 8.69% | str:13509 | 2810 | no | '0.980237154'; '0.984251969'; '0.980988593' |
| count_foundational_grade | 7.27% | str:11303 | 756 | no | '133'; '88'; '151' |
| count_of_students_expected_1 | 6.65% | str:10341 | 859 | no | '253'; '254'; '263' |
| percent_taking_alternative_assessment | 5.18% | str:8055 | 898 | no | '0.024193'; '0.024'; '0.015503' |
| percent_taking_alternative | 5.12% | str:7966 | 939 | no | '0.007434'; '0.007575'; '0.007936' |
| percent_met_tested_only | 4.61% | str:7170 | 2363 | no | '0.266129032'; '0.132'; '0.310077519' |
| percent_consistent_tested | 4.58% | str:7124 | 2360 | no | '0.1561338289962'; '0.2878787878787'; '0.2936507936507' |
| count_consistent_grade_level | 3.31% | str:5153 | 415 | no | '42'; '76'; '74' |

### attendance.json

| field | coverage | types | distinct_cap | overflow | examples |
| --- | --- | --- | --- | --- | --- |
| county | 100.0% | str:75204 | 1 | no | 'King' |
| dataasof | 100.0% | str:75204 | 5 | no | '2023-07-11T00:00:00.000'; '2023-06-21T00:00:00.000'; '2023-06-21' |
| districtcode | 100.0% | str:75204 | 1 | no | '17415' |
| districtname | 100.0% | str:75204 | 1 | no | 'Kent School District' |
| districtorganizationid | 100.0% | str:75204 | 1 | no | '100117' |
| esdname | 100.0% | str:75204 | 1 | no | 'Puget Sound Educational Service District 121' |
| gradelevel | 100.0% | str:75204 | 15 | no | '9'; '1'; '10' |
| organizationlevel | 100.0% | str:75204 | 2 | no | 'District'; 'School' |
| schoolname | 100.0% | str:75204 | 48 | no | 'District Total'; 'Grass Lake Elementary School'; 'Cedar Valley Elementary School' |
| schoolyear | 100.0% | str:75204 | 10 | no | '2014-15'; '2015-16'; '2016-17' |
| studentgroup | 100.0% | str:75204 | 39 | no | 'Migrant'; 'All Students'; 'English Language Learners' |
| studentgrouptype | 100.0% | str:75204 | 24 | no | 'Migrant'; 'AllStudents'; 'EnglishLearner' |
| currentschooltype | 94.61% | str:71147 | 4 | no | 'P'; 'A'; 'R' |
| schoolcode | 94.61% | str:71147 | 47 | no | '3708'; '3676'; '4345' |
| schoolorganizationid | 94.61% | str:71147 | 47 | no | '101591'; '101587'; '101605' |
| esdorganizationid | 89.82% | str:67551 | 1 | no | '100006' |
| measures | 59.37% | str:44652 | 1 | no | 'Regular Attendance' |
| suppression | 59.37% | str:44652 | 17 | no | 'Suppressed: N<10'; 'No Suppression'; 'Suppressed: >95%' |
| denominator | 46.94% | str:35300 | 1917 | no | '41'; '44'; '51' |
| numerator | 45.68% | str:34354 | 1782 | no | '37'; '46'; '54' |
| measure | 40.63% | str:30552 | 1 | no | 'Regular Attendance' |
| suppressionreason | 20.09% | str:15111 | 19 | no | 'No Suppression'; 'Suppressed: >92%'; 'Suppressed: >94%' |
| datreason | 10.36% | str:7788 | 85 | no | 'None'; 'Top/Bottom Range: >72.7%'; 'Top/Bottom Range: >76.9%' |
| dat_reason | 10.18% | str:7653 | 75 | no | 'None'; 'Cross Student Group - N<10'; 'Cross Grade Level - N<10' |
| label | 10.18% | str:7653 | 637 | no | '70.2%'; '60.8%'; '50.0%' |
| organizationname | 10.18% | str:7653 | 45 | no | 'Canyon Ridge Middle School'; 'Carriage Crest Elementary School'; 'Cedar Heights Middle School' |
| percent | 7.74% | str:5821 | 1677 | no | '0.7018'; '0.608'; '0.6076' |

### discipline.json

| field | coverage | types | distinct_cap | overflow | examples |
| --- | --- | --- | --- | --- | --- |
| county | 100.0% | str:74033 | 1 | no | 'King' |
| dateextracted | 100.0% | str:74033 | 10 | no | '2023-09-09T23:32:31.310Z'; '2023-09-10T01:00:19.500Z'; '2023-09-10T02:11:21.683Z' |
| disciplinedatnotes | 100.0% | str:74033 | 6 | no | 'None'; 'Top/Bottom Range'; 'N<10' |
| disciplinedenominator | 100.0% | str:74033 | 1778 | no | '74'; '*'; '443' |
| disciplinenumerator | 100.0% | str:74033 | 297 | no | '4'; '*'; '5' |
| disciplinerate | 100.0% | str:74033 | 2506 | no | '5.41%'; '<4.4%'; '<3.8%' |
| districtcode | 100.0% | str:74033 | 1 | no | '17415' |
| districtname | 100.0% | str:74033 | 1 | no | 'Kent School District' |
| districtorganizationid | 100.0% | str:74033 | 1 | no | '100117' |
| esdname | 100.0% | str:74033 | 1 | no | 'Puget Sound Educational Service District 121' |
| esdorganizationid | 100.0% | str:74033 | 1 | no | '100006' |
| gradelevel | 100.0% | str:74033 | 14 | no | 'Kindergarten'; '1st Grade'; '2nd Grade' |
| organizationlevel | 100.0% | str:74033 | 2 | no | 'School'; 'District' |
| rateexcluded10daysormore | 100.0% | str:74033 | 993 | no | '0.00%'; 'Top/Bottom Range'; 'N<10' |
| rateexcluded1dayorless | 100.0% | str:74033 | 1282 | no | '75.00%'; 'Top/Bottom Range'; '80.00%' |
| rateexcluded2to3days | 100.0% | str:74033 | 1226 | no | '25.00%'; 'Top/Bottom Range'; '20.00%' |
| rateexcluded4to5days | 100.0% | str:74033 | 1087 | no | '0.00%'; 'Top/Bottom Range'; 'N<10' |
| rateexcluded6to10days | 100.0% | str:74033 | 1022 | no | '0.00%'; 'Top/Bottom Range'; 'N<10' |
| schoolname | 100.0% | str:74033 | 48 | no | 'Carriage Crest Elementary School'; 'Cedar Heights Middle School'; 'Cedar Valley Elementary School' |
| schoolyear | 100.0% | str:74033 | 10 | no | '2014-15'; '2015-16'; '2016-17' |
| student_group | 100.0% | str:74033 | 29 | no | 'Non-Highly Capable'; 'All Students'; 'American Indian/ Alaskan Native' |
| currentschooltype | 94.74% | str:70140 | 4 | no | 'P'; 'R'; 'A' |
| schoolcode | 94.74% | str:70140 | 47 | no | '4353'; '4440'; '3676' |
| schoolorganizationid | 94.74% | str:70140 | 47 | no | '101606'; '101610'; '101587' |
| excluded10daysormore | 92.16% | str:68232 | 94 | no | '0'; '*'; '5' |
| excluded1dayorless | 92.16% | str:68232 | 139 | no | '3'; '*'; '4' |
| excluded2to3days | 92.16% | str:68232 | 181 | no | '1'; '*'; '9' |
| excluded4to5days | 92.16% | str:68232 | 109 | no | '0'; '*'; '7' |
| excluded6to10days | 92.16% | str:68232 | 108 | no | '0'; '*'; '3' |

### enrollment.json

| field | coverage | types | distinct_cap | overflow | examples |
| --- | --- | --- | --- | --- | --- |
| all_students | 100.0% | str:4562 | 730 | no | '2198'; '2135'; '2035' |
| american_indian_alaskan_native | 100.0% | str:4562 | 28 | no | '13'; '9'; '4' |
| asian | 100.0% | str:4562 | 343 | no | '363'; '395'; '411' |
| black_african_american | 100.0% | str:4562 | 253 | no | '267'; '210'; '228' |
| county | 100.0% | str:4562 | 1 | no | 'King' |
| dataasof | 100.0% | str:4562 | 4 | no | '2023-12-22T00:00:00.000'; '2024-06-18T00:00:00.000'; '2025-06-02T00:00:00.000' |
| districtcode | 100.0% | str:4562 | 1 | no | '17415' |
| districtname | 100.0% | str:4562 | 1 | no | 'Kent School District' |
| districtorganizationid | 100.0% | str:4562 | 1 | no | '100117' |
| english_language_learners | 100.0% | str:4562 | 400 | no | '172'; '122'; '128' |
| esdname | 100.0% | str:4562 | 1 | no | 'Puget Sound Educational Service District 121' |
| esdorganizationid | 100.0% | str:4562 | 1 | no | '100006' |
| female | 100.0% | str:4562 | 492 | no | '1055'; '1030'; '968' |
| gender_x | 100.0% | str:4562 | 20 | no | '0'; '1'; '4' |
| gradelevel | 100.0% | str:4562 | 16 | no | '10th Grade'; '11th Grade'; '12th Grade' |
| highly_capable | 100.0% | str:4562 | 232 | no | '0'; '1'; '37' |
| hispanic_latino_of_any_race | 100.0% | str:4562 | 379 | no | '443'; '359'; '304' |
| homeless | 100.0% | str:4562 | 70 | no | '22'; '23'; '18' |
| low_income | 100.0% | str:4562 | 620 | no | '1164'; '1023'; '928' |
| male | 100.0% | str:4562 | 509 | no | '1143'; '1105'; '1067' |
| migrant | 100.0% | str:4562 | 21 | no | '6'; '2'; '5' |
| military_parent | 100.0% | str:4562 | 45 | no | '0'; '1'; '3' |
| mobile | 100.0% | str:4562 | 135 | no | '136'; '101'; '118' |
| non_english_language_learners | 100.0% | str:4562 | 687 | no | '2026'; '2013'; '1907' |
| non_highly_capable | 100.0% | str:4562 | 736 | no | '2198'; '2135'; '2035' |
| non_homeless | 100.0% | str:4562 | 736 | no | '2176'; '2112'; '2017' |
| non_low_income | 100.0% | str:4562 | 568 | no | '1034'; '1112'; '1107' |
| non_migrant | 100.0% | str:4562 | 725 | no | '2192'; '2133'; '2033' |
| non_military_parent | 100.0% | str:4562 | 742 | no | '2198'; '2135'; '2035' |
| non_mobile | 100.0% | str:4562 | 719 | no | '2062'; '2034'; '1917' |
| non_section_504 | 100.0% | str:4562 | 728 | no | '2093'; '2033'; '1947' |
| organizationlevel | 100.0% | str:4562 | 2 | no | 'District'; 'School' |
| schoolname | 100.0% | str:4562 | 49 | no | 'District Total'; 'Fairwood Elementary School'; 'Soos Creek Elementary School' |
| schoolyear | 100.0% | str:4562 | 12 | no | '2014-15'; '2015-16'; '2016-17' |
| section_504 | 100.0% | str:4562 | 157 | no | '105'; '102'; '88' |
| students_with_disabilities | 100.0% | str:4562 | 255 | no | '209'; '176'; '200' |
| students_without_disabilities | 100.0% | str:4562 | 704 | no | '1989'; '1959'; '1835' |
| two_or_more_races | 100.0% | str:4562 | 204 | no | '187'; '198'; '121' |
| white | 100.0% | str:4562 | 462 | no | '877'; '923'; '914' |
| currentschooltype | 95.27% | str:4346 | 5 | no | 'P'; 'A'; 'S' |
| schoolcode | 95.27% | str:4346 | 48 | no | '3678'; '3707'; '3708' |
| schoolorganizationid | 95.27% | str:4346 | 48 | no | '101589'; '101590'; '101591' |
| fostercare | 86.06% | str:3926 | 22 | no | '10'; '0'; '14' |
| native_hawaiian_pacific_islander | 86.06% | str:3926 | 104 | no | '48'; '41'; '44' |
| non_fostercare | 86.06% | str:3926 | 190 | no | '2188'; '2128'; '2028' |
| dat | 32.35% | str:1476 | 3 | no | 'DAT Applied: Foster Care - N < 10'; 'DAT Applied: > 95%'; 'DAT Applied: N < 10' |
| foster_care | 13.94% | str:636 | 3 | no | '0'; '52'; '42' |
| native_hawaiian_other_pacific | 13.94% | str:636 | 62 | no | '49'; '65'; '68' |
| non_foster_care | 13.94% | str:636 | 27 | no | '1936'; '1945'; '2289' |

### graduation.json

| field | coverage | types | distinct_cap | overflow | examples |
| --- | --- | --- | --- | --- | --- |
| cohort | 100.0% | str:9670 | 6 | no | 'Five Year'; 'Four Year'; 'Seven Year' |
| county | 100.0% | str:9670 | 1 | no | 'King' |
| dataasof | 100.0% | str:9670 | 4 | no | '2024-01-19T00:00:00.000'; '2024-01-19'; '2024-11-27T00:00:00.000' |
| districtcode | 100.0% | str:9670 | 1 | no | '17415' |
| districtname | 100.0% | str:9670 | 1 | no | 'Kent School District' |
| districtorganizationid | 100.0% | str:9670 | 1 | no | '100117' |
| esdname | 100.0% | str:9670 | 1 | no | 'Puget Sound Educational Service District 121' |
| esdorganizationid | 100.0% | str:9670 | 1 | no | '100006' |
| organizationlevel | 100.0% | str:9670 | 2 | no | 'School'; 'District' |
| schoolname | 100.0% | str:9670 | 13 | no | 'Individualized Graduation & Degree Program'; 'Kent-Meridian High School'; 'Kent Mountain View Academy (Closed after 2020-2021 school year)' |
| schoolyear | 100.0% | str:9670 | 11 | no | '2020-21'; '2019-20'; '2018-19' |
| studentgroup | 100.0% | str:9670 | 30 | no | 'Non Section 504'; 'Section 504'; 'All Students' |
| studentgrouptype | 100.0% | str:9670 | 12 | no | '504'; 'All'; 'ELL' |
| schoolcode | 98.8% | str:9554 | 13 | no | '5275'; '2797'; '3014' |
| schoolorganizationid | 98.8% | str:9554 | 13 | no | '105491'; '101574'; '101576' |
| graduationrate | 95.18% | str:9204 | 2564 | no | '0.165'; '0.066'; '0.1556420233463' |
| graduate | 91.4% | str:8838 | 660 | no | 'NULL'; '40'; '35' |
| transferout | 91.4% | str:8838 | 211 | no | 'NULL'; '10'; '15' |
| year4dropout | 91.33% | str:8832 | 118 | no | 'NULL'; '69'; '58' |
| year3dropout | 91.3% | str:8829 | 48 | no | 'NULL'; '26'; '20' |
| year1dropout | 91.02% | str:8802 | 38 | no | 'NULL'; '4'; '1' |
| year2dropout | 90.98% | str:8798 | 36 | no | 'NULL'; '2'; '3' |
| year5dropout | 90.81% | str:8781 | 100 | no | 'NULL'; '60'; '77' |
| year6dropout | 90.57% | str:8758 | 81 | no | 'NULL'; '27'; '49' |
| year7dropout | 90.39% | str:8741 | 56 | no | 'NULL'; '31'; '40' |
| finalcohort | 82.16% | str:7945 | 685 | no | 'NULL'; '257'; '282' |
| transferin | 82.16% | str:7945 | 228 | no | 'NULL'; '37'; '42' |
| beggininggrade9 | 79.9% | str:7726 | 648 | no | 'NULL'; '230'; '255' |
| suppression | 70.69% | str:6836 | 161 | no | 'Cross Group'; 'No DAT'; '<1.54%' |
| continuing | 29.94% | str:2895 | 62 | no | 'NULL'; '0'; '33' |
| dropout | 29.94% | str:2895 | 144 | no | 'NULL'; '203'; '140' |
| dat | 29.31% | str:2834 | 120 | no | 'Cross Group'; 'No DAT'; '<17.6%' |
| beginninggrade9 | 11.5% | str:1112 | 181 | no | 'NULL'; '218'; '268' |
| final_cohort | 9.23% | str:893 | 196 | no | 'NULL'; '234'; '192' |
| transferredin | 9.23% | str:893 | 87 | no | 'NULL'; '30'; '23' |

### growth.json

| field | coverage | types | distinct_cap | overflow | examples |
| --- | --- | --- | --- | --- | --- |
| county | 100.0% | str:50665 | 1 | no | 'King' |
| dataasof | 100.0% | str:50665 | 4 | no | '2019-12-06T00:00:00.000'; '2023-12-11T00:00:00.000'; '2024-08-30T00:00:00.000' |
| districtcode | 100.0% | str:50665 | 1 | no | '17415' |
| districtname | 100.0% | str:50665 | 1 | no | 'Kent School District' |
| districtorganizationid | 100.0% | str:50665 | 1 | no | '100117' |
| esdname | 100.0% | str:50665 | 1 | no | 'Puget Sound Educational Service District 121' |
| esdorganizationid | 100.0% | str:50665 | 1 | no | '100006' |
| gradelevel | 100.0% | str:50665 | 12 | no | '6th Grade'; 'All Grades'; '4th Grade' |
| organizationlevel | 100.0% | str:50665 | 2 | no | 'School'; 'District' |
| schoolname | 100.0% | str:50665 | 40 | no | 'Grass Lake Elementary School'; 'Carriage Crest Elementary School'; 'Cedar Heights Middle School' |
| schoolyear | 100.0% | str:50665 | 8 | no | '2014-15'; '2018-19'; '2017-18' |
| studentgroup | 100.0% | str:50665 | 35 | no | 'Non-Low Income'; 'Non Migrant'; 'Military Parent' |
| studentgrouptype | 100.0% | str:50665 | 12 | no | 'Low Income'; 'Migrant'; 'Military' |
| subject | 100.0% | str:50665 | 2 | no | 'English Language Arts'; 'Math' |
| schoolcode | 97.24% | str:49265 | 40 | no | '3708'; '4353'; '4440' |
| schoolorganizationid | 97.24% | str:49265 | 40 | no | '101591'; '101606'; '101610' |
| currentschooltype | 95.12% | str:48193 | 2 | no | 'P'; 'A' |
| mediansgp | 82.63% | str:41864 | 176 | no | '58.5'; '49'; '55' |
| percenthighgrowth | 82.63% | str:41864 | 3468 | no | '0.378'; '0.327'; '0.362' |
| percentlowgrowth | 82.63% | str:41864 | 3476 | no | '0.284'; '0.32'; '0.261' |
| percenttypicalgrowth | 82.63% | str:41864 | 2863 | no | '0.338'; '0.353'; '0.377' |
| numberhighgrowth | 75.07% | str:38035 | 767 | no | '56'; '49'; '50' |
| numberlowgrowth | 75.07% | str:38035 | 811 | no | '42'; '48'; '36' |
| numbertypicalgrowth | 75.07% | str:38035 | 758 | no | '50'; '53'; '52' |
| studentcount | 75.07% | str:38035 | 1358 | no | '148'; '150'; '138' |
| suppression | 41.11% | str:20826 | 7 | no | 'Suppressed: N<10'; 'Suppressed: Cross Group'; 'Suppressed: Cross Grade' |
| datreason | 28.66% | str:14521 | 4 | no | 'NULL'; 'N<10'; 'Cross Group' |

### sqss.json

| field | coverage | types | distinct_cap | overflow | examples |
| --- | --- | --- | --- | --- | --- |
| county | 100.0% | str:109011 | 1 | no | 'King' |
| dataasof | 100.0% | str:109011 | 5 | no | '2023-07-11T00:00:00.000'; '2023-06-21T00:00:00.000'; '2023-06-21' |
| districtcode | 100.0% | str:109011 | 1 | no | '17415' |
| districtname | 100.0% | str:109011 | 1 | no | 'Kent School District' |
| districtorganizationid | 100.0% | str:109011 | 1 | no | '100117' |
| esdname | 100.0% | str:109011 | 1 | no | 'Puget Sound Educational Service District 121' |
| gradelevel | 100.0% | str:109011 | 15 | no | '11'; '9'; '1' |
| organizationlevel | 100.0% | str:109011 | 2 | no | 'District'; 'School' |
| schoolname | 100.0% | str:109011 | 48 | no | 'District Total'; 'Grass Lake Elementary School'; 'Cedar Valley Elementary School' |
| schoolyear | 100.0% | str:109011 | 10 | no | '2014-15'; '2015-16'; '2016-17' |
| studentgroup | 100.0% | str:109011 | 39 | no | 'Foster Care'; 'Migrant'; 'All Students' |
| studentgrouptype | 100.0% | str:109011 | 24 | no | 'Foster'; 'Migrant'; 'AllStudents' |
| currentschooltype | 94.47% | str:102985 | 4 | no | 'P'; 'A'; 'R' |
| schoolcode | 94.47% | str:102985 | 47 | no | '3708'; '3676'; '4345' |
| schoolorganizationid | 94.47% | str:102985 | 47 | no | '101591'; '101587'; '101605' |
| esdorganizationid | 89.64% | str:97719 | 1 | no | '100006' |
| measures | 59.41% | str:64762 | 3 | no | 'Dual Credit'; 'Regular Attendance'; 'Ninth Grade on Track' |
| suppression | 59.41% | str:64762 | 26 | no | 'Suppressed: N<10'; 'No Suppression'; 'Suppressed: >95%' |
| denominator | 41.0% | str:44695 | 2218 | no | '41'; '44'; '51' |
| measure | 40.59% | str:44249 | 3 | no | 'Regular Attendance'; 'Dual Credit'; 'Ninth Grade on Track' |
| numerator | 40.13% | str:43749 | 2079 | no | '37'; '46'; '54' |
| suppressionreason | 19.91% | str:21705 | 26 | no | 'No Suppression'; 'Suppressed: >92%'; 'Suppressed: >94%' |
| dat_reason | 10.36% | str:11292 | 114 | no | 'No Students'; 'None'; 'Cross Student Group - N<10' |
| organizationname | 10.36% | str:11292 | 45 | no | 'Canyon Ridge Middle School'; 'Carriage Crest Elementary School'; 'Cedar Heights Middle School' |
| label | 10.35% | str:11283 | 748 | no | 'No Students'; '70.2%'; '60.8%' |
| datreason | 10.32% | str:11252 | 133 | no | 'None'; 'Top/Bottom Range: >72.7%'; 'Top/Bottom Range: >76.9%' |
| percent | 6.45% | str:7026 | 2071 | no | '0.7018'; '0.608'; '0.6076' |
| percenttakingctetechprep | 6.35% | str:6922 | 3977 | no | '0.1'; '0.05'; '0.2752293577981' |
| percenttakingcollegeinth | 6.26% | str:6823 | 3255 | no | '0.1'; '0.05'; '0.03' |
| percenttakingap | 6.23% | str:6791 | 3294 | no | '0.1'; '0.05'; '0.03' |
| percenttakingrunningstart | 6.11% | str:6660 | 2493 | no | '0.1'; '0.05'; '0.03' |
| percenttakingib | 5.97% | str:6504 | 1205 | no | '0.1'; '0.05'; '0.03' |
| percenttakingcambridge | 5.87% | str:6397 | 10 | no | '0.1'; '0.05'; '0.03' |
| numbertakingctetechprep | 4.33% | str:4722 | 939 | no | '30'; '152'; '324' |
| numbertakingap | 3.28% | str:3572 | 626 | no | '162'; '100'; '103' |
| numbertakingcollegeinthe | 3.24% | str:3537 | 575 | no | '155'; '111'; '13' |
| numbertakingrunningstart | 2.5% | str:2723 | 492 | no | '12'; '8'; '82' |
| apcoursenumber | 1.51% | str:1648 | 166 | no | '0'; '615'; '356' |
| cambridgecoursenumber | 1.51% | str:1648 | 1 | no | '0' |
| cihscoursenumber | 1.51% | str:1648 | 230 | no | '0'; '7'; '3' |
| ctecoursenumber | 1.51% | str:1648 | 301 | no | '0'; '13'; '4' |
| ibcoursenumber | 1.51% | str:1648 | 53 | no | '0'; '81'; '153' |
| runningstartcoursenumber | 1.51% | str:1648 | 159 | no | '0'; '3'; '17' |
| numbertakingib | 1.05% | str:1148 | 249 | no | '25'; '24'; '3' |
| apcoursepercent | 0.82% | str:899 | 245 | no | '0.031'; '0.034'; '0.1' |
| cambridgecoursepercent | 0.82% | str:899 | 100 | no | '0.031'; '0.034'; '0.1' |
| cihscoursepercent | 0.82% | str:899 | 375 | no | '0.031'; '0.034'; '0.1' |
| ctecoursepercent | 0.82% | str:899 | 362 | no | '0.031'; '0.034'; '0.1' |
| ibcoursepercent | 0.82% | str:899 | 154 | no | '0.031'; '0.034'; '0.1' |
| runningstartcoursepercent | 0.82% | str:899 | 289 | no | '0.031'; '0.034'; '0.1' |

### teacher_demographics.json

| field | coverage | types | distinct_cap | overflow | examples |
| --- | --- | --- | --- | --- | --- |
| avgyearsexperience | 100.0% | str:4745 | 280 | no | '12.3'; '11.9'; 'NULL' |
| county | 100.0% | str:4745 | 1 | no | 'King' |
| dataasof | 100.0% | str:4745 | 2 | no | '2025-12-11'; '2025-12-16' |
| demographiccategory | 100.0% | str:4745 | 13 | no | 'All'; 'Female'; 'Gender X' |
| demographiccategoryid | 100.0% | str:4745 | 13 | no | '0'; '9'; '10' |
| demographiccategorytype | 100.0% | str:4745 | 3 | no | 'All'; 'Gender'; 'RaceEthnicity' |
| esdname | 100.0% | str:4745 | 1 | no | 'Puget Sound Educational Service District 121' |
| esdorganizationid | 100.0% | str:4745 | 1 | no | '100006' |
| iseevalidated | 100.0% | str:4745 | 2 | no | '1'; '0' |
| leacode | 100.0% | str:4745 | 1 | no | '17415' |
| leaname | 100.0% | str:4745 | 1 | no | 'Kent School District' |
| leaorganizationid | 100.0% | str:4745 | 1 | no | '100117' |
| ma_count | 100.0% | str:4745 | 109 | no | '1176'; '836'; '0' |
| ma_percent | 100.0% | str:4745 | 331 | no | '0.769'; '0.772'; 'NULL' |
| organizationid | 100.0% | str:4745 | 49 | no | '100117'; '101569'; '101571' |
| organizationlevel | 100.0% | str:4745 | 2 | no | 'LEA'; 'School' |
| organizationlevelid | 100.0% | str:4745 | 2 | no | '3'; '4' |
| organizationname | 100.0% | str:4745 | 49 | no | 'Kent School District'; 'Regional Justice Center'; 'Meridian Elementary School' |
| rowid | 100.0% | str:4745 | 4745 | no | '260859'; '260860'; '260861' |
| schoolcode | 100.0% | str:4745 | 49 | no | 'NULL'; '1807'; '2565' |
| schoolname | 100.0% | str:4745 | 49 | no | 'NULL'; 'Regional Justice Center'; 'Meridian Elementary School' |
| schoolorganizationid | 100.0% | str:4745 | 49 | no | 'NULL'; '101569'; '101571' |
| schoolyear | 100.0% | str:4745 | 8 | no | '2024-25'; '2023-24'; '2022-23' |
| sumyearsexperience | 100.0% | str:4745 | 1684 | no | '18718.5'; '12873.5'; 'NULL' |
| teachercount | 100.0% | str:4745 | 143 | no | '1530'; '1083'; '0' |
| teacherpercent | 100.0% | str:4745 | 523 | no | '1.000'; '0.708'; '0.000' |
| teachertotalcount | 100.0% | str:4745 | 81 | no | '1530'; 'NULL'; '31' |

### teacher_experience.json

| field | coverage | types | distinct_cap | overflow | examples |
| --- | --- | --- | --- | --- | --- |
| county | 100.0% | str:4602 | 1 | no | 'King' |
| dataasof | 100.0% | str:4602 | 1 | no | '2025-12-17' |
| esdname | 100.0% | str:4602 | 1 | no | 'Puget Sound Educational Service District 121' |
| esdorganizationid | 100.0% | str:4602 | 1 | no | '100006' |
| experiencebin | 100.0% | str:4602 | 13 | no | '0.0 - 4.9'; '5.0 - 9.9'; '10.0 - 14.9' |
| iseevalidated | 100.0% | str:4602 | 1 | no | '1' |
| leacode | 100.0% | str:4602 | 1 | no | '17415' |
| leaname | 100.0% | str:4602 | 1 | no | 'Kent School District' |
| leaorganizationid | 100.0% | str:4602 | 1 | no | '100117' |
| organizationid | 100.0% | str:4602 | 48 | no | '100117'; '106971'; '101606' |
| organizationlevel | 100.0% | str:4602 | 2 | no | 'LEA'; 'School' |
| organizationlevelid | 100.0% | str:4602 | 2 | no | '3'; '4' |
| organizationname | 100.0% | str:4602 | 48 | no | 'Kent School District'; 'Canyon Ridge Middle School'; 'Carriage Crest Elementary School' |
| rowid | 100.0% | str:4602 | 4602 | no | '241307'; '241308'; '241309' |
| schoolcode | 100.0% | str:4602 | 48 | no | 'NULL'; '5738'; '4353' |
| schoolname | 100.0% | str:4602 | 48 | no | 'NULL'; 'Canyon Ridge Middle School'; 'Carriage Crest Elementary School' |
| schoolorganizationid | 100.0% | str:4602 | 48 | no | 'NULL'; '106971'; '101606' |
| schoolyear | 100.0% | str:4602 | 8 | no | '2024-25'; '2023-24'; '2022-23' |
| teachercount | 100.0% | str:4602 | 88 | no | '395'; '358'; '234' |
| teacherpercent | 100.0% | str:4602 | 357 | no | '25.8'; '23.4'; '15.3' |
| teachertotalcount | 100.0% | str:4602 | 80 | no | '1530'; '46'; '24' |

### wakids.json

| field | coverage | types | distinct_cap | overflow | examples |
| --- | --- | --- | --- | --- | --- |
| county | 100.0% | str:210202 | 1 | no | 'King' |
| dataasof | 100.0% | str:210202 | 2 | no | '2025-11-13T14:14:27.847'; '2025-12-31T16:50:58.367' |
| districtname | 100.0% | str:210202 | 1 | no | 'Kent School District' |
| districtorganizationid | 100.0% | str:210202 | 1 | no | '100117' |
| domain | 100.0% | str:210202 | 7 | no | 'Cognitive'; 'Language'; 'Literacy' |
| esdname | 100.0% | str:210202 | 1 | no | 'Puget Sound Educational Service District 121' |
| esdorganizationid | 100.0% | str:210202 | 1 | no | '100006' |
| measure | 100.0% | str:210202 | 17 | no | 'CognitiveDevelopmentLevel'; 'CognitiveReadinessFlag'; 'LanguageDevelopmentLevel' |
| organizationid | 100.0% | str:210202 | 30 | no | '100117'; '101571'; '101575' |
| organizationlevel | 100.0% | str:210202 | 2 | no | 'District'; 'School' |
| schoolname | 100.0% | str:210202 | 30 | no | 'District Total'; 'Meridian Elementary School'; 'East Hill Elementary School' |
| schoolyear | 100.0% | str:210202 | 12 | no | '2014-15'; '2015-16'; '2016-17' |
| studentgroup | 100.0% | str:210202 | 21 | no | 'All Students'; 'English Language Learners'; 'Non-English Language Learners' |
| studentgrouptype | 100.0% | str:210202 | 8 | no | 'All'; 'Bilingual'; 'FederalRaceEthnicity' |
| washingtonstatecode | 100.0% | str:210202 | 1 | no | '103300' |
| washingtonstatename | 100.0% | str:210202 | 1 | no | 'State Total' |
| measurevalue | 99.48% | str:209108 | 14 | no | 'Blue'; 'Green'; 'orange' |
| schoolorganizationid | 94.41% | str:198456 | 29 | no | '101571'; '101575'; '101580' |
| percent | 80.94% | str:170144 | 5000 | yes | '0.48098'; '0.16195'; '0.04390' |
| denominator | 67.94% | str:142813 | 725 | no | '1025'; '1035'; '1026' |
| numerator | 67.94% | str:142813 | 1092 | no | '493'; '166'; '45' |
| developmentlevel | 55.37% | str:116380 | 5 | no | '4 Year Olds'; '3 Year Olds'; '0-2 Year Olds' |
| suppress | 32.06% | str:67389 | 2 | no | 'N<10'; 'Cross Student Group' |

## Candidate Record Identity Fields

| dataset | candidate fields |
| --- | --- |
| assessment.json | county, districtcode, districtname, districtorganizationid, esdname, esdorganizationid, gradelevel, organizationlevel, schoolname, schoolyear, studentgroup, studentgrouptype, testadministration, testsubject, schoolorganizationid, currentschooltype, schoolcode, suppression |
| attendance.json | county, districtcode, districtname, districtorganizationid, esdname, gradelevel, organizationlevel, schoolname, schoolyear, studentgroup, studentgrouptype, currentschooltype, schoolcode, schoolorganizationid, esdorganizationid, measures, suppression, measure, organizationname |
| discipline.json | county, districtcode, districtname, districtorganizationid, esdname, esdorganizationid, gradelevel, organizationlevel, schoolname, schoolyear, currentschooltype, schoolcode, schoolorganizationid |
| enrollment.json | county, districtcode, districtname, districtorganizationid, esdname, esdorganizationid, gender_x, gradelevel, organizationlevel, schoolname, schoolyear, currentschooltype, schoolcode, schoolorganizationid |
| graduation.json | cohort, county, districtcode, districtname, districtorganizationid, esdname, esdorganizationid, organizationlevel, schoolname, schoolyear, studentgroup, studentgrouptype, schoolcode, schoolorganizationid, suppression |
| growth.json | county, districtcode, districtname, districtorganizationid, esdname, esdorganizationid, gradelevel, organizationlevel, schoolname, schoolyear, studentgroup, studentgrouptype, subject, schoolcode, schoolorganizationid, currentschooltype, suppression |
| sqss.json | county, districtcode, districtname, districtorganizationid, esdname, gradelevel, organizationlevel, schoolname, schoolyear, studentgroup, studentgrouptype, currentschooltype, schoolcode, schoolorganizationid, esdorganizationid, measures, suppression, measure, organizationname |
| teacher_demographics.json | county, demographiccategory, demographiccategoryid, demographiccategorytype, esdname, esdorganizationid, iseevalidated, leacode, leaname, leaorganizationid, organizationid, organizationlevel, organizationlevelid, organizationname, rowid, schoolcode, schoolname, schoolorganizationid, schoolyear |
| teacher_experience.json | county, esdname, esdorganizationid, experiencebin, iseevalidated, leacode, leaname, leaorganizationid, organizationid, organizationlevel, organizationlevelid, organizationname, rowid, schoolcode, schoolname, schoolorganizationid, schoolyear |
| wakids.json | county, districtname, districtorganizationid, domain, esdname, esdorganizationid, measure, organizationid, organizationlevel, schoolname, schoolyear, studentgroup, studentgrouptype, washingtonstatecode, washingtonstatename, schoolorganizationid |

## Year Ranges

| dataset | field | min | max | sample values | coverage_pct |
| --- | --- | --- | --- | --- | --- |
| assessment.json | schoolyear | 2014 | 2024 | 2014, 2015, 2016, 2017, 2018, 2020, 2021, 2022, 2023, 2024 | 100.0 |
| attendance.json | schoolyear | 2014 | 2023 | 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023 | 100.0 |
| discipline.json | schoolyear | 2014 | 2023 | 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023 | 100.0 |
| enrollment.json | schoolyear | 2014 | 2025 | 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025 | 100.0 |
| graduation.json | schoolyear | 2014 | 2024 | 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024 | 100.0 |
| growth.json | schoolyear | 2014 | 2024 | 2014, 2015, 2016, 2017, 2018, 2022, 2023, 2024 | 100.0 |
| sqss.json | schoolyear | 2014 | 2023 | 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023 | 100.0 |
| teacher_demographics.json | schoolyear | 2017 | 2024 | 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024 | 100.0 |
| teacher_experience.json | schoolyear | 2017 | 2024 | 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024 | 100.0 |
| wakids.json | schoolyear | 2014 | 2025 | 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025 | 100.0 |

## Kent School District Signals

| dataset | matching records | signals | examples |
| --- | --- | --- | --- |
| assessment.json | 155528 | districtcode=17415:155528, districtname=Kent School District:155528, districtorganizationid=100117:155528 | ['districtcode=17415', 'districtname=Kent School District', 'districtorganizationid=100117'] |
| attendance.json | 75204 | districtcode=17415:75204, districtname=Kent School District:75204, districtorganizationid=100117:75204, organizationname=Kent School District:422 | ['districtcode=17415', 'districtname=Kent School District', 'districtorganizationid=100117']; ['organizationname=Kent School District', 'districtcode=17415', 'districtname=Kent School District', 'districtorganiz... |
| discipline.json | 74033 | districtcode=17415:74033, districtname=Kent School District:74033, districtorganizationid=100117:74033 | ['districtcode=17415', 'districtname=Kent School District', 'districtorganizationid=100117'] |
| enrollment.json | 4562 | districtcode=17415:4562, districtname=Kent School District:4562, districtorganizationid=100117:4562 | ['districtcode=17415', 'districtname=Kent School District', 'districtorganizationid=100117'] |
| graduation.json | 9670 | districtcode=17415:9670, districtname=Kent School District:9670, districtorganizationid=100117:9670 | ['districtcode=17415', 'districtname=Kent School District', 'districtorganizationid=100117'] |
| growth.json | 50665 | districtcode=17415:50665, districtname=Kent School District:50665, districtorganizationid=100117:50665 | ['districtcode=17415', 'districtname=Kent School District', 'districtorganizationid=100117'] |
| sqss.json | 109011 | districtcode=17415:109011, districtname=Kent School District:109011, districtorganizationid=100117:109011, organizationname=Kent School District:635 | ['districtcode=17415', 'districtname=Kent School District', 'districtorganizationid=100117']; ['organizationname=Kent School District', 'districtcode=17415', 'districtname=Kent School District', 'districtorganiz... |
| teacher_demographics.json | 4745 | leaname=Kent School District:4745, organizationid=100117:104, organizationname=Kent School District:104 | ['organizationid=100117', 'organizationname=Kent School District', 'leaname=Kent School District']; ['leaname=Kent School District'] |
| teacher_experience.json | 4602 | leaname=Kent School District:4602, organizationid=100117:104, organizationname=Kent School District:104 | ['organizationid=100117', 'organizationname=Kent School District', 'leaname=Kent School District']; ['leaname=Kent School District'] |
| wakids.json | 210202 | districtname=Kent School District:210202, districtorganizationid=100117:210202, organizationid=100117:11746 | ['organizationid=100117', 'districtorganizationid=100117', 'districtname=Kent School District']; ['districtorganizationid=100117', 'districtname=Kent School District'] |

## Null, Missing, And Suppression Patterns

| dataset | field | null_like | blank | suppressed | suppression examples |
| --- | --- | --- | --- | --- | --- |
| assessment.json | studentgroup | 1528 | 0 | 0 |  |
| assessment.json | schoolorganizationid | 936 | 0 | 0 |  |
| assessment.json | percentmetstandard | 0 | 0 | 49783 | '<10%'; 'Suppressed: N<10' |
| assessment.json | suppression | 46659 | 0 | 62525 | '<10%'; 'N<10'; 'Cross Student Group - N<10' |
| assessment.json | percentlevel1 | 6058 | 0 | 0 |  |
| assessment.json | percentlevel2 | 6058 | 0 | 0 |  |
| assessment.json | percentlevel3 | 6058 | 0 | 0 |  |
| assessment.json | percentlevel4 | 6058 | 0 | 0 |  |
| assessment.json | count_of_students_expected | 8459 | 0 | 0 |  |
| assessment.json | count_of_students_expected_to_test_including_previously_passed | 8220 | 0 | 0 |  |
| assessment.json | dat | 15931 | 0 | 20745 | 'N<10'; 'Cross Student Group - N<10'; '<10%' |
| assessment.json | percent_consistent_grade_level_knowledge_and_above | 0 | 0 | 10381 | 'N<10'; '<10%'; '<10.0%' |
| assessment.json | percent_foundational_grade | 0 | 0 | 10844 | 'N<10'; '<10%' |
| assessment.json | count_consistent_grade_level_knowledge_and_above | 8459 | 0 | 0 |  |
| assessment.json | percent_consistent_grade | 0 | 0 | 5714 | 'N<10'; '<10%' |
| assessment.json | percent_consistent_tested_only | 6058 | 0 | 0 |  |
| assessment.json | percentnoscore | 6058 | 0 | 0 |  |
| assessment.json | percentparticipation | 6058 | 0 | 0 |  |
| attendance.json | suppression | 0 | 0 | 43995 | 'Suppressed: N<10'; 'No Suppression'; 'Suppressed: >95%' |
| attendance.json | suppressionreason | 0 | 0 | 14951 | 'No Suppression'; 'Suppressed: >92%'; 'Suppressed: >94%' |
| attendance.json | datreason | 4069 | 0 | 3208 | 'N<10'; 'Cross Student Group - N<10'; 'Cross Organization - N<10' |
| attendance.json | dat_reason | 4296 | 0 | 2831 | 'Cross Student Group - N<10'; 'Cross Grade Level - N<10'; 'N<10' |
| attendance.json | label | 0 | 0 | 1757 | 'N<10' |
| discipline.json | disciplinedatnotes | 9466 | 0 | 20231 | 'N<10'; 'Cross Student Group - N<10'; 'Cross Grade Level - N<10' |
| discipline.json | disciplinedenominator | 0 | 0 | 64567 | '*' |
| discipline.json | disciplinenumerator | 0 | 0 | 64567 | '*' |
| discipline.json | disciplinerate | 0 | 0 | 18738 | 'N<10'; '<10.0%'; '<10.7%' |
| discipline.json | rateexcluded10daysormore | 0 | 0 | 20231 | 'N<10'; 'Cross Student Group - N<10'; 'Cross Grade Level - N<10' |
| discipline.json | rateexcluded1dayorless | 0 | 0 | 20231 | 'N<10'; 'Cross Student Group - N<10'; 'Cross Grade Level - N<10' |
| discipline.json | rateexcluded2to3days | 0 | 0 | 20231 | 'N<10'; 'Cross Student Group - N<10'; 'Cross Grade Level - N<10' |
| discipline.json | rateexcluded4to5days | 0 | 0 | 20231 | 'N<10'; 'Cross Student Group - N<10'; 'Cross Grade Level - N<10' |
| discipline.json | rateexcluded6to10days | 0 | 0 | 20231 | 'N<10'; 'Cross Student Group - N<10'; 'Cross Grade Level - N<10' |
| discipline.json | excluded10daysormore | 0 | 0 | 58766 | '*' |
| discipline.json | excluded1dayorless | 0 | 0 | 58766 | '*' |
| discipline.json | excluded2to3days | 0 | 0 | 58766 | '*' |
| discipline.json | excluded4to5days | 0 | 0 | 58766 | '*' |
| discipline.json | excluded6to10days | 0 | 0 | 58766 | '*' |
| graduation.json | schoolcode | 980 | 0 | 0 |  |
| graduation.json | schoolorganizationid | 980 | 0 | 0 |  |
| graduation.json | graduationrate | 3619 | 0 | 0 |  |
| graduation.json | graduate | 6101 | 0 | 0 |  |
| graduation.json | transferout | 6101 | 0 | 0 |  |
| graduation.json | year4dropout | 6178 | 0 | 0 |  |
| graduation.json | year3dropout | 6415 | 0 | 0 |  |
| graduation.json | year1dropout | 7261 | 0 | 0 |  |
| graduation.json | year2dropout | 6814 | 0 | 0 |  |
| graduation.json | year5dropout | 7122 | 0 | 0 |  |
| graduation.json | year6dropout | 8047 | 0 | 0 |  |
| graduation.json | year7dropout | 8493 | 0 | 0 |  |
| graduation.json | finalcohort | 5473 | 0 | 0 |  |
| graduation.json | transferin | 5490 | 0 | 0 |  |
| graduation.json | beggininggrade9 | 5222 | 0 | 0 |  |
| graduation.json | suppression | 0 | 0 | 2055 | '<10.7%'; 'N<10'; '<10.0%' |
| graduation.json | continuing | 2134 | 0 | 0 |  |
| graduation.json | dropout | 2134 | 0 | 0 |  |
| graduation.json | dat | 0 | 0 | 856 | 'N<10'; '<10.3%'; '<10.0%' |
| graduation.json | beginninggrade9 | 879 | 0 | 0 |  |
| graduation.json | final_cohort | 628 | 0 | 0 |  |
| graduation.json | transferredin | 631 | 0 | 0 |  |
| growth.json | schoolcode | 1072 | 0 | 0 |  |
| growth.json | schoolorganizationid | 1072 | 0 | 0 |  |
| growth.json | mediansgp | 7792 | 0 | 0 |  |
| growth.json | percenthighgrowth | 8610 | 0 | 0 |  |
| growth.json | percentlowgrowth | 8610 | 0 | 0 |  |
| growth.json | percenttypicalgrowth | 8610 | 0 | 0 |  |
| growth.json | numberhighgrowth | 11049 | 0 | 0 |  |
| growth.json | numberlowgrowth | 11049 | 0 | 0 |  |
| growth.json | numbertypicalgrowth | 11049 | 0 | 0 |  |
| growth.json | studentcount | 11045 | 0 | 0 |  |
| growth.json | suppression | 4237 | 0 | 14885 | 'Suppressed: N<10'; 'Suppressed: Cross Group'; 'Suppressed: Cross Grade' |
| growth.json | datreason | 7942 | 0 | 3647 | 'N<10' |
| sqss.json | suppression | 0 | 0 | 64105 | 'Suppressed: N<10'; 'No Suppression'; 'Suppressed: >95%' |
| sqss.json | suppressionreason | 0 | 0 | 21545 | 'No Suppression'; 'Suppressed: >92%'; 'Suppressed: >94%' |
| sqss.json | dat_reason | 5167 | 0 | 3365 | 'Cross Student Group - N<10'; 'Cross Grade Level - N<10'; 'N<10' |
| sqss.json | label | 0 | 0 | 2107 | 'N<10'; '<10.0%'; '<10.3%' |
| sqss.json | datreason | 6772 | 0 | 3797 | 'N<10'; 'Cross Student Group - N<10'; 'Cross Organization - N<10' |
| teacher_demographics.json | avgyearsexperience | 2265 | 0 | 0 |  |
| teacher_demographics.json | ma_count | 143 | 0 | 0 |  |
| teacher_demographics.json | ma_percent | 1882 | 0 | 0 |  |
| teacher_demographics.json | schoolcode | 104 | 0 | 0 |  |
| teacher_demographics.json | schoolname | 104 | 0 | 0 |  |
| teacher_demographics.json | schoolorganizationid | 104 | 0 | 0 |  |
| teacher_demographics.json | sumyearsexperience | 2265 | 0 | 0 |  |
| teacher_demographics.json | teachercount | 143 | 0 | 0 |  |
| teacher_demographics.json | teacherpercent | 143 | 0 | 0 |  |
| teacher_demographics.json | teachertotalcount | 143 | 0 | 0 |  |
| teacher_experience.json | schoolcode | 104 | 0 | 0 |  |
| teacher_experience.json | schoolname | 104 | 0 | 0 |  |
| teacher_experience.json | schoolorganizationid | 104 | 0 | 0 |  |
| wakids.json | suppress | 0 | 0 | 40058 | 'N<10' |

## Numeric Field Patterns

| dataset | field | numeric | numeric strings | percent strings | min | max | types |
| --- | --- | --- | --- | --- | --- | --- | --- |
| assessment.json | county | 0 | 0 | 0 | None | None | str:155528 |
| assessment.json | districtcode | 0 | 155528 | 0 | 17415.0 | 17415.0 | str:155528 |
| assessment.json | districtorganizationid | 0 | 155528 | 0 | 100117.0 | 100117.0 | str:155528 |
| assessment.json | esdorganizationid | 0 | 155528 | 0 | 100006.0 | 100006.0 | str:155528 |
| assessment.json | gradelevel | 0 | 109489 | 0 | 1.0 | 12.0 | str:155528 |
| assessment.json | organizationlevel | 0 | 0 | 0 | None | None | str:155528 |
| assessment.json | schoolorganizationid | 0 | 144551 | 0 | 101569.0 | 106971.0 | str:145487 |
| assessment.json | schoolcode | 0 | 144551 | 0 | 1807.0 | 5738.0 | str:144551 |
| assessment.json | percentmetstandard | 0 | 61542 | 61542 | 1.1 | 95.1 | str:114615 |
| assessment.json | percentlevel1 | 0 | 83418 | 0 | 0.0 | 0.9247311827956 | str:89476 |
| assessment.json | percentlevel2 | 0 | 83418 | 0 | 0.0 | 0.9285714285714 | str:89476 |
| assessment.json | percentlevel3 | 0 | 83418 | 0 | 0.0 | 0.8 | str:89476 |
| assessment.json | percentlevel4 | 0 | 83418 | 0 | 0.0 | 0.8260869565217 | str:89476 |
| assessment.json | percent_no_score | 0 | 75836 | 0 | 0.0 | 0.9698275862068 | str:75836 |
| assessment.json | count_of_students_expected | 0 | 62490 | 0 | 10.0 | 14302.0 | str:70949 |
| assessment.json | percentmettestedonly | 0 | 61542 | 0 | 0.0144404332129 | 1.0 | str:61542 |
| assessment.json | count_of_students_expected_to_test_including_previously_passed | 0 | 52079 | 0 | 10.0 | 14302.0 | str:60299 |
| assessment.json | countmetstandard | 0 | 46659 | 0 | 2.0 | 8285.0 | str:46659 |
| assessment.json | percent_consistent_grade_level_knowledge_and_above | 0 | 15024 | 15024 | 1.15 | 91.7 | str:27166 |
| assessment.json | percent_foundational_grade | 0 | 15654 | 15654 | 3.6 | 100.0 | str:26964 |
| assessment.json | count_consistent_grade_level_knowledge_and_above | 0 | 10678 | 0 | 3.0 | 6139.0 | str:19137 |
| assessment.json | percent_consistent_grade | 0 | 7318 | 7318 | 1.8 | 89.7 | str:13747 |
| assessment.json | percent_consistent_tested_only | 0 | 7582 | 0 | 0.02849002849 | 0.9111111111111 | str:13640 |
| assessment.json | percentnoscore | 0 | 7582 | 0 | 0.0 | 0.696 | str:13640 |
| assessment.json | percentparticipation | 0 | 7582 | 0 | 0.304 | 1.0 | str:13640 |
| assessment.json | percent_participation | 0 | 13509 | 0 | 0.35106383 | 1.0 | str:13509 |
| assessment.json | count_foundational_grade | 0 | 11303 | 0 | 3.0 | 8408.0 | str:11303 |
| assessment.json | count_of_students_expected_1 | 0 | 10341 | 0 | 10.0 | 13075.0 | str:10341 |
| assessment.json | percent_taking_alternative_assessment | 0 | 8055 | 0 | 0.0 | 1.0 | str:8055 |
| assessment.json | percent_taking_alternative | 0 | 7966 | 0 | 0.0 | 1.0 | str:7966 |
| assessment.json | percent_met_tested_only | 0 | 7170 | 0 | 0.016949153 | 0.929577465 | str:7170 |
| assessment.json | percent_consistent_tested | 0 | 7124 | 0 | 0.0193370165745 | 0.9285714285714 | str:7124 |
| assessment.json | count_consistent_grade_level | 0 | 5153 | 0 | 4.0 | 5545.0 | str:5153 |
| attendance.json | county | 0 | 0 | 0 | None | None | str:75204 |
| attendance.json | districtcode | 0 | 75204 | 0 | 17415.0 | 17415.0 | str:75204 |
| attendance.json | districtorganizationid | 0 | 75204 | 0 | 100117.0 | 100117.0 | str:75204 |
| attendance.json | gradelevel | 0 | 56001 | 0 | 1.0 | 12.0 | str:75204 |
| attendance.json | organizationlevel | 0 | 0 | 0 | None | None | str:75204 |
| attendance.json | schoolcode | 0 | 71147 | 0 | 1807.0 | 5738.0 | str:71147 |
| attendance.json | schoolorganizationid | 0 | 71147 | 0 | 101569.0 | 106971.0 | str:71147 |
| attendance.json | esdorganizationid | 0 | 67551 | 0 | 100006.0 | 100006.0 | str:67551 |
| attendance.json | denominator | 0 | 35300 | 0 | 0.0 | 28244.0 | str:35300 |
| attendance.json | numerator | 0 | 34354 | 0 | 2.0 | 24454.0 | str:34354 |
| attendance.json | label | 0 | 5370 | 5370 | 11.8 | 94.6 | str:7653 |
| attendance.json | percent | 0 | 5821 | 0 | 0.1176 | 0.9948 | str:5821 |
| discipline.json | county | 0 | 0 | 0 | None | None | str:74033 |
| discipline.json | disciplinedatnotes | 0 | 0 | 0 | None | None | str:74033 |
| discipline.json | disciplinedenominator | 0 | 9466 | 0 | 10.0 | 30001.0 | str:74033 |
| discipline.json | disciplinenumerator | 0 | 9466 | 0 | 3.0 | 1250.0 | str:74033 |
| discipline.json | disciplinerate | 0 | 15280 | 15280 | 0.02 | 63.64 | str:74033 |
| discipline.json | districtcode | 0 | 74033 | 0 | 17415.0 | 17415.0 | str:74033 |
| discipline.json | districtorganizationid | 0 | 74033 | 0 | 100117.0 | 100117.0 | str:74033 |
| discipline.json | esdorganizationid | 0 | 74033 | 0 | 100006.0 | 100006.0 | str:74033 |
| discipline.json | gradelevel | 0 | 0 | 0 | None | None | str:74033 |
| discipline.json | organizationlevel | 0 | 0 | 0 | None | None | str:74033 |
| discipline.json | rateexcluded10daysormore | 0 | 12388 | 12388 | 0.0 | 100.0 | str:74033 |
| discipline.json | rateexcluded1dayorless | 0 | 12388 | 12388 | 0.0 | 100.0 | str:74033 |
| discipline.json | rateexcluded2to3days | 0 | 12388 | 12388 | 0.0 | 100.0 | str:74033 |
| discipline.json | rateexcluded4to5days | 0 | 12388 | 12388 | 0.0 | 100.0 | str:74033 |
| discipline.json | rateexcluded6to10days | 0 | 12388 | 12388 | 0.0 | 100.0 | str:74033 |
| discipline.json | schoolcode | 0 | 70140 | 0 | 1807.0 | 5738.0 | str:70140 |
| discipline.json | schoolorganizationid | 0 | 70140 | 0 | 101569.0 | 106971.0 | str:70140 |
| discipline.json | excluded10daysormore | 0 | 9466 | 0 | 0.0 | 152.0 | str:68232 |
| discipline.json | excluded1dayorless | 0 | 9466 | 0 | 0.0 | 320.0 | str:68232 |
| discipline.json | excluded2to3days | 0 | 9466 | 0 | 0.0 | 430.0 | str:68232 |
| discipline.json | excluded4to5days | 0 | 9466 | 0 | 0.0 | 218.0 | str:68232 |
| discipline.json | excluded6to10days | 0 | 9466 | 0 | 0.0 | 182.0 | str:68232 |
| enrollment.json | all_students | 0 | 4562 | 0 | 0.0 | 28151.0 | str:4562 |
| enrollment.json | american_indian_alaskan_native | 0 | 4562 | 0 | 0.0 | 127.0 | str:4562 |
| enrollment.json | asian | 0 | 4562 | 0 | 0.0 | 6217.0 | str:4562 |
| enrollment.json | black_african_american | 0 | 4562 | 0 | 0.0 | 3468.0 | str:4562 |
| enrollment.json | county | 0 | 0 | 0 | None | None | str:4562 |
| enrollment.json | districtcode | 0 | 4562 | 0 | 17415.0 | 17415.0 | str:4562 |
| enrollment.json | districtorganizationid | 0 | 4562 | 0 | 100117.0 | 100117.0 | str:4562 |
| enrollment.json | english_language_learners | 0 | 4562 | 0 | 0.0 | 8076.0 | str:4562 |
| enrollment.json | esdorganizationid | 0 | 4562 | 0 | 100006.0 | 100006.0 | str:4562 |
| enrollment.json | female | 0 | 4562 | 0 | 0.0 | 13564.0 | str:4562 |
| enrollment.json | gender_x | 0 | 4562 | 0 | 0.0 | 48.0 | str:4562 |
| enrollment.json | gradelevel | 0 | 0 | 0 | None | None | str:4562 |
| enrollment.json | highly_capable | 0 | 4562 | 0 | 0.0 | 3107.0 | str:4562 |
| enrollment.json | hispanic_latino_of_any_race | 0 | 4562 | 0 | 0.0 | 6364.0 | str:4562 |
| enrollment.json | homeless | 0 | 4562 | 0 | 0.0 | 631.0 | str:4562 |
| enrollment.json | low_income | 0 | 4562 | 0 | 0.0 | 15842.0 | str:4562 |
| enrollment.json | male | 0 | 4562 | 0 | 0.0 | 14720.0 | str:4562 |
| enrollment.json | migrant | 0 | 4562 | 0 | 0.0 | 56.0 | str:4562 |
| enrollment.json | military_parent | 0 | 4562 | 0 | 0.0 | 346.0 | str:4562 |
| enrollment.json | mobile | 0 | 4562 | 0 | 0.0 | 1188.0 | str:4562 |
| enrollment.json | non_english_language_learners | 0 | 4562 | 0 | 0.0 | 23075.0 | str:4562 |
| enrollment.json | non_highly_capable | 0 | 4562 | 0 | 0.0 | 27776.0 | str:4562 |
| enrollment.json | non_homeless | 0 | 4562 | 0 | 0.0 | 27759.0 | str:4562 |
| enrollment.json | non_low_income | 0 | 4562 | 0 | 0.0 | 13973.0 | str:4562 |
| enrollment.json | non_migrant | 0 | 4562 | 0 | 0.0 | 28095.0 | str:4562 |
| enrollment.json | non_military_parent | 0 | 4562 | 0 | 0.0 | 28151.0 | str:4562 |
| enrollment.json | non_mobile | 0 | 4562 | 0 | 0.0 | 27084.0 | str:4562 |
| enrollment.json | non_section_504 | 0 | 4562 | 0 | 0.0 | 26911.0 | str:4562 |
| enrollment.json | organizationlevel | 0 | 0 | 0 | None | None | str:4562 |
| enrollment.json | section_504 | 0 | 4562 | 0 | 0.0 | 1469.0 | str:4562 |
| enrollment.json | students_with_disabilities | 0 | 4562 | 0 | 0.0 | 3851.0 | str:4562 |
| enrollment.json | students_without_disabilities | 0 | 4562 | 0 | 0.0 | 25124.0 | str:4562 |
| enrollment.json | two_or_more_races | 0 | 4562 | 0 | 0.0 | 2719.0 | str:4562 |
| enrollment.json | white | 0 | 4562 | 0 | 0.0 | 10329.0 | str:4562 |
| enrollment.json | schoolcode | 0 | 4346 | 0 | 1807.0 | 5738.0 | str:4346 |
| graduation.json | county | 0 | 0 | 0 | None | None | str:9670 |
| graduation.json | districtcode | 0 | 9670 | 0 | 17415.0 | 17415.0 | str:9670 |
| graduation.json | districtorganizationid | 0 | 9670 | 0 | 100117.0 | 100117.0 | str:9670 |
| graduation.json | esdorganizationid | 0 | 9670 | 0 | 100006.0 | 100006.0 | str:9670 |
| graduation.json | organizationlevel | 0 | 0 | 0 | None | None | str:9670 |
| graduation.json | studentgrouptype | 0 | 801 | 0 | 504.0 | 504.0 | str:9670 |
| graduation.json | schoolcode | 0 | 8574 | 0 | 1807.0 | 5729.0 | str:9554 |
| graduation.json | schoolorganizationid | 0 | 8574 | 0 | 101569.0 | 106918.0 | str:9554 |
| graduation.json | graduationrate | 0 | 5585 | 0 | 0.010909090909 | 0.9894736842105 | str:9204 |
| graduation.json | graduate | 0 | 2737 | 0 | 3.0 | 1713.0 | str:8838 |
| graduation.json | transferout | 0 | 2737 | 0 | 0.0 | 349.0 | str:8838 |
| graduation.json | year4dropout | 0 | 2654 | 0 | 1.0 | 159.0 | str:8832 |
| graduation.json | year3dropout | 0 | 2414 | 0 | 1.0 | 67.0 | str:8829 |
| graduation.json | year1dropout | 0 | 1541 | 0 | 1.0 | 90.0 | str:8802 |
| graduation.json | year2dropout | 0 | 1984 | 0 | 1.0 | 35.0 | str:8798 |
| graduation.json | year5dropout | 0 | 1659 | 0 | 1.0 | 157.0 | str:8781 |
| graduation.json | year6dropout | 0 | 711 | 0 | 1.0 | 128.0 | str:8758 |
| graduation.json | year7dropout | 0 | 248 | 0 | 1.0 | 81.0 | str:8741 |
| graduation.json | finalcohort | 0 | 2472 | 0 | 10.0 | 1936.0 | str:7945 |
| graduation.json | transferin | 0 | 2455 | 0 | 1.0 | 351.0 | str:7945 |
| graduation.json | beggininggrade9 | 0 | 2504 | 0 | 9.0 | 1878.0 | str:7726 |
| graduation.json | continuing | 0 | 761 | 0 | 0.0 | 106.0 | str:2895 |
| graduation.json | dropout | 0 | 761 | 0 | 0.0 | 218.0 | str:2895 |
| graduation.json | beginninggrade9 | 0 | 233 | 0 | 24.0 | 1763.0 | str:1112 |
| graduation.json | final_cohort | 0 | 265 | 0 | 12.0 | 1881.0 | str:893 |
| graduation.json | transferredin | 0 | 262 | 0 | 1.0 | 258.0 | str:893 |
| growth.json | county | 0 | 0 | 0 | None | None | str:50665 |
| growth.json | districtcode | 0 | 50665 | 0 | 17415.0 | 17415.0 | str:50665 |
| growth.json | districtorganizationid | 0 | 50665 | 0 | 100117.0 | 100117.0 | str:50665 |
| growth.json | esdorganizationid | 0 | 50665 | 0 | 100006.0 | 100006.0 | str:50665 |
| growth.json | gradelevel | 0 | 8852 | 0 | 4.0 | 8.0 | str:50665 |
| growth.json | organizationlevel | 0 | 0 | 0 | None | None | str:50665 |
| growth.json | schoolcode | 0 | 48193 | 0 | 2565.0 | 5738.0 | str:49265 |
| growth.json | schoolorganizationid | 0 | 48193 | 0 | 101571.0 | 106971.0 | str:49265 |
| growth.json | mediansgp | 0 | 34072 | 0 | 3.0 | 97.0 | str:41864 |
| growth.json | percenthighgrowth | 0 | 33254 | 0 | 0.0 | 2532.0 | str:41864 |
| growth.json | percentlowgrowth | 0 | 33254 | 0 | 0.0 | 3030.0 | str:41864 |
| growth.json | percenttypicalgrowth | 0 | 33254 | 0 | 0.0 | 2760.0 | str:41864 |
| growth.json | numberhighgrowth | 0 | 26986 | 0 | 0.0 | 3312.0 | str:38035 |
| growth.json | numberlowgrowth | 0 | 26986 | 0 | 0.0 | 3535.0 | str:38035 |
| growth.json | numbertypicalgrowth | 0 | 26986 | 0 | 0.0 | 3298.0 | str:38035 |
| growth.json | studentcount | 0 | 26990 | 0 | 10.0 | 9594.0 | str:38035 |
| sqss.json | county | 0 | 0 | 0 | None | None | str:109011 |
| sqss.json | districtcode | 0 | 109011 | 0 | 17415.0 | 17415.0 | str:109011 |
| sqss.json | districtorganizationid | 0 | 109011 | 0 | 100117.0 | 100117.0 | str:109011 |
| sqss.json | gradelevel | 0 | 66504 | 0 | 1.0 | 12.0 | str:109011 |
| sqss.json | organizationlevel | 0 | 0 | 0 | None | None | str:109011 |
| sqss.json | schoolcode | 0 | 102985 | 0 | 1807.0 | 5738.0 | str:102985 |
| sqss.json | schoolorganizationid | 0 | 102985 | 0 | 101569.0 | 106971.0 | str:102985 |
| sqss.json | esdorganizationid | 0 | 97719 | 0 | 100006.0 | 100006.0 | str:97719 |
| sqss.json | denominator | 0 | 44695 | 0 | 0.0 | 28244.0 | str:44695 |
| sqss.json | numerator | 0 | 43749 | 0 | 0.0 | 24454.0 | str:43749 |
| sqss.json | label | 0 | 6416 | 6416 | 0.7 | 98.4 | str:11283 |
| sqss.json | percent | 0 | 7026 | 0 | 0.0069 | 0.9948 | str:7026 |
| sqss.json | percenttakingctetechprep | 0 | 6922 | 0 | 0.01 | 0.9 | str:6922 |
| sqss.json | percenttakingcollegeinth | 0 | 6823 | 0 | 0.00990099 | 0.9 | str:6823 |
| sqss.json | percenttakingap | 0 | 6791 | 0 | 0.01 | 0.9 | str:6791 |
| sqss.json | percenttakingrunningstart | 0 | 6660 | 0 | 0.006688963 | 0.7777777777777 | str:6660 |
| sqss.json | percenttakingib | 0 | 6504 | 0 | 0.004459309 | 0.7857142857142 | str:6504 |
| sqss.json | percenttakingcambridge | 0 | 6397 | 0 | 0.01 | 0.1 | str:6397 |
| sqss.json | numbertakingctetechprep | 0 | 4722 | 0 | 2.0 | 4351.0 | str:4722 |
| sqss.json | numbertakingap | 0 | 3572 | 0 | 1.0 | 2133.0 | str:3572 |
| sqss.json | numbertakingcollegeinthe | 0 | 3537 | 0 | 1.0 | 1867.0 | str:3537 |
| sqss.json | numbertakingrunningstart | 0 | 2723 | 0 | 1.0 | 1209.0 | str:2723 |
| sqss.json | apcoursenumber | 0 | 1648 | 0 | 0.0 | 971.0 | str:1648 |
| sqss.json | cambridgecoursenumber | 0 | 1648 | 0 | 0.0 | 0.0 | str:1648 |
| sqss.json | cihscoursenumber | 0 | 1648 | 0 | 0.0 | 2034.0 | str:1648 |
| sqss.json | ctecoursenumber | 0 | 1648 | 0 | 0.0 | 4105.0 | str:1648 |
| sqss.json | ibcoursenumber | 0 | 1648 | 0 | 0.0 | 234.0 | str:1648 |
| sqss.json | runningstartcoursenumber | 0 | 1648 | 0 | 0.0 | 1139.0 | str:1648 |
| sqss.json | numbertakingib | 0 | 1148 | 0 | 1.0 | 428.0 | str:1148 |
| sqss.json | apcoursepercent | 0 | 899 | 0 | 0.002 | 0.893 | str:899 |
| sqss.json | cambridgecoursepercent | 0 | 899 | 0 | 0.0 | 0.3 | str:899 |
| sqss.json | cihscoursepercent | 0 | 899 | 0 | 0.005 | 0.897 | str:899 |
| sqss.json | ctecoursepercent | 0 | 899 | 0 | 0.007 | 0.885 | str:899 |
| sqss.json | ibcoursepercent | 0 | 899 | 0 | 0.001 | 0.524 | str:899 |
| sqss.json | runningstartcoursepercent | 0 | 899 | 0 | 0.001 | 0.735 | str:899 |
| teacher_demographics.json | avgyearsexperience | 0 | 2480 | 0 | 0.0 | 43.9 | str:4745 |
| teacher_demographics.json | county | 0 | 0 | 0 | None | None | str:4745 |
| teacher_demographics.json | demographiccategoryid | 0 | 4745 | 0 | 0.0 | 12.0 | str:4745 |
| teacher_demographics.json | esdorganizationid | 0 | 4745 | 0 | 100006.0 | 100006.0 | str:4745 |
| teacher_demographics.json | iseevalidated | 0 | 4745 | 0 | 0.0 | 1.0 | str:4745 |
| teacher_demographics.json | leacode | 0 | 4745 | 0 | 17415.0 | 17415.0 | str:4745 |
| teacher_demographics.json | leaorganizationid | 0 | 4745 | 0 | 100117.0 | 100117.0 | str:4745 |
| teacher_demographics.json | ma_count | 0 | 4602 | 0 | 0.0 | 1176.0 | str:4745 |
| teacher_demographics.json | ma_percent | 0 | 2863 | 0 | 0.0 | 1.0 | str:4745 |
| teacher_demographics.json | organizationid | 0 | 4745 | 0 | 100117.0 | 106971.0 | str:4745 |
| teacher_demographics.json | organizationlevel | 0 | 0 | 0 | None | None | str:4745 |
| teacher_demographics.json | organizationlevelid | 0 | 4745 | 0 | 3.0 | 4.0 | str:4745 |
| teacher_demographics.json | rowid | 0 | 4745 | 0 | 1665.0 | 296556.0 | str:4745 |
| teacher_demographics.json | schoolcode | 0 | 4641 | 0 | 1807.0 | 5738.0 | str:4745 |
| teacher_demographics.json | schoolorganizationid | 0 | 4641 | 0 | 101569.0 | 106971.0 | str:4745 |
| teacher_demographics.json | sumyearsexperience | 0 | 2480 | 0 | 0.0 | 18718.5 | str:4745 |
| teacher_demographics.json | teachercount | 0 | 4602 | 0 | 0.0 | 1654.0 | str:4745 |
| teacher_demographics.json | teacherpercent | 0 | 4602 | 0 | 0.0 | 1.0 | str:4745 |
| teacher_demographics.json | teachertotalcount | 0 | 4602 | 0 | 1.0 | 1654.0 | str:4745 |
| teacher_experience.json | county | 0 | 0 | 0 | None | None | str:4602 |
| teacher_experience.json | esdorganizationid | 0 | 4602 | 0 | 100006.0 | 100006.0 | str:4602 |
| teacher_experience.json | iseevalidated | 0 | 4602 | 0 | 1.0 | 1.0 | str:4602 |
| teacher_experience.json | leacode | 0 | 4602 | 0 | 17415.0 | 17415.0 | str:4602 |
| teacher_experience.json | leaorganizationid | 0 | 4602 | 0 | 100117.0 | 100117.0 | str:4602 |
| teacher_experience.json | organizationid | 0 | 4602 | 0 | 100117.0 | 106971.0 | str:4602 |
| teacher_experience.json | organizationlevel | 0 | 0 | 0 | None | None | str:4602 |
| teacher_experience.json | organizationlevelid | 0 | 4602 | 0 | 3.0 | 4.0 | str:4602 |
| teacher_experience.json | rowid | 0 | 4602 | 0 | 1652.0 | 255710.0 | str:4602 |
| teacher_experience.json | schoolcode | 0 | 4498 | 0 | 1807.0 | 5738.0 | str:4602 |
| teacher_experience.json | schoolorganizationid | 0 | 4498 | 0 | 101569.0 | 106971.0 | str:4602 |
| teacher_experience.json | teachercount | 0 | 4602 | 0 | 0.0 | 499.0 | str:4602 |
| teacher_experience.json | teacherpercent | 0 | 4602 | 0 | 0.0 | 100.0 | str:4602 |
| teacher_experience.json | teachertotalcount | 0 | 4602 | 0 | 1.0 | 1654.0 | str:4602 |
| wakids.json | county | 0 | 0 | 0 | None | None | str:210202 |
| wakids.json | districtorganizationid | 0 | 210202 | 0 | 100117.0 | 100117.0 | str:210202 |
| wakids.json | esdorganizationid | 0 | 210202 | 0 | 100006.0 | 100006.0 | str:210202 |
| wakids.json | organizationid | 0 | 210202 | 0 | 100117.0 | 106750.0 | str:210202 |
| wakids.json | organizationlevel | 0 | 0 | 0 | None | None | str:210202 |
| wakids.json | washingtonstatecode | 0 | 210202 | 0 | 103300.0 | 103300.0 | str:210202 |
| wakids.json | measurevalue | 0 | 28732 | 0 | 0.0 | 6.0 | str:209108 |
| wakids.json | schoolorganizationid | 0 | 198456 | 0 | 101571.0 | 106750.0 | str:198456 |
| wakids.json | percent | 0 | 170144 | 0 | 0.00051 | 1.0 | str:170144 |
| wakids.json | denominator | 0 | 142813 | 0 | 10.0 | 2022.0 | str:142813 |
| wakids.json | numerator | 0 | 142813 | 0 | 1.0 | 1781.0 | str:142813 |
| wakids.json | developmentlevel | 0 | 0 | 0 | None | None | str:116380 |

## Provenance And Citation Recommendations

- Preserve the raw source file name and SHA-256 checksum for every staged record.
- Store dataset name, reporting year field/value, district and school identifiers, organization level, grade/subgroup/category dimensions, and a deterministic record hash.
- Add a future manifest ID and ingestion run ID before any database write so citations can point to the exact corpus snapshot and transformation run.
- Keep citations record-addressable: source file + checksum + dataset + natural key + record hash is the minimum useful citation bundle.

## Ingestion Design Recommendations

- Use a staging model that stores raw JSON records unchanged, plus extracted normalized columns for identity, time, entity, subgroup, metric, and value.
- Build dataset-specific normalized tables or views only after raw staging proves stable; the shared base dimensions are district, school, organization level, school year, grade, student/subgroup category, and metric.
- Generate a record hash from canonicalized raw JSON and store a schema version for each dataset profile.
- Treat numeric strings, percentages, suppressed values, and blank strings as normalization gates, not silent casts.
- Require pre-write validation gates: expected files, checksums, top-level array shape, record counts, required identity fields, year ranges, and suppression handling.
- Design the query layer to return both normalized values and provenance metadata so answer citations can quote the exact dataset record.

## Risks And Open Questions

- Natural keys vary by dataset; several tables need dataset-specific category/metric fields beyond district, school, and year.
- Suppression and masking conventions appear as strings and blanks; they need explicit semantic mapping before numeric analysis.
- Year fields appear stable as `schoolyear`, but downstream design should not assume every future OSPI export keeps the same name.
- Some numeric values are string-encoded and may include percentages or non-numeric placeholders.
- Metadata beyond filename/checksum is thin; future ingestion should attach a manifest ID, download/source URL metadata, and run ID.
- This profile parsed the corpus locally and read-only; it does not prove production ingestion behavior.
