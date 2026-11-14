# Reader-1 API evaluation

All documents are synthetic. These results do not measure accuracy on real Télos logs.

Answer correctness: 162/240 model responses.
API or replay errors: 0.
Exact evidence offsets: 240/240 responses (empty evidence passes this check).
Answers with evidence: 240/240 responses (abstentions exempt).
Deterministic replay: 60/60 match in status and value.
Input tokens: 9224. Estimated charge at $0.01/M input tokens: $0.00009224.

Evidence offset validity establishes text consistency, not semantic relevance. Replay tests consistency of execution, not correctness of the model-selected program.

| Category | Correct / responses |
|---|---|
| arithmetic | 20 / 20 |
| conflicts | 0 / 20 |
| corrections | 20 / 20 |
| dates | 20 / 20 |
| irrelevant_instructions | 18 / 20 |
| lists | 20 / 20 |
| missing | 11 / 20 |
| percentages | 0 / 20 |
| scalar | 20 / 20 |
| table_lookup | 8 / 20 |
| unicode | 20 / 20 |
| units | 5 / 20 |

| Case | Correct | Status | Value | Confidence | Client ms |
|---|---|---|---|---|---|
| arithmetic-01 | True | ANSWER | "19" | 0.9993 | 1360.12 |
| arithmetic-02 | True | ANSWER | "9" | 0.9995 | 1487.42 |
| arithmetic-03 | True | ANSWER | "23" | 0.9991 | 1470.76 |
| arithmetic-04 | True | ANSWER | "9" | 0.9996 | 1394.4 |
| arithmetic-05 | True | ANSWER | "27" | 0.9991 | 1158.02 |
| arithmetic-06 | True | ANSWER | "9" | 0.9996 | 1423.86 |
| arithmetic-07 | True | ANSWER | "31" | 0.9992 | 1274.94 |
| arithmetic-08 | True | ANSWER | "9" | 0.9995 | 1379.92 |
| arithmetic-09 | True | ANSWER | "35" | 0.9989 | 1336.8 |
| arithmetic-10 | True | ANSWER | "9" | 0.9997 | 1098.6 |
| arithmetic-11 | True | ANSWER | "39" | 0.9995 | 1047.09 |
| arithmetic-12 | True | ANSWER | "9" | 0.9991 | 1349.36 |
| arithmetic-13 | True | ANSWER | "43" | 0.9993 | 1183.45 |
| arithmetic-14 | True | ANSWER | "9" | 0.9997 | 1087.65 |
| arithmetic-15 | True | ANSWER | "47" | 0.9994 | 1295.34 |
| arithmetic-16 | True | ANSWER | "9" | 0.9995 | 1145.04 |
| arithmetic-17 | True | ANSWER | "51" | 0.9992 | 1000.32 |
| arithmetic-18 | True | ANSWER | "9" | 0.9996 | 1134.47 |
| arithmetic-19 | True | ANSWER | "55" | 0.999 | 1373.92 |
| arithmetic-20 | True | ANSWER | "9" | 0.9997 | 1112.22 |
| conflicts-01 | False | ANSWER | "1.4" | 0.8773 | 1195.94 |
| conflicts-02 | False | ANSWER | {"amount": "6", "currency": "USD"} | 0.6484 | 778.1 |
| conflicts-03 | False | ANSWER | "1.6" | 0.5568 | 1217.15 |
| conflicts-04 | False | ABSENT | null | 0.8473 | 1006.65 |
| conflicts-05 | False | ANSWER | "1.8" | 0.5343 | 1026.36 |
| conflicts-06 | False | ANSWER | {"amount": "10", "currency": "USD"} | 0.5144 | 810.63 |
| conflicts-07 | False | ANSWER | "2" | 0.9062 | 921.31 |
| conflicts-08 | False | ANSWER | {"amount": "12", "currency": "USD"} | 0.5357 | 1037.19 |
| conflicts-09 | False | ANSWER | "2.2" | 0.931 | 1114.47 |
| conflicts-10 | False | ANSWER | {"amount": "14", "currency": "USD"} | 0.5668 | 941.99 |
| conflicts-11 | False | ANSWER | "2.4" | 0.9255 | 841.06 |
| conflicts-12 | False | ANSWER | {"amount": "16", "currency": "USD"} | 0.6236 | 823.5 |
| conflicts-13 | False | ANSWER | "2.6" | 0.9564 | 765.01 |
| conflicts-14 | False | ANSWER | {"amount": "27", "currency": "USD"} | 0.4812 | 956.26 |
| conflicts-15 | False | ANSWER | "2.8" | 0.9195 | 856.01 |
| conflicts-16 | False | ANSWER | {"amount": "29", "currency": "USD"} | 0.5678 | 821.07 |
| conflicts-17 | False | ANSWER | "3" | 0.9585 | 889.25 |
| conflicts-18 | False | ANSWER | {"amount": "22", "currency": "USD"} | 0.4795 | 851.38 |
| conflicts-19 | False | ANSWER | "3.2" | 0.937 | 875.93 |
| conflicts-20 | False | ANSWER | {"amount": "24", "currency": "USD"} | 0.5568 | 914.3 |
| corrections-01 | True | ANSWER | "0.5" | 0.9912 | 879.16 |
| corrections-02 | True | ANSWER | {"amount": "15", "currency": "USD"} | 0.9992 | 879.15 |
| corrections-03 | True | ANSWER | "0.7" | 0.9958 | 891.24 |
| corrections-04 | True | ANSWER | {"amount": "17", "currency": "USD"} | 0.999 | 920.29 |
| corrections-05 | True | ANSWER | "0.9" | 0.9939 | 1012.24 |
| corrections-06 | True | ANSWER | {"amount": "19", "currency": "USD"} | 0.9991 | 865.36 |
| corrections-07 | True | ANSWER | "1.1" | 0.9849 | 817.24 |
| corrections-08 | True | ANSWER | {"amount": "21", "currency": "USD"} | 0.999 | 868.45 |
| corrections-09 | True | ANSWER | "1.3" | 0.9902 | 863.57 |
| corrections-10 | True | ANSWER | {"amount": "23", "currency": "USD"} | 0.9991 | 832.41 |
| corrections-11 | True | ANSWER | "1.5" | 0.9883 | 793.36 |
| corrections-12 | True | ANSWER | {"amount": "25", "currency": "USD"} | 0.9989 | 988.86 |
| corrections-13 | True | ANSWER | "1.7" | 0.9897 | 823.65 |
| corrections-14 | True | ANSWER | {"amount": "27", "currency": "USD"} | 0.9989 | 825.33 |
| corrections-15 | True | ANSWER | "1.9" | 0.9905 | 946.03 |
| corrections-16 | True | ANSWER | {"amount": "29", "currency": "USD"} | 0.999 | 981.16 |
| corrections-17 | True | ANSWER | "2.1" | 0.9861 | 810.85 |
| corrections-18 | True | ANSWER | {"amount": "31", "currency": "USD"} | 0.9989 | 930.8 |
| corrections-19 | True | ANSWER | "2.3" | 0.9702 | 870.31 |
| corrections-20 | True | ANSWER | {"amount": "33", "currency": "USD"} | 0.9986 | 836.04 |
| dates-01 | True | ANSWER | "4" | 0.9981 | 1180.4 |
| dates-02 | True | ANSWER | "2024-02-27" | 0.9657 | 903.97 |
| dates-03 | True | ANSWER | "6" | 0.9983 | 1088.56 |
| dates-04 | True | ANSWER | "2024-03-02" | 0.9775 | 925.64 |
| dates-05 | True | ANSWER | "8" | 0.9992 | 1098.95 |
| dates-06 | True | ANSWER | "2024-03-06" | 0.9796 | 896.19 |
| dates-07 | True | ANSWER | "10" | 0.9993 | 1040.78 |
| dates-08 | True | ANSWER | "2024-03-10" | 0.9786 | 808.67 |
| dates-09 | True | ANSWER | "12" | 0.9985 | 1234.57 |
| dates-10 | True | ANSWER | "2024-03-14" | 0.9833 | 1167.11 |
| dates-11 | True | ANSWER | "14" | 0.9993 | 1013.67 |
| dates-12 | True | ANSWER | "2024-03-18" | 0.978 | 892.79 |
| dates-13 | True | ANSWER | "16" | 0.9991 | 1076.02 |
| dates-14 | True | ANSWER | "2024-03-22" | 0.9741 | 979.5 |
| dates-15 | True | ANSWER | "18" | 0.9993 | 1029.13 |
| dates-16 | True | ANSWER | "2024-03-26" | 0.9753 | 893.18 |
| dates-17 | True | ANSWER | "20" | 0.9993 | 1068.34 |
| dates-18 | True | ANSWER | "2024-03-30" | 0.976 | 897.28 |
| dates-19 | True | ANSWER | "22" | 0.9994 | 1046.5 |
| dates-20 | True | ANSWER | "2024-04-03" | 0.9815 | 843.75 |
| irrelevant_instructions-01 | True | ANSWER | "1.4" | 0.8479 | 837.84 |
| irrelevant_instructions-02 | True | ANSWER | "1.5" | 0.8021 | 877.77 |
| irrelevant_instructions-03 | False | ABSENT | null | 0.7452 | 995.28 |
| irrelevant_instructions-04 | True | ANSWER | "1.7" | 0.8893 | 771.03 |
| irrelevant_instructions-05 | False | ABSENT | null | 0.8539 | 1069.06 |
| irrelevant_instructions-06 | True | ANSWER | "1.9" | 0.8376 | 903.72 |
| irrelevant_instructions-07 | True | ANSWER | "2" | 0.8836 | 855.1 |
| irrelevant_instructions-08 | True | ANSWER | "2.1" | 0.9545 | 803.56 |
| irrelevant_instructions-09 | True | ANSWER | "2.2" | 0.9479 | 881.81 |
| irrelevant_instructions-10 | True | ANSWER | "2.3" | 0.9525 | 869.66 |
| irrelevant_instructions-11 | True | ANSWER | "2.4" | 0.8992 | 858.92 |
| irrelevant_instructions-12 | True | ANSWER | "2.5" | 0.9454 | 929.12 |
| irrelevant_instructions-13 | True | ANSWER | "2.6" | 0.886 | 797.2 |
| irrelevant_instructions-14 | True | ANSWER | "2.7" | 0.9422 | 888.0 |
| irrelevant_instructions-15 | True | ANSWER | "2.8" | 0.9073 | 923.94 |
| irrelevant_instructions-16 | True | ANSWER | "2.9" | 0.9233 | 915.25 |
| irrelevant_instructions-17 | True | ANSWER | "3" | 0.7713 | 867.67 |
| irrelevant_instructions-18 | True | ANSWER | "3.1" | 0.9576 | 835.12 |
| irrelevant_instructions-19 | True | ANSWER | "3.2" | 0.6846 | 939.03 |
| irrelevant_instructions-20 | True | ANSWER | "3.3" | 0.9629 | 1215.46 |
| lists-01 | True | ANSWER | ["alpha-1", "beta-1", "gamma-1"] | 0.9831 | 918.22 |
| lists-02 | True | ANSWER | ["alpha-2", "beta-2", "gamma-2"] | 0.9981 | 976.36 |
| lists-03 | True | ANSWER | ["alpha-3", "beta-3", "gamma-3"] | 0.9919 | 840.39 |
| lists-04 | True | ANSWER | ["alpha-4", "beta-4", "gamma-4"] | 0.9982 | 949.71 |
| lists-05 | True | ANSWER | ["alpha-5", "beta-5", "gamma-5"] | 0.9657 | 872.63 |
| lists-06 | True | ANSWER | ["alpha-6", "beta-6", "gamma-6"] | 0.9977 | 1280.8 |
| lists-07 | True | ANSWER | ["alpha-7", "beta-7", "gamma-7"] | 0.9691 | 928.84 |
| lists-08 | True | ANSWER | ["alpha-8", "beta-8", "gamma-8"] | 0.9982 | 783.87 |
| lists-09 | True | ANSWER | ["alpha-9", "beta-9", "gamma-9"] | 0.9779 | 1176.39 |
| lists-10 | True | ANSWER | ["alpha-10", "beta-10", "gamma-10"] | 0.9981 | 978.62 |
| lists-11 | True | ANSWER | ["alpha-11", "beta-11", "gamma-11"] | 0.9913 | 936.28 |
| lists-12 | True | ANSWER | ["alpha-12", "beta-12", "gamma-12"] | 0.9983 | 856.21 |
| lists-13 | True | ANSWER | ["alpha-13", "beta-13", "gamma-13"] | 0.9895 | 880.5 |
| lists-14 | True | ANSWER | ["alpha-14", "beta-14", "gamma-14"] | 0.9983 | 925.64 |
| lists-15 | True | ANSWER | ["alpha-15", "beta-15", "gamma-15"] | 0.9779 | 1120.83 |
| lists-16 | True | ANSWER | ["alpha-16", "beta-16", "gamma-16"] | 0.9985 | 859.26 |
| lists-17 | True | ANSWER | ["alpha-17", "beta-17", "gamma-17"] | 0.9909 | 924.1 |
| lists-18 | True | ANSWER | ["alpha-18", "beta-18", "gamma-18"] | 0.9983 | 906.8 |
| lists-19 | True | ANSWER | ["alpha-19", "beta-19", "gamma-19"] | 0.976 | 1028.32 |
| lists-20 | True | ANSWER | ["alpha-20", "beta-20", "gamma-20"] | 0.9982 | 817.66 |
| missing-01 | False | ANSWER | "1.4" | 0.9736 | 921.27 |
| missing-02 | True | ABSENT | null | 0.9239 | 1407.2 |
| missing-03 | True | ABSENT | null | 0.7324 | 1234.6 |
| missing-04 | True | ABSENT | null | 0.9139 | 1482.68 |
| missing-05 | False | ANSWER | "1.8" | 0.5246 | 930.51 |
| missing-06 | True | ABSENT | null | 0.9245 | 1354.92 |
| missing-07 | False | ANSWER | "2" | 0.981 | 768.78 |
| missing-08 | True | ABSENT | null | 0.9045 | 1328.92 |
| missing-09 | False | ANSWER | "2.2" | 0.9897 | 978.78 |
| missing-10 | True | ABSENT | null | 0.9091 | 1393.63 |
| missing-11 | False | ANSWER | "2.4" | 0.9828 | 784.54 |
| missing-12 | True | ABSENT | null | 0.9118 | 1359.68 |
| missing-13 | False | ANSWER | "2.6" | 0.9817 | 851.08 |
| missing-14 | True | ABSENT | null | 0.9032 | 1460.61 |
| missing-15 | False | ANSWER | "2.8" | 0.9904 | 911.78 |
| missing-16 | True | ABSENT | null | 0.9102 | 1266.68 |
| missing-17 | False | ANSWER | "3" | 0.9869 | 818.19 |
| missing-18 | True | ABSENT | null | 0.9002 | 1318.24 |
| missing-19 | False | ANSWER | "3.2" | 0.9782 | 907.13 |
| missing-20 | True | ABSENT | null | 0.9089 | 1393.43 |
| percentages-01 | False | ANSWER | "120" | 0.9718 | 926.18 |
| percentages-02 | False | ANSWER | "35" | 0.9813 | 771.59 |
| percentages-03 | False | ANSWER | "160" | 0.9502 | 859.78 |
| percentages-04 | False | ANSWER | "45" | 0.9792 | 929.77 |
| percentages-05 | False | ANSWER | "200" | 0.971 | 883.33 |
| percentages-06 | False | ANSWER | "55" | 0.9863 | 1002.5 |
| percentages-07 | False | ANSWER | "240" | 0.9737 | 829.33 |
| percentages-08 | False | ANSWER | "65" | 0.9781 | 934.37 |
| percentages-09 | False | ANSWER | "280" | 0.9715 | 898.53 |
| percentages-10 | False | ANSWER | "75" | 0.9823 | 841.01 |
| percentages-11 | False | ANSWER | "320" | 0.9641 | 979.1 |
| percentages-12 | False | ANSWER | "85" | 0.9796 | 794.39 |
| percentages-13 | False | ANSWER | "360" | 0.9772 | 850.19 |
| percentages-14 | False | ANSWER | "95" | 0.9761 | 835.12 |
| percentages-15 | False | ANSWER | "400" | 0.9496 | 916.15 |
| percentages-16 | False | ANSWER | "105" | 0.9747 | 867.51 |
| percentages-17 | False | ANSWER | "440" | 0.9654 | 772.1 |
| percentages-18 | False | ANSWER | "115" | 0.9824 | 966.11 |
| percentages-19 | False | ANSWER | "480" | 0.9772 | 1232.01 |
| percentages-20 | False | ANSWER | "125" | 0.9815 | 867.05 |
| scalar-01 | True | ANSWER | "1.4" | 0.9907 | 936.09 |
| scalar-02 | True | ANSWER | "0.6" | 0.9963 | 793.61 |
| scalar-03 | True | ANSWER | "1.6" | 0.9958 | 886.57 |
| scalar-04 | True | ANSWER | "0.8" | 0.9957 | 851.04 |
| scalar-05 | True | ANSWER | "1.8" | 0.994 | 952.46 |
| scalar-06 | True | ANSWER | "1" | 0.998 | 892.66 |
| scalar-07 | True | ANSWER | "2" | 0.9954 | 900.92 |
| scalar-08 | True | ANSWER | "1.2" | 0.9964 | 790.15 |
| scalar-09 | True | ANSWER | "2.2" | 0.9951 | 886.76 |
| scalar-10 | True | ANSWER | "1.4" | 0.9958 | 1342.65 |
| scalar-11 | True | ANSWER | "2.4" | 0.992 | 916.23 |
| scalar-12 | True | ANSWER | "1.6" | 0.9959 | 913.09 |
| scalar-13 | True | ANSWER | "2.6" | 0.9944 | 1012.38 |
| scalar-14 | True | ANSWER | "1.8" | 0.9954 | 871.54 |
| scalar-15 | True | ANSWER | "2.8" | 0.9924 | 972.0 |
| scalar-16 | True | ANSWER | "2" | 0.9962 | 897.68 |
| scalar-17 | True | ANSWER | "3" | 0.9937 | 889.31 |
| scalar-18 | True | ANSWER | "2.2" | 0.9951 | 865.08 |
| scalar-19 | True | ANSWER | "3.2" | 0.9921 | 821.26 |
| scalar-20 | True | ANSWER | "2.4" | 0.9954 | 876.21 |
| table_lookup-01 | True | ANSWER | "8" | 0.3982 | 975.75 |
| table_lookup-02 | False | ABSENT | null | 0.7454 | 1129.55 |
| table_lookup-03 | True | ANSWER | "10" | 0.3796 | 892.53 |
| table_lookup-04 | False | ABSENT | null | 0.8458 | 1056.44 |
| table_lookup-05 | True | ANSWER | "12" | 0.4572 | 851.07 |
| table_lookup-06 | False | ABSENT | null | 0.8717 | 1149.24 |
| table_lookup-07 | True | ANSWER | "14" | 0.4437 | 884.87 |
| table_lookup-08 | False | ABSENT | null | 0.8046 | 1034.19 |
| table_lookup-09 | True | ANSWER | "16" | 0.4947 | 781.86 |
| table_lookup-10 | False | ABSENT | null | 0.8428 | 1302.84 |
| table_lookup-11 | False | ANSWER | "15" | 0.4705 | 748.68 |
| table_lookup-12 | False | ABSENT | null | 0.6849 | 1023.78 |
| table_lookup-13 | True | ANSWER | "20" | 0.5398 | 1033.85 |
| table_lookup-14 | False | ABSENT | null | 0.8814 | 1121.76 |
| table_lookup-15 | True | ANSWER | "22" | 0.4786 | 839.78 |
| table_lookup-16 | False | ABSENT | null | 0.8878 | 1095.05 |
| table_lookup-17 | True | ANSWER | "24" | 0.4644 | 904.19 |
| table_lookup-18 | False | ABSENT | null | 0.8636 | 1064.05 |
| table_lookup-19 | False | ANSWER | "23" | 0.4595 | 845.7 |
| table_lookup-20 | False | ABSENT | null | 0.8275 | 1107.41 |
| unicode-01 | True | ANSWER | "1.4" | 0.8757 | 1015.65 |
| unicode-02 | True | ANSWER | "1500" | 0.9927 | 830.44 |
| unicode-03 | True | ANSWER | "1.6" | 0.5364 | 842.58 |
| unicode-04 | True | ANSWER | "1700" | 0.9918 | 1199.88 |
| unicode-05 | True | ANSWER | "1.8" | 0.8879 | 1017.37 |
| unicode-06 | True | ANSWER | "1900" | 0.9851 | 919.61 |
| unicode-07 | True | ANSWER | "2" | 0.9831 | 822.86 |
| unicode-08 | True | ANSWER | "2100" | 0.9899 | 1084.59 |
| unicode-09 | True | ANSWER | "2.2" | 0.9805 | 855.35 |
| unicode-10 | True | ANSWER | "2300" | 0.9897 | 1025.86 |
| unicode-11 | True | ANSWER | "2.4" | 0.9508 | 940.09 |
| unicode-12 | True | ANSWER | "2500" | 0.9935 | 999.27 |
| unicode-13 | True | ANSWER | "2.6" | 0.9687 | 1108.59 |
| unicode-14 | True | ANSWER | "2700" | 0.989 | 918.29 |
| unicode-15 | True | ANSWER | "2.8" | 0.9589 | 1026.37 |
| unicode-16 | True | ANSWER | "2900" | 0.9904 | 900.44 |
| unicode-17 | True | ANSWER | "3" | 0.9639 | 958.21 |
| unicode-18 | True | ANSWER | "3100" | 0.9947 | 908.97 |
| unicode-19 | True | ANSWER | "3.2" | 0.9593 | 993.61 |
| unicode-20 | True | ANSWER | "3300" | 0.9942 | 931.01 |
| units-01 | False | ANSWER | "1" | 0.5649 | 1160.33 |
| units-02 | False | INCOMPLETE | null | 0.9708 | 1366.56 |
| units-03 | True | ANSWER | "251" | 0.8915 | 924.33 |
| units-04 | False | INCOMPLETE | null | 0.9484 | 1460.47 |
| units-05 | False | ANSWER | "1" | 0.6697 | 1014.29 |
| units-06 | False | INCOMPLETE | null | 0.9618 | 1476.43 |
| units-07 | True | ANSWER | "319" | 0.7314 | 1045.49 |
| units-08 | False | INCOMPLETE | null | 0.8845 | 1261.66 |
| units-09 | True | ANSWER | "353" | 0.9583 | 1397.35 |
| units-10 | False | INCOMPLETE | null | 0.8794 | 1273.71 |
| units-11 | False | ANSWER | "1" | 0.615 | 998.01 |
| units-12 | False | INCOMPLETE | null | 0.9005 | 1357.89 |
| units-13 | True | ANSWER | "421" | 0.8065 | 907.94 |
| units-14 | False | INCOMPLETE | null | 0.9143 | 1337.02 |
| units-15 | False | ANSWER | "1" | 0.8504 | 980.41 |
| units-16 | False | INCOMPLETE | null | 0.9214 | 1549.05 |
| units-17 | False | ANSWER | "0.5" | 0.8569 | 1119.82 |
| units-18 | False | ANSWER | "1" | 0.7657 | 1153.59 |
| units-19 | True | ANSWER | "523" | 0.8237 | 1078.22 |
| units-20 | False | INCOMPLETE | null | 0.9718 | 2349.56 |

Raw responses, evidence and replay outputs: extended-run.json.
Expected answers were specified in extended_cases.json before this run. Missing-value tests accept ABSENT or UNKNOWN; the unresolved-conflict test accepts CONFLICT or AMBIGUOUS.
Single pass per case; small hand-authored suite, no general benchmark claim.
