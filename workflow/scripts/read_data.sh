#!/bin/bash

echo '###### main_engagement ######'
python3 /N/u/baotruon/BigRed200/carisma/experiments/20241120_main_results/0.read_data_engagement.py --result_path /N/project/simsom/carisma/20241112_main_results --out_path /N/u/baotruon/BigRed200/carisma/experiments/20241120_main_results/engagement_20241112 --suffix 0.01

echo '###### prevalence_main ######'
python3 /N/u/baotruon/BigRed200/carisma/experiments/20241120_main_results/1.read_data_illegal_count.py --result_path /N/project/simsom/carisma/20241112_main_results --out_path /N/u/baotruon/BigRed200/carisma/experiments/20241120_main_results/prevalence_20241112 --suffix 0.01
