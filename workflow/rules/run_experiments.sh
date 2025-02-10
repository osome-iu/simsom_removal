# !/bin/bash

# Purpose: Automatically run all experiments to reproduce the results in the paper.
# 1. Create configuration folders for all experiments.
# 2. Run the run_exps.py script with an argument to specify the experiment to run: 
#     - vary_tau: main results
#     - vary_group_size
#     - vary_illegal_probability
#     - vary_network_type 

# NOTE: Must be run from the project root.

# Usage: 
#     - make file executable: chmod +x run_experiment.sh 
#     - call file: workflow/rules/run_experiments.sh

# Date: Feb 10, 2025
# Author: Bao Truong


### Constants
ABS_PATH=$"experiments"
CONFIG_DIR=$"${ABS_PATH}/config"
DATA_DIR=$"data"

# Generate configurations
python3 workflow/rules/make_config_illegal_removal.py $CONFIG_DIR $DATA_DIR

# Call the Python script for each set of experiments 
EXPS=("vary_tau" "vary_group_size" "vary_illegal_probability" "vary_network_type")

# Loop through each experiment type and call the Python script
for exp in "${EXPS[@]}"; do
  echo "** Run exp $exp **"
  python3 workflow/rules/run_exps.py "$exp" "$CONFIG_DIR"
done

# echo "Run "
# python3 workflow/rules/run_exps.py vary_tau $CONFIG_DIR
# python3 workflow/rules/run_exps.py vary_group_size $CONFIG_DIR
# python3 workflow/rules/run_exps.py vary_illegal_probability $CONFIG_DIR
# python3 workflow/rules/run_exps.py vary_network_type $CONFIG_DIR