"""
Script to run experiments based on a set of configurations

Date: Feb 10, 2025
Author: Bao Truong

"""

import os
import json
import subprocess
import sys

if __name__ == "__main__":
    exp_type = sys.argv[1]
    config_dir = sys.argv[2]
    exp_types = [
        "vary_tau",
        "vary_group_size",
        "vary_illegal_probability",
        "vary_network_type",
    ]
    if exp_type not in exp_types:
        raise ValueError(f"Invalid experiment type. Choose from {str(exp_types)}")

    ABS_PATH = f"{os.path.dirname(config_dir)}/{exp_type}"

    # Run exps based on configurations
    config_fname = os.path.join(config_dir, "all_configs.json")
    EXPS = json.load(open(config_fname, "r"))[exp_type]
    # Specify number of runs and number of threads to use
    sim_num = 1
    nthreads = 7

    # Make result directory
    RES_DIR = os.path.join(ABS_PATH, "results")
    TRACKING_DIR = os.path.join(ABS_PATH, "results_verbose")
    for path in [RES_DIR, TRACKING_DIR]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Example usage
    for exp_idx, exp_config in EXPS.items():
        exp_config_fpath = os.path.join(config_dir, exp_type, f"{exp_idx}.json")
        measurements = os.path.join(RES_DIR, f"{exp_idx}.json")
        tracking = os.path.join(TRACKING_DIR, f"{exp_idx}.json.gz")
        print(
            f"Running exp with -o {measurements} -v {tracking} --config {exp_config_fpath} --times {sim_num} --nthreads {nthreads}"
        )
        cmd = f"python3 workflow/scripts/driver.py -o {measurements} -v {tracking} --config {exp_config_fpath} --times {sim_num} --nthreads {nthreads}"
        subprocess.run(cmd, shell=True, check=True)
