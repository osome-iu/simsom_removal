"""
Script to run experiments based on a set of configurations

Date: Feb 10, 2025
Author: Bao Truong

Arguments:
    exp_type (str): The type of experiment to run. Must be one of the following:
                    "vary_tau", "vary_group_size", "vary_illegal_probability", "vary_network_type".
    config_dir (str): The directory containing the configuration files.
    no_runs (int): The number of times to run each experiment.

Usage:
    python3 run_exps.py <exp_type> <config_dir> <no_runs>
"""

import os
import json
import subprocess
import sys

if __name__ == "__main__":
    exp_type = sys.argv[1]
    config_dir = sys.argv[2]
    no_runs = sys.argv[3]

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
    # Specify the number of threads to use
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
            f"Running exp with -o {measurements} -v {tracking} --config {exp_config_fpath} --times {no_runs} --nthreads {nthreads}"
        )
        cmd = f"python3 workflow/scripts/driver.py -o {measurements} -v {tracking} --config {exp_config_fpath} --times {no_runs} --nthreads {nthreads}"
        subprocess.run(cmd, shell=True, check=True)
