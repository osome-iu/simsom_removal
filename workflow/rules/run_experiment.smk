"""
Example Snakemake workflow to run a simulation experiment.

Date: Feb 10, 2025
Author: Bao Truong

Arguments to specify
    exp_type (str): The type of experiment to run. Must be one of the following:
                    "vary_tau", "vary_group_size", "vary_illegal_probability", "vary_network_type".
    config_dir (str): The directory containing the configuration files.
    no_runs (int): The number of times to run each experiment.

Usage:
    First, make sure you have installed Snakemake. If not, you can install it via conda: 
    $ conda install -y -c bioconda snakemake-minimal

    To dry-run and print the commands that will be executed
    $ snakemake --snakefile workflow/rules/run_experiment.smk -n
    
    To run the workflow with a specific number of cores
    $ snakemake --snakefile workflow/rules/run_experiment.smk -j <no-cores>
"""

import json 

exp_type = "vary_tau"
config_dir = "experiments/config"
no_runs = 1

# Specify the number of threads to use for each run
nthreads = 7

ABS_PATH = f"{os.path.dirname(config_dir)}/{exp_type}"

# Run exps based on configurations
config_fname = os.path.join(config_dir, "all_configs.json")
EXP_IDXS = json.load(open(config_fname, "r"))[exp_type].keys()

# Make result directory
RES_DIR = os.path.join(ABS_PATH, "results")
TRACKING_DIR = os.path.join(ABS_PATH, "results_verbose")
for path in [RES_DIR, TRACKING_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

rule all:
    input: 
        results = expand(os.path.join(RES_DIR, '{exp_idx}.json'), exp_idx=EXP_IDXS)

rule run_simulation:
    input: os.path.join(config_dir, exp_type, f"{exp_idx}.json")
    output: 
        measurements = os.path.join(RES_DIR, '{exp_idx}.json'),
        tracking = os.path.join(TRACKING_DIR, '{exp_idx}.json.gz'),

    threads: nthreads
    shell: """
        python3 -m workflow.scripts.driver -o {output.measurements} -v {output.tracking} --config {input} --times {no_runs} --nthreads {nthreads}
    """