# Explore the effects of takedown delay on the persistence of illegal content

This repository contains code to reproduce the results in the paper ``Delayed takedown of illegal content on social media makes moderation ineffective''.

The model is an extension of [SimSoM: A <ins>Sim</ins>ulator of <ins>So</ins>cial <ins>M</ins>edia](https://github.com/osome-iu/SimSoM/)

## Overview of the repo
1. `data`: contains raw & derived datasets
2. `example`: contains a minimal example to start using the SimSoM model
3. `libs`: contains the extended SimSoM model package that can be imported into scripts
4. `experiments`: experiment configurations, results, supplementary data and .ipynb noteboooks to produce figures reported in the paper
5. `workflow`: scripts to run simulation and Snakemake rules to run sets of experiments

## 1. Install SimSoM

We include two ways to set up the environment and install the model

#### 1. Using Make (the simpliest way --- recommended)

Run `make` from the project directory (`SimSoM`)

#### 2. Using Conda

We use `conda`, a package manager to manage the development environment. Please make sure you have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) or [mamba](https://mamba.readthedocs.io/en/latest/installation.html#) installed on your machine

1. Create the environment with required packages: run `conda env create -n simsom -f environment.yml` to 
2. Install the `SimSoM` module: 
    - activate virtualenv: `conda activate simsom`
    - run `pip install -e ./libs/`

## 2. Reproduce results from the paper:

All scripts are run from the project root directory, `simsom_removal`

### 1. Run experiments 
1. From the root directory, unzip the data file: `unzip data/data.zip -d .`
2. Automatically run all experiments to reproduce the results in the paper by running 2 commands:
    - make file executable: `chmod +x workflow/rules/run_experiment.sh` 
    - run shell script: `workflow/rules/run_experiments.sh`

    This script does 2 things 
    1. Create configuration folders for all experiments (see `experiments/config` for the results of this step)

    2. Run the `run_exps.py` script with an argument to specify the experiment to run: 
        - vary_tau: main results
        - vary_group_size: robustness check for varying group sizes
        - vary_illegal_probability: robustness check for varying illegal probabilities 
        - vary_network_type: robustness check for varying network structures

### 2. Parse experiment data 
We are interested in the prevalence of illegal content and engagement metrics such as reach and impression. To aggregate these metrics, we need to parse the experiment verbose tracking files. 
To parse these files, run:
- For reach and impression: `python workflow/scripts/read_data_engagement.py --result_path experiments/<experiment_name> --out_path data/<experiment_name>` 
- For prevalence of illegal content: `python read_data_illegal_count.py --result_path experiments/<experiment_name> --out_path data/<experiment_name>`

### 3. Plot results 
- Run step 2 above to parse result, setting `out_path` to `experiments/figures/data/<experiment_name>`
- Run the notebooks in `experiments/figures` to visualize the experiment results in the paper 

## Other notes

### Data description

The empirical network is created from the [Replication Data](https://doi.org/10.7910/DVN/6CZHH5) for: [Right and left, partisanship predicts vulnerability to misinformation](https://doi.org/10.37016/mr-2020-55),
where: 
- `measures.tab` contains user information, i.e., one's partisanship and misinformation score. 
- `anonymized-friends.json` is the adjacency list. 

We reconstruct the empirical network from the above 2 files, resulting in `data/follower_network.gml`. The steps are specified in the [script to create empirical network](workflow/make_network.py)

### Step-by-step instruction and example of running SimSoM

Check out `example` to get started. 
- Example of the simulation and results: `example/run_simulation.ipynb`

### Troubleshooting


- SimSoM was written and tested with **Python>=3.6**
- The results in the paper are based on averages across multiple simulation runs. To reproduce those results, we suggest running the simulations in parallel, for example on a cluster, since they will need a lot of memory and CPU time.
