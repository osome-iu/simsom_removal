# SimSoM: A <ins>Sim</ins>ulator of <ins>So</ins>cial <ins>M</ins>edia

This repository contains code to reproduce the results in the paper [*Vulnerabilities of the Online Public Square to Manipulation*](https://arxiv.org/abs/1907.06130) by [Bao Tran Truong](https://btrantruong.github.io/), Xiaodan Lou, [Alessandro Flammini](https://cnets.indiana.edu/aflammin/), and [Filippo Menczer](https://cnets.indiana.edu/fil/).

## Overview of the repo
1. `data`: contains raw & derived datasets
2. `example`: contains a minimal example to start using the SimSoM model
3. `simsom`: the package for the SimSoM model that can be imported into scripts
4. `experiments`: experiment results and .ipynb noteboooks to produce figures reported in the paper
5. `workflow`: workflow files (Snakemake rules) and scripts

## Install 

- This code is written and tested with **Python>=3.6**
- We use `conda`, a package manager to manage the development environment. Please make sure you have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) or [mamba](https://mamba.readthedocs.io/en/latest/installation.html#) installed on your machine

### Using Make (recommended)

To set up the environment and install the model: run `make` from the project directory (`SimSoM`)
### Using Conda

1. Create the environment with required packages: run `conda env create -n simsom -f environment.yml` to 
2. Install the `SimSoM` module: 
    - activate virtualenv: `conda activate simsom`
    - run `pip install -e ./libs/`

## Data

The empirical network is created from the [Replication Data](https://doi.org/10.7910/DVN/6CZHH5) for: [Right and left, partisanship predicts vulnerability to misinformation](https://doi.org/10.37016/mr-2020-55),
where: 
- `measures.tab` contains user information, i.e., one's partisanship and misinformation score. 
- `anonymized-friends.json` is the adjacency list. 

We reconstruct the empirical network from the above 2 files, resulting in `data/follower_network.gml`. The steps are specified in the [script to create empirical network](workflow/make_network.py)

## Running the code

Check out `example` to get started. 
- Example of the simulation and results: `example/run_simulation.ipynb`


### Reproduce results from the paper:

1. From the root directory, unzip the data file: `unzip data/data.zip -d .`
2. Create config files specifying parameters for simulations: `workflow/scripts/make_finalconfig.py`
    - See `example/data/config.json` for example of a config file
3. Run a Snakemake rule corresponding to the simulations of interest. 
    - e.g.: `workflow/rules/shuffle_network.smk` runs simulations on different shuffled version of the empirical network

### Notes
The results in the paper are based on averages across multiple simulation runs. To reproduce those results, we suggest running the simulations in parallel, for example on a cluster, since they will need a lot of memory and CPU time.
