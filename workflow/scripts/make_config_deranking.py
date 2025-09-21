""" 
Make deranking experiment configurations
- Main result: effects of varying deranking severity (deranking_severity)
- Robustness check:
    - Effects of different deranking parameters (r_half, temp)
    - Effects of group size (s_L, s_H) 
    - Effects of illegal content prevalence (illegal_content_probability) 

Date: September 21, 2025
Author: Extended from existing pipeline for deranking experiments

"""

import sys
import os
# Add the workflow/rules directory to path to import config_vals
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'libs'))

import simsom.utils as utils
import config_vals as configs
import json


def save_config_to_subdir(config, config_name, saving_dir, exp_type):
    """
    Save each exp to a .json file
    """
    output_dir = os.path.join(saving_dir, f"{exp_type}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json.dump(config, open(os.path.join(output_dir, f"{config_name}.json"), "w"))


def make_deranking_exps(saving_dir, default_config):
    """
    Create configs for deranking experiments
    Outputs:
        - a master file (.json) for all configs
        - an experiment config (.json) save to a separate directory `{saving_dir}/{exp_type}/{config_id}.json`
    """
    all_exps = {}

    ##### MAIN RESULT - VARY DERANKING SEVERITY #####
    
    EXP_TYPE = "vary_deranking_severity"
    all_exps[EXP_TYPE] = {}

    # Deranking severity values: from no deranking (1.0) to strong deranking (0.01)
    DERANKING_SEVERITIES = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01]
    
    # Default deranking parameters
    DEFAULT_R_HALF = 10  # Messages ranked 10th or higher have 50% visibility
    DEFAULT_TEMP = 2     # Temperature parameter for sigmoid steepness
    
    # Use the actual network file for all experiments
    network_file = "data/network_20.gml"

    for deranking_severity in DERANKING_SEVERITIES:
        # with deranking
        cf = {
            "infosys_gml_fpath": network_file,
            "deranking_severity": deranking_severity,
            "r_half": DEFAULT_R_HALF,
            "temp": DEFAULT_TEMP,
            "moderate": False,  # Use deranking instead of moderation
            "modeling_legality": False,  # Use quality-based deranking
        }
        # use default config for the rest of the params
        config = utils.update_dict(cf, default_config)
        config_name = f"severity_{deranking_severity}"

        all_exps[EXP_TYPE][config_name] = config
        save_config_to_subdir(config, config_name, saving_dir, EXP_TYPE)

    # baseline without deranking (severity = 1.0):
    baseline_config = config.copy()
    baseline_config["deranking_severity"] = 1.0
    config_name = f"baseline"
    all_exps[EXP_TYPE][config_name] = baseline_config
    save_config_to_subdir(baseline_config, config_name, saving_dir, EXP_TYPE)

    ##### ROBUSTNESS - VARY R_HALF (VISIBILITY THRESHOLD) #####
    
    EXP_TYPE = "vary_r_half"
    all_exps[EXP_TYPE] = {}
    
    # Test different visibility thresholds
    R_HALF_VALUES = [5, 10, 20, 50]  # Messages ranked Nth or higher have 50% visibility
    DERANKING_SEVERITIES_ROBUST = [0.1, 0.5]  # Test with moderate and strong deranking

    for r_half in R_HALF_VALUES:
        for deranking_severity in DERANKING_SEVERITIES_ROBUST:
            cf = {
                "infosys_gml_fpath": network_file,
                "deranking_severity": deranking_severity,
                "r_half": r_half,
                "temp": DEFAULT_TEMP,
                "moderate": False,
                "modeling_legality": False,  # Use quality-based deranking
            }
            config = utils.update_dict(cf, default_config)
            config_name = f"rhalf_{r_half}__severity_{deranking_severity}"
            
            all_exps[EXP_TYPE][config_name] = config
            save_config_to_subdir(config, config_name, saving_dir, EXP_TYPE)

    ##### ROBUSTNESS - VARY TEMPERATURE (SIGMOID STEEPNESS) #####
    
    EXP_TYPE = "vary_temp"
    all_exps[EXP_TYPE] = {}
    
    # Test different sigmoid steepness values
    TEMP_VALUES = [1, 3, 5, 10, 20]  # Lower = steeper sigmoid, higher = gentler sigmoid
    
    for temp in TEMP_VALUES:
        for deranking_severity in DERANKING_SEVERITIES_ROBUST:
            cf = {
                "infosys_gml_fpath": network_file,
                "deranking_severity": deranking_severity,
                "r_half": DEFAULT_R_HALF,
                "temp": temp,
                "moderate": False,
                "modeling_legality": False,  # Use quality-based deranking
            }
            config = utils.update_dict(cf, default_config)
            config_name = f"temp_{temp}__severity_{deranking_severity}"
            
            all_exps[EXP_TYPE][config_name] = config
            save_config_to_subdir(config, config_name, saving_dir, EXP_TYPE)


    ##### ROBUSTNESS - VARY QUALITY THRESHOLD WITH DERANKING #####

    EXP_TYPE = "vary_quality_threshold_deranking"
    DERANKING_SEVERITIES_QUALITY = [0.1, 0.5]  # Test moderate and strong deranking
    all_exps[EXP_TYPE] = {}

    # Test different quality thresholds for what constitutes "bad content"
    QUALITY_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]  # Different cutoffs for bad vs good content
    
    for quality_threshold in QUALITY_THRESHOLDS:
        for deranking_severity in DERANKING_SEVERITIES_QUALITY:
            cf = {
                "infosys_gml_fpath": network_file,
                "deranking_severity": deranking_severity,
                "r_half": DEFAULT_R_HALF,
                "temp": DEFAULT_TEMP,
                "moderate": False,
                "modeling_legality": False,  # Use quality-based deranking
                "quality_threshold": quality_threshold,  # Custom threshold for bad content
            }
            config = utils.update_dict(cf, default_config)

            config_name = f"severity_{deranking_severity}__threshold_{quality_threshold}"
            all_exps[EXP_TYPE][config_name] = config
            save_config_to_subdir(config, config_name, saving_dir, EXP_TYPE)

        # baseline without deranking for each quality threshold:
        baseline_config = config.copy()
        baseline_config["deranking_severity"] = 1.0
        config_name = f"baseline__threshold_{quality_threshold}"
        all_exps[EXP_TYPE][config_name] = baseline_config
        save_config_to_subdir(baseline_config, config_name, saving_dir, EXP_TYPE)

    # Save master config file
    fp = os.path.join(saving_dir, "all_deranking_configs.json")
    json.dump(all_exps, open(fp, "w"))
    print(f"Finish saving deranking configs to {fp}")


if __name__ == "__main__":
    config_dir = sys.argv[1]

    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    print("Generating configs for deranking experiments.. ")
    make_deranking_exps(config_dir, configs.INFOSYS_DEFAULT)
    print(f"Saved all deranking configs to {config_dir}")
