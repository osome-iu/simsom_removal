""" 
Make exp config 
- Main result: effects of varying take down delays (moderation_half_life)
- Robustness check:
    - Effects of group size (s_L, s_H) 
    - Effects of illegal content prevalence (illegal_content_probability) 
    - Effects of network type (network_type)

Date: Feb 10, 2025
Author: Bao Truong

"""

import simsom.utils as utils
import config_vals as configs
import os
import json
import sys


def save_config_to_subdir(config, config_name, saving_dir, exp_type):
    """
    Save each exp to a .json file
    """
    output_dir = os.path.join(saving_dir, f"{exp_type}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json.dump(config, open(os.path.join(output_dir, f"{config_name}.json"), "w"))


def make_exps(saving_dir, default_config, network_dir):
    """
    Create configs for exps
    Outputs:
        - a master file (.json) for all configs
        - an experiment config (.json) save to a separate directory `{saving_dir}/{exp_type}/{config_id}.json`
    """
    all_exps = {}

    ##### FOR EACH CONFIG, CREATE A BASELINE (MODERATION = FALSE) #####
    ##### MAIN RESULT - VARY TAU #####

    EXP_TYPE = "vary_tau"
    all_exps[EXP_TYPE] = {}

    illegal_frac = default_config["quality_settings"]["total_illegal_frac"]
    network_name = f"empirical_{str(illegal_frac)}"  # TODO: change this to path?

    for moderation_scale in configs.HALFLIFE:

        # with moderation
        cf = {
            "infosys_gml_fpath": os.path.join(
                network_dir, EXP_TYPE, f"{network_name}.gml"
            ),
            "quality_settings": default_config["quality_settings"],
            "illegal_probability": illegal_frac,
            "moderation_half_life": moderation_scale,
            "moderate": True,
        }
        # use default config for the rest of the params
        config = utils.update_dict(cf, default_config)
        config_name = f"{moderation_scale}"

        all_exps[EXP_TYPE][config_name] = config
        save_config_to_subdir(config, config_name, saving_dir, EXP_TYPE)

    # baseline without moderation:
    config["moderate"] = False
    config["moderation_scale"] = None
    config_name = f"baseline"
    all_exps[EXP_TYPE][config_name] = config
    save_config_to_subdir(config, config_name, saving_dir, EXP_TYPE)

    ##### ROBUSTNESS - VARY GROUP SIZE #####

    EXP_TYPE = "vary_group_size"
    TAUS = [2, 8]
    all_exps[EXP_TYPE] = {}

    for network_config in configs.GROUP_SIZE_VALS:
        for moderation_scale in TAUS:
            highrisk_frac = network_config["sizes"][0]
            network_name = f"groupsize_{str(highrisk_frac)}"
            # with moderation
            cf = {
                "infosys_gml_fpath": os.path.join(
                    network_dir, EXP_TYPE, f"{network_name}.gml"
                ),
                "quality_settings": network_config,
                "illegal_probability": illegal_frac,
                "moderation_half_life": moderation_scale,
                "moderate": True,
            }
            # use default config for the rest of the params
            config = utils.update_dict(cf, default_config)

            config_name = f"{moderation_scale}__{highrisk_frac}"
            all_exps[EXP_TYPE][config_name] = config
            save_config_to_subdir(config, config_name, saving_dir, EXP_TYPE)

        # baseline without moderation:
        config["moderate"] = False
        config["moderation_half_life"] = -1
        config_name = f"baseline__{highrisk_frac}"
        all_exps[EXP_TYPE][config_name] = config
        save_config_to_subdir(config, config_name, saving_dir, EXP_TYPE)

    ##### ROBUSTNESS - VARY ILLEGAL CONTENT PROBABILITY #####

    EXP_TYPE = "vary_illegal_probability"
    TAUS = [2, 8]
    all_exps[EXP_TYPE] = {}

    for network_config in configs.ILLEGAL_PROBABILITY_VALS:
        for moderation_scale in TAUS:
            illegal_frac = network_config["total_illegal_frac"]
            network_name = (
                f"empirical_{str(illegal_frac)}"  # TODO: change empirical -> big net?
            )
            # with moderation
            cf = {
                "infosys_gml_fpath": os.path.join(
                    network_dir, EXP_TYPE, f"{network_name}.gml"
                ),
                "quality_settings": network_config,
                "moderation_half_life": moderation_scale,
                "moderate": True,
            }
            # use default config for the rest of the params
            config = utils.update_dict(cf, default_config)

            config_name = f"{moderation_scale}__{illegal_frac}"
            all_exps[EXP_TYPE][config_name] = config
            save_config_to_subdir(config, config_name, saving_dir, EXP_TYPE)

        # baseline without moderation:
        config["moderate"] = False
        config["moderation_half_life"] = -1
        config_name = f"baseline__{illegal_frac}"
        all_exps[EXP_TYPE][config_name] = config
        save_config_to_subdir(config, config_name, saving_dir, EXP_TYPE)

    ##### ROBUSTNESS - VARY NETWORK TYPE #####
    EXP_TYPE = "vary_network_type"
    TAUS = [2, 8]
    all_exps[EXP_TYPE] = {}

    for net_name, net_config in configs.NETWORK_VALS.items():
        for moderation_scale in TAUS:
            illegal_frac = net_config["quality_settings"]["total_illegal_frac"]
            # with moderation
            cf = {
                "infosys_gml_fpath": os.path.join(
                    network_dir, EXP_TYPE, f"{net_name}_{illegal_frac}.gml"
                ),
                "moderation_half_life": moderation_scale,
                "moderate": True,
            }

            net_config.update(cf)
            # use default config for the rest of the params
            config = utils.update_dict(net_config, default_config, fill_na=False)

            config_name = f"{moderation_scale}__{net_name}"
            all_exps[EXP_TYPE][config_name] = config
            save_config_to_subdir(config, config_name, saving_dir, EXP_TYPE)

        # baseline without moderation:
        config["moderate"] = False
        config["moderation_half_life"] = -1
        config_name = f"baseline__{net_name}"
        all_exps[EXP_TYPE][config_name] = config
        save_config_to_subdir(config, config_name, saving_dir, EXP_TYPE)

    fp = os.path.join(saving_dir, "all_configs.json")
    json.dump(all_exps, open(fp, "w"))
    print(f"Finish saving config to {fp}")


if __name__ == "__main__":
    config_dir = sys.argv[1]
    network_dir = sys.argv[2]

    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    print("Generating configs for experiments.. ")
    make_exps(config_dir, configs.INFOSYS_DEFAULT, network_dir)
    print(f"Saved all configs to {config_dir}")
