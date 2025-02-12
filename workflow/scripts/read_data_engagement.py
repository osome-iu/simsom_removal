"""
Purpose: read the results of the simulation and extract the engagement data
    Output: a dataframe of engagement data for a simulation. 
    Each row represents stats for all the messages in the simulation that belong to a particular type (illegal or legal)
    The columns are: type, no_exposures, no_reach, no_reshares, no_unique_reshares, fpath, base_name, illegal_prob
    - type (str): the type of content (illegal or legal)
    - no_exposures (list): a list of exposures, where an element the number of exposure a message received
    - no_reach (list): a list of reach, where an element is the number of unique agents that saw a message
    - fpath (str): the path to the file that contains the data    
    - s_H (float): relative size of high-risk group
    - illegal_prob (float): the illegal content probability of the system

    Usage: python read_data_engagement.py --result_path experiments/<experiment_name> --out_path data/<experiment_name>
    Results are saved in the directory of this script in the 'data' folder
    Log files are saved in the 'log' folder

Date: Feb 10, 2025
Author: Bao Truong

"""

# - no_reshares (list): a list of reshares, where an element is the number of reshares a message received
# - no_unique_reshares (list): a list of unique reshares, where an element is the number of unique agents that reshared a message

import simsom.utils as utils
import pandas as pd
import os
import glob
import argparse
import tqdm


def get_engagement(df, content_type="illegal"):
    # mean exposure for illegal messages
    focal = df[df.legality == content_type][
        ["seen_by_agents", "spread_via_agents", "id"]
    ]
    # return a series of no. exposures, no. reach, no. reshares for each message
    no_exposures = focal.seen_by_agents.apply(lambda x: len(x))
    no_reach = focal.seen_by_agents.apply(lambda x: len(set(x)))

    record = {
        "type": content_type,
        "no_exposures": no_exposures.values,
        "no_reach": no_reach.values,
    }
    engagement_df = pd.DataFrame([record])
    return engagement_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process experiment results paths.")
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Directory containing experiment results, e.g., `experiments/vary_group_size`",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Directory to save parsed results",
    )

    args = parser.parse_args()
    RES_DIR = args.result_path
    OUT_DIR = args.out_path

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    LOG_DIR = f"{OUT_DIR}/log"

    # get formatted today's date
    now = utils.get_now()
    logger = utils.get_file_logger(
        log_dir=LOG_DIR,
        full_log_path=os.path.join(LOG_DIR, f"engagement_data--{now}.log"),
        also_print=True,
    )

    dfs = []
    all_fpaths = glob.glob(f"{RES_DIR}/results_verbose/*.json.gz")
    for fpath in tqdm.tqdm(
        all_fpaths, desc=f"Processing {len(all_fpaths)} from {RES_DIR}"
    ):
        try:
            verbose = utils.read_json_compressed(fpath)
            df = pd.DataFrame(verbose["all_messages"])
            focal_dfs = []
            for content_type in ["illegal", "legal"]:
                focal_dfs.append(get_engagement(df, content_type=content_type))
            if len(focal_dfs) == 0:
                logger.info(f"\t No data for {os.path.basename(fpath)}")
                continue
            res = pd.concat(focal_dfs)

            # Save relevant experiment settings
            res["fpath"] = fpath * len(res)
            res["s_H"] = verbose["quality_settings"]["sizes"][0] * len(res)
            res["illegal_prob"] = verbose["quality_settings"][
                '"total_illegal_frac"'
            ] * len(res)
            res["moderation_half_life"] = verbose["moderation_half_life"] * len(res)

            dfs.append(res)
        except Exception as e:
            logger.info(f"\t Error reading results from file {os.path.basename(fpath)}")
            logger.info(e)
            continue
    df = pd.concat(dfs).reset_index(drop=True)

    out_fpath = f"{OUT_DIR}/{os.path.basename(RES_DIR)}__engagement.parquet"
    df.to_parquet(out_fpath)

    logger.info(f"Finish saving to {out_fpath} (len: {len(df)})!")
logger.info("All done!")
