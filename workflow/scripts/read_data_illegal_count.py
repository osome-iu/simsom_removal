"""
Purpose: read the results of the simulation and extract the engagement data
    Save results from each folder into a single parquet file (to avoid memory issues)
    Output: a dataframe of engagement data for a simulation. 
    Each row represents stats for all the messages in the simulation that belong to a particular type (illegal or legal)
    The columns are: illegal_frac, unique_illegal_frac, illegal_count, unique_illegal_count, fpath, moderation_half_life, quality, no_steps, base_name, illegal_prob
    - illegal_frac (float): the fraction of illegal content over all content
    - unique_illegal_frac (float): the fraction of unique illegal content over all unique content
    - illegal_count (int): the number of illegal content
    - unique_illegal_count (int): the number of unique illegal content
    - moderation_half_life (float): illegal content half-life
    - quality (float): the quality of the system
    - no_steps (int): the number of steps to convergence
    - fpath (str): the path to the file that contains the data    
    - s_H (float): relative size of high-risk group
    - illegal_prob (float): the illegal content probability of the system

    Usage:
    python read_data_illegal_count.py --result_path experiments/<experiment_name> --out_path data/<experiment_name>
    Log files are saved in the '{out_path}/log' folder
"""

import simsom.utils as utils
import numpy as np
import pandas as pd
import os
import glob
import argparse
import tqdm

PARAMS = [
    "moderation_half_life",
    "quality",
    "no_steps",
]


def get_illegal_prevalence(data):
    """
    Gets relevant stats to calculate illegal prevalence: the fraction of illegal messages over all messages.
    We report the relative illegal prevalence, which is the ratio between the two numbers in the scenario when mod=True over mod=False

    Args:
        data: dictionary containing verbose results as returned by the simulator
        Note: data needs to have at least the following keys: all_messages, feeds_message_ids, feeds_shares, feeds_ages

    Returns:
        Df containing the following columns: 'illegal_frac', 'unique_illegal_frac', 'illegal_count', 'unique_illegal_count'

    """
    import itertools

    try:
        df = pd.DataFrame.from_records(data["all_messages"])
        illegal_ids = list(df[df.legality == "illegal"]["id"])
        logger.info(f"\t\tNo. illegal content created: {len(illegal_ids)}")
        feed_lists = [i for i in data["feeds_message_ids"].values()]
        alive_content = []
        map(alive_content.extend, feed_lists)
        alive_content = list(itertools.chain.from_iterable(feed_lists))
        logger.info(f"\t\tNo. alive content: {len(alive_content)}")

        result = pd.DataFrame()
        # if somehow there are no illegal content or no alive content, return -1
        if len(illegal_ids) == 0 or len(alive_content) == 0:
            return result
        else:
            assert isinstance(illegal_ids[0], type(alive_content[0]))
            # Note: use the commented code (faster) if considering each illegal content only once — this underestimates the prevalence
            # illegal_alive_content = list(set(alive_content) & set(illegal_ids))

            illegal_alive_content = [i for i in alive_content if i in illegal_ids]
            record = {
                "illegal_frac": len(illegal_alive_content) / len(alive_content),
                "unique_illegal_frac": len(list(set(illegal_alive_content)))
                / len(list(set(alive_content))),
                "illegal_count": len(illegal_alive_content),
                "unique_illegal_count": len(list(set(illegal_alive_content))),
            }
            # print values
            for k, v in record.items():
                logger.info(f"\t\t{k}: {np.round(v, 4)}")
            result = pd.DataFrame.from_records([record])

    except Exception as e:
        logger.info(e)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process experiment results paths.")
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Directory containing experiment results (likely to be in /N/project/simsom/carisma)",
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
        full_log_path=os.path.join(LOG_DIR, f"prevalence_data--{now}.log"),
        also_print=True,
    )

    ## GET ILLEGAL CONTENT FRACTION
    ## READING ALL SUBFOLDERS
    all_fpaths = glob.glob(f"{RES_DIR}/results_verbose/*.json.gz")

    dfs = []
    for fpath in tqdm.tqdm(
        all_fpaths, desc=f"Processing {len(all_fpaths)} files from {RES_DIR}"
    ):
        try:
            verbose = utils.read_json_compressed(fpath)
            # calculate prevalence
            res = get_illegal_prevalence(verbose)

            # Save relevant experiment settings
            for param in PARAMS:
                if param == "no_steps":
                    res["no_steps"] = len(verbose["quality_timestep"]) * len(res)
                else:
                    res[param] = verbose[param] * len(res)
            res["fpath"] = fpath * len(res)
            res["s_H"] = verbose["quality_settings"]["sizes"][0] * len(res)
            res["illegal_prob"] = verbose["quality_settings"][
                '"total_illegal_frac"'
            ] * len(res)
            dfs.append(res)
        except Exception as e:
            logger.info(f"\tError reading results from file {os.path.basename(fpath)}")
            logger.info(e)
            continue
    df = pd.concat(dfs).reset_index(drop=True)

    out_fpath = f"{OUT_DIR}/{os.path.basename(RES_DIR)}__prevalence.parquet"
    df.to_parquet(out_fpath)

    logger.info(f"Finish saving to {out_fpath} (len: {len(df)})!")
    logger.info("All done!")
