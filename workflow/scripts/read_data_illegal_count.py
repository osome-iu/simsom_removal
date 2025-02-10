""" Read experiment results from folders matching a pattern `{result_path}*/results_verbose/*0.01*.json.gz`
    Save results from each folder into a single parquet file (to avoid memory issues)
     - Get no. steps to convergence of each exp
     - Read all subfolders in the  folder of an experiment (instead of specifying folder names)

    Usage:
    python read_data_illegal_count.py --result_path /N/project/simsom/carisma/04302024_illegal_removal --out_path /N/u/baotruon/BigRed200/carisma/experiments/20241126_main_results
    Log files are saved in the '{out_path}/log' folder
"""

import simsom.utils as utils
import numpy as np
import pandas as pd
import os
import glob
import argparse

PARAMS = [
    "moderation_half_life",
    "quality",
    "graph_gml",  # to get illegal_content_probability
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


## Transform file names to match mod=True vs mod=False
def transform_file_name(string):
    import re

    # Regular expression to match the desired prefix and remove unwanted trailing patterns including ".json.gz"
    # This pattern captures the initial part of the filename up to "__diff_true"
    # and ignores optional trailing patterns like "_1", "--EXTRA", "--1" (where "--EXTRA" and "--1" can repeat)
    # and the file extension ".json.gz". It also correctly handles filenames without trailing patterns.
    pattern = r"^(.*?__diff_true)(?:_[0-9]|(--EXTRA)|(--1))*\.json\.gz$"

    # Perform the regex search and substitution
    match = re.match(pattern, string)
    if match:
        # Return the base prefix part of the match
        return match.group(1)
    else:
        # If no match, return the original string without modifications
        return string


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
    # if os.path.exists(OUT_DIR):
    #     raise ValueError("Output directory already exists. Please remove it first.")
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
    folders = glob.glob(f"{RES_DIR}*/results_verbose/*")
    logger.info(f"** Reading from {len(folders)} folders ... ** ")

    for jdx, folder in enumerate(folders):
        logger.info(f"  Reading from {folder}...")
        if os.path.exists(f"{OUT_DIR}/{os.path.basename(folder)}"):
            logger.info(f"\t{os.path.basename(folder)} already processed. Skipping...")
            continue
        dfs = []
        all_fpaths = glob.glob(f"{folder}/*.json.gz")
        for fpath in all_fpaths:
            if jdx % 10 == 0:
                logger.info(f"\tProcessed {jdx}/{len(all_fpaths)} files")
            try:
                verbose = utils.read_json_compressed(fpath)
                # calculate prevalence
                res = get_illegal_prevalence(verbose)
                res["fpath"] = fpath * len(res)
                # get other info
                for param in PARAMS:
                    if param == "no_steps":
                        res["no_steps"] = len(verbose["quality_timestep"]) * len(res)
                    else:
                        res[param] = verbose[param] * len(res)
                dfs.append(res)
            except Exception as e:
                logger.info(
                    f"\tError reading results from file {os.path.basename(fpath)}"
                )
                logger.info(e)
                continue
        if len(dfs) == 0:
            logger.info(f"\tNo data for {os.path.basename(folder)}")
            continue
        df = pd.concat(dfs).reset_index(drop=True)

        out_fpath = f"{OUT_DIR}/{os.path.basename(folder)}__prevalence.parquet"
        df.to_parquet(out_fpath)

        logger.info(f"Finish saving to {out_fpath} (len: {len(df)})!")
    logger.info("All done!")
