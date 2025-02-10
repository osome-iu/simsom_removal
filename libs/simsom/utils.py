"""
Utility functions to help with I/o, plotting and statistic tests
"""

import numpy as np
import sys
import logging
import os
import json
import gzip
import datetime
import inspect
from typing import Tuple, List


### I/O
def write_json_compressed(fpath: str, data: dict) -> object:
    # write compressed json for hpc - using with statement for better resource management
    try:
        with gzip.open(fpath, "w") as fout:
            fout.write(json.dumps(data).encode("utf-8"))
        print(f"Successfully wrote to {fpath}")
        return True
    except Exception as e:
        print("Failed to write json compressed", e)
        return e


### HELPERS FOR EXP CONFIG
def read_json_compressed(fpath: str) -> object:
    data = None
    try:
        with gzip.open(fpath, "r") as fin:
            json_bytes = fin.read()
            json_str = json_bytes.decode("utf-8")
            data = json.loads(json_str)
        return data
    except Exception as e:
        print("Failed to read json compressed", e)
        return e


def update_dict(adict, default_dict, fill_na=True):
    # only update the dictionary if key doesn't exist
    # use to fill out the rest of the params we're not interested in
    # Fill NaN value if it exists in another dict

    for k, v in default_dict.items():
        if k not in adict.keys():
            adict.update({k: v})
        if fill_na is True and adict[k] is None:
            adict.update({k: v})
    return adict


def remove_illegal_kwargs(adict, amethod):
    # remove a keyword from a dict if it is not in the signature of a method
    new_dict = {}
    argspec = inspect.getargspec(amethod)
    legal = argspec.args
    for k, v in adict.items():
        if k in legal:
            new_dict[k] = v
    return new_dict


def get_now():
    # return timestamp
    return int(datetime.datetime.now().timestamp())


def get_logger(name):
    # Create a custom logger
    logger = logging.getLogger(name)
    # Create handlers
    handler = logging.StreamHandler()
    # Create formatters and add it to handlers
    logger_format = logging.Formatter("%(asctime)s@%(name)s:%(levelname)s: %(message)s")
    handler.setFormatter(logger_format)
    # Add handlers to the logger
    logger.addHandler(handler)
    # Set level
    level = logging.getLevelName("INFO")
    logger.setLevel(level)
    return logger


def get_file_logger(log_dir, full_log_path, also_print=False, tqdm=False):
    """Create logger."""

    # Create log_dir if it doesn't exist already
    try:
        os.makedirs(f"{log_dir}")
    except:
        pass

    # Create logger and set level
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # Configure file handler
    formatter = logging.Formatter(
        fmt="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
        datefmt="%Y-%m-%d_%H:%M:%S",
    )
    fh = logging.FileHandler(f"{full_log_path}")
    fh.setFormatter(formatter)
    fh.setLevel(level=logging.INFO)
    logger.addHandler(fh)

    # if tqdm:
    #     logger.addHandler(TqdmLoggingHandler())
    # If also_print is true, the logger will also print the output to the
    # console in addition to sending it to the log file
    if also_print:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level=logging.INFO)
        logger.addHandler(ch)

    return logger


def safe_open(path, mode="w"):
    """Open "path" for writing or reading, creating any parent directories as needed.
    mode =[w, wb, r, rb]
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, mode)


def entropy(x: List[float]) -> float:
    # x: list of proportion
    entropy = np.sum(x * np.log(x))
    return entropy


def normalize(v) -> float:
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        return v
    return v / norm
