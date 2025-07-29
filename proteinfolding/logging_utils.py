# logging_utils.py

import logging
import subprocess

def setup_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO)

def log_info(message):
    logging.info(message)

def get_git_commit():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
    return commit.decode("utf-8")
