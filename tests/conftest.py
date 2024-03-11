import logging
import os
import sys
from pathlib import Path
from typing import IO, Dict

LOGFILES: Dict[str, IO] = {}


def pytest_configure(config):
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id and worker_id not in LOGFILES:
        logfile = Path(f"build/tests/{worker_id}.log")
        logfile.parent.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            format=config.getini("log_file_format"),
            filename=logfile,
            level=config.getini("log_file_level"),
        )
        LOGFILES[worker_id] = logfile.open("wb")
        os.dup2(LOGFILES[worker_id].fileno(), sys.stdout.fileno())
        os.dup2(LOGFILES[worker_id].fileno(), sys.stderr.fileno())
