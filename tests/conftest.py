import logging
import os
from pathlib import Path


def pytest_configure(config):
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id:
        logfile = Path(f"build/tests/{worker_id}.log")
        logfile.parent.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            format=config.getini("log_file_format"),
            filename=logfile,
            level=config.getini("log_file_level"),
        )
        global logfilef
        logfilef = logfile.open("wb")
        os.dup2(logfilef.fileno(), 1)
        os.dup2(logfilef.fileno(), 2)
