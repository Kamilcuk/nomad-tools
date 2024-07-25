import logging
import os
import sys
from typing import Counter

import click

log = logging.getLogger(__name__)


DEBUGFLAGS: Counter[str] = Counter()


def __log(name: str, txt: str):
    print(f"{name.upper()}: {txt}", flush=True, file=sys.stderr)


def add(txt: str):
    if txt:
        # __log("FLAGDEBUG", txt)
        DEBUGFLAGS.update(txt.lower().split(","))


def debug(name: str) -> int:
    if "all" in DEBUGFLAGS:
        return 100
    return DEBUGFLAGS[name.lower()]


def logdebug(name: str, txt: str, ths: int = 0) -> bool:
    if debug(name) > ths:
        __log(name, txt)
        return True
    return False


def click_debug_option(envname: str):
    return click.option(
        "--debug",
        hidden=True,
        expose_value=False,
        default=lambda: add(os.environ.get(envname, "")),
        callback=lambda _a, _b, value: add(value),
        help="Comma separated list of debug flags",
    )
