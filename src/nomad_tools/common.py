# ruff: noqa: F403
import json
import logging
from typing import Any

from .common_base import *  # noqa: W0614
from .common_click import *  # noqa: W0614
from .common_nomad import *  # noqa: W0614

log = logging.getLogger(__name__)


def json_loads(txt: str) -> Any:
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        log.exception(f"Could not json.loads: {txt!r}")
        raise
