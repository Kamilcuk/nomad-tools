import json
import logging

from .common_base import *
from .common_click import *
from .common_nomad import *

log = logging.getLogger(__name__)


def json_loads(txt: str) -> Any:
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        log.exception(f"Could not json.loads: {txt!r}")
        raise
