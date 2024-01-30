import os
from typing import Counter

import click


class Flagdebug(Counter[str]):
    """
    Small wrapper over counter to split the string on comma and return the count with call operator.
    Basically holds flags with the count.
    """

    def __init__(self, txt: str = ""):
        self.add(txt)

    def add(self, txt: str):
        self.update(txt.lower().split(","))
        return self

    def __call__(self, name: str) -> int:
        if "all" in self:
            return 100
        return self[name.lower()]


debug = Flagdebug()
"""All options are in this object"""


def click_debug_option(envname: str):
    return click.option(
        "--debug",
        hidden=True,
        default=lambda: debug.add(os.environ.get(envname, "")),
        callback=lambda _a, _b, value: debug.add(value) if value else debug,
        help="Comma separated list of debug flags",
    )
