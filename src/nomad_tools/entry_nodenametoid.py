from __future__ import annotations

from typing import Dict, Tuple

import click

from .common import mynomad
from .common_click import help_h_option, completor


def get_nodenametoid() -> Dict[str, str]:
    return {v["Name"]: v["ID"] for v in mynomad.get("nodes")}


@click.command(
    "nodenametoid",
    help="""
    Convert node names to id.
    """,
)
@click.option(
    "--prefix",
    is_flag=True,
    help="Match the node name on this prefix, not exact match. Exactly one node has to match",
)
@click.argument(
    "nodename",
    nargs=-1,
    required=True,
    shell_complete=completor(lambda: list(get_nodenametoid().keys())),
)
@help_h_option()
def cli(prefix: bool, nodename: Tuple[str, ...]):
    nodenametoid = get_nodenametoid()
    for name in nodename:
        if prefix:
            names = [x for x in nodenametoid.keys() if x.startswith(name)]
            if len(names) == 0:
                raise Exception(f"No nodes start with {name} prefix")
            if len(names) != 1:
                raise Exception(f"Multiple nodes match name prefix {name}: {names}")
            name = names[0]
        print(nodenametoid[name])
