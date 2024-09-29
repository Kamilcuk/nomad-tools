from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

import click
import clickdc

from . import entry_constrainteval
from .common import help_h_option, verbose_option
from .common_click import completor
from .entry_constrainteval import ConstraintArgs, NodeCacheArgs, NodesAttributes


def get_all_nodes_attributes(args: Optional[NodeCacheArgs] = None):
    args = args or NodeCacheArgs()
    nodesattributes = NodesAttributes.load(args)
    allattributes = sorted(list(set(y for x in nodesattributes for y in x.attributes)))
    return allattributes


@click.command(
    "listattributes",
    help="""
List nodes that have given attributes and show these attributes values.

With no arguments, lists all possible attributes in all nodes.

Alias to constrainteval arg1 is_set '' arg2 is_set '' ...
""",
)
@click.argument(
    "attributes",
    nargs=-1,
    shell_complete=completor(get_all_nodes_attributes),
)
@clickdc.adddc("args", NodeCacheArgs)
@verbose_option()
@help_h_option()
def cli(args: NodeCacheArgs, attributes: Tuple[str, ...]):
    logging.basicConfig()
    if attributes:
        return entry_constrainteval.main(
            args,
            ConstraintArgs(tuple(y for x in attributes for y in [x, "is_set", ""])),
        )
    else:
        allattributes = get_all_nodes_attributes()
        if args.json:
            print(json.dumps(allattributes))
        else:
            for x in allattributes:
                print(x)
