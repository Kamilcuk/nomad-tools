from __future__ import annotations

import json
import logging
from typing import Dict, List, Tuple

import click
import clickdc

from .common import help_h_option, mynomad, verbose_option
from .common_click import completor
from .entry_constrainteval import NodeCacheArgs, NodesAttributes
from .mytabulate import mytabulate


def get_all_node_names():
    return [v["Name"] for v in mynomad.get("nodes")]


@click.command(
    "listnodeattributes",
    help="""
List attributes of specific nodes or all nodes.

Uses same cache as constrainteval.
""",
)
@click.argument(
    "nodenameorid",
    nargs=-1,
    shell_complete=completor(get_all_node_names),
)
@clickdc.adddc("args", NodeCacheArgs)
@verbose_option()
@help_h_option()
def cli(args: NodeCacheArgs, nodenameorid: Tuple[str, ...]):
    logging.basicConfig()
    nodesattributes = NodesAttributes.load(args)
    arr: List[Dict[str, str]] = []
    if nodenameorid:
        for input in nodenameorid:
            node = next(
                node
                for node in nodesattributes
                if node.attributes["node.unique.name"] == input
                or node.attributes["node.unique.id"] == input
            )
            arr.append(node.attributes)
    else:
        # Get all attributes of all nodes.
        allattributes: Dict[str, str] = {}
        for x in nodesattributes:
            allattributes.update(x.attributes)
        arr.append(allattributes)
    allkeys: List[str] = sorted(list(set(y for x in arr for y in x.keys())))
    output: List[List[str]] = [
        ["name", *allkeys],
        *[[x["node.unique.name"], *[str(x.get(k, "")) for k in allkeys]] for x in arr],
    ]
    if args.json:
        print(json.dumps(arr))
    else:
        print(mytabulate(output))
