from __future__ import annotations

import json
import logging
from typing import Dict, List, Tuple

import click
import clickdc

from .common import common_options, mynomad, verbose_option
from .common_click import completor
from .entry_constrainteval import Args, NodesAttributes
from .mytabulate import mytabulate


@click.command(
    "listattributes",
    help="""
List attributes of a specific node or all nodes.
Works similarly to constrainteval.
""",
)
@click.argument(
    "nodenameorid",
    nargs=-1,
    shell_complete=completor(lambda: [v["Name"] for v in mynomad.get("nodes")]),
)
@clickdc.adddc("args", Args)
@verbose_option()
@common_options()
def cli(args: Args, nodenameorid: Tuple[str, ...]):
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
