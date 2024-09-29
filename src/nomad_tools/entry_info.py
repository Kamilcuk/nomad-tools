from __future__ import annotations

import logging
from typing import List

import click

from .aliasedgroup import AliasedGroup
from . import common_click
from .common_nomad import mynomad

log = logging.getLogger(__name__)


@click.command(
    "info",
    cls=AliasedGroup,
    help="Get information about current Nomad state",
)
@common_click.help_h_option()
@common_click.verbose_option()
def cli():
    pass


@cli.command(help="Like topology web interface")
def topology():
    logging.basicConfig()
    allocations = mynomad.get(
        "allocations", params=dict(resources=True, task_states=False)
    )
    nodes = mynomad.get("nodes", params=dict(resources=True))
    tooutput: List[List[str]] = []
    for node in nodes:
        nodeallocs = [x for x in allocations if x["NodeID"] == node["ID"]]
        alloc_taskname_tasks = [
            (alloc, taskname, task)
            for alloc in nodeallocs
            for taskname, task in alloc["AllocatedResources"]["Tasks"].items()
        ]
        alloc_taskname_tasks.sort(
            key=lambda x: x[2]["Memory"]["MemoryMB"], reverse=True
        )
        used_cpu = sum(
            (task["Cpu"]["CpuShares"] for _, _, task in alloc_taskname_tasks),
            0,
        )
        used_memory = sum(
            (task["Memory"]["MemoryMB"] for _, _, task in alloc_taskname_tasks),
            0,
        )
        tooutput.append(
            [
                node["Datacenter"],
                node["Name"],
                node["Status"],
                f'{used_memory}MB/{node["NodeResources"]["Memory"]["MemoryMB"]}MB',
                f'{used_cpu}MHz/{node["NodeResources"]["Cpu"]["CpuShares"]}MHz',
                f"{len(nodeallocs)}allocs",
                *[
                    "["
                    + " ".join(
                        [
                            alloc["Name"],
                            taskname,
                            alloc["Namespace"],
                            f'{task["Memory"]["MemoryMB"]}MB',
                            f'{task["Cpu"]["CpuShares"]}MHz',
                        ]
                    )
                    + "]"
                    for alloc, taskname, task in alloc_taskname_tasks
                ],
            ]
        )
    tooutput.sort(key=lambda x: int(x[3].split("M")[0]))
    for x in tooutput:
        print(*x)
