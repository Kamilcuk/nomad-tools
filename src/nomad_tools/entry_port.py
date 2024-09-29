#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import json
import logging
import re
from typing import List

import click.shell_completion

from . import nomadlib
from .common import (
    alias_option,
    help_h_option,
    mynomad,
    namespace_option,
    nomad_find_job,
)

log = logging.getLogger(__name__)
args: argparse.Namespace


def gen_alloc(alloc: nomadlib.Alloc) -> List[str]:
    out = []
    log.debug(f"Parsing {alloc.ID}")
    ports = ((alloc.get("AllocatedResources") or {}).get("Shared") or {}).get("Ports")
    for port in ports or []:
        label = port["Label"]
        name = alloc["Name"]
        params = dict(
            **{
                k: v
                for k, v in alloc.items()
                if isinstance(k, str)
                and (v is None or isinstance(v, (float, int, str)))
            },
            label=label,
            host=port["HostIP"],
            port=port["Value"],
        )
        # If label if given, filter.
        if args.label:
            if args.label != label:
                log.debug(
                    f"Filtered {port} because label {label} does not match {args.label}"
                )
                continue
        if args.name:
            if not args.name.search(name):
                log.debug(
                    f"Filtered {port} because name {name} does not match {args.name}"
                )
                continue
        log.debug(f"Adding {port} with {params}")
        if args.json:
            txt = json.dumps(params)
        else:
            try:
                txt = args.format.format(**params)
            except KeyError:
                log.exception(f"format={args.format!r} params={params}")
                raise
        out += [txt]
    return out


def id_completor(
    ctx: click.Context, param: click.Parameter, incomplete: str
) -> List[click.shell_completion.CompletionItem]:
    args_alloc = ctx and ctx.params.get("alloc")
    arr = mynomad.get("allocations") if args_alloc else mynomad.get("jobs")
    ids = [id for id in (x["ID"] for x in arr) if id.startswith(incomplete)]
    return [click.shell_completion.CompletionItem(id) for id in ids]


default_format = "{host}:{port}"
long_format = "{host} {port} {label} {Name} {ID}"


@click.command(
    "port",
    help="""
Print dynamic ports allocated by Nomad for a specific job or allocation.
If no ports are found, exit with 2 exit status.
If label argument is given, outputs only redirects which label is equal to given label.

\b
The following variables are available for --format option:
  host    IP address allocated by Nomad
  port    port number allocated by Nomad
  label   the port name used in Nomad job definition
  ID      allocation ID that allocates this port
  Name    the name of the allocation from Nomad API, composed of <job>.<group>[<index>]
And any other key from Nomad allocation API definition which value is a string or a number.

\b
Exits with the following exit status:
  0  if at least one redirection was found,
  1  on python exception, missing job,
  2  if no redirections were found.
""",
)
@click.option(
    "-f",
    "--format",
    default=default_format,
    help=f"The template used to format output. Templated with python .format() function. [default: {default_format!r}]",
)
@alias_option(
    "-l",
    "--long",
    aliased=dict(format=long_format),
)
@click.option(
    "-j",
    "--json",
    is_flag=True,
    help="Output a json",
)
@click.option(
    "-s", "--separator", default="\n", help="Line separator. [default: newline]"
)
@click.option("-v", "--verbose", count=True, help="Be more verbose.")
@click.option("--alloc", is_flag=True, help="The argument is an allocation, not job id")
@click.option(
    "--all",
    help="Show ports of all allocations associated with the job, not only running or pending allocations.",
    is_flag=True,
)
@click.option(
    "-n",
    "--name",
    type=re.compile,
    help="Show only ports which name matches this regex.",
)
@help_h_option()
@namespace_option()
@click.argument("id", shell_complete=id_completor)
@click.argument("label", required=False)
def cli(id: str, **kwargs):
    global args
    args = argparse.Namespace(**kwargs)
    logging.basicConfig(
        level=(
            logging.DEBUG
            if args.verbose > 0
            else logging.INFO
            if args.verbose == 0
            else logging.WARN
            if args.verbose == 1
            else logging.ERROR
        ),
    )
    out: List[str] = []
    if args.alloc:
        allocs = mynomad.get("allocations", params={"prefix": id})
        assert len(allocs) > 0, f"Allocation with id {id} not found"
        assert len(allocs) < 2, f"Multiple allocations found starting with id {id}"
        alloc = nomadlib.Alloc(mynomad.get(f"allocation/{allocs[0]['ID']}"))
        out = gen_alloc(alloc)
    else:
        jobid = nomad_find_job(id)
        for alloc in mynomad.get(f"job/{jobid}/allocations"):
            if args.all or nomadlib.Alloc(alloc).is_pending_or_running():
                # job/*/allocations does not have AllocatedResources information.
                alloc = nomadlib.Alloc(mynomad.get(f"allocation/{alloc['ID']}"))
                out += gen_alloc(alloc)
    if not out:
        exit(2)
    print(*out, sep=args.separator)


if __name__ == "__main__":
    cli()
