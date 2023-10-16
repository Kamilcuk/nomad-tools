#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import logging
from typing import Any, List, Optional, Union

import click.shell_completion

from . import nomadlib
from .common import common_options, mynomad

log = logging.getLogger(__name__)


def gen_alloc(alloc: nomadlib.Alloc) -> List[str]:
    out = []
    log.debug(f"Parsing {alloc.ID}")
    for i in alloc.get("AllocatedResources", {}).get("Shared", {}).get("Ports", []):
        params = dict(
            **{
                k: v
                for k, v in alloc.items()
                if isinstance(v, str) and isinstance(k, str)
            },
            label=i["Label"],
            host=i["HostIP"],
            port=i["Value"],
        )
        log.debug(f"Found")
        try:
            txt = args.format.format(**params)
        except KeyError:
            log.exception(f"format={args.format!r} params={params}")
            raise
        out += [txt]
    return out


class IdArgument(click.ParamType):
    name = "id"

    def __is_alloc(self, ctx: Optional[click.Context]):
        return ctx and ctx.params.get("alloc")

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Union[nomadlib.Alloc, str]:
        id = value
        if self.__is_alloc(ctx):
            allocs = mynomad.get(f"allocations", params={"prefix": id})
            assert len(allocs) > 0, f"Allocation with id {id} not found"
            assert len(allocs) < 2, f"Multiple allocations found starting with id {id}"
            return nomadlib.Alloc(allocs[0])
        return mynomad.find_job(id)

    def shell_complete(
        self, ctx: click.Context, param: click.Parameter, incomplete: str
    ) -> List[click.shell_completion.CompletionItem]:
        arr = (
            mynomad.get("allocations") if self.__is_alloc(ctx) else mynomad.get("jobs")
        )
        ids = [id for id in (x["ID"] for x in arr) if id.startswith(incomplete)]
        return [click.shell_completion.CompletionItem(id) for id in ids]


@click.command(
    help="Print dynamic ports allocated by Nomad for a specific job or allocation."
)
@click.option(
    "-f",
    "--format",
    default="{host}:{port} {label} {Name} {ID}",
    show_default=True,
    help="""
        The python .format() to print the output with.
        By default print host:port followed by Label, allocation Name and allocation ID
        """,
)
@click.option("-s", "--separator", default="\n", show_default=True)
@click.option("-v", "--verbose", count=True, help="Be more verbose.")
@click.option("--alloc", is_flag=True, help="The argument is an allocation, not job id")
@click.option(
    "--all",
    help="Show all job allocation ports, not only running or pending allocations.",
    is_flag=True,
)
@common_options()
@click.argument("id", type=IdArgument())
def cli(id: Union[nomadlib.Alloc, str], **kwargs):
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
    if args.alloc:
        assert isinstance(id, nomadlib.Alloc)
        alloc = id
        out = gen_alloc(alloc)
    else:
        assert isinstance(id, str)
        jobid = mynomad.find_job(id)
        out = []
        for alloc in mynomad.get(f"job/{jobid}/allocations"):
            if args.all or nomadlib.Alloc(alloc).is_pending_or_running():
                # job/*/allocations does not have AllocatedResources information.
                alloc = nomadlib.Alloc(mynomad.get(f"allocation/{alloc['ID']}"))
                out += gen_alloc(alloc)
    if out:
        print(*out, sep=args.separator)


if __name__ == "__main__":
    cli()
