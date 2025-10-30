#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import re
import shlex
import subprocess
from typing import (
    Any,
    Iterable,
    List,
)

import click
from click.shell_completion import CompletionItem

from . import colors
from .common import composed
from .common_click import EPILOG

log = logging.getLogger(__name__)


def cli_jobfile_disabled(root: Any, name: str, help: str):
    # Disabled, because maintenance.
    nomad_job_run_flags = [
        click.option("-check-index", type=int),
        click.option("-detach", is_flag=True),
        click.option("-eval-priority", type=int),
        click.option("-json", is_flag=True),
        click.option("-hcl1", is_flag=True),
        click.option("-hcl2-strict", is_flag=True),
        click.option("-policy-override", is_flag=True),
        click.option("-preserve-counts", is_flag=True),
        click.option("-consul-token"),
        click.option("-vault-token"),
        click.option("-vault-namespace"),
        click.option("-var", multiple=True),
        click.option("-var-file", type=click.File()),
    ]
    cli_jobfile_help = """
    JOBFILE can be file with a HCL or JSON nomad job or
    it can be a string containing a HCL or JSON nomad job.
    """
    return composed(
        root.command(
            name,
            help + cli_jobfile_help,
            context_settings=dict(ignore_unknown_options=True),
        ),
        *nomad_job_run_flags,
        click.argument(
            "jobfile",
            shell_complete=click.File().shell_complete,
        ),
    )


def complete_nomad_args(
    self, ctx: click.Context, param: click.Parameter, incomplete: str
) -> List[CompletionItem]:
    c = [
        CompletionItem(x)
        for x in """
        -address
        -ca-cert
        -ca-path
        -check-index
        -client-cert
        -client-key
        -diff
        -force-color
        -hcl2-strict
        -json
        -namespace
        -no-color
        -output
        -policy-override
        -region
        -tls-server-name
        -tls-skip-verify
        -token
        -var
        -var-file
        -vault-namespace
        -verbose
        """.split()
        if x.startswith(incomplete)
    ]
    if c:
        return c
    return [CompletionItem(incomplete, type="file")]


def command_forward_args_to_nomad(root: Any, name: str, where: str, help: str):
    return composed(
        root.command(
            name,
            help=help.strip()
            + "\n"
            + f"""
            All following arguments are forwarded to 'nomad job {where}' command.
            Note that 'nomad job {where}' has arguments starting with a single dash.
            """,
            context_settings=dict(ignore_unknown_options=True),
            epilog=EPILOG,
        ),
        click.help_option("-h", "--help"),
        click.argument(
            "args",
            nargs=-1,
            required=True,
            shell_complete=complete_nomad_args,
        ),
    )


def remove_ansi_escapes(line):
    """https://superuser.com/a/1657976/892783"""
    re1 = re.compile(r"\x1b\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]")
    re2 = re.compile(r"\x1b[PX^_].*?\x1b\\")
    re3 = re.compile(r"\x1b\][^\a]*(?:\a|\x1b\\)")
    re4 = re.compile(r"\x1b[\[\]A-Z\\^_@]")
    # re5: zero-width ASCII characters
    # see https://superuser.com/a/1388860
    re5 = re.compile(r"[\x00-\x1f\x7f-\x9f\xad]+")
    for r in [re1, re2, re3, re4, re5]:
        line = r.sub("", line)
    return line


def has_nomad_job_changed(quiet: bool, args: Iterable[str]) -> bool:
    """Return True if nomad job plan detected any changes to the job definition"""
    args = list(args)
    jobfile: str = args[-1]
    cmd: List[str] = [
        *"nomad job plan".split(),
        *(["-force-color"] if colors.has_color() else []),
        *args,
    ]
    log.debug(f"+ {' '.join(shlex.quote(x) for x in cmd)}")
    rr = subprocess.run(cmd, text=True, stdout=subprocess.PIPE)
    if rr.returncode != 0 and rr.returncode != 1:
        exit(rr.returncode)
    haschanged = rr.returncode == 1
    if not haschanged and rr.returncode == 0:
        stdout = remove_ansi_escapes(rr.stdout)
        haschanged = stdout.startswith("+/-")
        if not haschanged:
            # https://github.com/hashicorp/nomad/blob/ce8c3995a23d0dbf3cb81f4dea35f5bd054b52fb/command/job_plan.go#L701
            haschanged = any(
                line.endswith(i)
                for i in [
                    " (forces create/destroy update)",
                    " (forces destroy)",
                    " (forces create)",
                ]
                for line in stdout.splitlines()
            )
    log.debug(f"Nomad job {jobfile} has{' not' if not haschanged else ''} changed")
    if not quiet or haschanged:
        print(rr.stdout)
    return haschanged
