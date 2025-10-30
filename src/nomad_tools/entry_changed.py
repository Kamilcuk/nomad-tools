#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Tuple

import click

from .common_click import h_help_quiet_verbose_logging_options, is_verbose
from .forward_args_to_nomad import command_forward_args_to_nomad, has_nomad_job_changed


@command_forward_args_to_nomad(
    click,
    name="changed",
    where="plan",
    help="""
    Check if the Nomad job definition has changed compared to what is running.

    This command executes `nomad job plan` and checks if the jobs is changed.
    The job is assumed to be changed when `nomad job plan`:
      - exits with 1,
      - or exits with 0 and prints a string '+/-' in the output.

    This command is meant to be used in CI/CD pipelines.
    It exists with 0 when the job definition is correct.
    It outputs the output of `nomad job plan` if job is changed.
    Otherwise there is no ouptut on stdout.

    \b
    In scripts, you should check if the output of this command is not empty:
        if tmp=$(nomadtools changed ./job.nomad.hcl); then
        if [[ -z "$tmp" ]]; then
            echo "Job definition is ok and it was not changed"
        else
            echo "Job definition is ok and it changed"
        fi
        else
        echo "nomad job plan failed"
        fi

    \b
    The command exits with:
       0      when the job definition is correct
       non-0  when `nomad job plan` exted with error
    """,
)
@h_help_quiet_verbose_logging_options()
def cli(args: Tuple[str, ...]):
    return has_nomad_job_changed(is_verbose(), args)
