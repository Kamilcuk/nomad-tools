#!/usr/bin/python3

import argparse
import os
from typing import Set

from .common_base import (
    NOMAD_NAMESPACE,
    eprint,
    get_package_file,
    print_version,
    quotearr,
    shell_completion,
)

NOMAD_HELP = """\
Usage: nomad [-version] [-help] [-autocomplete-(un)install] <command> [args]

Common commands:
    run         Run a new job or update an existing job
    stop        Stop a running job
    status      Display the status output for a resource
    alloc       Interact with allocations
    job         Interact with jobs
    node        Interact with nodes
    agent       Runs a Nomad agent

Other commands:
    acl                 Interact with ACL policies and tokens
    action              Run a pre-defined command from a given context
    agent-info          Display status information about the local agent
    config              Interact with configurations
    deployment          Interact with deployments
    eval                Interact with evaluations
    exec                Execute commands in task
    fmt                 Rewrites Nomad config and job files to canonical format
    license             Interact with Nomad Enterprise License
    login               Login to Nomad using an auth method
    monitor             Stream logs from a Nomad agent
    namespace           Interact with namespaces
    operator            Provides cluster-level tools for Nomad operators
    plugin              Inspect plugins
    quota               Interact with quotas
    recommendation      Interact with the Nomad recommendation endpoint
    scaling             Interact with the Nomad scaling endpoint
    sentinel            Interact with Sentinel policies
    server              Interact with servers
    service             Interact with registered services
    setup               Interact with setup helpers
    system              Interact with the system API
    tls                 Generate Self Signed TLS Certificates for Nomad
    ui                  Open the Nomad Web UI
    var                 Interact with variables
    version             Prints the Nomad version
    volume              Interact with volumes
"""


def get_nomad_commands() -> Set[str]:
    return set(
        x.strip().split(" ", 1)[0]
        for x in NOMAD_HELP.splitlines()
        if x and x.startswith("    ")
    )


def handle_bash_completion():
    """Custom bash completion with same interface as click"""
    arg = os.environ.get("_NOMADT_COMPLETE")
    if arg is None:
        return
    if arg != "bash_source":
        exit()
    script = get_package_file("nomadt_completion.sh")
    print(script)
    exit()


def cli():
    handle_bash_completion()
    parser = argparse.ArgumentParser(
        description="""
        Wrapper around nomad to execute nomad-anything as nomadt anything.
        If a 'nomad cmd' exists, then 'nomadt cmd' will forward to it.
        Otherwise, it will try to execute 'nomad-cmd' command.
        It is a wrapper that works similar to git.
        """
    )
    parser.add_argument(
        "-N", "--namespace", help="Set NOMAD_NAMESPACE before executing the command"
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument(
        "--autocomplete-info",
        action="store_true",
        help="Print shell completion information and exit",
    )
    parser.add_argument(
        "--autocomplete-install",
        action="store_true",
        help="Install bash shell completion and exit",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print the command before executing"
    )
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to execute")
    args = parser.parse_args()
    if args.version:
        print_version()
        exit()
    if args.autocomplete_info:
        shell_completion.print()
        exit()
    if args.autocomplete_install:
        shell_completion.install()
        exit()
    if args.namespace:
        os.environ[NOMAD_NAMESPACE] = args.namespace
    cmd = args.cmd
    if not cmd:
        parser.print_usage()
        exit(1)
    if cmd[0] in get_nomad_commands():
        cmd = ["nomad", *cmd]
    else:
        cmd = [f"nomad-{cmd[0]}", *cmd[1:]]
    if args.verbose:
        eprint(f"+ {quotearr(cmd)}")
    try:
        os.execvp(cmd[0], cmd)
    except FileNotFoundError:
        eprint(f"nomadt: {cmd[0]}: command not found")
        exit(127)


if __name__ == "__main__":
    cli()
