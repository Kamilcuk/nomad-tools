#!/usr/bin/python3

import argparse
import os
from typing import Any, Callable, List, Set, Tuple

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


def get_nomad_subcommands() -> List[str]:
    return [
        x.strip().split(" ", 1)[0]
        for x in NOMAD_HELP.splitlines()
        if x and x.startswith("    ")
    ]


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


def mk_action(callback: Callable[[], Any]):
    class MyAction(argparse.Action):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, nargs=0, **kwargs)

        def __call__(self, *args):
            callback()
            exit()

    return MyAction


def get_nomad_alias_commands() -> List[str]:
    ret: Set[str] = set()
    prefix = "nomad-"
    for d in sorted(list(set(os.environ.get("PATH", os.defpath).split(os.pathsep)))):
        if d == "":
            d = os.getcwd()
        if os.path.exists(d) and os.access(d, os.X_OK):
            for f in os.listdir(d):
                if (
                    f.startswith(prefix)
                    and os.path.isfile(os.path.join(d, f))
                    and os.access(os.path.join(d, f), os.X_OK)
                ):
                    ret.add(f[len(prefix) :])
    return sorted(list(ret))


class CommandsHelpAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, nargs=0, **kwargs)

    @staticmethod
    def __printcols(cols: List[Tuple[str, str]]):
        len1 = max(len(x[0]) for x in cols)
        for k, v in cols:
            print(f"    {k:{len1}}    {v}")

    def __call__(self, parser, *args):
        parser.print_help()
        print()
        print("\n".join(NOMAD_HELP.splitlines()[2:]).strip())
        print()
        print("Commands starting with nomad- prefix:")
        self.__printcols([(n, f"Run nomad-{n}") for n in get_nomad_alias_commands()])
        print()
        exit()


def cli():
    handle_bash_completion()
    parser = argparse.ArgumentParser(
        add_help=False,
        description="""
        Wrapper around nomad to execute nomad-anything as nomadt anything.
        If a 'nomad cmd' exists, then 'nomadt cmd' will forward to it.
        Otherwise, it will try to execute 'nomad-cmd' command.
        It is a wrapper that works similar to git.
        """,
    )
    parser.add_argument(
        "-h",
        "--help",
        action=CommandsHelpAction,
        help="show this help message and exit",
    )
    parser.add_argument(
        "-N", "--namespace", help="Set NOMAD_NAMESPACE before executing the command"
    )
    parser.add_argument(
        "--version", action=mk_action(print_version), help="Print version and exit"
    )
    parser.add_argument(
        "--autocomplete-info",
        action=mk_action(shell_completion.print),
        help="Print shell completion information and exit",
    )
    parser.add_argument(
        "--autocomplete-install",
        action=mk_action(shell_completion.install),
        help="Install bash shell completion and exit",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print the command before executing"
    )
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to execute")
    args = parser.parse_args()
    if args.namespace:
        os.environ[NOMAD_NAMESPACE] = args.namespace
    cmd: List[str] = args.cmd
    if not cmd:
        parser.print_usage()
        exit(1)
    if cmd[0] in get_nomad_subcommands():
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
