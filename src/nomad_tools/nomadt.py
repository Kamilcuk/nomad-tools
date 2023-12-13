#!/usr/bin/python3

import argparse
import os
import shutil
import subprocess
import sys
from typing import Set

from click.shell_completion import CompletionItem

from .common import get_package_file, get_version, print_shell_completion, quotearr

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
    action              Run a pre-defined action from a Nomad task
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
    if arg is None or arg != "bash_source":
        return
    script = get_package_file("nomadt_completion.sh")
    print(script)
    exit(0)


###############################################################################


def get_additional_commands() -> Set[str]:
    """
    Search all executables in PATH environment variable and returns a List[str] of executables starting with prefix.
    """
    path_env = os.environ.get("PATH", os.defpath)
    executables = set()
    prefix = "nomad-"
    for path in path_env.split(os.pathsep):
        try:
            for file in os.listdir(path):
                if file.startswith(prefix):
                    executables.add(file.lstrip(prefix))
        except FileNotFoundError:
            pass
    return executables


def __click_complete_cmd(ctx, param, incomplete):
    """unused"""
    BashComplete_source_template = """\
        %(complete_func)s() {
            local IFS=$'\\n'
            local response
            response=$(env COMP_WORDS="${COMP_WORDS[*]}" COMP_CWORD="$COMP_CWORD" %(complete_var)s=bash_complete $1)
            for completion in $response; do
                IFS=',' read type value <<< "$completion"
                case $type in
                dir) COMPREPLY=(); compopt -o dirnames; ;;
                file) COMPREPLY=(); compopt -o default; ;;
                plain) COMPREPLY+=($value); ;;
                subcmd)
                    # Use bash-completion project for completion
                    if declare -f _init_completion 2>/dev/null >&2; then
                        declare COMP_WORDS=($value)
                        COMP_CWORD=${#COMP_WORDS[@]}
                        local IFS=' '
                        COMP_LINE=${COMP_WORDS[*]}
                        COMP_POINT=${#COMP_LINE}
                        echo "$value ${COMP_WORDS[*]} $COMP_CWORD $COMP_LINE"
                        local cur prev words cword split
                        _init_completion || return
                        _command_offset 0
                    fi
                esac
            done
        }
        %(complete_func)s_setup() {
            complete -o nosort -F %(complete_func)s %(prog_name)s
        }
        %(complete_func)s_setup;
    """
    print(ctx, ctx.params, file=sys.stderr)
    cmd = list(ctx.params["cmd"])
    if not cmd:
        arr = sorted(list(get_additional_commands() | get_nomad_commands()))
        return [x for x in arr if x.startswith(incomplete)]
    else:
        if cmd[0] in get_nomad_commands():
            COMP_LINE = " ".join(cmd)
            # Run go completion. https://github.com/posener/complete/blob/v1/complete.go#L15
            arr = subprocess.run(
                ["nomad"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                env=dict(
                    COMP_LINE=COMP_LINE,
                    COMP_POINT=str(len(COMP_LINE)),
                ),
            ).stdout.splitlines()
            return [x for x in arr if x.startswith(incomplete)]
        else:
            cmd[0] = f"nomad-{cmd[0]}"
            cmd.append(incomplete)
            if shutil.which(cmd[0]):
                return [CompletionItem(quotearr(cmd), "subcmd")]
    return []


###############################################################################


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
        "--shell-completion",
        action="store_true",
        help="Print shell completion information and exit",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print the command before executing"
    )
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to execute")
    args = parser.parse_args()
    if args.version:
        prog_name = parser.prog
        print(f"{prog_name}, version {get_version()}")
        exit()
    if args.shell_completion:
        print_shell_completion()
        exit()
    if args.namespace:
        os.environ["NOMAD_NAMESPACE"] = args.namespace
    cmd = args.cmd
    if not cmd:
        parser.print_usage()
        exit(1)
    if cmd[0] in get_nomad_commands():
        cmd = ["nomad", *cmd]
    else:
        cmd = [f"nomad-{cmd[0]}", *cmd[1:]]
    if args.verbose:
        print(f"+ {quotearr(cmd)}", file=sys.stderr)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    cli()
