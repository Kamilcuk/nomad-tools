#!/usr/bin/python3

import os
import shutil
from typing import Dict, List

import click

from .common import common_options, namespace_option


def get_nomad_commands() -> Dict[str, str]:
    nomad_help = """\
        run         Run a new job or update an existing job
        stop        Stop a running job
        status      Display the status output for a resource
        alloc       Interact with allocations
        job         Interact with jobs
        node        Interact with nodes
        agent       Runs a Nomad agent
        acl                 Interact with ACL policies and tokens
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
        system              Interact with the system API
        tls                 Generate Self Signed TLS Certificates for Nomad
        ui                  Open the Nomad Web UI
        var                 Interact with variables
        version             Prints the Nomad version
        volume              Interact with volumes
    """
    ret = {}
    for x in nomad_help.splitlines():
        if x:
            tmp = x.strip().split(" ", 1)
            if tmp and len(tmp) == 2:
                ret[tmp[0]] = tmp[1].strip()
    return ret


def get_additional_commands() -> Dict[str, str]:
    """
    Search all executables in PATH environment variable and returns a List[str] of executables starting with prefix.
    """
    path_env = os.environ.get("PATH", os.defpath)
    executables = []
    prefix = "nomad-"
    for path in path_env.split(os.pathsep):
        try:
            for file in os.listdir(path):
                if file.startswith(prefix):
                    executables.append(file.lstrip(prefix))
        except FileNotFoundError:
            pass
    return {exec: f"**Run {prefix}{exec}" for exec in executables}


commands: Dict[str, str] = {
    **get_nomad_commands(),
    **get_additional_commands(),
}

###############################################################################


@click.group(
    help=f"""
If 'nomad-command[0]' exists, run it.
Otherwise, run 'nomad' with the arguments.
""",
)
@namespace_option()
@common_options()
def cli(**kwargs):
    pass


for cmd, help in commands.items():

    @cli.command(
        cmd,
        help=help,
        context_settings=dict(
            ignore_unknown_options=True,
            allow_extra_args=True,
        ),
    )
    @click.argument("command", nargs=-1, type=click.UNPROCESSED)
    @click.pass_context
    def func(ctx: click.Context, command: List[str]):
        name = ctx.command.name
        assert name
        command = [name, *command]
        cmd = f"nomad-{command[0]}"
        if shutil.which(cmd):
            os.execvp(cmd, command)
        else:
            os.execvp("nomad", command)


if __name__ == "__main__":
    cli()
