#!/usr/bin/env python3
import os
import sys

NOMAD_HELP = """
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


def get_nomad_subcommands():
    return set(x.split()[0] for x in NOMAD_HELP.splitlines() if x.startswith("    "))


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] in get_nomad_subcommands():
            os.execvp("nomad", ["nomad", *sys.argv[1:]])
    from . import entry

    entry.main()


if __name__ == "__main__":
    main()
