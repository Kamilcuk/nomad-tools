#!/usr/bin/env python3

import click
import clickforward
from click.shell_completion import BashComplete

from . import (
    entry_constrainteval,
    entry_go,
    nomad_cp,
    nomad_dockers,
    nomad_downloadrelease,
    nomad_gitlab_runner,
    nomad_port,
    nomad_vardir,
    nomad_watch,
)
from .common_click import EPILOG, common_options, main_options
from .common_nomad import namespace_option

clickforward.init()

# Fix bash splitting completion on colon.
# Use __reassemble_comp_words_by_ref from bash-completion.
# Pass COMP_POINT as environment variable.
BashComplete.source_template = r"""\
    %(complete_func)s() {
        local cword words=()
        if [[ $(type -t __reassemble_comp_words_by_ref) == function ]]; then
            __reassemble_comp_words_by_ref "=:" words cword
        else
            words=("${COMP_WORDS[@]}")
            cword=${COMP_CWORD}
        fi
        local IFS=$'\n'
        response=$(COMP_POINT=$COMP_POINT COMP_WORDS="${words[*]}" COMP_CWORD="$cword" %(complete_var)s=bash_complete $1)
        for completion in $response; do
            IFS=',' read type value <<< "$completion"
            case $type in
            dir) COMPREPLY=(); compopt -o dirnames; ;;
            file) COMPREPLY=(); compopt -o default; ;;
            plain) COMPREPLY+=("$value"); ;;
            nospace) compopt -o nospace; ;;
            esac
        done
    }
    %(complete_func)s_setup() {
        complete -o nosort -F %(complete_func)s %(prog_name)s
    }
    %(complete_func)s_setup;
"""


@click.group(
    "nomadtools",
    help="Collection of useful tools for HashiCorp Nomad.",
    epilog=EPILOG,
)
@namespace_option()
@common_options()
@main_options()
def cli():
    pass


cli.add_command(entry_constrainteval.cli)
cli.add_command(entry_go.cli)
cli.add_command(nomad_cp.cli)
cli.add_command(nomad_dockers.cli)
cli.add_command(nomad_downloadrelease.cli)
cli.add_command(nomad_gitlab_runner.cli)
cli.add_command(nomad_port.cli)
cli.add_command(nomad_vardir.cli)
cli.add_command(nomad_watch.cli)


def main():
    cli(max_content_width=9999)


if __name__ == "__main__":
    main()
