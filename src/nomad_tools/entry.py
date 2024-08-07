#!/usr/bin/env python3
import click
import clickforward
from click.shell_completion import BashComplete

from . import (
    entry_constrainteval,
    entry_go,
    entry_cp,
    entry_dockers,
    entry_downloadrelease,
    entry_githubrunner,
    entry_gitlab_runner,
    entry_port,
    entry_vardir,
    entry_watch,
)
from .common_click import EPILOG, common_options, main_options
from .common_nomad import namespace_option
from .aliasedgroup import AliasedGroup

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


@click.command(
    "nomadtools",
    cls=AliasedGroup,
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
cli.add_command(entry_cp.cli)
cli.add_command(entry_dockers.cli)
cli.add_command(entry_downloadrelease.cli)
cli.add_command(entry_githubrunner.cli)
cli.add_command(entry_gitlab_runner.cli)
cli.add_command(entry_port.cli)
cli.add_command(entry_vardir.cli)
cli.add_command(entry_watch.cli)


def main():
    cli(max_content_width=9999)


if __name__ == "__main__":
    main()
