import click
import clickforward
from click.shell_completion import BashComplete

from . import (
    entry_go,
    nomad_cp,
    nomad_dockers,
    nomad_downloadrelease,
    nomad_gitlab_runner,
    nomad_port,
    nomad_vardir,
    nomad_watch,
)
from .common_click import common_options, main_options
from .common_nomad import namespace_option

clickforward.init()

# Fix bash splitting completion on colon.
# Use __reassemble_comp_words_by_ref from bash-completion.
# Pass COMP_POINT as environment variable.
BashComplete.source_template = r"""\
    %(complete_func)s() {
        if [[ $(type -t __reassemble_comp_words_by_ref) != function ]]; then
            return -1
        fi
        local cword words=()
        __reassemble_comp_words_by_ref "=:" words cword
        local IFS=$'\n'
        response=$(env COMP_POINT=$COMP_POINT COMP_WORDS="${words[*]}" COMP_CWORD="$cword" %(complete_var)s=bash_complete $1)
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
    "nomadt",
    help="Nomad tools - collection of tools I find usefull when working with HashiCorp Nomad.",
    epilog="""
Written by Kamil Cukrowski 2023. Licensed under GNU GPL version or later.
""",
)
@namespace_option()
@common_options()
@main_options()
def cli():
    pass


cli.add_command(entry_go.cli)
cli.add_command(nomad_cp.cli)
cli.add_command(nomad_dockers.cli)
cli.add_command(nomad_downloadrelease.cli)
cli.add_command(nomad_gitlab_runner.cli)
cli.add_command(nomad_port.cli)
cli.add_command(nomad_vardir.cli)
cli.add_command(nomad_watch.cli)

if __name__ == "__main__":
    cli()
