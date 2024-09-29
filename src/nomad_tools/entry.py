#!/usr/bin/env python3
import click
import clickforward
from click.shell_completion import BashComplete

from .aliasedlazygroup import AliasedLazyGroup
from .common_click import EPILOG, help_h_option, main_options
from .common_nomad import namespace_option

clickforward.init()

# Problem: mising nospace handling in python.click.
# Solution: custom bash completion code.
# Problem: bash COMP_WORDBREAKS splits on colon `:`.
# Solution: use bash-completion helpers.
BashComplete.source_template = r"""\
    %(complete_func)s() {
        local cword words=()
        # __reassemble_comp_words_by_ref renamed to _comp__reassemble_words in newer bash-completion
        if [[ $(type -t __reassemble_comp_words_by_ref) == function ]]; then
            __reassemble_comp_words_by_ref "=:" words cword
        elif [[ $(type -t _comp__reassemble_words) == function ]]; then
            _comp__reassemble_words "=:" words cword
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

subcommands = """
    constrainteval
    cp
    dockers
    downloadrelease
    githubrunner
    gitlab-runner
    go
    info
    listattributes
    listnodeattributes
    nodenametoid
    port
    task
    vardir
    watch
    """.split()


@click.command(
    "nomadtools",
    cls=AliasedLazyGroup,
    lazy_subcommands={cmd: f"{__package__}.entry_{cmd.replace('-', '_')}.cli" for cmd in subcommands},
    help="Collection of useful tools for HashiCorp Nomad.",
    epilog=EPILOG,
)
@namespace_option()
@help_h_option()
@main_options()
def cli():
    pass


def main():
    cli(max_content_width=9999)


if __name__ == "__main__":
    main()
