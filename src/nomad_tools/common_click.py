import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Optional

import click

from .common_base import composed, print_version, shell_completion

EPILOG = "Written by Kamil Cukrowski 2024. Licensed under GNU GPL version or later."


def complete_set_namespace(ctx: click.Context):
    namespace = ctx.params.get("namespace")
    if namespace:
        os.environ["NOMAD_NAMESPACE"] = namespace


def completor(
    cb: Callable[[], Iterable[str]],
) -> Callable[[click.Context, str, str], Optional[List[str]]]:
    def completor_cb(
        ctx: click.Context, param: str, incomplete: str
    ) -> Optional[List[str]]:
        complete_set_namespace(ctx)
        try:
            return [x for x in cb() if x.startswith(incomplete)]
        except Exception:
            return []

    return completor_cb


def click_callback_wrap_exit(cb: Callable[[], None]):
    """Execute a callback from click callback function and exit"""

    def wrap(ctx: click.Context, param: click.Parameter, value: str):
        if not value or ctx.resilient_parsing:
            return
        cb()
        ctx.exit()

    return wrap


def main_options():
    return composed(
        click.option(
            "--version",
            is_flag=True,
            callback=click_callback_wrap_exit(print_version),
            expose_value=False,
            is_eager=True,
            help="Print program version then exit.",
        ),
        click.option(
            "--autocomplete-info",
            is_flag=True,
            callback=click_callback_wrap_exit(shell_completion.print),
            expose_value=False,
            is_eager=True,
            help="Print shell completion information.",
        ),
        click.option(
            "--autocomplete-install",
            is_flag=True,
            callback=click_callback_wrap_exit(shell_completion.install),
            expose_value=False,
            is_eager=True,
            help="Install shell completion.",
        ),
    )


def help_h_option():
    return click.help_option("-h", "--help")


def verbose_option():
    return click.option(
        "-v",
        "--verbose",
        count=True,
        expose_value=False,
        is_eager=True,
        callback=lambda ctx, opt, value: (
            logging.root.setLevel(max(logging.DEBUG, logging.root.level - 10 * value))
        ),
    )


def quiet_option():
    return click.option(
        "-q",
        "--quiet",
        count=True,
        expose_value=False,
        is_eager=True,
        callback=lambda ctx, opt, value: (
            logging.root.setLevel(
                min(logging.CRITICAL, logging.root.level + 10 * value)
            )
        ),
    )


def __alias_option_callback(
    aliased: Dict[str, Any],
    ctx: click.Context,
    param: click.Parameter,
    value: Any,
):
    """Callback called from alias_option option."""
    if value:
        for paramname, val in aliased.items():
            param = next(p for p in ctx.command.params if p.name == paramname)
            param.default = val


def alias_option(
    *param_decls: str,
    aliased: Dict[str, Any],
    **attrs: Any,
):
    """Add this to click options to have an alias for other options"""
    aliasedhelp = " ".join(
        "--"
        + k.replace("_", "-")
        + ("" if v is True else f"={v}" if isinstance(v, int) else f"={v!r}")
        for k, v in aliased.items()
    )
    return click.option(
        *param_decls,
        is_flag=True,
        help=f"Alias to {aliasedhelp}",
        callback=lambda *args: __alias_option_callback(aliased, *args),
        **attrs,
    )
