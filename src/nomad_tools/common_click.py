import os
from typing import Any, Callable, Dict, Iterable, List, Optional

import click

from .common_base import NOMAD_NAMESPACE, composed, print_version, shell_completion


def complete_set_namespace(ctx: click.Context):
    namespace = ctx.params.get("namespace")
    if namespace:
        os.environ[NOMAD_NAMESPACE] = namespace


def completor(
    cb: Callable[[], Iterable[str]]
) -> Callable[[click.Context, str, str], Optional[List[str]]]:
    def completor_cb(
        ctx: click.Context, param: str, incomplete: str
    ) -> Optional[List[str]]:
        complete_set_namespace(ctx)
        try:
            return [x for x in cb() if x.startswith(incomplete)]
        except Exception:
            pass

    return completor_cb


def click_callback_wrap_exit(cb: Callable[[], None]):
    """Execute a callback from click callback function and exit"""

    def wrap(ctx: click.Context, param: click.Parameter, value: str):
        if not value or ctx.resilient_parsing:
            return
        cb()
        ctx.exit()

    return wrap


def common_options():
    return composed(
        click.help_option("-h", "--help"),
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
