import json
import logging
import os
from typing import Any, Callable, Dict, Iterable

import click
import pkg_resources

from . import nomadlib

log = logging.getLogger(__name__)
mynomad = nomadlib.NomadConn()


def nomad_find_job(id: str) -> str:
    """Find job named jobprefix if namespace is *."""
    if os.environ["NOMAD_NAMESPACE"] != "*":
        mynomad.namespace = os.environ["NOMAD_NAMESPACE"]
        return id
    jobs = mynomad.get("jobs", params={"prefix": id})
    jobsstr = " ".join(f"{job['ID']}@{job['Namespace']}" for job in jobs)
    try:
        found = next(job for job in jobs if job["ID"] == id)
        mynomad.namespace = found["Namespace"]
        os.environ["NOMAD_NAMESPACE"] = found["Namespace"]
        return found["ID"]
    except StopIteration:
        log.exception(f"Job named {id} not found")
        raise


def _complete_set_namespace(ctx: click.Context):
    namespace = ctx.params.get("namespace")
    if namespace:
        os.environ["NOMAD_NAMESPACE"] = namespace


def completor(cb: Callable[[], Iterable[str]]):
    def completor_cb(ctx: click.Context, param: str, incomplete: str):
        _complete_set_namespace(ctx)
        try:
            return [x for x in cb() if x.startswith(incomplete)]
        except Exception:
            pass

    return completor_cb


def namespace_option():
    return click.option(
        "-N",
        "--namespace",
        help="Finds Nomad namespace matching given prefix and sets NOMAD_NAMESPACE environment variable.",
        envvar="NOMAD_NAMESPACE",
        show_default=True,
        default="default",
        shell_complete=completor(
            lambda: (x["Name"] for x in mynomad.get("namespaces"))
        ),
        callback=lambda ctx, param, value: os.environ.__setitem__(
            "NOMAD_NAMESPACE", value
        ),
    )


def complete_job():
    return completor(lambda: (x["ID"] for x in mynomad.get("jobs")))


def composed(*decs):
    """Merge decorators into one decorator"""

    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f

    return deco


def get_version():
    return pkg_resources.get_distribution(__package__).version


def __print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    # Copied from version_option()
    prog_name = ctx.find_root().info_name
    click.echo(f"{prog_name}, version {get_version()}")
    ctx.exit()


def common_options():
    return composed(
        click.help_option("-h", "--help"),
        click.option(
            "--version",
            is_flag=True,
            callback=__print_version,
            expose_value=False,
            is_eager=True,
            help="Print program version then exit.",
        ),
    )


def json_loads(txt: str) -> Any:
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        log.exception(f"Could not json.loads: {txt!r}")
        raise


###############################################################################


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
