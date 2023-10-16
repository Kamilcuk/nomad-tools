import os
from typing import Callable, Iterable

import click
import pkg_resources

from . import nomadlib

mynomad = nomadlib.NomadConn()


def nomad_find_namespace(prefix: str):
    """Finds a nomad namespace by prefix"""
    if prefix == "*":
        return prefix
    namespaces = mynomad.get("namespaces")
    names = [x["Name"] for x in namespaces]
    namesstr = " ".join(names)
    matchednames = [x for x in names if x.startswith(prefix)]
    matchednamesstr = " ".join(matchednames)
    assert (
        len(matchednames) > 0
    ), f"Couldn't find namespace maching prefix {prefix} from {namesstr}"
    assert (
        len(matchednames) < 2
    ), f"Prefix {prefix} matched multiple namespaces: {matchednamesstr}"
    return matchednames[0]


def _complete_set_namespace(ctx: click.Context):
    namespace = ctx.params.get("namespace")
    if namespace:
        try:
            os.environ["NOMAD_NAMESPACE"] = nomad_find_namespace(namespace)
        except Exception:
            pass


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
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f

    return deco


def get_version():
    return pkg_resources.get_distribution(__package__).version


def _print_version(ctx, param, value):
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
            callback=_print_version,
            expose_value=False,
            is_eager=True,
            help="Print program version then exit.",
        ),
    )
