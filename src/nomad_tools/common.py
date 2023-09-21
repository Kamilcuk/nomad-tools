import os
from typing import Callable, Iterable

import click

from .nomadlib import NomadConn

mynomad = NomadConn()


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
