import functools
import json
import logging
import os
import pkgutil
import shlex
import shutil
import subprocess
import sys
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, TypeVar

import click
import pkg_resources

from . import nomadlib

log = logging.getLogger(__name__)
mynomad = nomadlib.NomadConn()


def nomad_find_job(id: str) -> str:
    """Find job named jobprefix if namespace is *."""
    jobs = mynomad.get("jobs", params={"prefix": id})
    matches = [job for job in jobs if job["ID"] == id]
    if len(matches) == 0:
        raise nomadlib.JobNotFound(f"Job named {id} not found")
    assert len(matches) < 2, f"Found multiple jobs named {id}"
    found = matches[0]
    os.environ["NOMAD_NAMESPACE"] = mynomad.namespace = found["Namespace"]
    return found["ID"]


def _complete_set_namespace(ctx: click.Context):
    namespace = ctx.params.get("namespace")
    if namespace:
        os.environ["NOMAD_NAMESPACE"] = namespace


def completor(
    cb: Callable[[], Iterable[str]]
) -> Callable[[click.Context, str, str], Optional[List[str]]]:
    def completor_cb(
        ctx: click.Context, param: str, incomplete: str
    ) -> Optional[List[str]]:
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
        help="Set NOMAD_NAMESPACE environment variable.",
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


def quotearr(cmd: List[str]):
    return " ".join(shlex.quote(x) for x in cmd)


@functools.lru_cache()
def get_package_file(file: str) -> str:
    """Get a file relative to current package"""
    res = pkgutil.get_data(__package__, file)
    assert res is not None, f"Could not find {file}"
    return res.decode()


def composed(*decs):
    """Merge decorators into one decorator"""

    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f

    return deco


def andjoin(arr: Iterable[Any]) -> str:
    arr = list(arr)
    if not len(arr):
        return ""
    if len(arr) == 1:
        return str(arr[0])
    return ", ".join(str(x) for x in arr[:-1]) + " and " + str(arr[-1])


def get_version():
    return pkg_resources.get_distribution(__package__).version


def __print_version(ctx: click.Context, param: click.Parameter, value: str):
    if not value or ctx.resilient_parsing:
        return
    # Copied from version_option()
    prog_name = ctx.find_root().info_name
    click.echo(f"{prog_name}, version {get_version()}")
    ctx.exit()


def get_entrypoints() -> List[str]:
    txt = """
nomadt = "nomad_tools:nomadt.cli"
nomad-watch = "nomad_tools:nomad_watch.cli"
nomad-cp = "nomad_tools:nomad_cp.cli"
nomad-vardir = "nomad_tools:nomad_vardir.cli"
nomad-gitlab-runner = "nomad_tools:nomad_gitlab_runner.main"
nomad-port = "nomad_tools:nomad_port.cli"
nomad-dockers = "nomad_tools:nomad_dockers.cli"
nomad-downloadrelease = "nomad_tools:bin_downloadrelease.cli"
"""
    return [
        name
        for line in txt.splitlines()
        for name in [line.split("=")[0].strip()]
        if name
    ]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class shell_completion:
    @staticmethod
    def install_script() -> List[str]:
        dir = "~/.local/share/bash-completion/completions"
        script: List[str] = []
        script.append(f"mkdir -vp {dir}")
        for name in get_entrypoints():
            upname = name.upper().replace("-", "_")
            script.append(
                f"echo 'eval \"$(_{upname}_COMPLETE=bash_source {name})\"' > {dir}/{name}"
            )
        return script

    @staticmethod
    def install():
        if shutil.which("bash"):
            for line in shell_completion.install_script():
                eprint(f"+ {line}")
                subprocess.check_call(["bash", "-c", line])

    @staticmethod
    def print():
        print("This project uses click python module.")
        print(
            "See https://click.palletsprojects.com/en/8.1.x/shell-completion/ on how to install completion."
        )
        print("For bash-completion, execute the following:")
        for line in shell_completion.install_script():
            print(f"   {line}")

    @staticmethod
    def click_install(ctx: click.Context, param: click.Parameter, value: str):
        if not value or ctx.resilient_parsing:
            return
        shell_completion.install()
        ctx.exit()

    @staticmethod
    def click_print(ctx: click.Context, param: click.Parameter, value: str):
        if not value or ctx.resilient_parsing:
            return
        shell_completion.print()
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
        click.option(
            "--autocomplete-info",
            is_flag=True,
            callback=shell_completion.click_print,
            expose_value=False,
            is_eager=True,
            help="Print shell completion information.",
        ),
        click.option(
            "--autocomplete-install",
            is_flag=True,
            callback=shell_completion.click_install,
            expose_value=False,
            is_eager=True,
            help="Install shell completion.",
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


###############################################################################


T = TypeVar("T")
R = TypeVar("R")


class cached_property(Generic[T, R]):
    """
    Descriptor (non-data) for building an attribute on-demand on first use.
    No cached_property in pip has correct typing, so I wrote my own.
    """

    def __init__(self, factory: Callable[[T], R]):
        """
        <factory> is called such: factory(instance) to build the attribute.
        """
        self._attr_name = factory.__name__
        self._factory = factory

    def __get__(self, instance: T, owner) -> R:
        # Build the attribute.
        attr: R = self._factory(instance)

        # Cache the value; hide ourselves.
        setattr(instance, self._attr_name, attr)

        return attr
