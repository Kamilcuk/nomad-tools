# Only basic import functions, so that nomadt is as fast as possible.
import functools
import os
import pkgutil
import shlex
import subprocess
import sys
from typing import Any, Callable, Generic, Iterable, List, TypeVar, cast

NOMAD_NAMESPACE = "NOMAD_NAMESPACE"


def quotearr(cmd: List[str]):
    return " ".join(shlex.quote(x) for x in cmd)


@functools.lru_cache()
def get_package_file(file: str) -> str:
    """Get a file relative to current package"""
    package = __package__
    assert package
    res = pkgutil.get_data(package, file)
    assert res is not None, f"Could not find {file}"
    return res.decode()


def composed(*decs):
    """Merge decorators into one decorator"""

    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f

    return deco


def andjoin(arr: Iterable[Any], fin: str = " and ") -> str:
    arr = list(arr)
    if not len(arr):
        return ""
    if len(arr) == 1:
        return str(arr[0])
    return ", ".join(str(x) for x in arr[:-1]) + fin + str(arr[-1])


def get_version():
    # Load lazily, to optimize for import speed.
    package = __package__
    assert package
    try:
        import importlib.metadata  # pyright: ignore

        return importlib.metadata.version(package)
    except ImportError:
        import pkg_resources  # pyright: ignore

        return pkg_resources.get_distribution(package).version


def print_version():
    # Copied from version_option()
    print(f"{os.path.basename(sys.argv[0])}, version {get_version()}")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class shell_completion:
    @staticmethod
    def install_script() -> List[str]:
        dir = "~/.local/share/bash-completion/completions"
        script: List[str] = []
        script.append(f"mkdir -vp {dir}")
        name = "nomadtools"
        upname = name.upper().replace("-", "_")
        script.append(
            f"echo 'eval \"$(_{upname}_COMPLETE=bash_source {name})\"' > {dir}/{name}"
        )
        return script

    @staticmethod
    def install():
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


def dict_remove_none(data: T) -> T:
    """Remove all elements that are set to None"""
    if isinstance(data, dict):
        ret = {
            k: dict_remove_none(v) for k, v in data.items() if v is not None and v != {}
        }
    elif isinstance(data, list):
        ret = [dict_remove_none(e) for e in data if e is not None]
    else:
        ret = data
    return cast(T, ret)
