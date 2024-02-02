import dataclasses
import os
import subprocess
import sys


@dataclasses.dataclass
class Colors:
    """ASCII TERM colors"""

    bold: str = "bold"
    reset: str = "sgr0"
    black: str = "setaf 0"
    red: str = "setaf 1"
    green: str = "setaf 2"
    orange: str = "setaf 3"
    blue: str = "setaf 4"
    magenta: str = "setaf 5"
    cyan: str = "setaf 6"
    white: str = "setaf 7"
    brightblack: str = "setaf 8"
    brightred: str = "setaf 9"
    brightgreen: str = "setaf 10"
    brightorange: str = "setaf 11"
    brightblue: str = "setaf 12"
    brightmagenta: str = "setaf 13"
    brightcyan: str = "setaf 14"
    brightwhite: str = "setaf 15"


empty = Colors(**{f.name: "" for f in dataclasses.fields(Colors)})
"""Empty colors"""


def init_ex() -> Colors:
    """Return Colors with ANSI escape sequences extracted from tput"""
    if not sys.stdout.isatty() or not sys.stderr.isatty() or os.environ.get("NO_COLOR"):
        return empty
    tputdict = dataclasses.asdict(Colors())
    tputscript = "\nlongname\nlongname\n".join(tputdict.values())
    longname = subprocess.run(
        "tput longname".split(),
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    ).stdout
    # Protect against empty longname.
    if not longname:
        return empty
    ret = subprocess.run(
        "tput -S".split(),
        input=tputscript,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    ).stdout
    retarr = ret.split(longname + longname)
    if len(tputdict.keys()) != len(retarr):
        return empty
    return Colors(*retarr)


def init() -> Colors:
    try:
        return init_ex()
    except Exception:
        return empty
