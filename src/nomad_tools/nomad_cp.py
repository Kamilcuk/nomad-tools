#!/usr/bin/env python3

from __future__ import annotations

import argparse
import enum
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from shlex import quote, split
from typing import Any, List, Optional

import click
from click.shell_completion import BashComplete, CompletionItem
from typing_extensions import override

from . import nomadlib, taskexec
from .common import (
    cached_property,
    common_options,
    mynomad,
    namespace_option,
    nomad_find_job,
)
from .common_base import quotearr

log = logging.getLogger(Path(__file__).name)


###############################################################################


@dataclass
class Mypath(ABC):
    """pathlib.Path with some special methods"""

    path: str
    """
    This is string represented of the path.
    This has to be a string to properly handle /. suffixes in strings given by user.
    """

    def run(self, cmd: str):
        log.debug(f"+ {cmd}")
        subprocess.run(split(cmd), check=True)

    def check_output(self, cmd: str) -> str:
        log.debug(f"+ {cmd}")
        return subprocess.check_output(split(cmd), text=True)

    def popen(
        self, cmd: str, stdout: Optional[int] = None, stdin: Optional[int] = None
    ) -> Any:
        log.debug(f"+ {cmd}")
        return subprocess.Popen(split(cmd), stdout=stdout, stdin=stdin)

    def isstdin(self) -> bool:
        return str(self.path) == "-"

    def isnomad(self) -> bool:
        return False

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def is_file(self) -> bool:
        return os.path.isfile(self.path)

    def is_dir(self) -> bool:
        return os.path.isdir(self.path)

    def mkdir(self):
        Path(self.path).mkdir(parents=True, exist_ok=True)

    @property
    def name(self):
        return os.path.basename(self.path)

    def __str__(self):
        return str(self.path)

    def __itruediv__(self, name: str):
        self.path = os.path.join(self.path, name)
        return self

    def quoteparent(self):
        return quote(os.path.dirname(self.path))

    def quotename(self):
        return quote(os.path.basename(self.path))

    def quotepath(self):
        return quote(self.path)


class FileType(enum.Enum):
    file = enum.auto()
    dir = enum.auto()
    none = enum.auto()


@dataclass
class NomadMypath(Mypath):
    """Represents a path inside a specific Nomad task"""

    allocation: str
    task: str

    @override
    def run(self, cmd: str):
        log.debug(f"+ {self.allocation} {self.task} {cmd}")
        taskexec.run(self.allocation, self.task, split(cmd))

    @override
    def check_output(self, cmd: str) -> str:
        log.debug(f"+ {self.allocation} {self.task} {cmd}")
        return taskexec.check_output(self.allocation, self.task, split(cmd), text=True)

    @override
    def popen(
        self, cmd: str, stdout: Optional[int] = None, stdin: Optional[int] = None
    ) -> Any:
        log.debug(f"+ {self.allocation} {self.task} {cmd}")
        return taskexec.NomadPopen(
            self.allocation, self.task, split(cmd), stdout=stdout, stdin=stdin
        )

    @override
    def isstdin(self) -> bool:
        return False

    @override
    def isnomad(self) -> bool:
        return True

    def _nomadrunsh(self, script: str, *args: str) -> str:
        cmd = [
            "sh",
            "-c",
            script,
            *args,
        ]
        cmdstr = quotearr(cmd)
        log.debug(f"+ {cmdstr}")
        return self.check_output(cmdstr).strip()

    @cached_property
    def _nomadstat(self) -> FileType:
        assert self.isnomad()
        # Execute a script inside the container to determine the type of the file.
        script = 'if [ -f "$1" ]; then echo file; elif [ -d "$1" ]; then echo dir; else echo none; fi'
        stat = self._nomadrunsh(script, "--", self.path)
        log.debug(f"stat = {stat}")
        return FileType[stat]

    @override
    def exists(self) -> bool:
        return self._nomadstat != FileType.none

    @override
    def is_file(self) -> bool:
        return self._nomadstat == FileType.file

    @override
    def is_dir(self) -> bool:
        return self._nomadstat == FileType.dir

    @override
    def mkdir(self):
        v = "v" if args.verbose else ""
        self._nomadrunsh(f"mkdir -{v}p {quote(self.path)}")

    @staticmethod
    def __colon(arg: str):
        """Replaces single colon by esacped for printing"""
        return arg.replace(":", r"\:")

    @override
    def __str__(self):
        return f":{self.__colon(self.allocation)}:{self.__colon(self.task)}:{quote(self.__colon(str(self.path)))}"


###############################################################################


class NomadOrHostMyPath(click.ParamType):
    _description = textwrap.dedent(
        """\
        :ALLOCATION:SRC_PATH
        :ALLOCATION:TASK:SRC_PATH
        :ALLOCATION:GROUP:TASK:SRC_PATH
        JOB:SRC_PATH
        JOB:TASK:SRC_PATH
        JOB:GROUP:TASK:SRC_PATH
        SRC_PATH
        -
        """
    )

    name = "path"

    @dataclass
    class Elem:
        """Represents splitted up parts of the input"""

        path: str
        isalloc: bool = False
        id: Optional[str] = None
        group: Optional[str] = None
        task: Optional[str] = None

    @staticmethod
    def __split_on_colon(arg: str) -> List[str]:
        """Splits string on un-escaped colon, and then replaced secaped colons by colons"""
        return [x.replace(r"\:", ":") for x in re.split(r"(?<!\\):", arg)]

    @classmethod
    def __split_arg(cls, arg: str) -> NomadOrHostMyPath.Elem:
        """Converts the argument into it's parts represented with object"""
        arr = cls.__split_on_colon(arg)
        el = cls.Elem(arr[-1])
        if len(arr) > 1:
            if arr[0] == "":
                el.isalloc = True
                el.id = arr[1]
                arr = arr[1:]
            else:
                el.isalloc = False
                el.id = arr[0]
            el.group = arr[1] if len(arr) > 3 else None
            el.task = arr[2] if len(arr) > 3 else arr[1] if len(arr) > 2 else None
            assert len(arr) <= 5, f"Could not parse argument, too many colons: {arg}"
        return el

    @classmethod
    def __parse_arg(cls, arg: str) -> Mypath:
        """Converts the argument into NomadMypath or Mypath object"""
        el = cls.__split_arg(arg)
        if not el.id:
            return Mypath(el.path)
        # The job here is to find the allocation associated with specified parameters.
        if el.isalloc:
            allocations = [
                nomadlib.Alloc(x)
                for x in mynomad.get("allocations", params=dict(prefix=el.id))
            ]
            allocations = [x for x in allocations if x["ID"].startswith(el.id)]
            assert len(allocations) >= 1, f"No allocation starts with {el.id}"
            assert len(allocations) == 1, f"Multiple allocations start with {el.id}"
            allocation = allocations[0]
            assert allocation.is_running(), f"Allocation {allocation.ID} is not running"
            assert (
                len(allocation.get_taskstates()) != 0
            ), f"Allocation {allocation.ID} has no running tasks"
            allocations = [allocation]
        else:
            jobid = nomad_find_job(el.id)
            allocations = [
                nomadlib.Alloc(x) for x in mynomad.get(f"job/{jobid}/allocations")
            ]
            allocations = [x for x in allocations if x.is_running()]
            assert len(allocations) >= 1, f"Job {jobid} has no running allocations"
        if el.group:
            allocations = [x for x in allocations if x.TaskGroup == el.group]
            assert len(allocations) >= 1, f"No allocations matching group {el.group}"
        if el.task:
            allocations = [x for x in allocations if el.task in x.get_tasknames()]
            assert len(allocations) >= 1, f"No allocations running task {el.task}"
        assert (
            len(allocations) == 1
        ), f"Found multiple allocations mathing {arg!r}: {' '.join(x.ID for x in allocations)}"
        allocation = allocations[0]
        task = el.task if el.task else allocation.get_tasknames()[0]
        return NomadMypath(el.path, allocation.ID, task)

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Mypath:
        """Entrypoint for click option conversion"""
        return value if isinstance(value, Mypath) else self.__parse_arg(value)

    @staticmethod
    def __filter(
        arr: List[str], prefix: str, suffix: str = ":"
    ) -> List[CompletionItem]:
        """Filter list of string by prefix and convert to CompletionItem. Also remove duplicates"""
        return [
            CompletionItem(shlex.quote(f"{x}{suffix}"))
            for x in sorted(list(set(x for x in arr if x.startswith(prefix))))
        ]

    @staticmethod
    def debug(msg: str):
        if os.environ.get("COMP_DEBUG"):
            print(msg, file=sys.stderr)

    @classmethod
    def __compgen(cls, incomplete: str) -> List[CompletionItem]:
        """Given incomplete line, generate compgen of files inside a running allocation if possible"""
        try:
            mypath = cls.__parse_arg(incomplete)
        except AssertionError:
            return []
        assert mypath.isnomad()
        # This is some magic to handle find arguments.
        path: str = mypath.path
        parts: List[str] = path[::-1].split("/", 1)
        noslash: bool = len(parts) == 1
        searchdir: str = "." if noslash else parts[1][::-1] + "/"
        searchname: str = parts[0][::-1]
        searchnamenoglob = (
            searchname.replace("\\", "\\\\")
            .replace(r"*", r"\*")
            .replace(r"?", r"\?")
            .replace(r"[", r"\[")  # ]]
        )
        cmd = [
            "sh",
            "-c",
            textwrap.dedent(
                """\
                find "$1" -maxdepth 1 -mindepth 1 -type f -name "$2*"
                find "$1" -maxdepth 1 -mindepth 1 -type d -name "$2*" |
                    while IFS= read -r line; do echo "$line/"; done
                """
            ),
            "--",
            searchdir,
            searchnamenoglob,
        ]
        cmdstr = quotearr(cmd)
        cls.debug(f"+ {cmdstr}")
        try:
            output = mypath.check_output(cmdstr)
        except subprocess.CalledProcessError:
            return []
        # In case of noslash is given, the initial part is going to be './'. Strip it.
        ret = [x[2 if noslash else 0 :] for x in output.splitlines()]
        nospace = (
            [CompletionItem("", type="nospace")]
            if len(ret) > 1 or any(x[-1] == "/" for x in ret)
            else []
        )
        return nospace + cls.__filter(ret, path, suffix="")

    @classmethod
    def gen_shell_complete(cls, incomplete: str) -> List[CompletionItem]:
        """Generate completion given value"""
        arr = cls.__split_on_colon(incomplete)
        files = [CompletionItem(incomplete, type="file")]
        if len(arr) == 1:
            # JOB...
            # PATH...
            jobs = [x["ID"] for x in mynomad.get("jobs", params=dict(prefix=arr[0]))]
            return (
                files
                + [CompletionItem("", type="nospace")]
                + cls.__filter(jobs, arr[0])
            )
        elif arr[0] == "":
            if len(arr) == 2:
                # :ALLOCATION...
                allocidprefix = arr[1]
                allocations = [
                    x["ID"]
                    for x in mynomad.get("allocations", params=dict(prefix=arr[1]))
                    if x["ID"].startswith(allocidprefix)
                ]
                return [CompletionItem("", type="nospace")] + cls.__filter(
                    allocations, arr[1]
                )
            elif len(arr) in [3, 4, 5]:
                # :ALLOCATION:PATH...
                # :ALLOCATION:GROUP...
                # :ALLOCATION:GROUP:PATH...
                # :ALLOCATION:GROUP:TASK...
                # :ALLOCATION:GROUP:TASK:PATH...
                add = []
                if "/" not in arr[-1]:
                    allocidprefix = arr[1]
                    allocations = [
                        nomadlib.Alloc(x)
                        for x in mynomad.get(
                            "allocations", params=dict(prefix=allocidprefix)
                        )
                        if x["ID"].startswith(allocidprefix)
                    ]
                    assert (
                        len(allocations) == 1
                    ), f"Found multiple or none allocation matching prefix {allocidprefix}"
                    allocation = allocations[0]
                    if len(arr) <= 4:
                        add = [CompletionItem("", type="nospace")] + cls.__filter(
                            allocation.get_tasknames(), arr[-1]
                        )
                return add + cls.__compgen(incomplete)
        elif len(arr) in [2, 3, 4]:
            # JOB:PATH...
            # JOB:GROUP...
            # JOB:GROUP:PATH...
            # JOB:GROUP:TASK...
            # JOB:GROUP:TASK:PATH...
            add = []
            if "/" not in arr[-1]:
                jobid = nomad_find_job(arr[0])
                job = nomadlib.Job(mynomad.get(f"job/{jobid}"))
                tasks = [
                    t.Name
                    for tg in job.TaskGroups
                    if len(arr) < 3 or tg.Name == arr[2]
                    for t in tg.Tasks
                ]
                add = [CompletionItem("", type="nospace")] + cls.__filter(
                    tasks, arr[-1]
                )
            return add + cls.__compgen(incomplete)
        return []

    def shell_complete(
        self, ctx: click.Context, param: click.Parameter, incomplete: str
    ) -> List[CompletionItem]:
        try:
            return self.gen_shell_complete(incomplete)
        except Exception:
            pass
        return []


# Fix bash splitting completion on colon.
# Use __reassemble_comp_words_by_ref from bash-completion.
BashComplete.source_template = """\
    %(complete_func)s() {
        if [[ $(type -t __reassemble_comp_words_by_ref) != function ]]; then
            return -1
        fi
        local cword words=()
        __reassemble_comp_words_by_ref ":" words cword
        local IFS=$'\\n'
        response=$(env COMP_WORDS="${words[*]}" COMP_CWORD="$cword" %(complete_var)s=bash_complete $1)
        for completion in $response; do
            IFS=',' read type value <<< "$completion"
            case $type in
            dir) COMPREPLY=(); compopt -o dirnames; ;;
            file) COMPREPLY=(); compopt -o default; ;;
            plain) COMPREPLY+=($value); ;;
            nospace) compopt -o nospace; ;;
            esac
        done
    }
    %(complete_func)s_setup() {
        complete -o nosort -F %(complete_func)s %(prog_name)s
    }
    %(complete_func)s_setup;
"""


###############################################################################


def pipe(cmda: str, cmdb: str):
    """Run a pipe of cmda | cmdb"""
    arra = shlex.split(cmda)
    arrb = shlex.split(cmdb)
    log.debug(f"+ {quotearr(arra)} | {quotearr(arrb)}")
    with subprocess.Popen(arrb, stdin=subprocess.PIPE) as ps:
        subprocess.check_call(arra, stdin=subprocess.PIPE, stdout=ps.stdin)
    if ps.returncode:
        raise subprocess.CalledProcessError(ps.returncode, ps.args)


def run(cmd: str):
    """Run a command"""
    log.debug(f"+ {cmd}")
    subprocess.check_call(shlex.split(cmd), stdin=subprocess.DEVNULL)


def nomadpipe(src: Mypath, dst: Mypath, srccmd: str, dstcmd: str):
    pipe(f"{srccmd}", f"{dstcmd}")
    with src.popen(srccmd, stdout=subprocess.PIPE) as srcp:
        with dst.popen(dstcmd, stdin=subprocess.PIPE) as dstp:
            shutil.copyfileobj(srcp.stdout, dstp.stdin)
    for p in [srcp, dstp]:
        if p.returncode:
            raise subprocess.CalledProcessError(p.returncode, p.cmd)


def get_tar_x_opt():
    return (
        # If dryrun, pass vt to print to stdout.
        ("vt" if args.dryrun else "x")
        + ("v" if args.verbose else "")
        # If archive, pass p to preserve permissions when not root.
        # If not archive, pass o to do not preserve permissions even as root.
        # See tar docs.
        + ("p" if args.archive else "o")
    )


def nomadtarpipe(src: Mypath, dst: Mypath, tar_c: str, tar_x: str):
    global args
    x = get_tar_x_opt()
    nomadpipe(src, dst, f"tar -cf - {tar_c}", f"tar -{x}f - {tar_x}")


def copy_mode(src: Mypath, dst: Mypath):
    # logic from https://docs.docker.com/engine/reference/commandline/cp/
    # SRC_PATH specifies a file
    if src.is_file():
        # DEST_PATH does not exist and ends with /
        if not dst.exists() and str(dst)[-1:] == "/":
            exit("Error condition: the destination directory must exist.")
        # DEST_PATH is a directory
        if dst.is_dir():
            # the file is copied into this directory using the basename from SRC_PATH
            dst /= src.name
        if (
            # DEST_PATH does not exist
            # the file is saved to a file created at DEST_PATH
            not dst.exists()
            # DEST_PATH exists and is a file
            # the destination is overwritten with the source file’s contents
            or dst.is_file()
        ):
            # Copy one file.
            log.info(f"File {src} -> {dst}")
            nomadpipe(
                src,
                dst,
                f"cat {src.quotepath()}",
                f"sh -c 'cat >\"$1\"' -- {dst.quotepath()}"
                if not args.dryrun
                else f"true -- {dst.quotepath()}",
            )
        else:
            exit(f"Not a file or directory: {dst}")
    # SRC_PATH specifies a directory
    elif src.is_dir():
        # DEST_PATH does not exist
        if not dst.exists():
            # DEST_PATH is created as a directory and the contents of the source directory are copied into this directory
            log.info(f"New mkdir {src} -> {dst}")
            if not args.dryrun:
                log.debug(f"mkdir {dst}")
                dst.mkdir()
            tar_c = f"-C {src.quotepath()} ."
            # And unpack to parent directory.
            components: int = str(src.path).count("/")
            tar_x = f"-C {dst.quoteparent()} --strip-components={components}"
            nomadtarpipe(src, dst, tar_c, tar_x)
        # DEST_PATH exists and is a file
        elif dst.is_file():
            exit("Error condition: cannot copy a directory to a file")
        # DEST_PATH exists and is a directory
        elif dst.is_dir():
            # SRC_PATH does not end with /. (that is: slash followed by dot)
            if str(src)[-2:] != "/.":
                # the source directory is copied into this directory
                log.info(f"New dir {src} -> {dst}/{src.name}")
                tar_c = f"-C {src.quoteparent()} -- {src.quotename()}"
                tar_x = f"-C {dst.quotepath()}"
                nomadtarpipe(src, dst, tar_c, tar_x)
            # SRC_PATH does end with /. (that is: slash followed by dot)
            else:
                # the content of the source directory is copied into this directory
                log.info(f"Content of {src}/. -> {dst}/.")
                tar_c = f"-C {src.quotepath()} ."
                tar_x = f"-C {dst.quotepath()}"
                nomadtarpipe(src, dst, tar_c, tar_x)
        else:
            exit(f"Not a file or directory: {dst}")
    else:
        exit(f"Not a file or directory: {src}")


def stream_mode(src: Mypath, dst: Mypath):
    # One of source or dest is a -
    if src.isstdin():
        log.info(f"Stream stdin -> {dst}")
        assert dst.isstdin(), "Both operands are '-'"
        x = get_tar_x_opt()
        dst.run(f"tar -{x}f - -C {dst.quotepath()}")
    elif dst.isstdin():
        log.info(f"Stream {src} -> stdout")
        src.run(f"tar -cf - -C {src.quoteparent()} -- {src.quotename()}")
    else:
        assert 0, "Internal error - neither source nor dest is equal to -"


def test(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    script = """
        set -xeuo pipefail
        cmd="$1"
        cmd() { python -m nomad_tools.nomad_cp -vv "$@"; }
        tmpd=$(mktemp -d)
        trap 'rm -vrf "$tmpd"' EXIT
        mkdir "$tmpd/src" "$tmpd/dst"
        echo 123 > "$tmpd/src/1"

        echo "testing file copy without destination"
        cmd "$tmpd/src/1" "$tmpd/dst"
        test -e "$tmpd/dst/1"
        rm -v "$tmpd/dst/1"

        echo "tesing file copy to a different file"
        cmd "$tmpd/src/1" "$tmpd/dst/2"
        test -e "$tmpd/dst/2"
        rm -v "$tmpd/dst/2"

        echo "testing dir copy to a dir"
        cmd "$tmpd/src" "$tmpd/dst"
        test -e "$tmpd/dst/src/1"
        rm -vr "$tmpd/dst/src"

        echo "testing content of a dir copy to a dir"
        cmd "$tmpd/src/." "$tmpd/dst"
        test -e "$tmpd/dst/1"
        rm -v "$tmpd/dst/1"
        """
    subprocess.check_call(
        [
            "bash",
            "-c",
            script,
            "--",
            sys.argv[0],
        ]
    )
    ctx.exit()


###############################################################################


@click.command(
    help=f"""
Copy files/folders between a nomad allocation and the local filesystem.
Use '-' as the source to read a tar archive from stdin
and extract it to a directory destination in a container.
Use '-' as the destination to stream a tar archive of a
container source to stdout.
The logic mimics docker cp.

\b
Both source and dest take one of the forms:
{textwrap.indent(NomadOrHostMyPath._description, '   ')}

To use colon in any part of the part, escape it with backslash.

\b
Examples:
    nomad-cp -n :9190d781:/tmp ~/tmp
    nomad-cp -vn -Nservices promtail:/. ~/tmp
""",
    epilog="""
Written by Kamil Cukrowski 2023. Licensed under GNU GPL version or later.
""",
)
@click.option(
    "-n",
    "--dryrun",
    is_flag=True,
    help="Do tar -vt for unpacking. Usefull for listing files for debugging.",
)
@click.option("-v", "--verbose", count=True)
@namespace_option()
@click.option(
    "-a",
    "--archive",
    is_flag=True,
    help="Archive mode (copy all uid/gid information)",
)
@click.option("--test", is_flag=True, help="Run tests", is_eager=True, callback=test)
@click.argument("source", type=NomadOrHostMyPath())
@click.argument("dest", type=NomadOrHostMyPath())
@namespace_option()
@common_options()
def cli(source: Mypath, dest: Mypath, **kwargs):
    global args
    args = argparse.Namespace(**kwargs)
    logging.basicConfig(
        level=logging.DEBUG
        if args.verbose > 1
        else logging.INFO
        if args.verbose > 0
        else logging.WARNING,
        format="%(levelname)s %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    )
    if source.isstdin() or dest.isstdin():
        stream_mode(source, dest)
    else:
        copy_mode(source, dest)


if __name__ == "__main__":
    cli()
