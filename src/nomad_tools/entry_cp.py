#!/usr/bin/env python3

from __future__ import annotations

import collections
import contextlib
import enum
import logging
import os
import re
import shlex
import string
import subprocess
import sys
import textwrap
import traceback
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from shlex import quote, split
from typing import IO, Any, Dict, Iterator, List, Optional
from urllib.parse import parse_qs, urlparse

import click
import clickdc
from click.shell_completion import CompletionItem
from typing_extensions import Protocol, override

from . import nomadlib, taskexec
from .common import (
    cached_property,
    help_h_option,
    mynomad,
    namespace_option,
    nomad_find_job,
)
from .common_base import quotearr
from .transferstats import transfer_stats

log = logging.getLogger(Path(__file__).name)
ARGS: "Args"

###############################################################################


def popen_raise_for_returncode(pp: subprocess.Popen):
    if pp.returncode:
        raise subprocess.CalledProcessError(pp.returncode, pp.args)


class MyPopen(Protocol):
    stdin: Optional[IO[bytes]]
    stdout: Optional[IO[bytes]]


###############################################################################


@dataclass
class Mypath(ABC):
    """pathlib.Path with some special methods"""

    path: str
    """
    This is string represented of the path.
    This has to be a string to properly handle /. suffixes in strings given by user.
    """

    def check_output(self, cmd: str) -> str:
        log.debug(f"+ {cmd}")
        return subprocess.check_output(split(cmd), text=True)

    @contextlib.contextmanager
    def popen(
        self,
        cmd: str,
        stdin: int = subprocess.DEVNULL,
        stdout: int = subprocess.DEVNULL,
    ) -> Iterator[MyPopen]:
        log.debug(f"+ {cmd}")
        with subprocess.Popen(split(cmd), stdin=stdin, stdout=stdout) as pp:
            yield pp
        popen_raise_for_returncode(pp)

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
        # Dirname may return an empty string. Return dot in such case.
        return quote(os.path.dirname(self.path) or ".")

    def quotename(self):
        return quote(os.path.basename(self.path) or ".")

    def quotepath(self):
        return quote(self.path)


class FileType(enum.Enum):
    file = enum.auto()
    dir = enum.auto()
    notexists = enum.auto()
    other = enum.auto()


@dataclass
class NomadMypath(Mypath):
    """Represents a path inside a specific Nomad task"""

    allocation: str
    task: str

    @override
    def check_output(self, cmd: str) -> str:
        log.debug(f"+ {self.allocation}/{self.task} {cmd}")
        with taskexec.TaskExec(self.allocation, self.task, split(cmd)) as pp:
            pp.close_stdin()
            buf: bytes = pp.read()
        pp.raise_for_returncode(buf)
        return buf.decode(errors="replace")

    @override
    @contextlib.contextmanager
    def popen(
        self,
        cmd: str,
        stdin: int = subprocess.DEVNULL,
        stdout: int = subprocess.DEVNULL,
    ) -> Iterator[MyPopen]:
        log.debug(f"+ {self.allocation}/{self.task} {cmd}")
        i = str(bool(stdin != subprocess.DEVNULL)).lower()
        cmdarr: List[str] = [
            *f"nomad alloc exec -i={quote(i)} -t=false -task={quote(self.task)}".split(),
            self.allocation,
            *split(cmd),
        ]
        log.debug(f"+ {quotearr(cmdarr)}")
        with subprocess.Popen(cmdarr, stdin=stdin, stdout=stdout) as pp:
            yield pp
        popen_raise_for_returncode(pp)

    @override
    def isstdin(self) -> bool:
        return False

    @override
    def isnomad(self) -> bool:
        return True

    def _nomadrunsh(self, script: str, *args: str) -> str:
        cmd = ["sh", "-c", script, *args]
        cmdstr = quotearr(cmd)
        log.debug(f"+ {cmdstr}")
        return self.check_output(cmdstr).strip()

    @cached_property
    def _nomadstat(self) -> FileType:
        assert self.isnomad()
        # Execute a script inside the container to determine the type of the file.
        script = (
            'if [ -f "$1" ]; then echo file;'
            ' elif [ -d "$1" ]; then echo dir;'
            ' elif ! [ -e "$1" ]; then echo notexists;'
            " else echo other; fi"
        )
        stat = self._nomadrunsh(script, "sh", self.path)
        log.debug(f"stat = {stat}")
        return FileType[stat]

    @override
    def exists(self) -> bool:
        return self._nomadstat != FileType.notexists

    @override
    def is_file(self) -> bool:
        return self._nomadstat == FileType.file

    @override
    def is_dir(self) -> bool:
        return self._nomadstat == FileType.dir

    @override
    def mkdir(self):
        v = "v" if ARGS.verbose else ""
        self._nomadrunsh(f"mkdir -{v}p {quote(self.path)}")

    @staticmethod
    def __colon(arg: str):
        """Replaces single colon by escaped for printing"""
        return arg.replace(":", r"\:")

    @override
    def __str__(self):
        return f":{self.__colon(self.allocation)}:{self.__colon(self.task)}:{quote(self.__colon(str(self.path)))}"

    def compgen(self) -> List[str]:
        """Given incomplete line, generate compgen of files inside a running allocation if possible"""
        # This is some magic to handle find arguments.
        path: str = self.path
        parts: List[str] = path[::-1].split("/", 1)
        noslash: bool = len(parts) == 1
        searchdir: str = "." if noslash else (parts[1][::-1] + "/")
        searchname: str = parts[0][::-1]
        searchnamenoglob: str = (
            searchname.replace("\\", "\\\\")
            .replace(r"*", r"\*")
            .replace(r"?", r"\?")
            .replace(r"[", r"\[")  # ]]
        )
        # Repetition in output means it is a directory.
        # Technically, maxdepth and mindepth are not POSIX.
        cmd: str = (
            f'find {quote(searchdir)} -maxdepth 1 -mindepth 1 -name {quote(searchnamenoglob)}"*"'
            f' "(" -type f  -print ")" -o "(" -type d -print -print ")"'
        )
        output = self.check_output(cmd)
        # Add trailing / to directories.
        ret: List[str] = [
            k if v == 1 else (k + "/")
            for k, v in collections.Counter(output.split("\n")).items()
            if k
        ]
        # In case of noslash is given, the initial part is going to be './'. Strip it.
        ret = [x[2 if noslash else 0 :] for x in ret]
        return ret


###############################################################################

NOSPACE = CompletionItem("", type="nospace")


@dataclass
class ArgPath:
    """Represents splitted up parts of the input"""

    _description = textwrap.dedent(
        """\
        :ALLOCATION[:TASK]:[PATH]
        JOB[:[GROUP][:TASK]]:[PATH]
        task://JOB[@NAMESPACE]/PATH[?group=GROUP][&alloc=ALLOC][&task=TASK][&hostname=HOSTNAME][&node=NODE]
        PATH
        -
        """
    )

    arg: str
    path: str
    arr: Optional[List[str]] = None
    alloc: Optional[str] = None
    job: Optional[str] = None
    group: Optional[str] = None
    task: Optional[str] = None
    namespace: Optional[str] = None
    hostname: Optional[str] = None
    node: Optional[str] = None

    @staticmethod
    def mk(arg: str) -> ArgPath:
        """Converts the argument into it's parts represented with object"""
        if not arg.startswith("task://"):
            # Split string on un-escaped colon, and then replaced secaped colons by colons
            arr: List[str] = [x.replace(r"\:", ":") for x in re.split(r"(?<!\\):", arg)]
            assert len(arr) <= 4, f"Could not parse argument: too many colons: {arg!r}"
            path: str = arr[-1]
            if len(arr) > 1:
                if arr[0] == "":
                    return ArgPath(
                        arg=arg,
                        arr=arr,
                        path=path,
                        alloc=arr[1],
                        task=arr[2] if len(arr) > 3 else None,
                    )
                else:
                    return ArgPath(
                        arg=arg,
                        arr=arr,
                        path=path,
                        job=arr[0],
                        group=arr[1] if len(arr) > 2 else None,
                        task=arr[2] if len(arr) > 3 else None,
                    )
            return ArgPath(arg=arg, arr=arr, path=path)
        o = urlparse(arg)
        assert o.scheme == "task", "Internal error"
        assert o.port is None, f"Port in URL doesn't make sense: {arg!r}"
        qsl: Dict[str, List[str]] = parse_qs(o.query)
        qs: Dict[str, str] = {k: v[-1] for k, v in qsl.items()}
        fields = "alloc task group hostname node".split()
        for k in qs.keys():
            assert k in fields, f"Unknown URL parameter {k}: {arg!r}"
        return ArgPath(
            arg=arg,
            path=o.path[1:] if o.path and o.path[0] == "/" else o.path,
            job=o.username if o.username else o.hostname,
            namespace=o.hostname if o.username else None,
            arr=None,
            **qs,
        )

    def __post_init__(self):
        assert (
            (not self.alloc and not self.job)
            or (self.alloc and not self.job)
            or (not self.alloc and self.job)
        ), f"Internal error: initialized in both alloc and job mode: {self.arg!r} {self.alloc} {self.job}"
        if self.alloc:
            alloweddigits = string.hexdigits + "-"
            assert all(
                c in alloweddigits for c in self.alloc
            ), f"Allocation ID can only be one of {alloweddigits!r}: {self.alloc}"

    @cached_property
    def __find_jobid(self):
        assert self.job
        return nomad_find_job(self.job)

    @cached_property
    def __allocations(self):
        """Return running allocations with ID starting with self.alloc"""
        assert self.alloc is not None
        # It is only possible to prefix using even length. Query uneven length using filter expression.
        alloc = "".join(c for c in self.alloc if c in string.hexdigits)
        params = dict(
            prefix=alloc[: len(alloc) // 2 * 2],
            filter=f'ID matches "^{self.alloc}"' if len(alloc) % 2 != 0 else None,
        )
        allocations: List[nomadlib.Alloc] = [
            nomadlib.Alloc(x) for x in mynomad.get("allocations", params=params)
        ]
        return self.__filter_alocations(allocations)

    def __filter_alocations(
        self, allocations: List[nomadlib.Alloc]
    ) -> List[nomadlib.Alloc]:
        return [
            alloc
            for alloc in allocations
            if alloc.is_running()
            and len(alloc.get_taskstates()) != 0
            and (not self.alloc or alloc.ID.startswith(self.alloc))
            and (not self.task or self.task in alloc.get_tasknames())
            and (not self.group or alloc.TaskGroup == self.group)
            and (not self.hostname or alloc.NodeName == self.hostname)
            and (not self.node or alloc.NodeID == self.node)
        ]

    @cached_property
    def __allocation(self):
        """Return the single running allocation with ID starting with self.alloc"""
        assert self.alloc
        allocs = self.__allocations
        assert len(allocs) > 0, f"Found no running allocations matching {self.arg!r}"
        assert (
            len(allocs) == 1
        ), f"Found multiple running allocations matching {self.arg!r}"
        return allocs[0]

    def to_mypath(self) -> Mypath:
        """Converts the argument into NomadMypath or Mypath object"""
        # The job here is to find the allocation associated with specified parameters.
        if self.alloc:
            allocations = [self.__allocation]
        elif self.job:
            jobid = self.__find_jobid
            allocations = [
                nomadlib.Alloc(x) for x in mynomad.get(f"job/{jobid}/allocations")
            ]
            allocations = [x for x in allocations if x.is_running()]
            assert len(allocations) >= 1, f"Job {jobid} has no running allocations"
        else:
            # It is a local filesystem path, just return.
            return Mypath(self.path)
        # Filter using group.
        if self.group:
            allocations = [x for x in allocations if x.TaskGroup == self.group]
            assert (
                len(allocations) >= 1
            ), f"No running allocations after matching group: {self.arg!r}"
        # Filter on the task.
        if self.task:
            allocations = [x for x in allocations if self.task in x.get_tasknames()]
            assert (
                len(allocations) >= 1
            ), f"No running allocations after matching task: {self.arg!r}"
        assert (
            len(allocations) == 1
        ), f"Found multiple running allocations mathing {self.arg!r}: {' '.join(x.ID for x in allocations)}"
        allocation = allocations[0]
        if self.task:
            task = self.task
        else:
            # Get running tasks of specific allocation.
            runningtasks: List[str] = [
                task
                for task, state in allocation.get_taskstates().items()
                if state.FinishedAt is None
            ]
            assert (
                len(runningtasks) != 0
            ), f"No running tasks found in allocation {allocation.ID} matching {self.arg!r}"
            assert len(runningtasks) == 1, (
                f"Multiple running tasks found in allocation {allocation.ID}"
                f" matching {self.arg!r}: {' '.join(runningtasks)}"
            )
            task = runningtasks[0]
        return NomadMypath(self.path, allocation.ID, task)

    @staticmethod
    def __filter(
        arr: List[str], prefix: str, suffix: str = ":"
    ) -> List[CompletionItem]:
        """Filter list of string by prefix and convert to CompletionItem. Also remove duplicates"""
        ret = [
            CompletionItem(shlex.quote(f"{x}{suffix}"))
            for x in sorted(list(set(x for x in arr if x.startswith(prefix))))
        ]
        return ret

    @staticmethod
    def debug(msg: str):
        if os.environ.get("COMP_DEBUG"):
            print(f"\n{msg}\n", file=sys.stderr)

    @classmethod
    def debugexception(cls, e: Exception):
        cls.debug(f"{traceback.format_exc()} Exception: {e}")

    def __compgen(self) -> List[CompletionItem]:
        """Given incomplete line, generate compgen of files inside a running allocation if possible"""
        assert self.alloc or self.job
        mypath = self.to_mypath()
        assert mypath.isnomad()
        assert isinstance(mypath, NomadMypath)
        return self.compgen_nomadmypath(mypath)

    @classmethod
    def compgen_nomadmypath(cls, mypath: NomadMypath) -> List[CompletionItem]:
        arr: List[str] = mypath.compgen()
        ret: List[CompletionItem] = cls.__filter(arr, mypath.path, suffix="")
        addnospace: bool = len(ret) > 1 or any(x.value[-1] == "/" for x in ret)
        if addnospace:
            ret = [NOSPACE, *ret]
        return ret

    def __complete_job_name(self, last: str, suffix: str = ":"):
        jobs = [nomadlib.Job(x) for x in mynomad.get("jobs", params=dict(prefix=last))]
        jobs = [
            x
            for x in jobs
            if not x.is_dead()
            and (not self.group or self.group in [g.Name for g in x.TaskGroups])
            and (
                not self.task
                or self.task in [t.Name for g in x.TaskGroups for t in g.Tasks]
            )
        ]
        return self.__filter([x.ID for x in jobs], last, suffix=suffix)

    def __gen_shell_complete_arr(self) -> List[CompletionItem]:
        """Generate completion for JOB:GROUP:TASK:PATH or :ALLOCATION:TASK:PATH"""
        arr = self.arr
        assert arr
        add: List[CompletionItem] = []
        last = self.path
        if len(arr) == 1:
            # PATH...
            if "/" not in arr[0]:
                # JOB...
                add = self.__complete_job_name(last)
            files = CompletionItem(self.arg, type="file")
            return [files] + add
        elif arr[0] == "" and len(arr) == 2:
            # :ALLOCATION...
            assert (
                self.alloc is not None
            ), f"Internal error: self.alloc is None on allocation path: {self}"
            return [NOSPACE, *self.__filter([x.ID for x in self.__allocations], self.alloc)]
        # If there is only a single matching allocation:task pair, just use it.
        try:
            # :ALLOCATION:PATH...
            # :ALLOCATION:TASK:PATH...
            # JOB:PATH...
            # JOB:GROUP:PATH...
            # JOB:GROUP:TASK:PATH...
            return self.__compgen()
        except Exception as e:
            ArgPath.debugexception(e)
        # Otherwise show completions for the task or group names.
        if arr[0] == "":
            if len(arr) == 3:
                # :ALLOCATION:TASK:...
                return self.__filter(self.__allocation.get_tasknames(), last)
        elif len(arr) in [2, 3]:
            jobid = self.__find_jobid
            job = nomadlib.Job(mynomad.get(f"job/{jobid}"))
            if len(arr) == 2:
                # JOB:GROUP...
                groups = [g.Name for g in job.TaskGroups]
                return self.__filter(groups, last)
            elif len(arr) == 3:
                # JOB:GROUP:TASK:...
                tasks = [
                    t.Name
                    for g in job.TaskGroups
                    if (not self.group or g.Name == self.group)
                    for t in g.Tasks
                ]
                return self.__filter(tasks, last)
        return []

    def __gen_shell_complete_url(self) -> List[CompletionItem]:
        self.debug(f"{self}")
        if self.arg.count("/") == 2:
            if "@" not in self.arg:
                return self.__complete_job_name(self.job or "", suffix="/")
            nss = mynomad.get("namespaces", params={"prefix": self.namespace or ""})
            return self.__filter(
                [x["Name"] for x in nss], self.namespace or "", suffix="/"
            )
        # If there is only a single matching allocation:task pair, just use it.
        try:
            return self.__compgen()
        except Exception as e:
            ArgPath.debugexception(e)
        return []

    def gen_shell_complete(self) -> List[CompletionItem]:
        """Generate completion given value"""
        if self.arr:
            return self.__gen_shell_complete_arr()
        else:
            return self.__gen_shell_complete_url()


class NomadOrHostMyPath(click.ParamType):
    """Click parameter representing Nomad or host path"""

    name = "path"

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Mypath:
        """Entrypoint for click option conversion"""
        return value if isinstance(value, Mypath) else ArgPath.mk(value).to_mypath()

    def shell_complete(
        self, ctx: click.Context, param: click.Parameter, incomplete: str
    ) -> List[CompletionItem]:
        try:
            return ArgPath.mk(incomplete).gen_shell_complete()
        except Exception as e:
            ArgPath.debugexception(e)
        return []


###############################################################################


def nomadpipe(src: Mypath, dst: Mypath, srccmd: str, dstcmd: str):
    stats: bool = (ARGS.stats is None and sys.stderr.isatty()) or ARGS.stats is True
    with dst.popen(dstcmd, stdin=subprocess.PIPE) as pdst:
        assert pdst.stdin
        if stats:
            with src.popen(srccmd, stdout=subprocess.PIPE) as psrc:
                assert psrc.stdout
                transfer_stats(psrc.stdout.fileno(), pdst.stdin.fileno(), ARGS.pv)
        else:
            with src.popen(srccmd, stdout=pdst.stdin.fileno()) as psrc:
                pass


def tar_out_opt() -> str:
    return (
        ("z" if ARGS.gzip else "")
        + ("j" if ARGS.bzip2 else "")
        + ("J" if ARGS.xz else "")
    )


def tar_in_opt() -> str:
    return (
        tar_out_opt()
        # If dryrun, pass vt to print to stdout.
        + ("vt" if ARGS.dryrun else "x")
        + ("v" if ARGS.verbose else "")
        # If archive, pass p to preserve permissions when not root.
        # If not archive, pass o to do not preserve permissions even as root.
        # See tar docs.
        + ("p" if ARGS.archive else "o")
    )


def nomadtarpipe(src: Mypath, dst: Mypath, tar_c: str, tar_x: str):
    nomadpipe(
        src,
        dst,
        f"tar -c{tar_out_opt()}f - {tar_c}",
        f"tar -{tar_in_opt()}f - {tar_x}",
    )


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
            prog: Optional[str] = (
                "gzip"
                if ARGS.gzip
                else "xz"
                if ARGS.xz
                else "bzip2"
                if ARGS.bzip2
                else None
            )
            srcscript: str = f"{prog} -c" if prog else "cat"
            dstscript: str = f"{prog} -d" if prog else "cat"
            nomadpipe(
                src,
                dst,
                f"{srcscript} {src.quotepath()}",
                (
                    "sh -c " + quote(f"{dstscript} >{dst.quotepath()}")
                    if not ARGS.dryrun
                    else f"true -- {quote(dstscript)} {dst.quotepath()}"
                ),
            )
        else:
            exit(f"Not a file or directory: {dst}")
    # SRC_PATH specifies a directory
    elif src.is_dir():
        # DEST_PATH does not exist
        if not dst.exists():
            # DEST_PATH is created as a directory and the contents of the source directory are copied into this directory
            log.info(f"New mkdir {src} -> {dst}")
            log.debug(f"mkdir {dst}")
            if not ARGS.dryrun:
                dst.mkdir()
            tar_c = f"-C {src.quotepath()} ."
            # And unpack to parent directory.
            components: int = src.path.count("/")
            tar_x = f"-C {dst.quotepath()} --strip-components={components}"
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
        assert not dst.isstdin(), f"Both operands are '-': {src} {dst}"
        with dst.popen(
            f"tar -{tar_in_opt()}f - -C {dst.quotepath()}",
            stdin=sys.stdin.fileno(),
        ):
            pass
    elif dst.isstdin():
        log.info(f"Stream {src} -> stdout")
        with src.popen(
            f"tar -c{tar_out_opt()}f - -C {src.quoteparent()} -- {src.quotename()}",
            stdout=sys.stdout.fileno(),
        ):
            pass
    else:
        assert 0, "Internal error - neither source nor dest is equal to -"


def rsync_mode(src: Mypath, dst: Mypath):
    assert not src.isstdin()
    assert not dst.isstdin()
    if isinstance(src, NomadMypath):
        srcn = True
        npath = src
    elif isinstance(dst, NomadMypath):
        srcn = False
        npath = dst
    else:
        assert False, "One of SOURCE and DEST can be Nomad path"
    rsh = f"nomad alloc exec -t=false -i=true -task={quote(npath.task)} {quote(npath.allocation)}"
    srcp = (":" if srcn else "") + src.quotepath().replace(":", r"\:")
    dstp = ("" if srcn else ":") + dst.quotepath().replace(":", r"\:")
    cmd = f"rsync {ARGS.rsyncargs} --rsh={quote(rsh)} {srcp} {dstp}"
    log.debug(f"+ {quotearr(split(cmd))}")
    if not ARGS.dryrun:
        subprocess.check_call(split(cmd))


###############################################################################


@dataclass
class Args:
    gzip: bool = clickdc.option("-z", help="Pass -z to tar")
    xz: bool = clickdc.option("-J", help="Pass -J to tar")
    bzip2: bool = clickdc.option("-j", help="Pass -j to tar")
    archive: bool = clickdc.option(
        "-a",
        help="Archive mode (copy all uid/gid information)",
    )
    #
    rsync: bool = clickdc.option(help="Rsync two paths")
    rsyncargs: str = clickdc.option(help="Shell quoted rsync options", default="")
    #
    pv: Optional[bool] = clickdc.option(
        "--pv/--no-pv",
        help="Use pv to show statistics. By default, if pv is available, it will be used.",
        default=None,
    )
    stats: Optional[bool] = clickdc.option(
        "--stats/--no-stats",
        help="Display statistics. By default, if stderr is a terminal, stats will be shown.",
        default=None,
    )
    #
    dryrun: bool = clickdc.option(
        "-n",
        help="Do tar -vt for unpacking. Usefull for listing files for debugging.",
    )
    verbose: int = clickdc.option("-v", "--verbose", count=True)
    quiet: int = clickdc.option("-q", "--quiet", count=True)
    source: Mypath = clickdc.argument(type=NomadOrHostMyPath())
    dest: Mypath = clickdc.argument(type=NomadOrHostMyPath())


@click.command(
    "cp",
    help=f"""
Copy files/folders between a nomad allocation and the local filesystem.
Use '-' as the source to read a tar archive from stdin
and extract it to a directory destination in a container.
Use '-' as the destination to stream a tar archive of a
container source to stdout.
The logic mimics docker cp.

\b
Both source and dest take one of the forms:
{textwrap.indent(ArgPath._description, '   ')}

To use colon in any part of the path, escape it with backslash.

\b
Examples:
    nomad-tools cp -n :9190d781:/tmp ~/tmp
    nomad-tools cp -vn -Nservices promtail:/. ~/tmp
""",
    epilog="""
Written by Kamil Cukrowski 2023. Licensed under GNU GPL version or later.
""",
)
@namespace_option()
@help_h_option()
@clickdc.adddc("args", Args)
def cli(args: Args):
    global ARGS
    ARGS = args
    verbose = 1 + ARGS.verbose - ARGS.quiet
    logging.basicConfig(
        level=(
            logging.DEBUG
            if verbose > 1
            else logging.INFO
            if verbose > 0
            else logging.WARNING
        ),
        format="%(levelname)s %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    )
    log.debug(f"ARGS={ARGS}")
    if ARGS.rsync:
        rsync_mode(ARGS.source, ARGS.dest)
    elif ARGS.source.isstdin() or ARGS.dest.isstdin():
        stream_mode(ARGS.source, ARGS.dest)
    else:
        copy_mode(ARGS.source, ARGS.dest)


if __name__ == "__main__":
    cli()
