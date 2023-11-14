#!/usr/bin/env python3

import argparse
import enum
import json
import logging
import re
import shlex
import subprocess
import sys
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from typing import List, Optional, Union

import click

from .common import common_options

log = logging.getLogger(Path(__file__).name)


def listquote(list: List[str]):
    return " ".join(quote(x) for x in list)


###############################################################################


@dataclass
class Mypath(ABC):
    path: Path

    def nomadrun(self, stdin: Union[bool, int] = False) -> str:
        return ""

    def isnomad(self):
        return False

    def exists(self):
        return self.path.exists()

    def is_file(self):
        return self.path.is_file()

    def is_dir(self):
        return self.path.is_dir()

    @property
    def name(self):
        return self.path.name

    def __str__(self):
        return str(self.path)

    def __truediv__(self, name: str):
        return type(self)(Path(f"{self}/{name}"))

    def quoteparent(self):
        return quote(str(self.path.parent))

    def quotename(self):
        return quote(str(self.path.name))

    def quotepath(self):
        return quote(str(self.path))


class FileType(enum.Enum):
    file = enum.auto()
    dir = enum.auto()
    none = enum.auto()


@dataclass
class NomadMypath(Mypath):
    nomadid: str
    _stat: Optional[FileType] = None

    def isnomad(self):
        return True

    def nomadrun(self, stdin: Union[bool, int] = False) -> str:
        # Return a properly escaped command line that allows to execute a command in the container.
        global args
        assert self.nomadid is not None
        idisjob = args.job or (
            # Try to be smart: if the id does not "look like" UUID number, assume it is a job name.
            not re.fullmatch("[0-9a-fA-F-]{1,36}", self.nomadid)
        )
        i = str(bool(stdin)).lower()
        return f"nomad alloc exec -t=false -i={i} " + (
            (
                (f"-namespace {quote(args.namespace)} " if args.namespace else "")
                + f"-job {quote(self.nomadid)}"
            )
            if idisjob
            else f"{quote(self.nomadid)}"
        )

    def _nomadstat(self) -> FileType:
        assert self.isnomad()
        if self._stat is None:
            # Execute a script inside the container to determine the type of the file.
            script = 'if [ -f "$@" ]; then echo file; elif [ -d "$@" ]; then echo dir; else echo none; fi'
            cmd = [
                *shlex.split(self.nomadrun()),
                "sh",
                "-c",
                script,
                "--",
                str(self.path),
            ]
            log.debug(f"+ {listquote(cmd)}")
            stat = subprocess.check_output(cmd, text=True).strip()
            log.debug(f"stat = {stat}")
            self._stat = FileType[stat]
        return self._stat

    def exists(self):
        return self._nomadstat() != FileType.none

    def is_file(self):
        return self._nomadstat() == FileType.file

    def is_dir(self):
        return self._nomadstat() == FileType.dir

    def __str__(self):
        return f"{self.nomadid}:{self.path}"


def mypath_factory(txt: str) -> Mypath:
    """
    ALLOCATION:SRC_PATH
    JOB:SRC_PATH
    JOB@TASK:SRC_PATH
    SRC_PATH
    -
    """
    arr = txt.split(":", 1)
    if len(arr) == 1:
        log.debug(f"Constructing Path({txt})")
        return Mypath(Path(txt))
    else:
        place = arr[0].split("@", 1)
        if len(place) == 1:
            log.debug(f"Constructing Nomad({arr[1]}, {arr[0]})")
            return NomadMypath(Path(arr[1]), arr[0])
        else:
            allocs = json.loads(
                subprocess.check_output(
                    "nomad operator api /v1/job/{place[0]}/allocations".split(),
                    text=True,
                )
            )
            allocs = [
                x
                for x in allocs
                if x["ClientStatus"] == "running" and place[1] in x["TaskStates"].keys()
            ]
            assert (
                len(allocs) < 2
            ), f"Multiple running allocations found for job {place[0]} and task {place[1]}: {' '.join(x['ID'] for x in allocs)}"
            assert (
                len(allocs) > 0
            ), f"No running allocations found for job {place[0]} and task {place[1]}"
            return NomadMypath(Path(arr[1]), allocs[0]["ID"])


###############################################################################


def pipe(cmda: str, cmdb: str):
    """Run a pipe of cmda | cmdb"""
    arra = shlex.split(cmda)
    arrb = shlex.split(cmdb)
    log.debug(f"+ {arrquote(arra)} | {arrquote(arrb)}")
    with subprocess.Popen(arrb, stdin=subprocess.PIPE) as ps:
        subprocess.check_call(arra, stdin=subprocess.PIPE, stdout=ps.stdin)
    if ps.returncode:
        raise subprocess.CalledProcessError(ps.returncode, ps.args)


def run(cmd: str):
    """Run a command"""
    log.debug(f"+ {cmd}")
    subprocess.check_call(shlex.split(cmd), stdin=subprocess.DEVNULL)


def get_tar_x_opt():
    return (
        # If dry_run, pass vt to print to stdout.
        ("vt" if args.dry_run else "x")
        + ("v" if args.verbose else "")
        # If archive, pass p to preserve permissions when not root.
        # If not archive, pass o to do not preserve permissions even as root.
        # See tar docs.
        + ("p" if args.archive else "o")
    )


def tarpipe(src: Mypath, dst: Mypath, tar_c: str, tar_x: str):
    global args
    x = get_tar_x_opt()
    pipe(
        f"{src.nomadrun()} tar -cf - {tar_c}",
        f"{dst.nomadrun(stdin=1)} tar -{x}f - {tar_x}",
    )


def copy_mode(src: Mypath, dst: Mypath):
    # logic from https://docs.docker.com/engine/reference/commandline/cp/
    # SRC_PATH specifies a file
    if src.is_file():
        # DEST_PATH does not exist and ends with /
        if not dst.exists() and args.dest[-1:] == "/":
            exit("Error condition: the destination directory must exist.")
        # DEST_PATH is a directory
        if dst.is_dir():
            # the file is copied into this directory using the basename from SRC_PATH
            dst = dst / src.name
        if (
            # DEST_PATH does not exist
            # the file is saved to a file created at DEST_PATH
            not dst.exists()
            # DEST_PATH exists and is a file
            # the destination is overwritten with the source file’s contents
            or dst.is_file()
        ):
            # Tar the file.
            log.info(f"File {src} -> {dst}")
            # --transform the file to the output filename.
            tar_c = f"-C {src.quoteparent()} --transform s:.\\*:{dst.quotename()}: -- {src.quotename()}"
            # Unpack the file at destination filename.
            tar_x = f"-C {dst.quoteparent()} -- {dst.quotename()}"
            tarpipe(src, dst, tar_c, tar_x)
        else:
            exit(f"Not a file or directory: {dst}")
    # SRC_PATH specifies a directory
    elif src.is_dir():
        # DEST_PATH does not exist
        if not dst.exists():
            # DEST_PATH is created as a directory and the contents of the source directory are copied into this directory
            log.info(f"New dir {src} -> {dst}")
            # Use --transform to prepent all paths with destination directory name.
            tar_c = f"-C {src.quotepath()} --transform s:^:{dst.quotename()}/: ."
            # And unpack to parent directory.
            tar_x = f"-C {dst.quoteparent()}"
            tarpipe(src, dst, tar_c, tar_x)
        # DEST_PATH exists and is a file
        elif dst.is_file():
            exit("Error condition: cannot copy a directory to a file")
        # DEST_PATH exists and is a directory
        elif dst.is_dir():
            # SRC_PATH does not end with /. (that is: slash followed by dot)
            if args.source[-2:] != "/.":
                # the source directory is copied into this directory
                log.info(f"New dir {src} -> {dst}/{src.name}")
                tar_c = f"-C {src.quoteparent()} -- {src.quotename()}"
                tar_x = f"-C {dst.quotepath()}"
                tarpipe(src, dst, tar_c, tar_x)
            # SRC_PATH does end with /. (that is: slash followed by dot)
            else:
                # the content of the source directory is copied into this directory
                log.info(f"Content of {src}/. -> {dst}/.")
                tar_c = f"-C {src.quotepath()} ."
                tar_x = f"-C {dst.quotepath()}"
                tarpipe(src, dst, tar_c, tar_x)
        else:
            exit(f"Not a file or directory: {dst}")
    else:
        exit(f"Not a file or directory: {src}")


def stream_mode(src: Mypath, dst: Mypath):
    # One of source or dest is a -
    if args.source == "-":
        log.info(f"Stream stdin -> {args.dest}")
        assert args.dest != "-", "Both operands are '-'"
        x = get_tar_x_opt()
        run(f"{dst.nomadrun(stdin=1)} tar -{x}f - -C {dst.quotepath()}")
    elif args.dest == "-":
        log.info(f"Stream {args.source} -> stdout")
        run(f"{src.nomadrun()} tar -cf - -C {src.quoteparent()} -- {src.quotename()}")
    else:
        assert 0, "Internal error - neither source nor dest is equal to -"


def test():
    script = """
        set -xeuo pipefail
        cmd="$1"
        tmpd=$(mktemp -d)
        trap 'rm -vrf "$tmpd"' EXIT
        mkdir "$tmpd/src" "$tmpd/dst"
        echo 123 > "$tmpd/src/1"

        echo "testing file copy without destination"
        "$cmd" -d "$tmpd/src/1" "$tmpd/dst"
        test -e "$tmpd/dst/1"
        rm -v "$tmpd/dst/1"

        echo "tesing file copy to a different file"
        "$cmd" -d "$tmpd/src/1" "$tmpd/dst/2"
        test -e "$tmpd/dst/2"
        rm -v "$tmpd/dst/2"

        echo "testing dir copy to a dir"
        "$cmd" -d "$tmpd/src" "$tmpd/dst"
        test -e "$tmpd/dst/src/1"
        rm -vr "$tmpd/dst/src"

        echo "testing content of a dir copy to a dir"
        "$cmd" -d "$tmpd/src/." "$tmpd/dst"
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


###############################################################################


@click.command(
    help="""
Copy files/folders between a nomad allocation and the local filesystem.
Use '-' as the source to read a tar archive from stdin
and extract it to a directory destination in a container.
Use '-' as the destination to stream a tar archive of a
container source to stdout.

\b
Both source and dest take one of the forms:
    ALLOCATION:SRC_PATH
    JOB:SRC_PATH
    JOB@TASK:SRC_PATH
    SRC_PATH
    -

\b
Examples:
  {log.name} -n 9190d781:/tmp ~/tmp
  {log.name} -vn -Nservices -job promtail:/. ~/tmp
""",
    epilog=f"""
Written by Kamil Cukrowski 2023. Licensed under GNU GPL version or later.
""",
)
@click.option(
    "-n",
    "--dry-run",
    is_flag=True,
    help="Do tar -vt for unpacking. Usefull for listing files for debugging.",
)
@click.option("-v", "--verbose", count=True)
@click.option(
    "-N",
    "-namespace",
    "--namespace",
    help="Nomad namespace",
)
@click.option(
    "-a",
    "--archive",
    is_flag=True,
    help="Archive mode (copy all uid/gid information)",
)
@click.option(
    "-j",
    "-job",
    "--job",
    is_flag=True,
    help="Use a **random** allocation from the specified job ID.",
)
@click.option("--test", is_flag=True, help="Run tests")
@click.argument("source")
@click.argument("dest")
@common_options()
def cli(**kwargs):
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
    if args.test:
        test()
        exit()
    src = mypath_factory(args.source)
    dst = mypath_factory(args.dest)
    if args.source == "-" or args.dest == "-":
        stream_mode(src, dst)
    else:
        copy_mode(src, dst)


if __name__ == "__main__":
    cli()
