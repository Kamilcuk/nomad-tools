#!/usr/bin/env python3

import argparse
import logging
import re
import shlex
import subprocess
import sys
from pathlib import Path
from shlex import quote
from typing import List

log = logging.getLogger(Path(__file__).name)


def listquote(list: List[str]):
    return " ".join(quote(x) for x in list)


class Mypath:
    def __init__(self, path: str):
        # CONTAINER:PATH or just PATH
        splitted = path.split(":", 1)
        if len(splitted) == 2:
            self.nomadid = splitted[0]
            self.path = Path(splitted[1])
        else:
            assert len(splitted) == 1
            self.nomadid = None
            self.path = Path(splitted[0])
        self._stat = None

    def isnomad(self):
        return self.nomadid is not None

    def nomadrun(self):
        # Return a properly escaped command line that allows to execute a command in the container.
        if self.isnomad():
            global args
            assert self.nomadid is not None
            idisjob = args.job or (
                # Try to be smart: if the id does not "look like" UUID number, assume it is a job name.
                not re.fullmatch("[0-9a-fA-F-]{1,36}", self.nomadid)
            )
            return "nomad alloc exec " + (
                (
                    (f"-namespace {quote(args.namespace)} " if args.namespace else "")
                    + f"-job {quote(self.nomadid)}"
                )
                if idisjob
                else f"{quote(self.nomadid)}"
            )
        else:
            return ""

    def _nomadstat(self):
        assert self.isnomad()
        if self._stat is None:
            # Execute a script inside the container to determine the type of the file.
            script = 'if [ -f "$@" ]; then echo FILE; elif [ -d "$@" ]; then echo DIR; else echo NONE; fi'
            cmd = [
                *shlex.split(self.nomadrun()),
                "sh",
                "-c",
                script,
                "--",
                str(self.path),
            ]
            log.debug(f"+ {listquote(cmd)}")
            self._stat = subprocess.check_output(cmd, text=True).strip()
            log.debug(f"stat = {self._stat}")
            assert self._stat in [
                "FILE",
                "DIR",
                "NONE",
            ], f"Invalid output from nomad subshell: {self._stat}"
        return self._stat

    def exists(self):
        return (
            (self._nomadstat() in ["FILE", "DIR"])
            if self.isnomad()
            else self.path.exists()
        )

    def is_file(self):
        return (self._nomadstat() == "FILE") if self.isnomad() else self.path.is_file()

    def is_dir(self):
        return (self._nomadstat() == "DIR") if self.isnomad() else self.path.is_dir()

    @property
    def name(self):
        return self.path.name

    def __str__(self):
        if self.nomadid is None:
            return str(self.path)
        else:
            return f"{self.nomadid}:{self.path}"

    def __truediv__(self, name: str):
        return type(self)(f"{self}/{name}")

    def quoteparent(self):
        return quote(str(self.path.parent))

    def quotename(self):
        return quote(str(self.path.name))

    def quotepath(self):
        return quote(str(self.path))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
Copy files/folders between a nomad allocation and the local filesystem.
Use '-' as the source to read a tar archive from stdin
and extract it to a directory destination in a container.
Use '-' as the destination to stream a tar archive of a
container source to stdout.
    """,
        epilog=f"""
Examples:
  {log.name} -n 9190d781:/tmp ~/tmp
  {log.name} -vn -Nservices -job promtail:/. ~/tmp

Written by Kamil Cukrowski 2023. Licensed under GNU GPL version or later.
""",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Do tar -vt for unpacking. Usefull for listing files for debugging.",
    )
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-N",
        "-namespace",
        "--namespace",
        help="Nomad namespace",
    )
    parser.add_argument(
        "-a",
        "--archive",
        action="store_true",
        help="Archive mode (copy all uid/gid information)",
    )
    parser.add_argument(
        "-j",
        "-job",
        "--job",
        action="store_true",
        help="Use a **random** allocation from the specified job ID.",
    )
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("source", help="ALLOCATION:SRC_PATH|JOB:SRC_PATH|SRC_PATH|-")
    parser.add_argument("dest", help="ALLOCATION:DEST_PATH|JOB:DEST_PATH|DEST_PATH|-")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG
        if args.debug
        else logging.INFO
        if args.verbose
        else logging.WARNING,
        format="%(levelname)s %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    )
    return args


def bash(script: str):
    cmd = ["bash", "-o", "pipefail", "-c", script]
    log.debug(f"+ {listquote(cmd)}")
    subprocess.check_call(cmd)


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
    script = (
        f"{src.nomadrun()} tar -cf - {tar_c} | {dst.nomadrun()} tar -{x}f - {tar_x}"
    )
    bash(script)


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
        bash(f"{dst.nomadrun()} tar -{x}f - -C {dst.quotepath()}")
    elif args.dest == "-":
        log.info(f"Stream {args.source} -> stdout")
        bash(f"{src.nomadrun()} tar -cf - -C {src.quoteparent()} -- {src.quotename()}")
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

def cli():
    global args
    args = parse_args()
    if args.test:
        test()
        exit()
    src = Mypath(args.source)
    dst = Mypath(args.dest)
    if args.source == "-" or args.dest == "-":
        stream_mode(src, dst)
    else:
        copy_mode(src, dst)

if __name__ == "__main__":
    cli()
