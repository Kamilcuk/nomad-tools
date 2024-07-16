#!/usr/bin/env python3

import argparse
import contextlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple

from nomad_tools import taskexec
from nomad_tools.transferstats import transfer_stats


@contextlib.contextmanager
def popen(cmd: List[str], **kwargs):
    print(f"+ {cmd}")
    with subprocess.Popen(cmd, **kwargs) as pp:
        try:
            yield pp
        finally:
            pp.wait(5)
            pp.terminate()
            pp.wait()


def test1():
    with popen(["yes"], stdout=subprocess.PIPE) as pout:
        with popen(["cat"], stdout=subprocess.DEVNULL, stdin=subprocess.PIPE) as pin:
            assert pout.stdout
            assert pin.stdin
            transfer_stats(pout.stdout.fileno(), pin.stdin.fileno(), ARGS.pv)


@contextlib.contextmanager
def busybox() -> Iterator[Tuple[str, str]]:
    pipe = os.pipe()
    os.set_inheritable(pipe[1], True)
    name = Path(sys.argv[0]).stem
    with popen(
        [
            "nomad-tools",
            "go",
            "--purge",
            f"--name={name}",
            f"--notifystarted={pipe[1]}",
            "--kill-signal=SIGKILL",
            "--init",
            "busybox:stable",
            "sleep",
            "infinity",
        ],
        close_fds=False,
    ):
        # Wait to be started.
        os.close(pipe[1])
        os.read(pipe[0], 1)
        os.close(pipe[0])
        yield taskexec.find_job(name)


def test2():
    with busybox() as spec:
        with taskexec.NomadPopen(*spec, ["yes"], stdout=subprocess.PIPE) as pout:
            with taskexec.NomadPopen(
                *spec, ["sh", "-c", "exec cat >/dev/null"], stdin=subprocess.PIPE
            ) as pin:
                assert pout.stdout
                assert pin.stdin
                transfer_stats(pout.stdout.fileno(), pin.stdin.fileno(), ARGS.pv)


def test3():
    with busybox() as spec:
        with popen(
            [
                *"nomad alloc exec -i=false -t=false -task".split(),
                spec[1],
                spec[0],
                "yes",
            ],
            stdout=subprocess.PIPE,
        ) as pout:
            with popen(
                [
                    *"nomad alloc exec -i=true -t=false -task".split(),
                    spec[1],
                    spec[0],
                    "sh",
                    "-c",
                    "exec cat >/dev/null",
                ],
                stdin=subprocess.PIPE,
            ) as pin:
                assert pout.stdout
                assert pin.stdin
                transfer_stats(pout.stdout.fileno(), pin.stdin.fileno(), ARGS.pv)


if __name__ == "__main__":
    os.environ.setdefault("NOMAD_NAMESPACE", "default")
    if os.environ["NOMAD_NAMESPACE"] == "*":
        os.environ["NOMAD_NAMESPACE"] = "default"
    parser = argparse.ArgumentParser()
    parser.add_argument("--pv", action="store_true", default=None, dest="pv")
    parser.add_argument("--no-pv", action="store_false", dest="pv")
    parser.add_argument("mode", type=int)
    global ARGS
    ARGS = parser.parse_args()
    print(ARGS)
    map: Dict[int, Callable[[], None]] = {
        0: lambda: None,
        1: test1,
        2: test2,
        3: test3,
    }
    map[ARGS.mode]()
