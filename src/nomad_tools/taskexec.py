from __future__ import annotations

import base64
import dataclasses
import io
import json
import logging
import subprocess
import sys
import urllib.parse
from typing import (
    IO,
    Any,
    BinaryIO,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import websocket
from typing_extensions import Literal, override

from . import nomadlib
from .common_base import eprint
from .common_nomad import mynomad, nomad_find_job
from .nomadlib.types import Alloc

log = logging.getLogger(__name__)
log.setLevel(level=logging.INFO)

DEVNULL = subprocess.DEVNULL
PIPE = subprocess.PIPE
CompletedProcess = subprocess.CompletedProcess
_DESTFILE = Union[int, IO[Any]]
_FILE = Optional[_DESTFILE]
_StrAny = Union[bytes, str]
_InputString = Optional[_StrAny]


def str_to_b64(txt: str) -> str:
    return base64.b64encode(txt.encode()).decode()


def b64_to_str(txt: str) -> str:
    return base64.b64decode(txt.encode()).decode()


def find_alloc_task(
    allocid: Optional[str], task: Optional[str] = None, job: Optional[str] = None
) -> Tuple[str, str]:
    """Find allocation and task name given the search parameters."""
    allocation: Optional[Alloc] = None
    if allocid:
        assert not job
    else:
        allocations = [Alloc(x) for x in mynomad.get(f"job/{job}/allocations")]
        assert allocations, f"Did not find any allocations for job {job}"
        if task:
            allocations = [x for x in allocations if task in x.get_tasknames()]
        allocation = next((alloc for alloc in allocations if alloc.is_running()), None)
        assert allocation, f"Did not find running allocation for job {job}"
        allocid = allocation.ID
    if not task:
        if not allocation:
            assert allocid
            allocation = Alloc(mynomad.get(f"allocation/{allocid}"))
        tasknames = allocation.get_tasknames()
        assert len(tasknames) == 1
        task = tasknames[0]
    assert len(allocid) == 36, f"{allocid}"
    assert task
    return allocid, task


def find_job_alloc(job: str, task: Optional[str] = None) -> str:
    """
    Given a job, find a running allocation that has given task.
    There has to be exactly one running allocation of that job.
    That allocation has to have that task.
    """
    return find_alloc_task(None, task, job)[0]


def find_job(job: str) -> Tuple[str, str]:
    """
    Given a job, return its running allocation id and task.
    There has to be exactly one running allocation of that job.
    That allocation has to have exactly one task.
    """
    return find_alloc_task(None, None, job)


T = TypeVar("T", bytes, str)


class NomadPopen(Generic[T]):
    """
    An implementation of subprocess.Popen on top of Nomad Exec Alocation API.
    https://developer.hashicorp.com/nomad/api-docs/allocations#exec-allocation
    https://docs.python.org/3/library/subprocess.html
    """

    @overload
    def __init__(
        self: NomadPopen[bytes],
        allocid: str,
        task: str,
        args: List[str],
        job: Optional[str] = ...,
        stdin: _FILE = ...,
        stdout: _FILE = ...,
        text: Literal[False] = ...,
    ) -> None:
        ...

    @overload
    def __init__(
        self: NomadPopen[str],
        allocid: str,
        task: str,
        args: List[str],
        job: Optional[str] = ...,
        stdin: _FILE = ...,
        stdout: _FILE = ...,
        text: Literal[True] = ...,
    ) -> None:
        ...

    @overload
    def __init__(
        self: NomadPopen[bytes],
        allocid: str,
        task: str,
        args: List[str],
        job: Optional[str] = ...,
        stdin: _FILE = ...,
        stdout: _FILE = ...,
        text: bool = ...,
    ) -> None:
        ...

    def __init__(
        self,
        allocid: str,
        task: str,
        args: List[str],
        job: Optional[str] = None,
        stdin: _FILE = None,
        stdout: _FILE = None,
        text: bool = False,
    ):
        assert allocid
        assert task
        assert args
        assert stdin is None or stdin == PIPE
        self.cmd = [allocid, task] + args
        self.__stdin_arg: _DESTFILE = DEVNULL if stdin is None else stdin
        self.__stdout_arg: _DESTFILE = sys.stdout if stdout is None else stdout
        #
        self.__closed: Set[str] = set()
        """Protect against double closing"""
        self.__returncode: Optional[int] = None
        self._reader = self.__readergen()
        self.text = text
        # Connect to the remote side.
        path = f"v1/client/allocation/{allocid}/exec?" + urllib.parse.urlencode(
            dict(task=task, command=json.dumps(args))
        )
        log.debug(f"CONNECT {path!r} text={text}")
        self.ws: Optional[websocket.WebSocket] = nomadlib.create_websocket_connection(
            path
        )
        # Initialize self.stdin
        self.stdin: Optional[IO[T]] = None
        if self.__stdin_arg == DEVNULL:
            self._close_stream("stdin")
        elif self.__stdin_arg == PIPE:
            stream = self.Stdin(self)
            if self.text:
                stream = io.TextIOWrapper(stream)
            self.stdin = cast(IO[T], stream)
        else:
            assert 0, f"Unhandled stdin={self.__stdin_arg}"
        # Initialize self.stdout
        self.stdout: Optional[IO[T]] = None
        if self.__stdout_arg == DEVNULL:
            self._close_stream("stdout")
        elif self.__stdout_arg is sys.stdout:
            pass
        elif self.__stdout_arg == PIPE:
            stream = self.Stdout(self)
            if self.text:
                stream = io.TextIOWrapper(stream)
            self.stdout = cast(IO[T], stream)
        else:
            assert 0, f"Unhandled stdout={self.__stdout_arg}"

    def __output(self, buf: bytes) -> Iterator[int]:
        log.debug(f"RRAW len={len(buf)} {self.__stdout_arg}")
        if self.__stdout_arg is sys.stdout:
            sys.stdout.buffer.write(buf)
        elif self.__stdout_arg == PIPE:
            for c in buf:
                yield c
        else:
            assert 0, f"Unhandled stdout={self.__stdout_arg}"

    def __readergen(self) -> Iterator[int]:
        while self.ws:
            line = self.ws.recv()
            if not line:
                break
            frame: dict = json.loads(line)
            fstderr: Optional[dict] = frame.get("stderr")
            if fstderr:
                data: Optional[str] = fstderr.get("data")
                if data:
                    eprint(b64_to_str(data).rstrip())
            fstdout: Optional[dict] = frame.get("stdout")
            if fstdout:
                data: Optional[str] = fstdout.get("data")
                if data:
                    buf = base64.b64decode(data.encode())
                    for c in self.__output(buf):
                        yield c
            fexited = frame.get("exited")
            if fexited:
                self.__returncode = frame["result"].get("exit_code", 0)
                break

    def __enter__(self):
        return self

    def communicate(
        self, input: Optional[Union[str, bytes]] = None
    ) -> Tuple[Optional[Union[str, bytes]], None]:
        if input:
            assert self.__stdin_arg == PIPE
            assert self.stdin
            assert isinstance(input, str) if self.text else isinstance(input, bytes)
            self.stdin.write(cast(T, input))
            self.stdin.close()
        output: Optional[T] = None
        if self.__stdout_arg == PIPE:
            assert self.stdout
            output = self.stdout.read()
            assert isinstance(output, str) if self.text else isinstance(output, bytes)
        return output, None

    @property
    def returncode(self):
        assert (
            self.__returncode is not None
        ), f"nomad alloc exec {self.cmd} not finished"
        return self.__returncode

    def wait(self):
        assert self.ws
        self._close_stream("stdin")
        for _ in self._reader:
            pass
        if self.__returncode is None:
            raise Exception(
                f"when executing nomad alloc exec {self.cmd}"
                "the websocket was closed by Nomad server without sending the exit code of the process."
            )
        log.debug("closing")
        self.ws.close()
        self.ws = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wait()

    def _close_stream(self, stream: str):
        """Send message to Nomad to close specific stream"""
        assert self.ws
        if stream in self.__closed:
            return
        self.__closed.add(stream)
        msg = json.dumps({stream: {"close": True}})
        log.debug(f"W {msg}")
        try:
            self.ws.send(msg)
        except BrokenPipeError:
            self.ws.close()
            self.ws = None
            raise

    def write(self, s: bytes) -> int:
        assert isinstance(s, bytes)
        assert self.ws
        msg = json.dumps({"stdin": {"data": base64.b64encode(s).decode()}})
        log.debug(f"W stdin:data:{s!r}")
        return self.ws.send(msg)

    def read(self, size: int = -1) -> bytes:
        acc: bytearray = bytearray()
        for c in self._reader:
            acc.append(c)
            if size != -1 and len(acc) == size:
                break
        accbytes = bytes(acc)
        log.debug(f"R({size}) {accbytes!r}")
        return accbytes

    @dataclasses.dataclass
    class Stdin(BinaryIO):
        p: NomadPopen

        @override
        def fileno(self) -> int:
            return -1

        @override
        def writable(self) -> bool:
            return True

        @override
        def write(self, s) -> int:
            assert isinstance(s, bytes)
            return self.p.write(s)

        @override
        def close(self):
            self.p._close_stream("stdin")

    @dataclasses.dataclass
    class Stdout(BinaryIO):
        p: NomadPopen

        @override
        def fileno(self) -> int:
            return -1

        @override
        def readable(self) -> bool:
            return True

        @override
        def read(self, size: int = -1) -> bytes:
            return self.p.read(size)

        @override
        def close(self):
            self.p._close_stream("stdout")

    def raise_for_returncode(self, output: Optional[Union[str, bytes]] = None):
        if self.returncode:
            raise subprocess.CalledProcessError(self.returncode, self.cmd, output, None)


###############################################################################


@overload
def run(
    allocid: str,
    task: str,
    cmd: List[str],
    *,
    stdin: _FILE = ...,
    input: Optional[bytes] = ...,
    stdout: _FILE = ...,
    check: bool = ...,
    text: Literal[False] = False,
) -> CompletedProcess[bytes]:
    ...


@overload
def run(
    allocid: str,
    task: str,
    cmd: List[str],
    *,
    text: Literal[True],
    stdin: _FILE = ...,
    input: str = ...,
    stdout: _FILE = ...,
    check: bool = ...,
) -> CompletedProcess[str]:
    ...


@overload
def run(
    allocid: str,
    task: str,
    cmd: List[str],
    *,
    text: bool,
    input: Optional[Union[bytes, str]] = ...,
    stdin: _FILE = ...,
    stdout: _FILE = ...,
    check: bool = ...,
) -> CompletedProcess[Any]:
    ...


def run(
    allocid: str,
    task: str,
    cmd: List[str],
    *,
    stdin: _FILE = None,
    input: Optional[Union[bytes, str]] = None,
    stdout: _FILE = None,
    check: bool = False,
    text: bool = False,
) -> CompletedProcess[Any]:
    with NomadPopen(
        allocid,
        task,
        cmd,
        stdin=PIPE if input else stdin,
        stdout=stdout,
        text=text,
    ) as p:
        output = p.communicate(input)[0]
    if check:
        p.raise_for_returncode(output)
    return CompletedProcess(cmd, p.returncode, output, None)


###############################################################################


@overload
def check_output(
    allocid: str,
    task: str,
    cmd: List[str],
    *,
    input: Optional[bytes] = ...,
    text: Literal[False] = False,
) -> bytes:
    ...


@overload
def check_output(
    allocid: str,
    task: str,
    cmd: List[str],
    *,
    text: Literal[True],
    input: Optional[str] = ...,
) -> str:
    ...


@overload
def check_output(
    allocid: str,
    task: str,
    cmd: List[str],
    *,
    input: T,
    text: bool,
) -> T:
    ...


@overload
def check_output(
    allocid: str,
    task: str,
    cmd: List[str],
    *,
    text: bool,
) -> Union[str, bytes]:
    ...


def check_output(
    allocid: str,
    task: str,
    cmd: List[str],
    *,
    input: Optional[Union[str, bytes]] = None,
    text: bool = False,
) -> Union[str, bytes]:
    return run(
        allocid, task, cmd, input=input, stdout=PIPE, text=text, check=True
    ).stdout


###############################################################################

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--text", action="store_false")
    parser.add_argument(
        "--mode",
        choices="check_output run".split(),
        default="check_output",
    )
    parser.add_argument("job")
    parser.add_argument("task")
    parser.add_argument("cmd", nargs="+")
    args = parser.parse_args()
    if args.trace:
        websocket.enableTrace(True)
    if args.debug:
        log.setLevel(level=logging.DEBUG)
    print(args)
    job = nomad_find_job(args.job)
    allocid = find_job_alloc(args.job, args.task)
    if args.mode == "check_output":
        output = ""
        try:
            output = check_output(
                allocid, args.task, args.cmd, input=args.input, text=args.text
            ).strip()
        finally:
            print(output)
    elif args.mode == "run":
        run(allocid, args.task, args.cmd, input=args.input, text=args.text)
