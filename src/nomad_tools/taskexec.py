from __future__ import annotations

import base64
from dataclasses import dataclass
import functools
import io
import json
import logging
import os
import sys
import subprocess
import threading
import urllib.parse
from typing import (
    IO,
    Any,
    BinaryIO,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    TextIO,
    Tuple,
    TypeVar,
    Union,
    cast,
)
import websocket
from typing_extensions import Literal, overload

from . import nomadlib
from .common_base import eprint
from .common_nomad import mynomad
from .nomadlib.datadict import DataDict
from .nomadlib.types import Alloc

log = logging.getLogger(__name__)
log.setLevel(level=logging.INFO)


def find_alloc_task(
    allocid: Optional[str], task: Optional[str] = None, job: Optional[str] = None
) -> Tuple[str, str]:
    """Find allocation and task name given the search parameters."""
    allocation: Optional[Alloc] = None
    if allocid:
        assert not job
    else:
        assert job
        allocations = [Alloc(x) for x in mynomad.get(f"job/{job}/allocations")]
        assert allocations, f"Did not find any allocations for job {job}: {allocations}"
        if task:
            allocations = [x for x in allocations if task in x.get_tasknames()]
        allocation = next((alloc for alloc in allocations if alloc.is_running()), None)
        assert allocation, f"Did not find running allocation for job {job}: allocation"
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


###############################################################################


class FrameData(DataDict):
    data: Optional[str] = None
    close: Optional[bool] = None


class FrameResult(DataDict):
    exit_code: int = 0


class ExecStreamingOutput(DataDict):
    stderr: Optional[FrameData] = None
    stdout: Optional[FrameData] = None
    exited: Optional[bool] = None
    result: Optional[FrameResult] = None


class TaskExec:
    """Abstraction over Nomad websocket API for running argument in task"""

    def __init__(self, allocid: str, task: str, args: List[str]):
        assert allocid
        assert task
        assert args
        self.name: str = f"{self.__class__.__name__}({allocid}/{task},{args})"
        self.__closed: Set[str] = set()
        """Protect against double closing"""
        self.returncode: Optional[int] = None
        # Connect to the remote side.
        path = f"v1/client/allocation/{allocid}/exec?" + urllib.parse.urlencode(
            dict(task=task, command=json.dumps(args))
        )
        log.debug(f"CONNECT {path!r}")
        self.ws: websocket.WebSocket = nomadlib.create_websocket_connection(path)
        self.__readergen = self.__reader()
        assert self.ws.connected

    def __reader(self) -> Iterator[bytes]:
        while self.ws.connected:
            line = self.ws.recv()
            log.debug(f"R: {line}")
            if not line:
                break
            frame = ExecStreamingOutput(json.loads(line))
            if frame.stderr:
                if frame.stderr.data:
                    txt: str = base64.b64decode(frame.stderr.data.encode()).decode(
                        errors="replace"
                    )
                    eprint(txt, end="")
            if frame.stdout:
                if frame.stdout.data:
                    yield base64.b64decode(frame.stdout.data.encode())
            if frame.exited and frame.result:
                self.returncode = frame.result.exit_code
                break
        self.terminate()

    def __send(self, msg: str):
        try:
            self.ws.send(msg)
        except BrokenPipeError:
            self.ws.close()
            raise

    def __close_stream(self, stream: str):
        """Send message to Nomad to close specific stream"""
        if not self.ws.connected:
            return
        if stream in self.__closed:
            return
        self.__closed.add(stream)
        msg = json.dumps({stream: {"close": True}})
        log.debug(f"W {msg}")
        self.__send(msg)

    def close_stdin(self):
        # Only stdin is handled in Nomad
        # https://github.com/hashicorp/nomad/blob/695bb7ffcf90fc9455152dadd2a504bc4499e3b3/plugins/drivers/execstreaming.go#L41
        return self.__close_stream("stdin")

    def terminate(self):
        if self.ws.connected:
            log.debug("closing ws")
            self.ws.close()

    def wait(self):
        if not self.ws.connected:
            return
        self.close_stdin()
        self.read()

    def read(self) -> bytes:
        return functools.reduce(bytes.__add__, self.read1(), b"")

    def read1(self) -> Iterator[bytes]:
        return self.__readergen

    def write(self, s: bytes):
        msg = json.dumps({"stdin": {"data": base64.b64encode(s).decode()}})
        log.debug(f"W stdin:data:{s!r}")
        self.__send(msg)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

    def raise_for_returncode(self, output: Optional[Union[str, bytes]] = None):
        if self.returncode is None:
            raise Exception(
                f"when executing {self.name}"
                "the websocket was closed by Nomad server without sending the exit code of the process."
            )
        if self.returncode:
            raise subprocess.CalledProcessError(self.returncode, self.name, output)


###############################################################################


DEVNULL = subprocess.DEVNULL
PIPE = subprocess.PIPE
CompletedProcess = subprocess.CompletedProcess
_IoAny = Union[IO[bytes], IO[str]]
_DESTFILE = Union[int, _IoAny]
_FILE = Optional[_DESTFILE]
_StrAny = Union[bytes, str]
_InputString = Optional[_StrAny]


T = TypeVar("T", bytes, str)


@dataclass
class MyIO:
    """Wrapper around IO[bytes] that optionally closes the io on with block"""

    io: IO[bytes]
    close: bool

    def __enter__(self):
        return self.io

    def __exit__(self, *_):
        if self.close:
            self.io.close()


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
        stdin: _FILE = ...,
        stdout: _FILE = ...,
    ) -> None: ...

    @overload
    def __init__(
        self: NomadPopen[str],
        allocid: str,
        task: str,
        args: List[str],
        stdin: _FILE = ...,
        stdout: _FILE = ...,
        text: Literal[True] = ...,
    ) -> None: ...

    @overload
    def __init__(
        self: NomadPopen[bytes],
        allocid: str,
        task: str,
        args: List[str],
        stdin: _FILE = ...,
        stdout: _FILE = ...,
        text: bool = ...,
    ) -> None: ...

    def __init__(
        self,
        allocid: str,
        task: str,
        args: List[str],
        stdin: _FILE = None,
        stdout: _FILE = None,
        text: bool = False,
    ):
        self.np = TaskExec(allocid, task, args)
        self.text = text
        self.__initialize_stdin(self.__stdin_to_fd(stdin))
        self.__initialize_stdout(self.__stdout_to_fd(stdout))

    def __stdin_to_fd(self, stdin: _FILE) -> Optional[MyIO]:
        self.stdin: Optional[IO[T]] = None
        if stdin == DEVNULL or stdin is None:
            return None
        elif isinstance(stdin, BinaryIO):
            return MyIO(stdin, close=False)
        elif isinstance(stdin, TextIO):
            return MyIO(stdin.buffer, close=False)
        elif isinstance(stdin, int) and stdin >= 0:
            return MyIO(os.fdopen(stdin, "rb"), close=True)
        elif stdin == PIPE:
            pipe = os.pipe()
            self.stdin = os.fdopen(pipe[1], "w" if self.text else "wb")
            return MyIO(os.fdopen(pipe[0], "rb"), close=True)
        else:
            assert 0, f"Unhandled stdin={stdin}"

    def __stdout_to_fd(self, stdout: _FILE) -> MyIO:
        self.stdout: Optional[IO[T]] = None
        if stdout is None:
            return MyIO(sys.stdout.buffer, close=False)
        elif isinstance(stdout, BinaryIO):
            return MyIO(stdout, close=False)
        elif isinstance(stdout, TextIO):
            return MyIO(stdout.buffer, close=False)
        elif isinstance(stdout, int) and stdout >= 0:
            return MyIO(os.fdopen(stdout, "wb"), close=False)
        elif stdout == PIPE:
            pipe = os.pipe()
            self.stdout = os.fdopen(pipe[0], "r" if self.text else "rb")
            return MyIO(os.fdopen(pipe[1], "wb"), close=True)
        elif stdout == DEVNULL:
            pass
        else:
            assert 0, f"Unhandled stdout={self.stdout}"
        return MyIO(open(os.devnull, "wb"), close=True)

    def __initialize_stdin(self, fd: Optional[MyIO]):
        if fd is not None:

            def writer():
                try:
                    with fd as f:
                        # Use read1 if available.
                        for buf in (
                            iter(f.read1, b"")
                            if isinstance(f, io.BufferedIOBase)
                            else f
                        ):
                            self.np.write(buf)
                except BrokenPipeError:
                    pass
                except Exception as e:
                    log.exception(e)
                finally:
                    self.np.close_stdin()

            threading.Thread(
                name=f"{self.np.name}WRITER", target=writer, daemon=True
            ).start()
        else:
            self.np.close_stdin()

    def __initialize_stdout(self, fd: MyIO):
        def reader():
            try:
                with fd as f:
                    for buf in self.np.read1():
                        f.write(buf)
            except BrokenPipeError:
                pass
            except Exception as e:
                log.exception(e)

        self.readthread = threading.Thread(
            name=f"{self.np.name}READER", target=reader, daemon=True
        )
        self.readthread.start()

    ###############################################################################

    @property
    def name(self):
        return self.np.name

    def __enter__(self):
        return self

    def communicate(
        self, input: Optional[Union[str, bytes]] = None
    ) -> Tuple[Optional[Union[str, bytes]], None]:
        if input:
            assert self.stdin
            assert isinstance(input, str if self.text else bytes)
            self.stdin.write(cast(T, input))
        if self.stdin:
            self.stdin.close()
        output: Optional[T] = None
        if self.stdout:
            output = self.stdout.read()
            assert isinstance(output, str if self.text else bytes)
        return output, None

    @property
    def returncode(self) -> Optional[int]:
        return self.np.returncode

    def wait(self, timeout: Optional[float] = None):
        self.readthread.join(timeout)

    def terminate(self):
        self.np.terminate()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.terminate()
        self.wait()

    def raise_for_returncode(self, output: Optional[Union[str, bytes]] = None):
        self.readthread.join()
        self.np.raise_for_returncode()


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
) -> CompletedProcess[bytes]: ...


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
) -> CompletedProcess[str]: ...


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
) -> CompletedProcess[Any]: ...


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
    return CompletedProcess(
        cmd, p.returncode if p.returncode is not None else 400, output, None
    )


###############################################################################


@overload
def check_output(
    allocid: str,
    task: str,
    cmd: List[str],
    *,
    text: Literal[False] = False,
    input: Optional[bytes] = ...,
) -> bytes: ...


@overload
def check_output(
    allocid: str,
    task: str,
    cmd: List[str],
    *,
    text: Literal[True],
    input: Optional[str] = ...,
) -> str: ...


def check_output(
    allocid: str,
    task: str,
    cmd: List[str],
    *,
    text: bool = False,
    input: Optional[Union[str, bytes]] = None,
) -> Union[str, bytes]:
    return run(
        allocid, task, cmd, input=input, stdout=PIPE, text=text, check=True
    ).stdout


###############################################################################

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-t", "--text", action="store_true")
    parser.add_argument(
        "-m",
        "--mode",
        choices="check_output run popen".split(),
        default="check_output",
    )
    parser.add_argument("--stdin", "--in", default=str(subprocess.PIPE))
    parser.add_argument("--stdout", "--out", default=str(subprocess.PIPE))
    parser.add_argument("job")
    parser.add_argument("task")
    parser.add_argument("cmd", nargs="+")
    args = parser.parse_args()
    if args.trace:
        websocket.enableTrace(True)
    if args.debug:
        log.setLevel(level=logging.DEBUG)
    print("ARGS", args)
    # spec = find_job(args.job)
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
    elif args.mode == "popen":

        def parse_stream(txt: str) -> int:
            map = {"pipe": subprocess.PIPE, "null": subprocess.DEVNULL, "none": None}
            return map[txt.lower()] if txt.lower() in map else int(txt)

        ppargs: Dict[str, Any] = dict(
            allocid=allocid,
            task=args.task,
            args=args.cmd,
            stdin=parse_stream(args.stdin),
            stdout=parse_stream(args.stdout),
            text=args.text,
        )
        print(f"PIPE={subprocess.PIPE} DEVNULL={subprocess.DEVNULL}")
        print(f"NomadPopen({ppargs})")
        with NomadPopen(**ppargs) as pp:
            out = pp.communicate(
                None
                if args.input is None
                else args.input
                if args.text
                else args.input.encode()
            )
            print(f"{out[0]!r}")
