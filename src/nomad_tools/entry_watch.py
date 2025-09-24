#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import base64
import contextlib
import datetime
import enum
import inspect
import io
import itertools
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Pattern,
    Set,
    TextIO,
    Tuple,
    TypeVar,
)

import click
import clickdc
import jinja2
import requests
from click.shell_completion import CompletionItem
from typing_extensions import override

from . import colors, exit_on_thread_exception, flagdebug, nomaddbjob, nomadlib
from .aliasedgroup import AliasedGroup
from .common_base import andjoin, cached_property, composed, eprint, quotearr
from .common_click import complete_set_namespace, help_h_option
from .common_nomad import (
    NoJobFound,
    completor,
    mynomad,
    namespace_option,
    nomad_find_job,
)
from .nomaddbjob import NomadDbJob
from .nomadlib import Event, EventTopic, EventType, MyStrEnum, ns2dt
from .nomadlib.types import strdict

log = logging.getLogger(__name__)


###############################################################################

START_S: float = 0
"""This program start time, assigned from cli()"""

T = TypeVar("T")


def set_not_in_add(s: Set[T], value: T) -> bool:
    """If value is in the set, return False. Otherwise add the value to the set and return True"""
    if value in s:
        return False
    s.add(value)
    return True


def print_all_threads_stacktrace(*_):
    text: List[str] = []
    text.append("Received SIGUSR1")
    for th in threading.enumerate():
        text.append(str(th))
        if th.ident:
            text += traceback.format_stack(sys._current_frames()[th.ident])
        text.append("")
    out = "\n".join(x for x in "\n".join(text).splitlines() if x)
    eprint("\n\n" + out + "\n\n")


def datetime_is_naive(d: datetime.datetime) -> bool:
    return d.tzinfo is None or d.tzinfo.utcoffset(d) is None


###############################################################################


@dataclass
class InterruptCounter:
    """Exit on the second time after sending a signal"""

    max = 5
    cnt: dict[signal.Signals, int] = field(default_factory=dict)
    """The number of last signal received"""

    def install(self):
        # Installed after DB initialization.
        assert DB
        assert ARGS
        signal.signal(signal.SIGINT, self.__handler)
        signal.signal(signal.SIGTERM, self.__handler)

    def __handler(self, signum: int, frame):
        sig = signal.Signals(signum)
        cur = self.cnt[sig] = self.cnt.setdefault(sig, 0) + 1
        doexit = cur >= self.max if self.max > 0 else False if ARGS.attach else True
        stat = f" {cur}/{self.max}" if ARGS.attach else ""
        post = " Exiting!" if doexit else ""
        log.error(f"Interrupted with {sig.name}{stat}{post}")
        if doexit:
            exit(ExitCode.interrupted)
        else:
            # Initialized in NomadWatch object.
            # Send empty event to trigger main event loop.
            DB.send_empty_event()


###############################################################################


class CommaList(click.ParamType):
    """A click option to pass a comma separated list to the option"""

    name = "CommaList"

    @override
    def __init__(self, values: List[Any], separator: str = ",", **kvargs):
        super().__init__(**kvargs)
        self.separator = separator
        self.values = values

    @override
    def get_metavar(self, param: click.Parameter) -> Optional[str]:
        return self.separator.join(self.values)

    @override
    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> Any:
        if isinstance(value, str):
            arr = value.split(",")
            bad = [a for a in arr if a not in self.values]
            if bad:
                badstr = self.separator.join(bad)
                valuesstr = self.separator.join(self.values)
                self.fail(f"{badstr} must be one of {valuesstr}")
            return arr
        return value

    @override
    def shell_complete(
        self, ctx: click.Context, param: click.Parameter, incomplete: str
    ) -> List[CompletionItem]:
        arr = incomplete.split(self.separator)
        start = [v for v in self.values if v not in arr and v.startswith(arr[-1])]
        ret = [CompletionItem(self.separator.join([*arr[:-1], x])) for x in start]
        if arr[-1] in self.values:
            ret += [CompletionItem(self.separator.join(arr) + self.separator)]
        return ret


###############################################################################


class LogWhat(MyStrEnum):
    deploy = enum.auto()
    eval = enum.auto()
    alloc = enum.auto()
    stdout = enum.auto()
    stderr = enum.auto()


class LOGENABLED:
    """Should logging of a specific stream be enabled?"""

    deploy: bool = True
    eval: bool = True
    alloc: bool = True
    stdout: bool = True
    stderr: bool = True


@dataclass(frozen=True, order=True)
class LogLine:
    """Represents a single line to log out"""

    # Now has to be first for sorting order
    now: datetime.datetime
    """The timestamp when the log line was created or received"""
    id: str
    """The id of allocation or evaluation or deployment"""
    message: str
    """The actual message"""
    what: LogWhat
    """Where does the log line comes from? Allocation or deployment or evaluation or stdout or stderr"""
    ModifyIndex: int
    """The modify index associated with the id"""
    nodename: Optional[str] = None
    """nodename, extracted from allocation"""
    jobversion: Optional[str] = None
    """The job version, best efforted extracted from data or from job history"""
    group: Optional[str] = None
    """The task group, for stdout and stderr logs"""
    task: Optional[str] = None
    """The task name, for stdout and stderr logs"""

    def __post_init__(self):
        assert self.what in [
            e.value for e in LogWhat
        ], f"{self.what} value is not a valid what"
        assert not datetime_is_naive(self.now), f"{self}"

    @property
    def mark(self):
        """Log source as one character"""
        marks: Dict[LogWhat, str] = {
            LogWhat.deploy: "deploy",
            LogWhat.eval: "eval",
            LogWhat.alloc: "A",
            LogWhat.stderr: "E",
            LogWhat.stdout: "O",
        }
        return marks[self.what]

    @property
    def color(self):
        """The color pallete I use for logs"""
        colors: Dict[LogWhat, str] = {
            LogWhat.deploy: COLORS.brightmagenta,
            LogWhat.eval: COLORS.magenta,
            LogWhat.alloc: COLORS.cyan,
            LogWhat.stderr: COLORS.orange,
            LogWhat.stdout: "",
        }
        return colors[self.what]

    @property
    def colorreset(self):
        """Empty string, if there were no colors used"""
        return COLORS.reset if self.color else ""

    @property
    def known_time(self):
        """Is the time of the message known or guessed?"""
        return self.what not in [LogWhat.stdout, LogWhat.stderr]

    def json(self):
        """Serialize the object to JSON"""
        tmp = asdict(self)
        tmp = {k: v for k, v in tmp.items() if v is not None}
        tmp["now"] = tmp["now"].isoformat(sep="T")
        return json.dumps(tmp)


class LogFormatter:
    """Logging output format specification templates using f-string. Very poor mans templating langauge."""

    JINJA2_LIB = """"""

    pre = (
        "{{log.color}}{{log.now.strftime(args.log_time_format) + '>' if args.log_time}}"
    )
    mark = "{{log.mark}}>"
    post = "{{log.message}}{{log.colorreset}}"
    task = "{{log.task + '>' if log.task}}"
    space = "{{' ' if not args.log_nospace}}"

    DEFAULT = (
        pre
        + mark
        + "{{log.id[:args.log_id_len]}}>"
        + "v{{log.jobversion}}>"
        + task
        + space
        + post
    )
    """
    Default log format. The log is templated with f-string using eval() below.
        O>45fbbd>v0>task1> hello world
    """

    ONE = pre + mark + task + space + post
    """
    Log format with -1 option.
        O>task1> hello world
    """

    ONLYMARK = pre + mark + post
    """ O>hello world """

    ZERO = pre + post
    """
    Log format with -0 option.
        hello world
    """

    JSON = "{{log.json()}}"

    @staticmethod
    def logging():
        """Logging format"""
        return (
            COLORS.brightblue
            + (
                "%(asctime)s"
                + (".%(msecs)06d" if ARGS.log_time_format.endswith(".%f") else "")
                + ">"
                if ARGS.log_time
                else ""
            )
            + ("watch>" if ARGS.verbose <= 0 else "%(module)s>")
            + ("%(lineno)03d>" if ARGS.verbose else "")
            + ("" if ARGS.log_nospace else " ")
            + "%(levelname)s %(message)s"
            + COLORS.reset
        )

    def __init__(self, fmt: str):
        self.tmpl = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(
            self.JINJA2_LIB + fmt
        )

    def output(self, logline: LogLine):
        try:
            out: str = self.tmpl.render(args=ARGS, log=logline, colors=COLORS)
        except KeyError:
            log.exception(f"Could not format logging line. ARGS={ARGS} LOG={log}")
            raise
        try:
            print(out, flush=True)
        except BrokenPipeError:
            # Python flushes standard streams on exit; redirect remaining output
            # to devnull to avoid another BrokenPipeError at shutdown
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            sys.exit(1)  # Python exits with error code 1 on EPIPE


@dataclass
class LogOptions:
    log_time: bool = clickdc.option(
        "-T",
        is_flag=True,
        help="""
            Additionally add timestamp to the logs.
            The timestamp of stdout and stderr streams is when the log was received,
            as Nomad does not store timestamp of task logs.
            """,
    )
    log_time_format: str = clickdc.option(
        default="%Y-%m-%dT%H:%M:%S%z",
        show_default=True,
        help="Format time with specific format. Passed to python datetime.strftime.",
    )
    log_time_hour: Any = clickdc.alias_option(
        "-H",
        aliased=dict(
            log_time_format="%H:%M:%S",
            log_time=True,
        ),
    )
    log_time_us: Any = clickdc.alias_option(
        "-U",
        aliased=dict(
            log_time=True,
        ),
    )
    log_format: str = clickdc.option(
        default=LogFormatter.DEFAULT,
        show_default=True,
        help="""
            The format to use when printing job logs.
            Templated with jinja2 template.
            See documentation.
            """,
    )
    log_id_len: int = clickdc.option(
        default=6,
        type=int,
        help="The length of id to log. UUIDv4 has 36 characters.",
    )
    log_id_long: Any = clickdc.alias_option("-l", aliased=dict(log_id_len=36))
    log_only_task: Any = clickdc.alias_option(
        "-1", aliased=dict(log_format=LogFormatter.ONE)
    )
    log_only_mark: Any = clickdc.alias_option(
        aliased=dict(log_format=LogFormatter.ONLYMARK)
    )
    log_none: Any = clickdc.alias_option(
        "-0", aliased=dict(log_format=LogFormatter.ZERO)
    )
    log_json: bool = clickdc.alias_option(aliased=dict(log_format=LogFormatter.JSON))
    log_nospace: bool = clickdc.option(help="Do not print space on log lines")
    log_j: Any = clickdc.alias_option(
        aliased=dict(log_nospace=True, log_format=LogFormatter.ONE)
    )
    out: List[str] = clickdc.option(
        "-o",
        type=CommaList("all alloc stdout stderr eval deploy nolog none".split()),
        default=["all"],
        show_default=True,
        help="""
            Comma separated list of streams of messages to print:
            deployment, evaluation, allocation, stdout and stderr logs of tasks.
            """,
    )


def init_logging():
    LOGENABLED.deploy = any(s in "all deploy nolog".split() for s in ARGS.out)
    LOGENABLED.eval = any(s in "all eval nolog".split() for s in ARGS.out)
    LOGENABLED.alloc = any(s in "all alloc nolog".split() for s in ARGS.out)
    LOGENABLED.stderr = any(s in "all stderr".split() for s in ARGS.out)
    LOGENABLED.stdout = any(s in "all stdout".split() for s in ARGS.out)
    #
    if ARGS.log_time_us:
        ARGS.log_time_format += ".%f"
    #
    ARGS.verbose -= ARGS.quiet
    if ARGS.verbose > 1:
        from http import client as http_client

        http_client.HTTPConnection.debuglevel = 1
    global LOGFORMAT
    LOGFORMAT = LogFormatter(ARGS.log_format)
    global COLORS
    COLORS = colors.init()
    logging.basicConfig(
        format=LOGFORMAT.logging(),
        datefmt=ARGS.log_time_format[:-3]
        if ARGS.log_time_format.endswith(".%f")
        else ARGS.log_time_format,
        level=(
            logging.DEBUG
            if ARGS.verbose > 0
            else logging.INFO
            if ARGS.verbose == 0
            else logging.WARN
            if ARGS.verbose == 1
            else logging.ERROR
        ),
    )


class MyloggerDelayer:
    """Delay outputting of logs for lines_timeout seconds"""

    @dataclass(frozen=True, order=True)
    class Key:
        """The key used in cache as an object"""

        known_time: bool

    def __init__(self):
        self.old_ModifyIndex: Optional[int] = None
        """There are new events and old events. Filter out old events"""
        self.cache: Dict[MyloggerDelayer.Key, List[LogLine]] = {}
        """Cache of lines to output"""
        self.newcache: List[LogLine] = []
        """Cache of lines to output which are newer than old_ModifyIndex"""
        self.thread = threading.Thread(
            name=self.__class__.__name__, target=self.__run, daemon=False
        )
        self.threadstarted = False
        """Thread that after lines_timeout outputs log lines"""
        self.lock = threading.Lock()
        """Synchronize thread with log producer"""
        self.finished: bool = False
        """Set to true when thread is finished for speed"""

    def __run(self):
        assert ARGS.lines >= 0
        assert ARGS.lines_timeout >= 0
        log.debug(f"{self.thread.name} sleeping")
        time.sleep(ARGS.lines_timeout)
        log.debug(f"{self.thread.name} starting")
        with self.lock:
            try:
                if ARGS.lines == 0 or len(self.cache) == 0:
                    return
                #
                lines = ARGS.lines
                knowntimelines = sorted(
                    self.cache.get(self.Key(True), []),
                    key=lambda x: x.now,
                )[-lines:]
                for line in knowntimelines:
                    LOGFORMAT.output(line)
                lines -= len(knowntimelines)
                if lines:
                    unknowntimelines = self.cache.get(self.Key(False), [])[-lines:]
                    for line in unknowntimelines:
                        LOGFORMAT.output(line)
                #
                self.newcache.sort(key=lambda x: x.now)
                for line in self.newcache:
                    LOGFORMAT.output(line)
            finally:
                cachetxt = [
                    len(self.cache.get(self.Key(True), "")),
                    len(self.cache.get(self.Key(False), "")),
                ]
                log.debug(
                    f"{self.thread.name} finished"
                    f" len(newcache)={len(self.newcache)} len(cache)={cachetxt} -n={ARGS.lines}"
                )
                self.newcache.clear()
                self.cache.clear()
                self.finished = True

    def start(self, ModifyIndex: Optional[int]):
        """Start the thread"""
        self.old_ModifyIndex = ModifyIndex
        if ARGS.lines < 0:
            # If args is lower than 0, we set finished right away.
            self.finished = True
        else:
            if not self.threadstarted:
                self.threadstarted = True
                self.thread.start()

    def join(self):
        """The only possible way the thread can get stuck is if print() output
        is stuck, as if writing to a pipe."""
        if ARGS.lines >= 0:
            self.thread.join()

    def log(self, line: LogLine):
        """Try to log the line. The line is either logged streight or added to cache"""
        if self.finished:
            LOGFORMAT.output(line)
            return
        with self.lock:
            if self.finished:
                LOGFORMAT.output(line)
                return
            if (
                self.old_ModifyIndex is not None
                and line.ModifyIndex > self.old_ModifyIndex
            ):
                self.newcache.append(line)
            else:
                self.cache.setdefault(self.Key(line.known_time), []).append(line)


MYLOGGERDELAYER = MyloggerDelayer()


class Mylogger:
    """Used to log from the various streams we want to log from"""

    @staticmethod
    def __log(**kwargs):
        line = LogLine(**kwargs)
        MYLOGGERDELAYER.log(line)

    ###############################################################################

    @classmethod
    def log_eval(cls, eval: nomadlib.Eval, message: str):
        return cls.__log(
            what=LogWhat.eval,
            id=eval.ID,
            now=ns2dt(eval.ModifyTime),
            jobversion=DB.find_jobversion_from_modifyindex(eval.ModifyIndex),
            message=message,
            ModifyIndex=eval.ModifyIndex,
        )

    @classmethod
    def log_deploy(cls, deploy: nomadlib.Deploy, message: str):
        job = DB.jobversions.get(deploy.JobVersion)
        return cls.__log(
            what=LogWhat.deploy,
            id=deploy.ID,
            now=ns2dt(job.SubmitTime) if job else datetime.datetime.now().astimezone(),
            jobversion=deploy.JobVersion,
            message=message,
            ModifyIndex=deploy.ModifyIndex,
        )

    @staticmethod
    def __add_allocdata(allocid: str) -> dict:
        """Given an allocid, extracts ModifyIndex and finds job version"""
        alloc = DB.allocations.get(allocid)
        return dict(
            ModifyIndex=alloc.ModifyIndex if alloc else None,
            nodename=alloc.NodeName if alloc else None,
            group=alloc.TaskGroup if alloc else None,
            jobversion=None if not alloc else DB.get_allocation_jobversion(alloc),
        )

    @classmethod
    def log_alloc(cls, allocid: str, **kwargs):
        return cls.__log(
            what=LogWhat.alloc,
            id=allocid,
            **cls.__add_allocdata(allocid),
            **kwargs,
        )

    @classmethod
    def log_std(cls, stderr: bool, allocid: str, **kwargs):
        return cls.__log(
            what=LogWhat.stderr if stderr else LogWhat.stdout,
            id=allocid,
            **cls.__add_allocdata(allocid),
            **kwargs,
        )


@dataclass(frozen=True)
class TaskKey:
    """
    Represent data to uniquely identify a task.
    TaskKey stores allocation ID. Altough we have to do a search later,
    the allocation data itself change often and in parallel to this object.
    """

    allocid: str
    task: str

    def __str__(self):
        return f"{self.allocid:.6}:{self.task}"

    def asdict(self):
        return asdict(self)

    def log_alloc(self, now: datetime.datetime, message: str):
        Mylogger.log_alloc(now=now, message=message, **self.asdict())

    def log_task(self, stderr: bool, message: str):
        Mylogger.log_std(
            stderr=stderr,
            message=message,
            now=datetime.datetime.now().astimezone(),
            **self.asdict(),
        )


###############################################################################


class TaskLogger(threading.Thread):
    """Represents a single logging stream from Nomad. Such stream is created separately for stdout and stderr."""

    def __init__(self, tk: TaskKey, stderr: bool):
        super().__init__(name=f"{tk}:{'stderr' if stderr else 'stdout'}")
        self.tk = tk
        """task key"""
        self.stderr: bool = stderr
        """is this stderr or stdout logger"""
        self.exitreqtime: Optional[float] = None
        """stop() was called"""
        self.started: bool = False
        """This logger is assumed to have started and printed logs"""
        self.startedtime: Optional[float] = None
        """A timer that will set self.stated"""
        self.stopped: bool = False

    @staticmethod
    def read_json_txt(stream: requests.Response) -> Iterator[str]:
        txt: str = ""
        for dataorbytes in stream.iter_content(decode_unicode=True):
            data: str
            try:
                data = dataorbytes.decode()
            except (UnicodeDecodeError, AttributeError):
                data = dataorbytes
            for c in data:
                txt += c
                # Nomad happens to be consistent, the jsons are flat.
                if c == "}":
                    yield txt
                    txt = ""
        if txt:
            yield txt

    @classmethod
    def read_json_stream(cls, stream: requests.Response) -> Iterator[Dict[str, Any]]:
        for txt in cls.read_json_txt(stream):
            try:
                ret = json.loads(txt)
                # log.debug(f"RECV: {ret}")
                yield ret
            except json.JSONDecodeError as e:
                if not re.match(
                    "failed to list entries:.*no such file or directory", txt
                ):
                    log.warn(f"error decoding json: {txt} {e}")

    def __typestr(self):
        return "stderr" if self.stderr else "stdout"

    def __run_in(self):
        usestart = ARGS.lines < 0 or START_S + ARGS.lines_timeout < time.time()
        with mynomad.stream(
            f"client/fs/logs/{self.tk.allocid}",
            params={
                "task": self.tk.task,
                "type": self.__typestr(),
                "follow": True,
                "origin": "start" if usestart else "end",
                "offset": 0 if usestart else 50000,
            },
        ) as stream:
            for event in self.read_json_stream(stream):
                if event:
                    line64: Optional[str] = event.get("Data")
                    if line64:
                        linebytes = base64.b64decode(line64.encode())
                        lines = linebytes.decode(errors="replace").splitlines()
                        for line in lines:
                            self.tk.log_task(self.stderr, line.rstrip())
                    fileevent: Optional[str] = event.get("FileEvent")
                    if fileevent == "file deleted":
                        # Deleted means end of stream.
                        break
                # Nomad json stream every second sends empty {}.
                # If requested to exit, then exit.
                if self.exitreqtime is not None:
                    if self.exitreqtime <= time.time():
                        break
        stream.raise_for_status()

    def run(self):
        """Listen to Nomad log stream and print the logs"""
        try:
            # Try getting the logs for 2 seconds, then give up.
            tries: int = int(ARGS.shutdown_timeout + 1)
            for _ in range(tries):
                try:
                    self.__run_in()
                    break
                except nomadlib.LogNotFound:
                    time.sleep(1)
            else:
                self.__run_in()
        except nomadlib.LogNotFound as e:
            # Gracefully handle missing logs errors from Nomad.
            # Logs are removed by garbage collector and when purging the job.
            code = e.response.status_code if e.response is not None else None
            text = e.response.text if e.response is not None else None
            self.tk.log_alloc(
                datetime.datetime.now().astimezone(),
                f"Error getting {self.__typestr()} logs: {code} {text!r}",
            )
        finally:
            log.debug(f"{self} stopped")
            self.stopped = True
            DB.send_empty_event()

    def stop(self, delay: bool):
        if self.exitreqtime is None or (
            delay is True and self.exitreqtime > time.time()
        ):
            log.debug(f"{self} stoppping delay={delay}")
            self.exitreqtime = time.time() + delay * ARGS.shutdown_timeout


@dataclass
class TaskHandler:
    """A handler for one task. Creates loggers, writes out task events, handle exit conditions"""

    loggers: Optional[List[TaskLogger]] = None
    """Array of loggers that log allocation logs."""
    messages: Set[int] = field(default_factory=set)
    """A set of message timestamp to know what has been printed."""
    exitcode: Optional[int] = None

    @staticmethod
    def _create_loggers(tk: TaskKey):
        ths: List[TaskLogger] = []
        if LOGENABLED.stdout:
            ths.append(TaskLogger(tk, False))
        if LOGENABLED.stderr:
            ths.append(TaskLogger(tk, True))
        for th in ths:
            th.start()
        log.debug(f"Started {len(ths)} loggers for {tk}")
        return ths

    def notify(self, tk: TaskKey, taskstate: nomadlib.AllocTaskState):
        """Receive notification that a task state has changed"""
        if LOGENABLED.alloc:
            for e in taskstate.Events or []:
                msgtime_ns = e.Time
                # Ignore message before ignore times.
                if (
                    msgtime_ns
                    and (
                        ARGS.lines < 0
                        or msgtime_ns >= int(START_S * 10**9)
                        or len(self.messages) < ARGS.lines
                    )
                    and set_not_in_add(self.messages, msgtime_ns)
                ):
                    for line in e.DisplayMessage.splitlines():
                        if line:
                            tk.log_alloc(ns2dt(msgtime_ns), f"{e.Type} {line}")
        if (
            self.loggers is None
            and taskstate.State in ["running", "dead"]
            and taskstate.was_started()
        ):
            self.loggers = self._create_loggers(tk)
        if taskstate.State == "dead":
            # log.debug(f"Task {tk} dead")
            self.stop(True)
        if self.exitcode is None and taskstate.State == "dead":
            terminatedevent = taskstate.find_event("Terminated")
            if terminatedevent:
                self.exitcode = terminatedevent.ExitCode

    def stop(self, delay: bool = False):
        for ll in self.loggers or []:
            ll.stop(delay)


@dataclass
class AllocWorker:
    """Represents a worker that prints out and manages state related to one allocation"""

    taskhandlers: Dict[TaskKey, TaskHandler] = field(default_factory=dict)

    def notify(self, alloc: nomadlib.Alloc):
        """Update the state with alloc"""
        for taskname, task in alloc.get_taskstates().items():
            if ARGS.task and not ARGS.task.search(taskname):
                continue
            if ARGS.node and not ARGS.node.search(alloc.NodeName):
                continue
            tk = TaskKey(allocid=alloc.ID, task=taskname)
            self.taskhandlers.setdefault(tk, TaskHandler()).notify(tk, task)


class ExitCode:
    success = 0
    exception = 1
    interrupted = 2
    """This program execution flow was interrupted - user clicked ctrl+c"""
    failed = 3
    """Starting a job failed - deployment was reverted"""
    any_failed_tasks = 124
    all_failed_tasks = 125
    any_unfinished_tasks = 126
    no_allocations = 127


@dataclass
class ExitcodeRet:
    code: int
    reason: str


@dataclass
class NotifierWorker:
    """An containers for storing a map of allocation workers"""

    workers: Dict[str, AllocWorker] = field(default_factory=dict)
    """Allocation ID to allocation worker of this allocation"""
    messages: Set[Tuple[int, str]] = field(default_factory=set)
    """A set of evaluation ModifyIndex to know what has been printed."""

    def lineno_key_not_printed(self, key: str) -> bool:
        """
        Nomad has no timestamp of messages. I want to keep track of messages printed not to print them twice.
        I check if a specific message was printed already by tracking unique key and file line number information.
        """
        lineno = inspect.currentframe().f_back.f_lineno  # type: ignore
        return set_not_in_add(self.messages, (lineno, key))

    def alloc_notified(self, alloc: nomadlib.Alloc):
        return alloc.ID in self.workers

    def notify_alloc(self, alloc: nomadlib.Alloc):
        if ARGS.group and not ARGS.group.search(alloc.TaskGroup):
            return
        #
        if LOGENABLED.eval:
            evaluation = DB.evaluations.get(alloc.EvalID)
            if evaluation and self.lineno_key_not_printed(f"{alloc.ID} {alloc.EvalID}"):
                Mylogger.log_eval(
                    evaluation,
                    f"Allocation {alloc.ID} started on {alloc.NodeName}",
                )
        #
        self.workers.setdefault(alloc.ID, AllocWorker()).notify(alloc)
        #
        if LOGENABLED.eval and alloc.is_finished():
            evaluation = DB.evaluations.get(alloc.EvalID)
            if evaluation and self.lineno_key_not_printed(f"{alloc.ID} {alloc.EvalID}"):
                Mylogger.log_eval(evaluation, f"Allocation {alloc.ID} finished")
        #
        if LOGENABLED.eval and alloc.FollowupEvalID:
            followupeval = DB.evaluations.get(alloc.FollowupEvalID)
            if followupeval:
                waituntil = followupeval.getWaitUntil()
                if waituntil and self.lineno_key_not_printed(followupeval.ID):
                    utcnow = datetime.datetime.now(datetime.timezone.utc)
                    delay = waituntil - utcnow
                    if delay > datetime.timedelta(0):
                        Mylogger.log_eval(
                            followupeval,
                            f"Nomad will attempt to reschedule in {delay} seconds",
                        )

    def notify_eval(self, evaluation: nomadlib.Eval):
        if (
            LOGENABLED.eval
            and evaluation.Status == nomadlib.EvalStatus.blocked.value
            and "FailedTGAllocs" in evaluation
            and self.lineno_key_not_printed(evaluation.ID)
        ):
            Mylogger.log_eval(
                evaluation,
                f"{evaluation.JobID}: Placement Failures: {len(evaluation.FailedTGAllocs)} unplaced",
            )
            for task, metric in evaluation.FailedTGAllocs.items():
                for msg in metric.format(True, f"{task}: ").splitlines():
                    Mylogger.log_eval(evaluation, msg)

    def notify_deploy(self, deployment: nomadlib.Deploy):
        # If the job has any service defined.
        if (
            LOGENABLED.eval
            and DB.job
            and deployment.Status
            in [
                nomadlib.DeploymentStatus.successful,
                nomadlib.DeploymentStatus.failed,
            ]
        ):
            """
            https://github.com/hashicorp/nomad/blob/e02dd2a331c778399fd271e85c75bff3e3783d80/ui/app/templates/components/job-deployment/deployment-metrics.hbs#L10
            """
            for task, tg in deployment.TaskGroups.items():
                Mylogger.log_deploy(
                    deployment,
                    f"{task} Canaries={len(tg.PlacedCanaries or [])}/{tg.DesiredCanaries}"
                    f" Placed={tg.PlacedAllocs} Desired={tg.DesiredTotal} Healthy={tg.HealthyAllocs}"
                    f" Unhealthy={tg.UnhealthyAllocs} {deployment.StatusDescription}",
                )

    def stop(self):
        for w in self.workers.values():
            for th in w.taskhandlers.values():
                th.stop()

    def get_threads(self) -> List[TaskLogger]:
        return [
            logger
            for w in self.workers.values()
            for th in w.taskhandlers.values()
            for logger in th.loggers or []
        ]

    def all_threads_stopped(self) -> bool:
        return all(th.stopped for th in self.get_threads())

    def join(self):
        threads = self.get_threads()
        # Logs stream outputs empty {} which allows to handle timeouts.
        thcnt = sum(len(w.taskhandlers) for w in self.workers.values())
        log.debug(
            f"Joining {len(self.workers)} allocations with {thcnt} taskhandlers and {len(threads)} loggers"
        )
        MYLOGGERDELAYER.join()
        timeend = time.time() + ARGS.shutdown_timeout
        timeout = None
        for thread in threads:
            timeout = timeend - time.time()
            if timeout > 0:
                log.debug(f"joining worker {thread.name} timeout={timeout}")
                thread.join(timeout)
            else:
                break
        log.debug(f"{len(threads)} threads joined with {timeout} timeout to spare")

    def __exitcode(self) -> ExitcodeRet:
        tasks_exitcodes: List[Optional[int]] = [
            th.exitcode for w in self.workers.values() for th in w.taskhandlers.values()
        ]
        if flagdebug.debug("exitcode"):
            for _, w in self.workers.items():
                for taskkey, taskhandler in w.taskhandlers.items():
                    eprint(f"EXITCODE: {taskkey} {taskhandler.exitcode}")
        if len(tasks_exitcodes) == 0:
            return ExitcodeRet(ExitCode.no_allocations, "there were no allocations")
        unfinished_tasks = [v for v in tasks_exitcodes if v is None]
        if unfinished_tasks:
            return ExitcodeRet(
                ExitCode.any_unfinished_tasks,
                f"There were {len(unfinished_tasks)} unfinished tasks",
            )
        only_one_task = len(tasks_exitcodes) == 1
        if only_one_task:
            assert tasks_exitcodes[0] is not None
            return ExitcodeRet(
                tasks_exitcodes[0],
                f"Single task exited with {tasks_exitcodes[0]} exit status",
            )
        failed_tasks = [v for v in tasks_exitcodes if v != 0]
        if len(failed_tasks) == len(tasks_exitcodes):
            return ExitcodeRet(
                ExitCode.all_failed_tasks,
                f"All {len(tasks_exitcodes)} tasks exited with nonzero exit status",
            )
        if failed_tasks:
            return ExitcodeRet(
                ExitCode.any_failed_tasks,
                f"{len(failed_tasks)} tasks exited with nonzero exit status",
            )
        return ExitcodeRet(
            ExitCode.success,
            f"All {len(tasks_exitcodes)} tasks exited with zero exit status",
        )

    def exitcode(self) -> int:
        er: ExitcodeRet = self.__exitcode()
        log.info(f"{er.reason.capitalize()}. Exit code is {er.code}.")
        return er.code


###############################################################################


class redirect_stdin_from_str(contextlib.AbstractContextManager):
    def __init__(self, txt: str):
        self.txt = txt

    def __enter__(self):
        self.old = sys.stdin
        sys.stdin = self.new = io.StringIO(self.txt)
        return self.new

    def __exit__(self, exctype, excinst, exctb):
        self.new.close()
        sys.stdin = self.old


def __nomad_job_run(args: List[str]) -> str:
    """Call nomad job run to start a Nomad job using nomad job run call with specified arguments"""
    cmd: List[str] = [*"nomad job run -detach -verbose".split(), *args]
    log.info(f"+ {' '.join(shlex.quote(x) for x in cmd)}")
    try:
        output = subprocess.check_output(cmd, text=True, stdin=sys.stdin)
    except subprocess.CalledProcessError as e:
        # nomad will print its error, we can just exit
        exit(e.returncode)
    # Extract evaluation id from Nomad output.
    for line in output.splitlines():
        log.info(line)
    founduuids = re.findall(
        "[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
        output.lower(),
        re.MULTILINE,
    )
    assert len(founduuids) == 1, (
        "Could not find UUID in nomad job run output."
        " This may be caused by the job being parametric or periodic,"
        " nomad command has upgraded its output,"
        " invalid command line arguments were passed,"
        " or there is an error in this script."
    )
    evalid = founduuids[0]
    return evalid


def nomad_start_job(opts: List[str]) -> nomadlib.Eval:
    """Start a nomad job using nomad job run parameters. Return evluation from running the job"""
    assert opts
    evalid: Optional[str] = None
    file: str = opts[-1]
    if ARGS.json or (len(opts) == 2 and opts[0] == "-json"):
        # If the input file is a json and we have no arguments, we can use the API ourselves.
        if ARGS.jobarg:
            data = file
        else:
            stream = contextlib.closing(sys.stdin) if file == "-" else open(file)
            with stream as f:
                data: str = f.read()
        job: dict = json.loads(data)
        if "Job" in job:
            job = job["Job"]
        format: str = "json"
        resp = mynomad.start_job(job, nomadlib.JobSubmission(data, format))
        evalid = resp["EvalID"]
        warnings = resp.get("Warnings")
        if warnings:
            log.warning(f"{warnings}")
    else:
        # Otherwise, call nomad executable to schedule the job.
        if ARGS.json:
            opts = ["-json"] + opts
        if ARGS.jobarg:
            with redirect_stdin_from_str(file):
                opts[:-1] = "-"
                evalid = __nomad_job_run(opts)
        else:
            evalid = __nomad_job_run(opts)
    evaluation = nomadlib.Eval(mynomad.get(f"evaluation/{evalid}"))
    mynomad.namespace = evaluation.Namespace
    return evaluation


###############################################################################


class _NomadJobWatcherDetail(ABC):
    """Watches over a job. Schedules watches over allocations. Spawns loggers."""

    endstatusstr: ClassVar[str]
    """Notify user up-until we are watching the job"""

    def __init__(self, jobid: Optional[str], eval: Optional[nomadlib.Eval] = None):
        assert (jobid and not eval) or (not jobid and eval)
        assert mynomad.namespace
        self.eval = eval
        """Evaluation or None"""
        self.jobid: str = self.eval.JobID if self.eval else jobid if jobid else ""
        """Watched job ID"""
        assert self.jobid
        self.job: Optional[nomadlib.Job] = None
        """Last valid job with matching watched job ID"""
        global DB
        DB = NomadDbJob(
            topics=[
                f"Job:{self.jobid}",
                f"Evaluation:{self.jobid}",
                f"Allocation:{self.jobid}",
                f"Deployment:{self.jobid}",
            ],
            select_event_cb=self.__db_select_event_job,
            init_cb=self.__db_init_cb,
            force_polling=True if ARGS.polling else None,
        )
        self.notifier = NotifierWorker()
        """Notification worker dispatching Nomad stream events"""
        self.no_follow_end: bool = False
        """Flag that is set once no_follow timeout passes"""
        #
        self.interruptcounter = InterruptCounter()
        if ARGS.no_follow:
            th = threading.Timer(ARGS.no_follow_timeout, self.__no_follow_timer)
            th.setDaemon(True)
            th.start()
        DB.start()
        self.interruptcounter.install()
        MYLOGGERDELAYER.start(self.eval.ModifyIndex if self.eval else None)

    def __no_follow_timer(self):
        self.no_follow_end = True
        DB.send_empty_event()

    def __db_init_cb(self) -> List[Event]:
        """Db initialization callback"""
        try:
            job: dict = mynomad.get(f"job/{self.jobid}")
            jobversions: dict = mynomad.get(f"job/{self.jobid}/versions")["Versions"]
            deployments: List[dict] = mynomad.get(f"job/{self.jobid}/deployments")
            evaluations: List[dict] = mynomad.get(f"job/{self.jobid}/evaluations")
            allocations: List[dict] = mynomad.get(f"job/{self.jobid}/allocations")
        except nomadlib.JobNotFound as e:
            # This is fine to fail if it was purged, potentially.
            # If the job was purged, generate one event so that listener can catch it.
            # In the follow mode, just ignore missing job.
            log.debug(f"JobNotFound exception: {e}")
            DB.job_deregistered_ModifyIndex = DB.job.ModifyIndex + 1 if DB.job else 0
            return []
        # Set the job if not set for the first time.
        if not self.job:
            self.job = nomadlib.Job(job)
            log.debug(
                f"Found job {self.job.description()} from {self.eval.ID if self.eval else None}"
                f" with {len(allocations)} allocations"
            )
        # Finally output the events for database to pick up.
        events: List[Event] = [
            Event(-2, EventTopic.Job, EventType.JobRegistered, job),
            *[
                Event(-3, EventTopic.Job, EventType.JobRegistered, job)
                for job in jobversions
            ],
            *[
                Event(-4, EventTopic.Deployment, EventType.DeploymentStatusUpdate, d)
                for d in deployments
            ],
            *[
                Event(-5, EventTopic.Evaluation, EventType.EvaluationUpdated, e)
                for e in evaluations
            ],
            *[
                Event(-6, EventTopic.Allocation, EventType.AllocationUpdated, a)
                for a in allocations
            ],
        ]
        events.sort(key=lambda event: event.data["ModifyIndex"])
        return events

    def __db_select_event_job_id(self, e: Event):
        """Select only events about observed job ID. Ignore all events about other jobs."""
        return DB.apply_selects(
            e,
            lambda job: job.ID == self.jobid,
            lambda eval: eval.JobID == self.jobid,
            lambda alloc: alloc.JobID == self.jobid,
            lambda deploy: deploy.JobID == self.jobid,
        )

    def __db_select_event_job(self, e: Event):
        return ARGS.all or self.__db_select_event_job_id(e)

    @cached_property
    def minjobmodifyindex(self) -> int:
        """
        Get the JobModifyIndex we are watching events from.
        It captures the moment in time where from we started watching the job.
        Return smallest value of JobModifyIndex of interesting events.
        """
        assert self.job, f"should be set in {self.__db_init_cb.__name__}"
        jobmodifyindex: int = self.job.JobModifyIndex
        if self.eval is not None and self.eval.JobModifyIndex is not None:
            jobmodifyindex = min(jobmodifyindex, self.eval.JobModifyIndex)
        # If there are active alocations with JobModifyIndex lower than current watched one,
        # also start watching them by lowering JobModfyIndex threashold.
        min_active_allocation_jobmodifyindex: int = min(
            (
                DB.get_allocation_jobmodifyindex(alloc, jobmodifyindex)
                for alloc in DB.allocations.values()
                if nomadlib.Alloc(alloc).is_pending_or_running()
            ),
            default=jobmodifyindex,
        )
        jobmodifyindex = min(jobmodifyindex, min_active_allocation_jobmodifyindex)
        # Set the JobModifyIndex to the last still running Job version.
        not_stopped_jobs = [job for job in DB.jobversions.values() if not job.Stop]
        if not_stopped_jobs:
            last_not_stopped_job = sorted(
                not_stopped_jobs, key=lambda job: job.ModifyIndex
            )[-1]
            jobmodifyindex = min(jobmodifyindex, last_not_stopped_job.JobModifyIndex)
        return jobmodifyindex

    def __handle_event(self, event: Event):
        """Handle a single event from event stream"""
        if event.topic == EventTopic.Allocation:
            allocation = nomadlib.Alloc(event.data)
            if (
                ARGS.all
                or self.notifier.alloc_notified(allocation)
                or DB.get_allocation_jobmodifyindex(allocation, -1)
                >= self.minjobmodifyindex
            ):
                self.notifier.notify_alloc(allocation)
        elif event.topic == EventTopic.Evaluation:
            evaluation = nomadlib.Eval(event.data)
            if self.eval and self.eval.ID == evaluation.ID:
                self.eval = evaluation
                self.notifier.notify_eval(evaluation)
            elif (
                ARGS.all
                or (
                    evaluation.JobModifyIndex is not None
                    and evaluation.JobModifyIndex >= self.minjobmodifyindex
                )
                or evaluation.is_blocked()
            ):
                self.notifier.notify_eval(evaluation)
        elif event.topic == EventTopic.Deployment:
            deployment = nomadlib.Deploy(event.data)
            if ARGS.all or deployment.JobModifyIndex >= self.minjobmodifyindex:
                self.notifier.notify_deploy(deployment)
        elif event.topic == EventTopic.Job:
            if self.job and DB.job and self.job.ModifyIndex < DB.job.ModifyIndex:
                # self.job follows newest job definition.
                self.job = DB.job

    def __loop_debug(self, events: List[nomaddbjob.Event]):
        if flagdebug.debug("loop"):
            info = dict(
                e=f"{events[0].topic.name}.{events[0].type.name}" if events else "-",
                minModifyIndex=self.minjobmodifyindex,
                evalDone=self.eval is None or not self.eval.is_pending_or_blocked(),
                jobDeregister=DB.job_deregistered_ModifyIndex,
                seenJob=self.job and DB.seen_job(),
                activeEvals=self.has_active_evaluations(),
                activeAllocs=self.has_active_allocations(),
                activeDeploys=self.has_active_deployments(),
                isPurged=DB.job_purged(),
                jobDead=self.job.is_dead() if self.job else None,
                notifiers=[
                    f"{th.name}={th.started}" for th in self.notifier.get_threads()
                ],
            )
            eprint(f"LOOP: {strdict(info)}")

    def loop(self) -> Iterator[None]:
        """Thread entrypoint that handles events from Nomad event stream database.
        Main event loop."""
        untilstr = (
            "forever"
            if ARGS.follow
            else f"for {ARGS.no_follow_timeout} seconds"
            if ARGS.no_follow
            else f"until it is {self.endstatusstr}"
        )
        log.info(f"Watching job {self.jobid}@{mynomad.namespace} {untilstr}")
        for events in DB.events():
            for event in events:
                self.__handle_event(event)
            self.__loop_debug(events)
            # If we started with an evaluation, it has to be finished.
            # Otherwise the events may be related to a previous job version.
            if self.eval is None or not self.eval.is_pending_or_blocked():
                yield
            if self.no_follow_end:
                break
        log.debug(f"Watching job {self.jobid}@{mynomad.namespace} exiting")

    def has_active_deployments(self):
        for deployment in DB.deployments.values():
            if not deployment.is_finished():
                if flagdebug.debug("active"):
                    eprint(f"ACTIVE: {deployment}")
                return True
        return False

    def has_active_evaluations(self):
        for evaluation in DB.evaluations.values():
            if evaluation.is_pending_or_blocked():
                if flagdebug.debug("active"):
                    eprint(f"ACTIVE: {evaluation}")
                return True
        return False

    def has_active_allocations(self):
        for alloc in DB.allocations.values():
            if alloc.is_pending_or_running():
                if flagdebug.debug("active"):
                    eprint(
                        f"ACTIVE: alloc {alloc.ID[:6]} status={alloc.ClientStatus}"
                        f" desired={alloc.DesiredStatus}"
                    )
                return True
        return False

    def has_no_active_allocations_nor_evaluations_nor_deployments(self):
        return (
            not self.has_active_allocations()
            and not self.has_active_evaluations()
            and not self.has_active_deployments()
        )


class NotifyEvent(MyStrEnum):
    started = enum.auto()
    purging = enum.auto()
    stopping = enum.auto()
    exit = enum.auto()

    @classmethod
    def txt(cls) -> str:
        return andjoin((repr(x + "") for x in cls), fin=" or ")


class _NomadJobWatcherEvents(_NomadJobWatcherDetail):
    @override
    def __init__(self, jobid: Optional[str], eval: Optional[nomadlib.Eval] = None):
        super().__init__(jobid, eval)
        self.purged_eval: Optional[str] = None
        """Return true if we sent a request to purge the job"""
        self.stopped_eval: Optional[str] = None
        """Return true if we sent a request to stop the job"""
        self.job_started: bool = False
        """If set to True, means that the job has started all main tasks"""
        self.notify: list[TextIO] = [
            open(int(x) if x.isdigit() else x) for x in sorted(list(set(ARGS.notify)))
        ]
        self.notifystarted: list[TextIO] = [
            open(int(x) if x.isdigit() else x)
            for x in sorted(list(set(ARGS.notifystarted)))
        ]

    def __notifyfile(self, arg: List[str], files: list[TextIO]):
        for file in files:
            line: str = json.dumps(arg)
            assert line.count("\n") == 0
            log.debug(f"notify fd {file.name} {line!r}")
            try:
                file.write(line)
            except (BrokenPipeError, IOError) as e:
                log.debug(f"notify exception {file.name} {e}")
                try:
                    file.close()
                except Exception:
                    pass
                files.remove(file)

    def __notifyuser(self, event: NotifyEvent):
        flagdebug.logdebug("notify", f"notify {event}")
        arg: List[str] = [self.jobid, event]
        cmd: List[str]
        for exe in ARGS.notifyexe:
            cmd = shlex.split(exe) + arg
            log.debug(f"notify exe {quotearr(cmd)}")
            subprocess.check_call(cmd)
        self.__notifyfile(arg, self.notify)
        if event == NotifyEvent.started:
            for exe in ARGS.notifyexestarted:
                cmd = shlex.split(exe)
                log.debug(f"notify exestarted {quotearr(cmd)}")
                subprocess.check_call(cmd)
            self.__notifyfile(arg, self.notifystarted)
            for file in self.notifystarted:
                file.close()

    @override
    def loop(self) -> Iterator[None]:
        try:
            for i in super().loop():
                if not self.job_started:
                    if self.__job_has_finished_starting():
                        self.__notifyuser(NotifyEvent.started)
                        self.job_started = True
                yield i
        finally:
            self.__notifyuser(NotifyEvent.exit)

    def stop_job(self, purge: bool = False):
        assert DB.initialized.is_set()
        if purge:
            if self.purged_eval:
                return
            self.__notifyuser(NotifyEvent.purging)
            log.info(f"Purging job {self.jobid}")
        else:
            if self.stopped_eval:
                return
            log.info(f"Stopping job {self.jobid}")
            self.__notifyuser(NotifyEvent.stopping)
        evalid = mynomad.stop_job(self.jobid, purge)["EvalID"]
        self.stopped_eval = evalid
        if purge:
            self.purged_eval = evalid

    @staticmethod
    def __nomad_job_group_main_tasks(group: nomadlib.JobTaskGroup):
        """Get a set of names of Nomad job group main tasks"""
        # Main tasks are tasks that:
        # - do not have lifecycle
        # - have lifecycle prestart with sidecar = true
        # - have lifecycle poststart
        # All these tasks have to be started.
        maintasks = set(
            t.Name
            for t in group.Tasks
            if "Lifecycle" not in t
            or t.Lifecycle is None
            or (
                t.Lifecycle.Hook == nomadlib.LifecycleHook.prestart
                and t.Lifecycle.get_sidecar() is True
            )
            or t.Lifecycle.Hook == nomadlib.LifecycleHook.poststart
        )
        assert len(maintasks) > 0, f"Internal error when getting main tasks of {group}"
        return maintasks

    def __job_has_finished_starting(self) -> bool:
        """
        Return True if the job is not starting anymore.
        self.started will be set to True, if the job was successfully started.
        """
        # log.debug(f"has_active_deployments={self.has_active_deployments()} has_active_evaluations={self.has_active_evaluations()} allocations={len(DB.allocations)}")  # noqa
        if (
            not self.job
            or self.has_active_deployments()
            or self.has_active_evaluations()
        ):
            flagdebug.logdebug(
                "started",
                "The job is still doing something. Wait for it. "
                f" job={self.job} deployments={self.has_active_deployments()} evals={self.has_active_evaluations()}",
            )
            return False
        allocations = DB.allocations.values()
        if (
            not allocations
            or any(alloc.is_pending() for alloc in allocations)
            or any(alloc.TaskStates is None for alloc in allocations)
        ):
            flagdebug.logdebug(
                "started",
                "There are still allocations which Tasks have not started yet. Wait for them."
                f" allocations={not not allocations}"
                f" pendings={sum(alloc.is_pending() for alloc in allocations)}"
                f" taskstates={sum(alloc.TaskStates is None for alloc in allocations)}",
            )
            return False
        # Check if there are tasks running for all Groups.
        groupmsgs: List[str] = []
        for group in self.job.TaskGroups:
            previousrunningallocs: List[nomadlib.Alloc] = [
                alloc
                for alloc in allocations
                if alloc.TaskGroup == group.Name
                and alloc.is_pending_or_running()
                and (
                    DB.get_allocation_jobmodifyindex(alloc, -1) < self.minjobmodifyindex
                    or DB.get_allocation_jobversion(alloc, -1) < self.job.Version
                )
            ]
            # Allocations related to previous job version have to be stopped.
            if previousrunningallocs:
                flagdebug.logdebug(
                    "started",
                    f"There are still allocations running for previous job version: {previousrunningallocs}",
                )
                return False
            #
            groupallocs: List[nomadlib.Alloc] = [
                alloc
                for alloc in allocations
                if alloc.TaskGroup == group.Name
                and DB.get_allocation_jobmodifyindex(alloc, -1)
                >= self.minjobmodifyindex
                and DB.get_allocation_jobversion(alloc, -1) == self.job.Version
            ]
            # There have to be at exactly group.Count allocations of this group for it to be deployed.
            # The allocation not necessarily have to be running - they may have finished.
            if len(groupallocs) != group.Count:
                # This group has no active evaluation and deployments (checked above).
                flagdebug.logdebug(
                    "started",
                    f"groupallocs={[x.ID for x in groupallocs]}"
                    f" {[DB.get_allocation_jobmodifyindex(alloc, -1) for alloc in groupallocs]}"
                    f" {[DB.get_allocation_jobversion(alloc, -1) for alloc in groupallocs]}",
                )
                log.error(
                    f"Job {self.job.description()} group {group.Name!r} started"
                    f" {len(groupallocs)} allocation out of {group.Count}."
                )
                return True
            maintasks: Set[str] = self.__nomad_job_group_main_tasks(group)
            for alloc in groupallocs:
                # List of started tasks.
                startedtasks: Set[str] = set(
                    name
                    for name, taskstate in alloc.get_taskstates().items()
                    if taskstate.was_started()
                )
                notrunningmaintasks = maintasks.difference(startedtasks)
                if notrunningmaintasks:
                    # There are main tasks that are not running.
                    if alloc.TaskStates is None or alloc.is_pending_or_running():
                        flagdebug.logdebug(
                            "started",
                            f"TaskStates={alloc.TaskStates} is_pending_or_running={alloc.is_pending_or_running()}",
                        )
                        # Wait for them.
                        return False
                    else:
                        # The allocation has finished - the tasks will never start.
                        log.error(
                            f"Job {self.job.description()} failed to start group"
                            f" {group.Name!r} tasks {' '.join(notrunningmaintasks)}"
                        )
                        return True
            groupallocsidsstr = andjoin(alloc.ID[:6] for alloc in groupallocs)
            groupmsgs.append(
                f"allocations {groupallocsidsstr} running group {group.Name!r} with {len(maintasks)} main tasks"
            )
        self.started = True
        msg = f"Job {self.job.description()} started " + andjoin(groupmsgs)
        # If we are running in eval mode, we can check the associated deployment for failure.
        if self.eval and self.eval.DeploymentID:
            deploy = DB.deployments.get(self.eval.DeploymentID)
            if deploy and deploy.Status in [
                nomadlib.DeploymentStatus.cancelled,
                nomadlib.DeploymentStatus.failed,
            ]:
                msg += f", but deployment failed: {deploy.StatusDescription!r}"
                self.started = False
        log.info(msg + ".")
        return True


class NomadJobWatcher(_NomadJobWatcherEvents):
    @abstractmethod
    def _get_exitcode_cb(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def _loop_end_cb(self):
        raise NotImplementedError()

    def run_and_exit(self, stopit: bool = False):
        try:
            for _ in self.loop():
                # If the job is purged, stop watching.
                if (
                    not ARGS.follow
                    and self.job
                    and (
                        DB.job_purged()
                        or (
                            self.purged_eval
                            and self.purged_eval in DB.evaluations
                            and DB.evaluations[self.purged_eval].is_finished()
                        )
                    )
                ):
                    if self.has_no_active_allocations_nor_evaluations_nor_deployments():
                        log.info(
                            f"Job {self.job.description()} purged with no active allocations, evaluations nor deployments. Exiting."
                        )
                    else:
                        log.info(f"Job {self.job.description()} purged. Exiting.")
                    break
                if stopit:
                    self.stop_job()
                if self.interruptcounter.cnt:
                    if ARGS.attach:
                        self.stop_job()
                    else:
                        exit(ExitCode.interrupted)
                if not ARGS.follow and self._loop_end_cb():
                    break
        finally:
            log.debug(f"stopping {self.__class__.__name__}")
            self.notifier.stop()
            DB.stop()
            if not isinstance(sys.exc_info()[1], SystemExit):
                self.notifier.join()
                DB.join()
        exit(
            ExitCode.interrupted
            if self.interruptcounter.cnt
            else self._get_exitcode_cb()
        )

    def job_is_finished(self):
        return (
            DB.seen_job()
            and self.job
            and self.job.is_dead()
            and self.has_no_active_allocations_nor_evaluations_nor_deployments()
        )

    def job_finished_successfully(self):
        s: nomadlib.JobSummarySummary = nomadlib.JobSummary(
            mynomad.get(f"job/{self.jobid}/summary")
        ).get_sum_summary()
        log.debug(f"{s}")
        return s.only_completed()

    def job_dead_message(self):
        assert self.job
        return f"Job {self.job.description()} dead with no active allocations, evaluations nor deployments. Exiting."


class NomadJobWatcherUntilFinished(NomadJobWatcher):
    """Watch a job until the job is dead or purged"""

    endstatusstr: ClassVar[str] = "finished"

    @override
    def _loop_end_cb(self) -> bool:
        if self.purged_eval:
            # Wait for beeing purged in the main loop.
            return False
        if self.job_is_finished():
            if not self.purged_eval and (
                ARGS.purge
                or (ARGS.purge_successful and self.job_finished_successfully())
            ):
                if self.notifier.all_threads_stopped():
                    self.stop_job(True)
            else:
                log.info(self.job_dead_message())
                return True
        return False

    @override
    def _get_exitcode_cb(self) -> int:
        exitcode: int = (
            ExitCode.success if ARGS.no_preserve_status else self.notifier.exitcode()
        )
        log.debug(f"exitcode={exitcode} no_preserve_status={ARGS.no_preserve_status}")
        return exitcode


class NomadJobWatcherUntilStarted(NomadJobWatcher):
    """Watches a job until the job is started"""

    endstatusstr: ClassVar[str] = "started"

    @override
    def __init__(self, jobid: Optional[str], eval: Optional[nomadlib.Eval] = None):
        super().__init__(jobid, eval)
        self.started: bool = False
        """Will be set to True if the job was _successfully_ started"""

    @override
    def _loop_end_cb(self) -> bool:
        if self.job_started:
            return True
        if self.job_is_finished():
            log.info(self.job_dead_message())
            return True
        return False

    @override
    def _get_exitcode_cb(self) -> int:
        exitcode = ExitCode.success if self.started else ExitCode.failed
        log.debug(f"started={self.started} exitcode={exitcode}")
        return exitcode


###############################################################################


@dataclass
class NomadAllocationWatcher:
    """Watch an allocation until it is finished"""

    alloc: nomadlib.Alloc
    allocworkers: NotifierWorker = field(default_factory=NotifierWorker)
    finished: bool = False

    def __post_init__(self):
        global DB
        DB = NomadDbJob(
            topics=[f"Allocation:{self.alloc.ID}"],
            select_event_cb=lambda e: e.topic == EventTopic.Allocation
            and e.data["ID"] == self.alloc.ID,
            init_cb=lambda: [
                Event(
                    -1,
                    EventTopic.Allocation,
                    EventType.AllocationUpdated,
                    mynomad.get(f"allocation/{self.alloc.ID}"),
                )
            ],
            force_polling=True if ARGS.polling else None,
        )
        #
        log.info(f"Watching allocation {self.alloc.ID}")
        DB.start()

    def run_and_exit(self):
        try:
            for events in DB.events():
                for event in events:
                    if event.topic == EventTopic.Allocation:
                        alloc = nomadlib.Alloc(event.data)
                        assert (
                            self.alloc.ID == alloc.ID
                        ), f"Internal error in Db filter: {alloc} {self.alloc}"
                        self.allocworkers.notify_alloc(alloc)
                        if alloc.is_finished():
                            log.info(
                                f"Allocation {alloc.ID} has status {alloc.ClientStatus}. Exiting."
                            )
                            self.finished = True
                            break
                if ARGS.no_follow or self.finished:
                    break
        finally:
            DB.stop()
            self.allocworkers.stop()
            DB.join()
            self.allocworkers.join()
        exit(self.allocworkers.exitcode())


###############################################################################


class JobPath:
    """Unused currently"""

    jobname: str
    group: Optional[Pattern[str]]
    task: Optional[Pattern[str]]

    def __init__(self, param: str):
        a = param.split("@")
        self.jobname = a[0]
        if len(a) == 2:
            self.task = re.compile(a[1])
        elif len(a) == 3:
            self.group = re.compile(a[1])
            self.task = re.compile(a[2])
        assert (
            1 <= len(a) <= 3
        ), f"Invalid job/job@task/job@group@task specification: {param}"

    @staticmethod
    def complete(ctx: click.Context, _: str, incomplete: str) -> List[str]:
        complete_set_namespace(ctx)
        try:
            jobs = mynomad.get("jobs")
        except requests.HTTPError:
            return []
        jobsids = [x["ID"] for x in jobs]
        arg = incomplete.split("@")
        complete = []
        if len(arg) == 1:
            complete = [x for x in jobsids]
            complete += [f"{x}@" for x in jobsids]
        elif len(arg) == 2 or len(arg) == 3:
            jobid = arg[0]
            jobsids = [x for x in jobsids if x == arg[0]]
            if len(jobsids) != 1:
                return []
            mynomad.namespace = next(x for x in jobs if x["ID"] == arg[0])["Namespace"]
            try:
                job = nomadlib.Job(mynomad.get(f"job/{jobid}"))
            except requests.HTTPError:
                return []
            if len(arg) == 2:
                tasks = [t.Name for tg in job.TaskGroups for t in tg.Tasks]
                groups = [tg.Name for tg in job.TaskGroups]
                complete = itertools.chain(tasks, groups)
            elif len(arg) == 3:
                complete = [f"{tg.Name}@{t}" for tg in job.TaskGroups for t in tg.Tasks]
            complete = [f"{arg[0]}@{x}" for x in complete]
        else:
            return []
        return [x for x in complete if x.startswith(incomplete)]


@dataclass
class NotifyOptions:
    notifyexe: Tuple[str, ...] = clickdc.option(
        help=f"""
            Command to execute that will shlex.split whenver state changes
            with two arguments:
            the watched jobid and one of {NotifyEvent.txt()}.
            """,
        multiple=True,
    )
    notify: Tuple[str, ...] = clickdc.option(
        help=f"""
            Pathname of the file or an integer file desriptor that whenever state changes
            will receive a JSON array with two elements:
            the watched jobid and one of {NotifyEvent.txt()}.
            """,
        multiple=True,
    )
    notifyexestarted: Tuple[str, ...] = clickdc.option(
        help="Execute this shlex.split command once the job is started.",
        multiple=True,
    )
    notifystarted: Tuple[str, ...] = clickdc.option(
        help="Send a single line message to this file or file descriptor when the job is started.",
        multiple=True,
    )


def click_validate(check: Callable[[Any], bool], msg: str):
    def validator(ctx, param, value):
        if not check(value):
            raise click.BadParameter(msg)
        return value

    return validator


@dataclass
class Args(LogOptions, NotifyOptions):
    all: bool = clickdc.option(
        "-a",
        is_flag=True,
        help="Print logs from all allocations, including previous versions of the job.",
    )
    verbose: int = clickdc.option("-v", count=True, help="Be more verbose.")
    quiet: int = clickdc.option("-q", count=True, help="Be less verbose.")
    attach: bool = clickdc.option(
        "-A",
        is_flag=True,
        help=f"""
             Stop the job on receiving an SIGINT or SIGTERM signal.
             Exit immediately after receiving a signal the {InterruptCounter.max} times.
             """,
    )
    purge: bool = clickdc.option(
        is_flag=True,
        help="""
             After the job is stopped, purge the job and wait until the job is purged.
             Relevant in job, run, stop and stopped modes.
             """,
    )
    purge_successful: bool = clickdc.option(
        is_flag=True,
        help="""
             After the job is stopped and is successfull, purge the job and wait until it is purged.
             Job is successfull if all job summary metrics are zero except nonzero complete metric.
             Relevant in job, run, stop and stopped modes.
             """,
    )
    jobarg: bool = clickdc.option(
        help="The jobfile argument is the Nomad job specification in HCL2 or JSON format"
    )
    json: bool = clickdc.option(
        help="The job file is in JSON format",
    )
    lines: int = clickdc.option(
        "-n",
        default=30,
        show_default=True,
        help="""
             Sets the tail location in best-efforted number of lines relative to the end of logs.
             Negative value prints all available log lines.
             """,
    )
    lines_timeout: float = clickdc.option(
        default=2,
        show_default=True,
        help="When using --lines the number of lines is best-efforted by ignoring lines for this specific time",
        callback=click_validate(lambda x: x >= 0, "timeout must be greater than 0"),
    )
    shutdown_timeout: float = clickdc.option(
        default=2,
        show_default=True,
        help="The time to wait to make sure task loggers received all logs when exiting.",
        callback=click_validate(lambda x: x >= 0, "timeout must be greater than 0"),
    )
    follow: bool = clickdc.option(
        help="Never exit",
    )
    no_follow: bool = clickdc.option(
        help="Just run once, get the logs in a best-effort style and exit.",
    )
    no_follow_timeout: float = clickdc.option(
        default=3,
        show_default=True,
        help="The time to run in --no-follow mode.",
        callback=click_validate(lambda x: x >= 0, "timeout must be greater than 0"),
    )
    task: Optional[re.Pattern] = clickdc.option(
        "-t",
        type=re.compile,
        help="Only watch tasks names matching this regex.",
    )
    group: Optional[re.Pattern] = clickdc.option(
        "-g",
        type=re.compile,
        help="Only watch group names matching this regex.",
    )
    node: Optional[re.Pattern] = clickdc.option(
        type=re.compile,
        help="Only watch tasks running on nodes whose names match this regex.",
    )
    polling: bool = clickdc.option(
        help="Instead of listening to Nomad event stream, periodically poll for events.",
    )
    no_preserve_status: bool = clickdc.option(
        "-x",
        help="Do not preserve tasks exit statuses.",
    )


@click.command(
    "watch",
    cls=AliasedGroup,
    help="""
Depending on the command, run or stop a Nomad job. Watch over the job and
print all job allocation events and tasks stdouts and tasks stderrs
logs. Depending on command, wait for a specific event to happen to finish
watching. This program is intended to help debugging issues with running
jobs in Nomad and for synchronizing with execution of batch jobs in Nomad.

Logs are printed in the format: 'mark>id>vversion>task> message'.
The mark in the log lines is equal to: 'deploy' for messages printed as
a result of deployment, 'eval' for messages printed from evaluations,
'A' from allocation, 'E' for stderr logs of a task and 'O' from stdout
logs of a task.

Some commands take argument of type JOB_ID_OR_FILE . If this argument is
an existing file and that ends with .hcl .nomad or .json, the job name
and optionally job namespaces are extracted from that file.
""",
    epilog="""
\b
Examples:
    nomadtools watch run ./some-job.nomad.hcl
    nomadtools watch job some-job
    nomadtools watch alloc af94b2
    nomadtools watch -N services --task redis -1f job redis

Written by Kamil Cukrowski 2024. Licensed under GNU GPL version 3 or later.
""",
)
@flagdebug.click_debug_option("NOMADTOOLS_DEBUG")
@clickdc.adddc("args", Args)
@namespace_option()
@help_h_option()
def cli(args: Args):
    signal.signal(signal.SIGUSR1, print_all_threads_stacktrace)
    exit_on_thread_exception.install()
    global ARGS
    ARGS = args
    assert not (ARGS.follow and ARGS.no_follow), "--follow and --no-follow conflict"
    #
    global START_S
    START_S = time.time()
    init_logging()


class JobIdOrFile(click.ParamType):
    # Problem: typing watch job ./file.nomad.hcl exist with no such job
    # Solution: if the ifile is a file and it endsw ith hcl or nomad, get job name from the file.
    name = "job_id_or_file"

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> str:
        exists = Path(value).exists()
        job = None
        if exists and any(value.endswith(x) for x in ".hcl .nomad".split()):
            try:
                log.info(f"Extracting job name from file {value} using nomad command")
                output = subprocess.check_output(
                    "nomad job run -output".split() + [value]
                )
                job = json.loads(output)
            except subprocess.CalledProcessError:
                pass
            # Problem: some job files require variables
            # Solution: just parse the file content with regex
            if not job:
                log.info(f"Extracting job name from file {value} using regex")
                jobid = jobnamespace = None
                with Path(value).open() as f:
                    for line in f:
                        if not jobid:
                            m = re.match(r'^job\s*"(.*)"\s*{\s*$', line)
                            if m:
                                jobid = m[1]
                        else:
                            m = re.match(r'^\s*namespace\s*=\s*"(.*)"\s*$', line)
                            if m:
                                jobnamespace = m[1]
                                break
                            if re.match(r'^\s*group\s*".*', line):
                                # Stop parsing the file if we get to groups
                                break
                if jobid:
                    job = {}
                    job["ID"] = jobid
                    if jobnamespace:
                        job["Namespace"] = jobnamespace
        if exists and value.endswith(".json"):
            log.info(f"Extracting job name from json file {value}")
            with Path(value).open() as f:
                job = json.load(f)
        if job:
            job = job.get("Job", job)
            namespace = job.get("Namespace", None)
            if namespace:
                os.environ["NOMAD_NAMESPACE"] = mynomad.namespace = namespace
            return job["ID"]
        else:
            return value

    def shell_complete(
        self, ctx: click.Context, param: click.Parameter, incomplete: str
    ) -> List[CompletionItem]:
        jobs = [
            CompletionItem(x["ID"])
            for x in mynomad.get("jobs")
            if x["ID"].startswith(incomplete)
        ]
        if jobs:
            return jobs
        return [CompletionItem(incomplete, type="file")]


cli_jobid = click.argument("jobid", metavar="JOB_ID_OR_FILE", type=JobIdOrFile())


def cli_jobfile_disabled(name: str, help: str):
    # Disabled, because maintenance.
    nomad_job_run_flags = [
        click.option("-check-index", type=int),
        click.option("-detach", is_flag=True),
        click.option("-eval-priority", type=int),
        click.option("-json", is_flag=True),
        click.option("-hcl1", is_flag=True),
        click.option("-hcl2-strict", is_flag=True),
        click.option("-policy-override", is_flag=True),
        click.option("-preserve-counts", is_flag=True),
        click.option("-consul-token"),
        click.option("-vault-token"),
        click.option("-vault-namespace"),
        click.option("-var", multiple=True),
        click.option("-var-file", type=click.File()),
    ]
    cli_jobfile_help = """
    JOBFILE can be file with a HCL or JSON nomad job or
    it can be a string containing a HCL or JSON nomad job.
    """
    return composed(
        cli.command(
            name,
            help + cli_jobfile_help,
            context_settings=dict(ignore_unknown_options=True),
        ),
        *nomad_job_run_flags,
        click.argument(
            "jobfile",
            shell_complete=click.File().shell_complete,
        ),
    )


def cli_jobfile_command(name: str, where: str, help: str):
    f"""Command that forwards all arguments to 'nomad job {where}' command."""
    return composed(
        cli.command(
            name,
            help=help.rstrip()
            + f"""

            All following arguments are forwarded to 'nomad {where}' command.
            Note that 'nomad job {where}' has arguments starting with a single dash.
            """,
            context_settings=dict(ignore_unknown_options=True),
        ),
        click.argument(
            "args",
            nargs=-1,
            shell_complete=click.File().shell_complete,
        ),
        click.argument(
            "jobfile",
            nargs=1,
            shell_complete=click.File().shell_complete,
        ),
    )


###############################################################################


@cli.command(
    "alloc",
    help="Watch over specific allocation. Like job mode, but only one allocation is filtered.",
)
@click.argument(
    "allocid",
    shell_complete=completor(lambda: (x["ID"] for x in mynomad.get("allocations"))),
)
def mode_alloc(allocid):
    allocs = mynomad.get("allocations", params={"prefix": allocid})
    assert len(allocs) > 0, f"Allocation with id {allocid} not found"
    assert len(allocs) < 2, f"Multiple allocations found starting with id {allocid}"
    alloc = nomadlib.Alloc(allocs[0])
    mynomad.namespace = alloc.Namespace
    NomadAllocationWatcher(alloc).run_and_exit()


@cli.command(
    "eval", help="Watch like job mode the job that results from a specific evaluation."
)
@click.argument(
    "evalid",
    shell_complete=completor(lambda: (x["ID"] for x in mynomad.get("evaluations"))),
)
def mode_eval(evalid):
    evaluation = mynomad.get("evaluations", params={"prefix": evalid})
    assert len(evaluation) > 0, f"Evaluation with id {evalid} not found"
    assert len(evaluation) < 2, f"Multiple evaluations found starting with id {evalid}"
    NomadJobWatcherUntilFinished(None, nomadlib.Eval(evaluation[0])).run_and_exit()


@cli_jobfile_command(
    "run",
    "run",
    help="Run a Nomad job and then act like stopped mode.",
)
def mode_run(args: Tuple[str, ...], jobfile: str):
    evaluation = nomad_start_job(list(args) + [jobfile])
    NomadJobWatcherUntilFinished(None, evaluation).run_and_exit()


@cli.command(
    "job",
    help="Alias to stopped command.",
)
@cli_jobid
def mode_job(jobid: str):
    jobid = nomad_find_job(jobid)
    NomadJobWatcherUntilFinished(jobid).run_and_exit()


@cli_jobfile_command(
    "start", "run", help="Start a Nomad job file and then act like started command."
)
def mode_start(args: Tuple[str, ...], jobfile: str):
    evaluation = nomad_start_job(list(args) + [jobfile])
    NomadJobWatcherUntilStarted(None, evaluation).run_and_exit()


@cli.command(
    "started",
    help=f"""
Watch a Nomad job until the job is started.
Job is started when it has no active deployments
and no active evaluations
and the number of allocations is equal to the number of groups multiplied by group count
and all main tasks in each allocation are running.
An active deployment is a deployment that has status equal to
initializing, running, pending, blocked or paused.
Main tasks are all tasks without lifetime property or sidecar prestart tasks or poststart tasks.

\b
Exit with the following status:
  {ExitCode.success}  when all tasks of the job have started running,
  {ExitCode.exception}  when python exception was thrown,
  {ExitCode.interrupted}  when process was interrupted,
  {ExitCode.failed}  when job was stopped or job deployment was reverted.
""",
)
@cli_jobid
def mode_started(jobid: str):
    jobid = nomad_find_job(jobid)
    NomadJobWatcherUntilStarted(jobid).run_and_exit()


@cli.command(
    "stop",
    help="Stop a Nomad job and then act like stopped command.",
)
@cli_jobid
def mode_stop(jobid: str):
    try:
        jobid = nomad_find_job(jobid)
    except NoJobFound:
        if ARGS.no_preserve_status and (ARGS.purge or ARGS.purge_successful):
            return
        else:
            raise
    NomadJobWatcherUntilFinished(jobid).run_and_exit(True)


@cli.command(
    "purge",
    help=f"""
Alias to `--purge stop`, with the following difference in exit status.

\b
If the option --no-preserve-status is given, then exit with the following status:
  {ExitCode.success}  when the job was purged or does not exist from the start.

The command `-x purge` exits with zero exit status if the job just does not exists.
""",
)
@cli_jobid
def mode_purge(jobid: str):
    ARGS.purge = True
    try:
        jobid = nomad_find_job(jobid)
    except NoJobFound:
        if ARGS.no_preserve_status:
            return
        else:
            raise
    NomadJobWatcherUntilFinished(jobid).run_and_exit(True)


@cli.command(
    "stopped",
    help=f"""
Watch a Nomad job until the job is stopped.
Job is stopped when the job is dead or, if the job was purged, does not exists anymore,
and the job has no running or pending allocations,
no active deployments and no active evaluations.

\b
If the option --no-preserve-status is given, then exit with the following status:
  {ExitCode.success}    when the job was stopped.
Otherwise, exit with the following status:
  {"?"}    when the job has one task, with that task exit status,
  {ExitCode.success}    when all tasks of the job exited with 0 exit status,
  {ExitCode.any_failed_tasks}  when any of the job tasks have failed,
  {ExitCode.all_failed_tasks}  when all job tasks have failed,
  {ExitCode.any_unfinished_tasks}  when any tasks are still running,
  {ExitCode.no_allocations}  when job has no started tasks.
In any case, exit with the following exit status:
  {ExitCode.exception}    when python exception was thrown,
  {ExitCode.interrupted}    when the process was interrupted.
""",
)
@cli_jobid
def mode_stopped(jobid: str):
    jobid = nomad_find_job(jobid)
    NomadJobWatcherUntilFinished(jobid).run_and_exit()


@cli_jobfile_command(
    "plan",
    "plan",
    help="""
    This is an alias to nomad job plan for ease of typing.
    """,
)
def mode_plan(args: Tuple[str, ...], jobfile: str):
    cmd: List[str] = [*"nomad job plan".split(), *args, jobfile]
    log.info(f"+ {' '.join(shlex.quote(x) for x in cmd)}")
    try:
        subprocess.check_call(cmd, text=True)
    except subprocess.CalledProcessError as e:
        # nomad will print its error, we can just exit
        exit(e.returncode)


###############################################################################

if __name__ == "__main__":
    cli.main()
