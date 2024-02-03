#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import base64
import datetime
import enum
import inspect
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
from http import client as http_client
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    TypeVar,
)

import click
import requests
from click.shell_completion import CompletionItem
from typing_extensions import override

from . import colors, exit_on_thread_exception, flagdebug, nomaddbjob, nomadlib
from .common_base import andjoin, cached_property, composed, eprint
from .common_click import alias_option, common_options, complete_set_namespace
from .common_nomad import (
    NoJobFound,
    complete_job,
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

ARGS = argparse.Namespace()
"""Arguments passed to this program"""

START_NS: int = 0
"""This program start time, assigned from cli()"""

COLORS = colors.init()

DB: NomadDbJob
"""Database listening to Nomad stream"""

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


###############################################################################


class InterruptTwice:
    """Exit on the second time after sending a signal"""

    received: int = 0
    """The number of last signal received"""

    @classmethod
    def install(cls):
        # Installed after DB initialization.
        signal.signal(signal.SIGINT, cls.__handler)
        signal.signal(signal.SIGTERM, cls.__handler)

    @classmethod
    def __handler(cls, signum: int, frame):
        log.error(f"Interrupted with {signal.Signals(signum).name}")
        if cls.received:
            exit(ExitCode.interrupted)
        else:
            cls.received = signum
            # Initialized in NomadWatch object.
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


class LOGFORMAT:
    """Logging output format specification templates using f-string. Very poor mans templating langauge."""

    pre = "{color}{now.strftime(args.log_time_format) + '>' if ARGS.log_time else ''}"
    mark = "{mark}>"
    post = "{message}{reset}"
    task = "{task + '>' if task else ''}"
    DEFAULT = (
        pre + mark + "{id:.{args.log_id_len}}>v{str(jobversion)}>" + task + " " + post
    )
    """
    Default log format. The log is templated with f-string using eval() below.
        O>45fbbd>v0>group1>task1> hello world
    """
    ONE = pre + mark + task + " " + post
    """
    Log format with -1 option.
        O>task1> hello world
    """
    ZERO = pre + post
    """
    Log format with -0 option.
        hello world
    """
    LOGGING = (
        COLORS.brightblue
        + "{'%(asctime)s>' if ARGS.log_time else ''}"
        + "{'nomad-watch' if ARGS.verbose <= 0 else '%(module)s'}>"
        + "%(lineno)03d>"
        + " %(levelname)s %(message)s"
        + COLORS.reset
    )
    """Logging format, first templated with f-string, then by logging."""
    colors: Dict[LogWhat, str] = {
        LogWhat.deploy: COLORS.brightmagenta,
        LogWhat.eval: COLORS.magenta,
        LogWhat.alloc: COLORS.cyan,
        LogWhat.stderr: COLORS.orange,
        LogWhat.stdout: "",
    }
    """Logs colors"""
    marks: Dict[LogWhat, str] = {
        LogWhat.deploy: "deploy",
        LogWhat.eval: "eval",
        LogWhat.alloc: "A",
        LogWhat.stderr: "E",
        LogWhat.stdout: "O",
    }
    """Logs marks"""

    @staticmethod
    def render(fmt: str, **kwargs) -> str:
        locals().update(kwargs)
        return eval(f"f{fmt!r}")


def click_log_options():
    """All logging options"""
    return composed(
        click.option(
            "-T",
            "--log-time",
            is_flag=True,
            help="""
                Additionally add timestamp to the logs.
                The timestamp of stdout and stderr streams is when the log was received,
                as Nomad does not store timestamp of task logs.
                """,
        ),
        click.option(
            "--log-time-format",
            default="%Y-%m-%dT%H:%M:%S%z",
            show_default=True,
            help="Format time with specific format. Passed to python datetime.strftime.",
        ),
        alias_option(
            "-H",
            "--log-time-hour",
            aliased=dict(
                log_time_format="%H:%M:%S",
                log_time=True,
            ),
        ),
        click.option(
            "--log-format",
            default=LOGFORMAT.DEFAULT,
            show_default=True,
            help="The format to use when printing job logs",
        ),
        click.option(
            "--log-id-len",
            default=6,
            help="The length of id to log. UUIDv4 has 36 characters.",
        ),
        alias_option(
            "-l",
            "--log-id-long",
            aliased=dict(log_id_len=36),
        ),
        alias_option(
            "-1",
            "--log-only-task",
            aliased=dict(log_format=LOGFORMAT.ONE),
        ),
        alias_option(
            "-0",
            "--log-none",
            aliased=dict(log_format=LOGFORMAT.ZERO),
        ),
        click.option(
            "-o",
            "--out",
            type=CommaList("all alloc stdout stderr eval deploy none".split()),
            default=["all"],
            show_default=True,
            help="""
                Comma separated list of streams of messages to print:
                deployment, evaluation, allocation, stdout and stderr logs of tasks.
                """,
        ),
    )


def init_logging():
    LOGENABLED.deploy = any(s in "all deploy".split() for s in ARGS.out)
    LOGENABLED.eval = any(s in "all eval".split() for s in ARGS.out)
    LOGENABLED.alloc = any(s in "all alloc".split() for s in ARGS.out)
    LOGENABLED.stderr = any(s in "all stderr".split() for s in ARGS.out)
    LOGENABLED.stdout = any(s in "all stdout".split() for s in ARGS.out)
    #
    ARGS.verbose -= ARGS.quiet
    if ARGS.verbose > 1:
        http_client.HTTPConnection.debuglevel = 1
    logging.basicConfig(
        format=LOGFORMAT.render(LOGFORMAT.LOGGING),
        datefmt=ARGS.log_time_format,
        level=(
            logging.DEBUG
            if ARGS.verbose > 0
            else (
                logging.INFO
                if ARGS.verbose == 0
                else logging.WARN if ARGS.verbose == 1 else logging.ERROR
            )
        ),
    )


@dataclass(frozen=True)
class Mylogger:
    """Used to log from the various streams we want to log from"""

    id: str
    message: str
    what: str
    now: datetime.datetime = field(default_factory=datetime.datetime.now)
    nodename: Optional[str] = None
    jobversion: Optional[str] = None
    group: Optional[str] = None
    task: Optional[str] = None

    def fmt(self):
        assert self.what in [
            e.value for e in LogWhat
        ], f"{self.what} value is not a valid what"
        mark: str = LOGFORMAT.marks[self.what]
        color: str = LOGFORMAT.colors[self.what]
        reset: str = COLORS.reset if color else ""
        return LOGFORMAT.render(
            ARGS.log_format,
            **asdict(self),
            mark=mark,
            color=color,
            reset=reset,
            args=ARGS,
        )

    @staticmethod
    def __log(what: str, **kwargs):
        mylogger = Mylogger(what=what, **kwargs)
        try:
            out = mylogger.fmt()
        except KeyError:
            log.exception(f"fmt={ARGS.log_format!r} params={ARGS}")
            raise
        try:
            print(out, flush=True)
        except BrokenPipeError:
            # Python flushes standard streams on exit; redirect remaining output
            # to devnull to avoid another BrokenPipeError at shutdown
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            sys.exit(1)  # Python exits with error code 1 on EPIPE

    @classmethod
    def log_eval(cls, eval: nomadlib.Eval, message: str):
        return cls.__log(
            LogWhat.eval,
            id=eval.ID,
            jobversion=DB.find_jobversion_from_modifyindex(eval.ModifyIndex),
            now=ns2dt(eval.ModifyTime),
            message=message,
        )

    @classmethod
    def log_deploy(cls, deploy: nomadlib.Deploy, message: str):
        job = DB.jobversions.get(deploy.JobVersion)
        return cls.__log(
            LogWhat.deploy,
            id=deploy.ID,
            jobversion=deploy.JobVersion,
            now=ns2dt(job.SubmitTime) if job else datetime.datetime.now(),
            message=message,
        )

    @classmethod
    def log_alloc(cls, allocid: str, **kwargs):
        return cls.__log(LogWhat.alloc, id=allocid, **kwargs)

    @classmethod
    def log_std(cls, stderr: bool, allocid: str, **kwargs):
        return cls.__log(
            LogWhat.stderr if stderr else LogWhat.stdout, id=allocid, **kwargs
        )


@dataclass(frozen=True)
class TaskKey:
    """Represent data to unique identify a task"""

    allocid: str
    jobversion: str
    nodename: str
    group: str
    task: str

    def __str__(self):
        return f"{self.allocid:.6}:v{self.jobversion}:{self.group}:{self.task}"

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
        #
        self.startevent = threading.Event()
        """This is an event which is sent after --shutdown-timeout time after log listening starts"""
        self.starttimer = threading.Timer(ARGS.shutdown_timeout, self.__starttimer_do)
        """A timer that will set startevent"""
        self.exitevent = threading.Event()
        """User should set this event to stop logging the task logs"""
        self.ignoredlines: List[str] = []
        """In the case of --lines argument, this is a buffer of spare lines to output"""
        self.first_line: bool = True
        """Ignore input lines if printing only trailing lines."""
        self.ignoretime_ns: int = START_NS + int(ARGS.lines_timeout * 1e9)
        """If ignore time is in the past, it is no longer relevant anyway."""
        #
        if ARGS.lines < 0 or self.ignoretime_ns < time.time_ns():
            self.ignoretime_ns = 0

    def __starttimer_do(self):
        self.startevent.set()
        DB.send_empty_event()

    @staticmethod
    def read_json_stream(stream: requests.Response):
        txt: str = ""
        for dataorbytes in stream.iter_content(decode_unicode=True):
            try:
                data: str = dataorbytes.decode()
            except (UnicodeDecodeError, AttributeError):
                data: str = dataorbytes
            for c in data:
                txt += c
                # Nomad happens to be consistent, the jsons are flat.
                if c == "}":
                    try:
                        ret = json.loads(txt)
                        # log.debug(f"RECV: {ret}")
                        yield ret
                    except json.JSONDecodeError as e:
                        log.warn(f"error decoding json: {txt} {e}")
                    txt = ""

    def taskout(self, lines: List[str]):
        """Output the lines"""
        # If ignoring and this is first received line or the ignoring time is still happenning.
        if self.ignoretime_ns and (
            self.first_line or time.time_ns() < self.ignoretime_ns
        ):
            # Accumulate args.lines into ignoredlines array.
            self.first_line = False
            self.ignoredlines = lines
            self.ignoredlines = self.ignoredlines[: ARGS.lines]
        else:
            if self.ignoretime_ns:
                # If not ignoring lines, flush the accumulated lines.
                lines = self.ignoredlines + lines
                self.ignoredlines.clear()
                # Disable further accumulation of ignored lines.
                self.ignoretime_ns = 0
            # Print the log lines.
            for line in lines:
                line = line.rstrip()
                self.tk.log_task(self.stderr, line)

    def __typestr(self):
        return "stderr" if self.stderr else "stdout"

    def __run_in(self):
        with mynomad.stream(
            f"client/fs/logs/{self.tk.allocid}",
            params={
                "task": self.tk.task,
                "type": self.__typestr(),
                "follow": True,
                "origin": "end" if self.ignoretime_ns else "start",
                "offset": 50000 if self.ignoretime_ns else 0,
            },
        ) as stream:
            self.starttimer.start()
            for event in self.read_json_stream(stream):
                if event:
                    line64: Optional[str] = event.get("Data")
                    if line64:
                        lines = base64.b64decode(line64.encode()).decode().splitlines()
                        self.taskout(lines)
                else:
                    # Nomad json stream periodically sends empty {}.
                    self.taskout([])
                if self.exitevent.is_set():
                    break

    def run(self):
        """Listen to Nomad log stream and print the logs"""
        try:
            self.__run_in()
        except nomadlib.LogNotFound as e:
            # Gracefully handle missing logs errors from Nomad.
            # Logs are removed by garbage collector and when purging the job.
            code = e.response.status_code if e.response is not None else None
            text = e.response.text if e.response is not None else None
            self.tk.log_alloc(
                datetime.datetime.now(),
                f"Error getting {self.__typestr()} logs: {code} {text!r}",
            )
        except requests.HTTPError:
            raise
        finally:
            self.startevent.set()

    def stop(self):
        self.exitevent.set()
        self.startevent.set()
        self.starttimer.cancel()


@dataclass
class TaskHandler:
    """A handler for one task. Creates loggers, writes out task events, handle exit conditions"""

    loggers: Optional[List[TaskLogger]] = None
    """Array of loggers that log allocation logs."""
    messages: Set[int] = field(default_factory=set)
    """A set of message timestamp to know what has been printed."""
    exitcode: Optional[int] = None
    stoptimer: Optional[threading.Timer] = None

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
                        or msgtime_ns >= START_NS
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
        if self.stoptimer is None and self.loggers and taskstate.State == "dead":
            # If the task is already finished, give myself max 3 seconds to query all the logs.
            # This is to reduce the number of connections.
            self.stoptimer = threading.Timer(3, self.stop)
            self.stoptimer.start()
        if self.exitcode is None and taskstate.State == "dead":
            terminatedevent = taskstate.find_event("Terminated")
            if terminatedevent:
                self.exitcode = terminatedevent.ExitCode

    def stop(self):
        if self.stoptimer:
            self.stoptimer.cancel()
        for ll in self.loggers or []:
            ll.stop()


@dataclass
class AllocWorker:
    """Represents a worker that prints out and manages state related to one allocation"""

    taskhandlers: Dict[TaskKey, TaskHandler] = field(default_factory=dict)

    def notify(self, alloc: nomadlib.Alloc):
        """Update the state with alloc"""
        for taskname, task in alloc.get_taskstates().items():
            if ARGS.task and not ARGS.task.search(taskname):
                continue
            tk = TaskKey(
                alloc.ID,
                str(DB.get_allocation_jobversion(alloc, "?")),
                alloc.NodeName,
                alloc.TaskGroup,
                taskname,
            )
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
            and evaluation.Status == nomadlib.EvalStatus.blocked
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

    def join(self):
        threads = self.get_threads()
        # Logs stream outputs empty {} which allows to handle timeouts.
        thcnt = sum(len(w.taskhandlers) for w in self.workers.values())
        log.debug(
            f"Joining {len(self.workers)} allocations with {thcnt} taskhandlers and {len(threads)} loggers"
        )
        timeend = time.time() + ARGS.shutdown_timeout
        timeout = None
        for thread in threads:
            timeout = timeend - time.time()
            if timeout > 0:
                log.debug(f"joining worker {thread.name} timeout={timeout}")
                thread.startevent.wait(timeout)
            else:
                break
        log.debug(f"{len(threads)} threads joined with {timeout} timeout to spare")

    def __exitcode(self) -> ExitcodeRet:
        tasks_exitcodes: List[Optional[int]] = [
            th.exitcode for w in self.workers.values() for th in w.taskhandlers.values()
        ]
        if flagdebug.debug("exitcode"):
            for allocid, w in self.workers.items():
                for taskkey, taskhandler in w.taskhandlers.items():
                    eprint(
                        f"EXITCODE: {allocid[:6]} {taskkey.group}/{taskkey.task} {taskhandler.exitcode}"
                    )
        if len(tasks_exitcodes) == 0:
            return ExitcodeRet(ExitCode.no_allocations, "there were no allocations")
        unfinished_tasks = [v for v in tasks_exitcodes if v is None]
        if unfinished_tasks:
            return ExitcodeRet(
                ExitCode.any_unfinished_tasks,
                f"there were {len(unfinished_tasks)} unfinished tasks",
            )
        only_one_task = len(tasks_exitcodes) == 1
        if only_one_task:
            assert tasks_exitcodes[0] is not None
            return ExitcodeRet(
                tasks_exitcodes[0],
                f"single task exited with {tasks_exitcodes[0]} exit status",
            )
        failed_tasks = [v for v in tasks_exitcodes if v != 0]
        if len(failed_tasks) == len(tasks_exitcodes):
            return ExitcodeRet(
                ExitCode.all_failed_tasks,
                f"all {len(tasks_exitcodes)} tasks exited with nonzero exit status",
            )
        if failed_tasks:
            return ExitcodeRet(
                ExitCode.any_failed_tasks,
                f"{len(failed_tasks)} tasks exited with nonzero exit status",
            )
        return ExitcodeRet(
            ExitCode.success,
            f"all {len(tasks_exitcodes)} tasks exited with zero exit status",
        )

    def exitcode(self) -> int:
        er: ExitcodeRet = self.__exitcode()
        log.info(f"{er.reason.capitalize()}. Exit code is {er.code}.")
        return er.code


###############################################################################


def __nomad_job_run(opts: Tuple[str]) -> str:
    """Call nomad job run to start a Nomad job using nomad job run call with specified arguments"""
    cmd: List[str] = "nomad job run -detach -verbose".split() + list(opts)
    log.info(f"+ {' '.join(shlex.quote(x) for x in cmd)}")
    try:
        output = subprocess.check_output(cmd, text=True)
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
        " or nomad command line has upgraded its output or there is an error in this script."
    )
    evalid = founduuids[0]
    return evalid


def nomad_start_job(opts: Tuple[str]) -> nomadlib.Eval:
    """Start a nomad job using nomad job run parameters. Return evluation from running the job"""
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
        self.no_follow_timeend: float = time.time() + ARGS.no_follow_timeout
        """If in --no-follow mode, exit after this time"""
        #
        DB.start()
        InterruptTwice.install()

    def __db_init_cb(self) -> List[Event]:
        """Db initialization callback"""
        try:
            job: dict = mynomad.get(f"job/{self.jobid}")
            jobversions: dict = mynomad.get(f"job/{self.jobid}/versions")["Versions"]
            deployments: List[dict] = mynomad.get(f"job/{self.jobid}/deployments")
            evaluations: List[dict] = mynomad.get(f"job/{self.jobid}/evaluations")
            allocations: List[dict] = mynomad.get(f"job/{self.jobid}/allocations")
        except nomadlib.JobNotFound:
            # This is fine to fail if it was purged, potentially.
            if DB.seen_job() and self.job:
                # If the job was purged, generate one event so that listener can catch it.
                return [
                    Event(
                        -1, EventTopic.Job, EventType.JobDeregistered, self.job.asdict()
                    )
                ]
            raise
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
                e=f"{events[0].topic.name}.{events[0].type.name}" if events else None,
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
                    f"{th.name}={int(th.startevent.is_set())}"
                    for th in self.notifier.get_threads()
                ],
            )
            eprint(f"LOOP: {strdict(info)}")

    def loop(self) -> Iterator[None]:
        """Thread entrypoint that handles events from Nomad event stream database"""
        untilstr = (
            "forever"
            if ARGS.all
            else (
                f"for {ARGS.no_follow_timeout} seconds"
                if ARGS.no_follow
                else f"until it is {self.endstatusstr}"
            )
        )
        log.info(f"Watching job {self.jobid}@{mynomad.namespace} {untilstr}")
        for events in DB.events():
            for event in events:
                self.__handle_event(event)
            self.__loop_debug(events)
            if (
                # --follow means never exit.
                not ARGS.follow
                # If we started with an evaluation, it has to be finished.
                # Otherwise the events may be related to a previous job version.
                and (self.eval is None or not self.eval.is_pending_or_blocked())
                # Make sure all the threads have finished outputting logs.
                and all(th.startevent.is_set() for th in self.notifier.get_threads())
            ):
                yield
                if self.job and DB.job_purged():
                    assert self.job
                    if self.has_no_active_allocations_nor_evaluations_nor_deployments():
                        log.info(
                            f"Job {self.job.description()} purged with no active allocations, evaluations nor deployments. Exiting."
                        )
                    else:
                        log.info(f"Job {self.job.description()} purged. Exiting.")
                    break
            if ARGS.no_follow and time.time() > self.no_follow_timeend:
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


class NomadJobWatcher(_NomadJobWatcherDetail):
    @override
    def __init__(self, jobid: Optional[str], eval: Optional[nomadlib.Eval] = None):
        super().__init__(jobid, eval)
        self.sent_purged_req = False
        """Return true if we sent a request to purge the job"""
        self.sent_stopped_req = False
        """Return true if we sent a request to stop the job"""

    @abstractmethod
    def _get_exitcode_cb(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def _loop_end_cb(self):
        raise NotImplementedError()

    def stop_job(self, purge: bool = False):
        assert DB.initialized.is_set()
        if purge:
            if self.sent_purged_req:
                return
            log.info(f"Purging job {self.jobid}")
            self.sent_stopped_req = True
            self.sent_purged_req = True
        else:
            if self.sent_stopped_req:
                return
            log.info(f"Stopping job {self.jobid}")
            self.sent_stopped_req = True
        mynomad.stop_job(self.jobid, purge)

    def stop_threads(self):
        log.debug(f"stopping {self.__class__.__name__}")
        self.notifier.stop()
        DB.stop()
        self.notifier.join()
        DB.join()
        mynomad.session.close()

    def get_exitcode(self) -> int:
        assert DB.stopevent.is_set(), f"{self.stop_threads.__name__} not called"
        if InterruptTwice.received:
            return ExitCode.interrupted
        return self._get_exitcode_cb()

    def run_and_exit(self, stopit: bool = False):
        try:
            for _ in self.loop():
                if stopit:
                    self.stop_job()
                if InterruptTwice.received:
                    if ARGS.attach or ARGS.purge or ARGS.purge_successful:
                        purge: bool = ARGS.purge or (
                            ARGS.purge_successful and self.job_finished_successfully()
                        )
                        self.stop_job(purge)
                    else:
                        break
                if self._loop_end_cb():
                    break
        finally:
            self.stop_threads()
        exit(self.get_exitcode())

    def job_is_finished(self):
        return (
            DB.seen_job()
            and self.job
            and self.job.is_dead()
            and self.has_no_active_allocations_nor_evaluations_nor_deployments()
        )

    def job_get_summary(self):
        s: nomadlib.JobSummarySummary = nomadlib.JobSummary(
            mynomad.get(f"job/{self.jobid}/summary")
        ).get_sum_summary()
        log.debug(f"{s}")
        return s

    def job_finished_successfully(self):
        DB.initialized.wait()
        assert self.job
        if self.job.Status != "dead":
            return False
        s = self.job_get_summary()
        return (
            s.Queued == 0
            and s.Complete != 0
            and s.Failed == 0
            and s.Running == 0
            and s.Starting == 0
            and s.Lost == 0
        )

    def job_running_successfully(self):
        DB.initialized.wait()
        assert self.job
        if self.job.is_dead():
            return self.job_finished_successfully()
        s = self.job_get_summary()
        return (
            s.Queued == 0
            and s.Failed == 0
            and s.Running != 0
            and s.Starting == 0
            and s.Lost == 0
        )

    def job_dead_message(self):
        assert self.job
        return f"Job {self.job.description()} dead with no active allocations, evaluations nor deployments. Exiting."


class NomadJobWatcherUntilFinished(NomadJobWatcher):
    """Watch a job until the job is dead or purged"""

    endstatusstr: ClassVar[str] = "finished"

    @override
    def _loop_end_cb(self) -> bool:
        if self.job_is_finished():
            if not self.sent_purged_req and (
                ARGS.purge
                or (ARGS.purge_successful and self.job_running_successfully())
            ):
                self.stop_job(True)
            if self.sent_purged_req:
                # Wait for beeing purged.
                return False
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
        if self.job_has_finished_starting():
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

    @staticmethod
    def nomad_job_group_main_tasks(group: nomadlib.JobTaskGroup):
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

    def job_has_finished_starting(self) -> bool:
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
            # The job is still doing something. Wait for it.
            return False
        allocations = DB.allocations.values()
        if (
            not allocations
            or any(alloc.is_pending() for alloc in allocations)
            or any(alloc.TaskStates is None for alloc in allocations)
        ):
            # There are still allocations which Tasks have not started yet. Wait for them.
            return False
        #
        groupmsgs: List[str] = []
        for group in self.job.TaskGroups:
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
                log.debug(
                    f"groupallocs={[x.ID for x in groupallocs]}"
                    f" {[DB.get_allocation_jobmodifyindex(alloc, -1) for alloc in groupallocs]}"
                    f" {[DB.get_allocation_jobversion(alloc, -1) for alloc in groupallocs]}"
                )
                log.error(
                    f"Job {self.job.description()} group {group.Name!r} started"
                    f" {len(groupallocs)} allocation out of {group.Count}."
                )
                return True
            maintasks: Set[str] = self.nomad_job_group_main_tasks(group)
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
        log.info(f"Job {self.job.description()} started " + andjoin(groupmsgs) + ".")
        self.started = True
        return True


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


@click.group(
    help="""
Depending on the command, run or stop a Nomad job. Watch over the job and
print all job allocation events and tasks stdouts and tasks stderrs
logs. Depending on command, wait for a specific event to happen to finish
watching. This program is intended to help debugging issues with running
jobs in Nomad and for synchronizing with execution of batch jobs in Nomad.

Logs are printed in the format: 'mark>id>vversion>group>task> message'.
The mark in the log lines is equal to: 'deploy' for messages printed as
a result of deployment, 'eval' for messages printed from evaluations,
'A' from allocation, 'E' for stderr logs of a task and 'O' from stdout
logs of a task.

\b
Examples:
    nomad-watch run ./some-job.nomad.hcl
    nomad-watch job some-job
    nomad-watch alloc af94b2
    nomad-watch -N services --task redis -1f job redis
""",
    epilog="""
    Written by Kamil Cukrowski 2023. Licensed under GNU GPL version 3 or later.
    """,
)
@namespace_option()
@click.option(
    "-a",
    "--all",
    is_flag=True,
    help="Print logs from all allocations, including previous versions of the job.",
)
@click.option("-v", "--verbose", count=True, help="Be more verbose.")
@click.option("-q", "--quiet", count=True, help="Be less verbose.")
@flagdebug.click_debug_option("NOMAD_WATCH_DEBUG")
@click.option(
    "-A",
    "--attach",
    is_flag=True,
    help="""
        Stop the job on receiving an SIGINT or SIGTERM signal.
        Exit immidately after receiving a signal the second time.
        """,
)
@click.option(
    "--purge-successful",
    "purge_successful",
    is_flag=True,
    help="""
        When stopping the job, purge it when all job summary metrics are zero except nonzero complete metric.
        Relevant in job, run, stop and stopped modes. Implies --attach.
        """,
)
@click.option(
    "--purge",
    is_flag=True,
    help="""
        Purge the job after it is finished.
        Relevant in job, run, stop and stopped modes. Implies --attach.
        """,
)
@click.option(
    "-n",
    "--lines",
    default=10,
    show_default=True,
    type=int,
    help="""
        Sets the tail location in best-efforted number of lines relative to the end of logs.
        Negative value prints all available log lines.
        """,
)
@click.option(
    "--lines-timeout",
    "lines_timeout",
    default=0.5,
    show_default=True,
    type=float,
    help="When using --lines the number of lines is best-efforted by ignoring lines for this specific time",
)
@click.option(
    "--shutdown-timeout",
    "shutdown_timeout",
    default=1,
    show_default=True,
    type=float,
    help="The time to wait to make sure task loggers received all logs when exiting.",
)
@click.option(
    "-f",
    "--follow",
    is_flag=True,
    help="Never exit",
)
@click.option(
    "--no-follow",
    "no_follow",
    is_flag=True,
    help="Just run once, get the logs in a best-effort style and exit.",
)
@click.option(
    "--no-follow-timeout",
    "no_follow_timeout",
    default=3,
    show_default=True,
    type=float,
    help="The time to run in --no-follow mode.",
)
@click.option(
    "-t",
    "--task",
    type=re.compile,
    help="Only watch tasks names matching this regex.",
)
@click.option(
    "-g",
    "--group",
    type=re.compile,
    help="Only watch group names matching this regex.",
)
@click.option(
    "--polling",
    is_flag=True,
    help="Instead of listening to Nomad event stream, periodically poll for events.",
)
@click.option(
    "-x",
    "--no-preserve-status",
    "no_preserve_status",
    is_flag=True,
    help="Do not preserve tasks exit statuses.",
)
@click_log_options()
@common_options()
def cli(**kwargs):
    signal.signal(signal.SIGUSR1, print_all_threads_stacktrace)
    exit_on_thread_exception.install()
    global ARGS
    ARGS = argparse.Namespace(**kwargs)
    assert (
        (not ARGS.follow and not ARGS.no_follow)
        or (ARGS.follow and not ARGS.no_follow)
        or (not ARGS.follow and ARGS.no_follow)
    ), "--follow and --no-follow conflict"
    #
    global START_NS
    START_NS = time.time_ns()
    init_logging()


cli_jobid = click.argument(
    "jobid",
    shell_complete=complete_job(),
)


def cli_jobfile(name: str, help: str):
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


def cli_command_run_nomad_job_run(name: str, help: str):
    return composed(
        cli.command(
            name,
            help=help.rstrip()
            + """\
            All following command arguments are passed to nomad job run command.
            Note that nomad job run has arguments with a single dash.
            """,
            context_settings=dict(ignore_unknown_options=True),
        ),
        click.argument(
            "cmd",
            nargs=-1,
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


@cli_command_run_nomad_job_run(
    "run",
    help="Run a Nomad job and then act like stopped mode.",
)
def mode_run(cmd: Tuple[str]):
    evaluation = nomad_start_job(cmd)
    NomadJobWatcherUntilFinished(None, evaluation).run_and_exit()


@cli.command(
    "job",
    help="Alias to stopped command.",
)
@cli_jobid
def mode_job(jobid: str):
    jobid = nomad_find_job(jobid)
    NomadJobWatcherUntilFinished(jobid).run_and_exit()


@cli_command_run_nomad_job_run(
    "start", help="Start a Nomad Job and then act like started command."
)
def mode_start(cmd: Tuple[str]):
    evaluation = nomad_start_job(cmd)
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
  {ExitCode.success    }  when all tasks of the job have started running,
  {ExitCode.exception  }  when python exception was thrown,
  {ExitCode.interrupted}  when process was interrupted,
  {ExitCode.failed     }  when job was stopped or job deployment was reverted.
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
    jobid = nomad_find_job(jobid)
    NomadJobWatcherUntilFinished(jobid).run_and_exit(True)


@cli.command(
    "purge",
    help=f"""
Alias to `--purge stop`, with the following difference in exit status.
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
  {ExitCode.success           }    when the job was stopped.
Otherwise, exit with the following status:
  {'?'                        }    when the job has one task, with that task exit status,
  {ExitCode.success           }    when all tasks of the job exited with 0 exit status,
  {ExitCode.any_failed_tasks    }  when any of the job tasks have failed,
  {ExitCode.all_failed_tasks    }  when all job tasks have failed,
  {ExitCode.any_unfinished_tasks}  when any tasks are still running,
  {ExitCode.no_allocations      }  when job has no started tasks.
In any case, exit with the following exit status:
  {ExitCode.exception         }    when python exception was thrown,
  {ExitCode.interrupted       }    when the process was interrupted.
""",
)
@cli_jobid
def mode_stopped(jobid: str):
    jobid = nomad_find_job(jobid)
    NomadJobWatcherUntilFinished(jobid).run_and_exit()


###############################################################################

if __name__ == "__main__":
    try:
        cli.main()
    finally:
        mynomad.session.close()
