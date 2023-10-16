#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime
import enum
import inspect
import itertools
import json
import logging
import re
import shlex
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from http import client as http_client
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    TypeVar,
)

import click
import requests

from . import exit_on_thread_exception, nomadlib
from .common import (
    ALIASED,
    _complete_set_namespace,
    alias_option,
    common_options,
    complete_job,
    completor,
    composed,
    mynomad,
    namespace_option,
)
from .nomaddbjob import NomadDbJob
from .nomadlib import Event, EventTopic, EventType, MyStrEnum, ns2dt

log = logging.getLogger(__name__)

###############################################################################

args = argparse.Namespace()
"""Arguments passed to this program"""

START_NS: int = 0
"""This program start time, assigned from cli()"""


@dataclasses.dataclass
class Colors:
    bold: str = "bold"
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
    reset: str = "sgr0"

    @staticmethod
    def init() -> Colors:
        tputdict = dataclasses.asdict(Colors())
        empty = Colors(**{k: "" for k in tputdict.keys()})
        if not sys.stdout.isatty() or not sys.stderr.isatty():
            return empty
        tputscript = "\n".join(tputdict.values()).replace(
            "\n", "\nlongname\nlongname\n"
        )
        try:
            longname = subprocess.run(
                f"tput longname".split(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout
            ret = subprocess.run(
                "tput -S".split(),
                input=tputscript,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return empty
        retarr = ret.split(f"{longname}{longname}")
        if len(tputdict.keys()) != len(retarr):
            return empty
        return Colors(**{k: v for k, v in zip(tputdict.keys(), retarr)})


COLORS = Colors.init()


T = TypeVar("T")


def set_not_in_add(s: Set[T], value: T) -> bool:
    """If value is in the set, return False. Otherwise add the value to the set and return True"""
    if value in s:
        return False
    s.add(value)
    return True


def andjoin(arr: Iterable[str]) -> str:
    arr = list(arr)
    if not len(arr):
        return ""
    if len(arr) == 1:
        return arr[0]
    return ", ".join(arr[:-1]) + " and " + arr[-1]


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

    pre = "{color}{now.strftime(args.log_time_format) + '>' if args.log_time else ''}{mark}>"
    post = " {message}{reset}"
    task = "{task + '>' if task else ''}"
    DEFAULT = (
        pre
        + "{id:.{args.log_id_len}}{'>' if args.log_id_len else ''}"
        + "{'#' + str(jobversion) + '>' if jobversion is not None else ''}"
        + "{group + '>' if group else ''}"
        + task
        + post
    )
    """Default log format. The log is templated with f-string using eval() below."""
    ONE = pre + task + post
    """Log format with -1 option"""
    ZERO = pre + post
    """Log format with -0 option"""
    LOGGING = (
        COLORS.blue
        + "{'%(asctime)s>' if args.log_time else ''}"
        + "nomad-watch>%(lineno)03d> %(levelname)s %(message)s"
        + COLORS.reset
    )
    """Logging format, first templated with f-string, then by logging"""
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
            help="Additionally add timestamp to the logs. The timestamp of stdout and stderr streams is when the log was received, as Nomad does not store timestamp of task logs.",
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
    )


def init_logging():
    LOGENABLED.deploy = any(s in "all deployment deploy d".split() for s in args.out)
    LOGENABLED.eval = any(s in "all evaluation eval e".split() for s in args.out)
    LOGENABLED.alloc = any(s in "all alloc a".split() for s in args.out)
    LOGENABLED.stderr = any(s in "all stderr err E 2".split() for s in args.out)
    LOGENABLED.stdout = any(s in "all stdout out O 1".split() for s in args.out)
    #
    args.verbose -= args.quiet
    if args.verbose > 1:
        http_client.HTTPConnection.debuglevel = 1
    logging.basicConfig(
        format=LOGFORMAT.render(LOGFORMAT.LOGGING),
        datefmt=args.log_time_format,
        level=(
            logging.DEBUG
            if args.verbose > 0
            else logging.INFO
            if args.verbose == 0
            else logging.WARN
            if args.verbose == 1
            else logging.ERROR
        ),
    )


@dataclasses.dataclass(frozen=True)
class Mylogger:
    """Used to log from the various streams we want to log from"""

    id: str
    message: str
    what: str
    now: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)
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
            args.log_format,
            **dataclasses.asdict(self),
            mark=mark,
            color=color,
            reset=reset,
            args=args,
        )

    @staticmethod
    def __log(what: str, **kwargs):
        mylogger = Mylogger(what=what, **kwargs)
        try:
            out = mylogger.fmt()
        except KeyError:
            log.exception(f"fmt={args.log_format!r} params={args}")
            raise
        print(out, flush=True)

    @classmethod
    def log_eval(cls, db: NomadDbJob, eval: nomadlib.Eval, message: str):
        return cls.__log(
            LogWhat.eval,
            id=eval.ID,
            jobversion=db.find_jobversion_from_modifyindex(eval.ModifyIndex),
            now=ns2dt(eval.ModifyTime),
            message=message,
        )

    @classmethod
    def log_deploy(cls, db: NomadDbJob, deploy: nomadlib.Deploy, message: str):
        job = db.jobversions.get(deploy.JobVersion)
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


@dataclasses.dataclass(frozen=True)
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
        return dataclasses.asdict(self)

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
        super().__init__(name=f"{tk}{1 + int(stderr)}")
        self.tk = tk
        self.stderr: bool = stderr
        self.exitevent = threading.Event()
        self.ignoredlines: List[str] = []
        self.first_line = True
        # Ignore input lines if printing only trailing lines.
        self.ignoretime_ns = START_NS + int(args.lines_timeout * 1e9)
        # If ignore time is in the past, it is no longer relevant anyway.
        if args.lines < 0 or self.ignoretime_ns < time.time_ns():
            self.ignoretime_ns = 0

    @staticmethod
    def read_json_stream(stream: requests.Response):
        txt: str = ""
        for data in stream.iter_content(decode_unicode=True):
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
            self.ignoredlines = self.ignoredlines[: args.lines]
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
            for event in self.read_json_stream(stream):
                if event:
                    line64: Optional[str] = event.get("Data")
                    if line64:
                        lines = base64.b64decode(line64.encode()).decode().splitlines()
                        self.taskout(lines)
                else:
                    # Nomad json stream periodically sends empty {}.
                    # No idea why, but I can implement timeout.
                    self.taskout([])
                    if self.exitevent.is_set():
                        break

    def run(self):
        """Listen to Nomad log stream and print the logs"""
        try:
            self.__run_in()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.tk.log_alloc(
                    datetime.datetime.now(),
                    f"{self.__typestr()} logs were garbage collected from Nomad",
                )
            else:
                raise

    def stop(self):
        self.exitevent.set()


@dataclasses.dataclass
class TaskHandler:
    """A handler for one task. Creates loggers, writes out task events, handle exit conditions"""

    loggers: Optional[List[TaskLogger]] = None
    """Array of loggers that log allocation logs."""
    messages: Set[int] = dataclasses.field(default_factory=set)
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
        events = taskstate.Events
        if LOGENABLED.alloc:
            for e in events:
                msgtime_ns = e.Time
                # Ignore message before ignore times.
                if (
                    msgtime_ns
                    and (
                        args.lines < 0
                        or msgtime_ns >= START_NS
                        or len(self.messages) < args.lines
                    )
                    and set_not_in_add(self.messages, msgtime_ns)
                ):
                    tk.log_alloc(ns2dt(msgtime_ns), f"{e.Type} {e.DisplayMessage}")
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
                self.exitcode = terminatedevent["ExitCode"]

    def stop(self):
        if self.stoptimer:
            self.stoptimer.cancel()
        for l in self.loggers or []:
            l.stop()


@dataclasses.dataclass
class AllocWorker:
    """Represents a worker that prints out and manages state related to one allocation"""

    db: NomadDbJob
    taskhandlers: Dict[TaskKey, TaskHandler] = dataclasses.field(default_factory=dict)

    def notify(self, alloc: nomadlib.Alloc):
        """Update the state with alloc"""
        for taskname, task in alloc.get_taskstates().items():
            if args.task and not args.task.search(taskname):
                continue
            tk = TaskKey(
                alloc.ID,
                str(self.db.get_allocation_job_version(alloc, "?")),
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


@dataclasses.dataclass
class NotifierWorker:
    """An containers for storing a map of allocation workers"""

    db: NomadDbJob
    """Link to database"""
    workers: Dict[str, AllocWorker] = dataclasses.field(default_factory=dict)
    messages: Set[Tuple[int, str]] = dataclasses.field(default_factory=set)
    """A set of evaluation ModifyIndex to know what has been printed."""

    def lineno_key_not_printed(self, key: str) -> bool:
        lineno = inspect.currentframe().f_back.f_lineno  # type: ignore
        return set_not_in_add(self.messages, (lineno, key))

    def notify_alloc(self, alloc: nomadlib.Alloc):
        if LOGENABLED.eval:
            evaluation = self.db.evaluations.get(alloc.EvalID)
            if evaluation and self.lineno_key_not_printed(alloc.EvalID):
                Mylogger.log_eval(
                    self.db,
                    evaluation,
                    f"Allocation {alloc.ID} started on {alloc.NodeName}",
                )
        #
        self.workers.setdefault(alloc.ID, AllocWorker(self.db)).notify(alloc)
        #
        if LOGENABLED.eval and alloc.is_finished():
            evaluation = self.db.evaluations.get(alloc.EvalID)
            if evaluation and self.lineno_key_not_printed(alloc.EvalID):
                Mylogger.log_eval(
                    self.db, evaluation, f"Allocation {alloc.ID} finished"
                )
        #
        if LOGENABLED.eval and alloc.FollowupEvalID:
            followupeval = self.db.evaluations.get(alloc.FollowupEvalID)
            if followupeval:
                waituntil = followupeval.getWaitUntil()
                if waituntil and self.lineno_key_not_printed(followupeval.ID):
                    utcnow = datetime.datetime.now(datetime.timezone.utc)
                    delay = waituntil - utcnow
                    if delay > datetime.timedelta(0):
                        Mylogger.log_eval(
                            self.db,
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
                self.db,
                evaluation,
                f"{evaluation.JobID}: Placement Failures: {len(evaluation.FailedTGAllocs)} unplaced",
            )
            for task, metric in evaluation.FailedTGAllocs.items():
                for msg in metric.format(True, f"{task}: ").splitlines():
                    Mylogger.log_eval(self.db, evaluation, msg)

    def notify_deploy(self, deployment: nomadlib.Deploy):
        # If the job has any service defined.
        if (
            LOGENABLED.eval
            and self.db.job
            and deployment.Status
            in [
                nomadlib.DeploymentStatus.successful,
                nomadlib.DeploymentStatus.failed,
            ]
        ):
            """https://github.com/hashicorp/nomad/blob/e02dd2a331c778399fd271e85c75bff3e3783d80/ui/app/templates/components/job-deployment/deployment-metrics.hbs#L10"""
            for task, tg in deployment.TaskGroups.items():
                Mylogger.log_deploy(
                    self.db,
                    deployment,
                    f"{task} Canaries={len(tg.PlacedCanaries or [])}/{tg.DesiredCanaries} Placed={tg.PlacedAllocs} Desired={tg.DesiredTotal} Healthy={tg.HealthyAllocs} Unhealthy={tg.UnhealthyAllocs} {deployment.StatusDescription}",
                )

    def stop(self):
        for w in self.workers.values():
            for th in w.taskhandlers.values():
                th.stop()

    def join(self):
        # Logs stream outputs empty {} which allows to handle timeouts.
        threads: List[Tuple[str, threading.Thread]] = [
            (f"{tk.task}[{i}]", logger)
            for w in self.workers.values()
            for tk, th in w.taskhandlers.items()
            for i, logger in enumerate(th.loggers or [])
        ]
        thcnt = sum(len(w.taskhandlers) for w in self.workers.values())
        log.debug(
            f"Joining {len(self.workers)} allocations with {thcnt} taskhandlers and {len(threads)} loggers"
        )
        timeend = time.time() + args.shutdown_timeout
        for desc, thread in threads:
            timeout = timeend - time.time()
            if timeout > 0:
                log.debug(f"joining worker {desc} timeout={timeout}")
                thread.join(timeout=timeout)
            else:
                log.debug("timeout passed for joining workers")
                break

    def exitcode(self) -> int:
        exitcodes: List[int] = [
            # If thread did not return, exit with -1.
            -1 if th.exitcode is None else th.exitcode
            for w in self.workers.values()
            for th in w.taskhandlers.values()
        ]
        if len(exitcodes) == 0:
            return ExitCode.no_allocations
        any_unfinished = any(v == -1 for v in exitcodes)
        if any_unfinished:
            return ExitCode.any_unfinished_tasks
        only_one_task = len(exitcodes) == 1
        if only_one_task:
            return exitcodes[0]
        all_failed = all(v != 0 for v in exitcodes)
        if all_failed:
            return ExitCode.all_failed_tasks
        any_failed = any(v != 0 for v in exitcodes)
        if any_failed:
            return ExitCode.any_failed_tasks
        return ExitCode.success


###############################################################################


###############################################################################


def nomad_job_run(opts: Tuple[str]) -> str:
    """Call nomad job run to start a Nomad job from a file or from stdin or from input"""
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
    assert len(founduuids) == 1, f"Could not find uuid in nomad job run output"
    evalid = founduuids[0]
    return evalid


def nomad_start_job(opts: Tuple[str]) -> nomadlib.Eval:
    evalid = nomad_job_run(opts)
    return nomadlib.Eval(mynomad.get(f"evaluation/{evalid}"))


def nomad_find_job(jobid: str) -> nomadlib.Job:
    jobid = mynomad.find_job(jobid)
    return nomadlib.Job(mynomad.find_last_not_stopped_job(jobid))


###############################################################################


class NomadJobWatcher(ABC):
    """Watches over a job. Schedules watches over allocations. Spawns loggers."""

    def __init__(
        self,
        job: Optional[nomadlib.Job],
        eval: Optional[nomadlib.Eval],
        endstatusstr: str,
    ):
        self.job = job
        self.eval = eval
        self.endstatusstr = endstatusstr
        assert self.eval or self.job
        if self.eval:
            assert not self.job
            self.jobid = self.eval.JobID
            mynomad.namespace = self.eval.Namespace
            self.jobmodifyindex = self.eval.JobModifyIndex
        else:
            assert self.job
            self.jobid = self.job.ID
            mynomad.namespace = self.job.Namespace
            self.jobmodifyindex = self.job.JobModifyIndex
        self.db = NomadDbJob(
            topics=[
                f"Job:{self.jobid}",
                f"Evaluation:{self.jobid}",
                f"Allocation:{self.jobid}",
                f"Deployment:{self.jobid}",
            ],
            select_event_cb=self.db_select_event_job,
            init_cb=self.db_init_cb,
            force_polling=True if args.polling else None,
        )
        self.notifier = NotifierWorker(self.db)
        self.done = threading.Event()
        """I am using threading.Event because you can't handle KeyboardInterrupt while Thread.join()."""
        self.thread = threading.Thread(
            target=self.__thread_run,
            name=f"{self.__class__.__name__}({self.jobid})",
            daemon=True,
        )
        self.dbseenjob: bool = False
        """The job was found at least once."""
        self.purged: bool = False
        """If set, we are waiting until the job is there and it means that JobNotFound is not an error - the job was removed."""
        self.purgedlock: threading.Lock = threading.Lock()
        """A special lock to synchronize callback with changing finish conditions"""
        self.donemsg = ""
        """Message printed when done watching. It is not printed right away, because we might decide to wait for purging later."""
        self.started: bool = False
        """The job finished because all allocations started, not because they failed."""
        #
        self.db.start()
        self.thread.start()

    def was_purged(self):
        with self.purgedlock:
            return self.purged

    def db_init_cb(self) -> List[Event]:
        """Db initialization callback"""
        try:
            job: dict = mynomad.get(f"job/{self.jobid}")
            jobversions: dict = mynomad.get(f"job/{self.jobid}/versions")
            evaluations: List[dict] = mynomad.get(f"job/{self.jobid}/evaluations")
            allocations: List[dict] = mynomad.get(f"job/{self.jobid}/allocations")
            deployments: List[dict] = mynomad.get(f"job/{self.jobid}/deployments")
        except nomadlib.JobNotFound:
            # This is fine to fail.
            if self.was_purged() and self.job:
                # If the job was purged, generate one event so that listener can catch it.
                return [
                    Event(EventTopic.Job, EventType.JobRegistered, self.job.asdict())
                ]
            raise
        if not self.job:
            self.job = nomadlib.Job(job)
            log.debug(
                f"Found job {self.job.description()} from {self.eval.ID if self.eval else None}"
            )
        if not allocations:
            log.debug(f"Job {self.job.description()} has no allocations")
        else:
            # If there are active alocations with JobModifyIndex lower than current watched one,
            # also start watching them by lowering JobModfyIndex threashold.
            minactiveallocationjobmodifyindex = min(
                (
                    next((e for e in evaluations if e["ID"] == a["EvalID"]), {}).get(
                        "JobModifyIndex", self.jobmodifyindex
                    )
                    for a in allocations
                    if nomadlib.Alloc(a).is_pending_or_running()
                ),
                default=self.jobmodifyindex,
            )
            self.jobmodifyindex = min(
                self.jobmodifyindex, minactiveallocationjobmodifyindex
            )
        return [
            *[
                Event(EventTopic.Deployment, EventType.DeploymentStatusUpdate, d)
                for d in deployments
            ],
            *[
                Event(EventTopic.Evaluation, EventType.EvaluationUpdated, e)
                for e in evaluations
            ],
            Event(EventTopic.Job, EventType.JobRegistered, job),
            *[
                Event(EventTopic.Allocation, EventType.AllocationUpdated, a)
                for a in allocations
            ],
            *[
                Event(EventTopic.Job, EventType.JobRegistered, job)
                for job in jobversions["Versions"]
            ],
        ]

    def db_select_event_jobid(self, e: Event):
        return self.db.apply_selects(
            e,
            lambda job: job.ID == self.jobid,
            lambda eval: eval.JobID == self.jobid,
            lambda alloc: alloc.JobID == self.jobid,
            lambda deploy: deploy.JobID == self.jobid,
        )

    def db_select_event_job(self, e: Event) -> bool:
        if args.all:
            return True
        job_filter: Callable[[nomadlib.Job], bool] = lambda _: True
        eval_filter: Callable[[nomadlib.Eval], bool] = lambda eval: (
            (self.eval is not None and self.eval.ID == eval.ID)
            # The JobModifyIndex has to be greater.
            or eval.get("JobModifyIndex", -1) >= self.jobmodifyindex
        )
        alloc_filter: Callable[[nomadlib.Alloc], bool] = lambda alloc: (
            # If allocation has JobVersion, then it has to match the version in the job.
            (self.job and alloc.get("JobVersion", -1) >= self.job.Version)
            or (
                # If the allocation has no JobVersion, find the maching evaluation.
                # The JobModifyIndex from the evaluation has to match.
                self.db.evaluations.get(alloc.EvalID, {}).get("JobModifyIndex", -1)
                >= self.jobmodifyindex
            )
        )
        deploy_filter: Callable[[nomadlib.Deploy], bool] = lambda deploy: (
            deploy.get("JobModifyIndex", -1) >= self.jobmodifyindex
        )
        return self.db_select_event_jobid(e) and self.db.apply_selects(
            e, job_filter, eval_filter, alloc_filter, deploy_filter
        )

    def finish_cb(self) -> bool:
        """Overloaded callback to call to determine if we should finish watching the job"""
        # Default is to return False and fallback to job_is_finished.
        return False

    def __thread_run(self):
        untilstr = (
            "forever"
            if args.all
            else f"for {args.shutdown_timeout} seconds"
            if args.no_follow
            else f"until it is {self.endstatusstr}"
        )
        log.info(f"Watching job {self.jobid}@{mynomad.namespace} {untilstr}")
        #
        no_follow_timeend = time.time() + args.shutdown_timeout
        for events in self.db.events():
            for event in events:
                if event.topic == EventTopic.Allocation:
                    alloc = nomadlib.Alloc(event.data)
                    self.notifier.notify_alloc(alloc)
                elif event.topic == EventTopic.Evaluation:
                    evaluation = nomadlib.Eval(event.data)
                    self.notifier.notify_eval(evaluation)
                    if self.eval and self.eval.ID == evaluation.ID:
                        self.eval = evaluation
                elif event.topic == EventTopic.Deployment:
                    deployment = nomadlib.Deploy(event.data)
                    self.notifier.notify_deploy(deployment)
                elif event.topic == EventTopic.Job:
                    if self.db.job:
                        self.dbseenjob = True
                    if (
                        self.job
                        and self.db.job
                        and self.job.ModifyIndex < self.db.job.ModifyIndex
                    ):
                        # self.job follows newest job definition.
                        self.job = self.db.job
            if (
                not self.done.is_set()
                and not args.follow
                # Evaluation is finished if was passed.
                and (self.eval is None or not self.eval.is_pending())
                and (
                    (self.finish_cb() or self.job_is_finished())
                    or (args.no_follow and time.time() > no_follow_timeend)
                )
            ):
                self.done.set()
        log.debug(f"Watching job {self.jobid}@{mynomad.namespace} exiting")

    @abstractmethod
    def _get_exitcode_cb(self) -> int:
        raise NotImplementedError()

    def get_exitcode(self) -> int:
        assert self.db.stopevent.is_set(), "stop not called"
        assert self.done.is_set(), "stop not called"
        return self._get_exitcode_cb()

    def wait(self):
        self.done.wait()

    def stop_job(self, purge: bool):
        self.db.initialized.wait()
        if purge:
            with self.purgedlock:
                self.purged = True
                self.done.clear()
        mynomad.stop_job(self.jobid, purge)

    def stop_threads(self):
        log.debug(f"stopping {self.__class__.__name__} done={self.done.is_set()}")
        self.notifier.stop()
        self.db.stop()
        self.notifier.join()
        self.db.join()
        mynomad.session.close()
        if self.donemsg:
            log.info(self.donemsg)

    def run_and_exit(self):
        try:
            self.wait()
        finally:
            self.stop_threads()
        exit(self.get_exitcode())

    def has_active_deployments(self):
        for deployments in self.db.deployments.values():
            if deployments.Status in [
                nomadlib.DeploymentStatus.initializing,
                nomadlib.DeploymentStatus.running,
                nomadlib.DeploymentStatus.pending,
                nomadlib.DeploymentStatus.blocked,
                nomadlib.DeploymentStatus.paused,
            ]:
                return True
        return False

    def has_active_evaluations(self):
        for evaluation in self.db.evaluations.values():
            if evaluation.is_pending():
                return True
        return False

    def has_no_active_allocations_and_evaluations_and_deployments(self):
        for allocation in self.db.allocations.values():
            if allocation.is_pending_or_running():
                return False
        return not self.has_active_deployments() and not self.has_active_evaluations()

    def _job_is_dead_message(self):
        assert self.job
        return f"Job {self.job.description()} is dead with no active allocations, evaluations nor deployments."

    def job_is_finished(self) -> bool:
        """Return True if the job is finished and nothing more will happen to it. Sets donemsg"""
        # Protect against the situation when the job JSON is not in a database.
        # init_cb first queries allocations, then the job itself.
        if (
            self.dbseenjob
            and self.job
            and self.has_no_active_allocations_and_evaluations_and_deployments()
        ):
            # Depending on purge argument, we wait for the job to stop existing
            # or for the job to be dead.
            if self.was_purged():
                if self.db.job is None:
                    self.donemsg = f"Job {self.job.description()} removed with no active allocations, evaluations nor deployments. Exiting."
                    return True
            else:
                if self.job.is_dead():
                    self.donemsg = f"{self._job_is_dead_message()} Exiting."
                    return True
        return False

    def is_current_allocation(self, alloc: nomadlib.Alloc) -> bool:
        """Return True if the allocation is for the current job version"""
        return self.job is not None and (
            alloc.get("JobVersion", -1) >= self.job.Version
            or self.db.evaluations.get(alloc.EvalID, {}).get("JobModifyIndex", -1)
            >= self.job.JobModifyIndex
        )

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
                and t.Lifecycle.get_sidecar() == True
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
        # log.debug(f"has_active_deployments={self.has_active_deployments()} has_active_evaluations={self.has_active_evaluations()} allocations={len(self.db.allocations)}")
        if (
            not self.job
            or self.has_active_deployments()
            or self.has_active_evaluations()
        ):
            return False
        allocations = self.db.allocations.values()
        if not allocations:
            return False
        groupmsgs: List[str] = []
        for group in self.job.TaskGroups:
            # Similar logic in db_filter_event_job()
            groupallocs: List[nomadlib.Alloc] = [
                alloc
                for alloc in allocations
                if alloc.TaskGroup == group.Name and self.is_current_allocation(alloc)
            ]
            if len(groupallocs) != group.Count:
                # This group has no current allocations,
                # and no active evaluation and deployments (checked above).
                log.info(
                    f"Job {self.job.description()} has failed to start group {group.Name!r}"
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
                    if alloc.is_pending_or_running():
                        return False
                    else:
                        # The allocation has finished - the tasks will never start.
                        log.info(
                            f"Job {self.job.description()} has failed to start group {group.Name!r} tasks {' '.join(notrunningmaintasks)}"
                        )
                        return True
            groupallocsidsstr = andjoin(alloc.ID[:6] for alloc in groupallocs)
            groupmsgs.append(
                f"allocations {groupallocsidsstr} running group {group.Name!r} with {len(maintasks)} main tasks"
            )
        msg = f"Job {self.job.description()} started " + andjoin(groupmsgs) + "."
        log.info(msg)
        self.started = True
        return True


class NomadJobWatcherUntilFinished(NomadJobWatcher):
    """Watcher a job until the job is dead or purged"""

    def __init__(
        self,
        job: Optional[nomadlib.Job],
        eval: Optional[nomadlib.Eval] = None,
    ):
        super().__init__(job, eval, "finished")

    def _get_exitcode_cb(self) -> int:
        exitcode: int = (
            (ExitCode.success if self.done.is_set() else ExitCode.interrupted)
            if args.no_preserve_status
            else self.notifier.exitcode()
        )
        log.debug(f"exitcode={exitcode}")
        return exitcode

    def job_finished_successfully(self):
        assert self.job
        if self.job.Status != "dead":
            return False
        s: nomadlib.JobSummarySummary = nomadlib.JobSummary(
            mynomad.get(f"job/{self.job.ID}/summary")
        ).get_sum_summary()
        log.debug(f"{self.job_finished_successfully.__name__}: {s}")
        return (
            s.Queued == 0
            and s.Complete != 0
            and s.Failed == 0
            and s.Running == 0
            and s.Starting == 0
            and s.Lost == 0
        )

    def job_running_successfully(self):
        assert self.job
        if self.job.Status == "dead":
            return self.job_finished_successfully()
        s: nomadlib.JobSummarySummary = nomadlib.JobSummary(
            mynomad.get(f"job/{self.jobid}/summary")
        ).get_sum_summary()
        log.debug(f"{self.job_running_successfully.__name__} {s}")
        return (
            s.Queued == 0
            and s.Failed == 0
            and s.Running != 0
            and s.Starting == 0
            and s.Lost == 0
        )


class NomadJobWatcherUntilStarted(NomadJobWatcher):
    """Watches a job until the job is started"""

    def __init__(
        self,
        job: Optional[nomadlib.Job],
        eval: Optional[nomadlib.Eval] = None,
    ):
        super().__init__(job, eval, "started")

    def finish_cb(self) -> bool:
        return self.job_has_finished_starting()

    def _get_exitcode_cb(self) -> int:
        exitcode = (
            ExitCode.interrupted
            if not self.done.is_set()
            else ExitCode.failed
            if not self.started
            else ExitCode.success
        )
        log.debug(
            f"done={self.done.is_set()} started={self.started} exitcode={exitcode}"
        )
        return exitcode


class NomadAllocationWatcher:
    """Watch an allocation until it is finished"""

    def __init__(self, alloc: nomadlib.Alloc):
        self.alloc = alloc
        self.db = NomadDbJob(
            topics=[f"Allocation:{alloc.ID}"],
            select_event_cb=lambda e: e.topic == EventTopic.Allocation
            and e.data["ID"] == alloc.ID,
            init_cb=lambda: [
                Event(
                    EventTopic.Allocation,
                    EventType.AllocationUpdated,
                    mynomad.get(f"allocation/{alloc.ID}"),
                )
            ],
            force_polling=True if args.polling else None,
        )
        self.allocworkers = NotifierWorker(self.db)
        self.finished = False
        #
        log.info(f"Watching allocation {alloc.ID}")
        self.db.start()

    def run_and_exit(self):
        try:
            for events in self.db.events():
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
                if args.no_follow or self.finished:
                    break
        finally:
            self.db.stop()
            self.allocworkers.stop()
            self.db.join()
            self.allocworkers.join()
        exit(self.allocworkers.exitcode())


###############################################################################


class JobPath:
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
        _complete_set_namespace(ctx)
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
    help=f"""
Depending on the command, run or stop a Nomad job. Watch over the job and
print all job allocation events and tasks stdouts and tasks stderrs
logs. Depending on command, wait for a specific event to happen to finish
watching. This program is intended to help debugging issues with running
jobs in Nomad and for synchronizing with execution of batch jobs in Nomad.

Logs are printed in the format: 'mark>id>#version>group>task> message'.
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
@click.option(
    "-o",
    "--out",
    type=click.Choice(
        "all alloc A stdout out O 1 stderr err E 2 evaluation eval e deployment deploy d none".split()
    ),
    default=["all"],
    multiple=True,
    show_default=True,
    help="Choose which stream of messages to print - evaluation, allocation, stdout, stderr. This option is cumulative.",
)
@click.option("-v", "--verbose", count=True, help="Be more verbose.")
@click.option("-q", "--quiet", count=True, help="Be less verbose.")
@click.option(
    "-A",
    "--attach",
    is_flag=True,
    help="Stop the job on interrupt and after it has finished. Relevant in run mode only.",
)
@click.option(
    "--purge-successful",
    is_flag=True,
    help="""
        When stopping the job, purge it when all job summary metrics are zero except nonzero complete metric.
        Relevant in run and stop modes. Implies --attach.
        """,
)
@click.option(
    "--purge",
    is_flag=True,
    help="When stopping the job, purge it. Relevant in run and stop modes. Implies --attach.",
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
    default=0.5,
    show_default=True,
    type=float,
    help="When using --lines the number of lines is best-efforted by ignoring lines for this specific time",
)
@click.option(
    "--shutdown-timeout",
    default=2,
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
    is_flag=True,
    help="Just run once, get the logs in a best-effort style and exit.",
)
@click.option(
    "-t",
    "--task",
    type=re.compile,
    help="Only watch tasks names matching this regex.",
)
@click.option(
    "--polling",
    is_flag=True,
    help="Instead of listening to Nomad event stream, periodically poll for events.",
)
@click.option(
    "-x",
    "--no-preserve-status",
    is_flag=True,
    help="Do not preserve tasks exit statuses.",
)
@click_log_options()
@common_options()
def cli(**kwargs):
    exit_on_thread_exception.install()
    global args
    args = argparse.Namespace(**{**kwargs, **ALIASED})
    assert (
        (not args.follow and not args.no_follow)
        or (args.follow and not args.no_follow)
        or (not args.follow and args.no_follow)
    ), f"--follow and --no-follow conflict"
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
@common_options()
def mode_alloc(allocid):
    allocs = mynomad.get(f"allocations", params={"prefix": allocid})
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
@common_options()
def mode_eval(evalid):
    evaluation = mynomad.get(f"evaluations", params={"prefix": evalid})
    assert len(evaluation) > 0, f"Evaluation with id {evalid} not found"
    assert len(evaluation) < 2, f"Multiple evaluations found starting with id {evalid}"
    NomadJobWatcherUntilFinished(None, evaluation).run_and_exit()


@cli_command_run_nomad_job_run(
    "run",
    help=f"Run a Nomad job and then act like started mode.",
)
def mode_run(cmd: Tuple[str]):
    evaluation = nomad_start_job(cmd)
    do = NomadJobWatcherUntilFinished(None, evaluation)
    try:
        do.wait()
    finally:
        # On normal execution, the job is stopped.
        # On KeyboardException, job is still running.
        if args.attach or args.purge or args.purge_successful:
            purge: bool = args.purge or (
                args.purge_successful and do.job_finished_successfully()
            )
            do.stop_job(purge)
            do.wait()
        do.stop_threads()
    exit(do.get_exitcode())


@cli.command(
    "job",
    help="Alias to stopped command.",
)
@cli_jobid
@common_options()
def mode_job(jobid: str):
    jobinit = nomad_find_job(jobid)
    NomadJobWatcherUntilFinished(jobinit).run_and_exit()


@cli_command_run_nomad_job_run(
    "start", help=f"Start a Nomad Job and then act like started command."
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
@common_options()
def mode_started(jobid: str):
    jobinit = nomad_find_job(jobid)
    NomadJobWatcherUntilStarted(jobinit).run_and_exit()


def mode_stop_in(jobid: str):
    jobinit = nomad_find_job(jobid)
    do = NomadJobWatcherUntilFinished(jobinit)
    purge: bool = args.purge or (
        args.purge_successful and do.job_running_successfully()
    )
    do.stop_job(purge)
    do.run_and_exit()


@cli.command(
    "stop",
    help="Stop a Nomad job and then act like stopped command.",
)
@cli_jobid
@common_options()
def mode_stop(jobid: str):
    mode_stop_in(jobid)


@cli.command(
    "purge",
    help="Alias to --purge stop.",
)
@cli_jobid
@common_options()
def mode_purge(jobid: str):
    args.purge = True
    mode_stop_in(jobid)


@cli.command(
    "stopped",
    help=f"""
Watch a Nomad job until the job is stopped.
Job is stopped when the job is dead or, if the job was purged, does not exists anymore,
and the job has no running or pending allocations,
no active deployments and no active evaluations.

\b
If the option --no-preserve-status is given, then exit with the following status:
  {ExitCode.success           }    when the job was stopped
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
@common_options()
def mode_stopped(jobid: str):
    jobinit = nomad_find_job(jobid)
    NomadJobWatcherUntilFinished(jobinit).run_and_exit()


###############################################################################

if __name__ == "__main__":
    try:
        cli.main()
    finally:
        mynomad.session.close()
