#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import base64
import dataclasses
import datetime
import itertools
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from http import client as http_client
from typing import Any, Callable, Dict, Iterable, List, Optional, Pattern, Set, Tuple

import click
import requests

from . import nomadlib
from .common import (
    _complete_set_namespace,
    common_options,
    complete_job,
    completor,
    composed,
    mynomad,
    namespace_option,
    nomad_find_namespace,
)
from .nomad_smart_start_job import nomad_smart_start_job

log = logging.getLogger(__name__)

###############################################################################

args = argparse.Namespace()


@dataclasses.dataclass
class Argsstream:
    stdout: bool = False
    stderr: bool = False
    alloc: bool = False


args_out: Argsstream = Argsstream()

args_lines_start_ns: int = 0


def _init_colors() -> Dict[str, str]:
    tputdict = {
        "bold": "bold",
        "black": "setaf 0",
        "red": "setaf 1",
        "green": "setaf 2",
        "orange": "setaf 3",
        "blue": "setaf 4",
        "magenta": "setaf 5",
        "cyan": "setaf 6",
        "white": "setaf 7",
        "reset": "sgr0",
    }
    empty = {k: "" for k in tputdict.keys()}
    if not sys.stdout.isatty() or not sys.stderr.isatty():
        return empty
    tputscript = "\n".join(tputdict.values()).replace("\n", "\nlongname\nlongname\n")
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
    return {k: v for k, v in zip(tputdict.keys(), retarr)}


COLORS = _init_colors()

###############################################################################


def ns2s(ns: int):
    return ns / 1000000000


def ns2dt(ns: int):
    return datetime.datetime.fromtimestamp(ns2s(ns)).astimezone()


###############################################################################


@dataclasses.dataclass(frozen=True)
class LogFormat:
    alloc: str
    stderr: str
    stdout: str
    module: str

    @classmethod
    def mk(
        cls,
        prefix: str = "%(allocid).6s:%(group)s:%(task)s:",
        log_timestamp: bool = False,
    ):
        now = "%(asctime)s:" if log_timestamp else ""
        alloc_now = "" if log_timestamp else " %(asctime)s"
        lf = cls(
            f"{now}{prefix}A{alloc_now} %(message)s",
            f"{now}{prefix}E %(message)s",
            f"{now}{prefix}O %(message)s",
            f"{now}%(module)s:%(lineno)03d: %(levelname)s %(message)s",
        )
        lf = cls(
            f"%(cyan)s{lf.alloc}%(reset)s",
            f"%(orange)s{lf.stderr}%(reset)s",
            lf.stdout,
            f"%(blue)s{lf.module}%(reset)s",
        )
        return lf

    def astuple(self):
        return dataclasses.astuple(self)


log_format = LogFormat.mk()


def click_log_options():
    """All logging options"""
    return composed(
        click.option(
            "-T",
            "--log-timestamp",
            is_flag=True,
            help="Additionally add timestamp of the logs from the task. The timestamp is when the log was received. Nomad does not store timestamp of logs sadly.",
        ),
        click.option(
            "--log-timestamp-format",
            default="%Y-%m-%dT%H:%M:%S%z",
            show_default=True,
        ),
        click.option(
            "-H",
            "--log-timestamp-hour",
            is_flag=True,
            help="Alias for --log-timestamp --log-timestamp-format %H:%M:%S",
        ),
        click.option("--log-format-alloc", default=log_format.alloc, show_default=True),
        click.option(
            "--log-format-stderr", default=log_format.stderr, show_default=True
        ),
        click.option(
            "--log-format-stdout", default=log_format.stdout, show_default=True
        ),
        click.option(
            "--log-long-alloc", is_flag=True, help="Log full length allocation id"
        ),
        click.option(
            "-G",
            "--log-no-group",
            is_flag=True,
            help="Do not log group",
        ),
        click.option(
            "--log-no-task",
            is_flag=True,
            help="Do not log task",
        ),
        click.option(
            "-1",
            "--log-only-task",
            is_flag=True,
            help="Prefix the lines only with task name.",
        ),
        click.option(
            "-0",
            "--log-none",
            is_flag=True,
            help="Log only stream prefix",
        ),
    )


def log_format_choose():
    global log_format
    log_format = LogFormat(
        args.log_format_alloc,
        args.log_format_stderr,
        args.log_format_stdout,
        log_format.module,
    )
    alloc = "%(allocid)s" if args.log_long_alloc else "%(allocid).6s"
    group = "" if args.log_no_group else "%(group)s:"
    task = "" if args.log_no_task else "%(task)s:"
    log_format = LogFormat.mk(f"{alloc}:{group}{task}", args.log_timestamp)
    args.log_timestamp = args.log_timestamp_hour or args.log_timestamp
    args.log_timestamp_format = (
        "%H:%M:%S" if args.log_timestamp_hour else args.log_timestamp_format
    )
    if args.log_only_task:
        log_format = LogFormat.mk("%(task)s:", args.log_timestamp)
    elif args.log_none:
        log_format = LogFormat.mk("", args.log_timestamp)
    else:
        log_format = LogFormat.mk(log_timestamp=args.log_timestamp)
    #
    logging.basicConfig(
        format=log_format.module,
        datefmt=args.log_timestamp_format,
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
    # https://stackoverflow.com/questions/17558552/how-do-i-add-custom-field-to-python-log-format-string
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        for k, v in COLORS.items():
            setattr(record, k, v)
        return record

    logging.setLogRecordFactory(record_factory)


@dataclasses.dataclass(frozen=True)
class TaskKey:
    """Represent data to unique identify a task"""

    allocid: str
    nodename: str
    group: str
    task: str

    def __str__(self):
        return f"{self.allocid:.6}:{self.group}:{self.task}"

    def _params(self, params: Dict[str, Any] = {}) -> Dict[str, Any]:
        return {
            **params,
            **dataclasses.asdict(self),
            **COLORS,
            "asctime": params["now"].strftime(args.log_timestamp_format),
        }

    def _log(self, fmt, **kvargs: Any):
        print(fmt % self._params(kvargs), flush=True)

    def log_alloc(self, now: datetime.datetime, message: str):
        self._log(log_format.alloc, now=now, message=message)

    def log_task(self, stderr: bool, message: str):
        self._log(
            log_format.stderr if stderr else log_format.stdout,
            message=message,
            now=datetime.datetime.now().astimezone(),
        )


###############################################################################


class Logger(threading.Thread):
    """Represents a single logging stream from Nomad. Such stream is created separately for stdout and stderr."""

    def __init__(self, tk: TaskKey, stderr: bool):
        super().__init__(name=f"{tk}{1 + int(stderr)}")
        self.tk = tk
        self.stderr: bool = stderr
        self.exitevent = threading.Event()
        self.ignoredlines: List[str] = []
        self.first_line = True
        # Ignore input lines if printing only trailing lines.
        global args_lines_start_ns
        self.ignoretime_ns = (
            0 if args.lines < 0 else args_lines_start_ns + int(args.lines_timeout * 1e9)
        )
        # If ignore time is in the past, it is no longer relevant anyway.
        if self.ignoretime_ns and self.ignoretime_ns < time.time_ns():
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

    def run(self):
        """Listen to Nomad log stream and print the logs"""
        with mynomad.stream(
            f"client/fs/logs/{self.tk.allocid}",
            params={
                "task": self.tk.task,
                "type": "stderr" if self.stderr else "stdout",
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

    def stop(self):
        self.exitevent.set()


class TaskHandler:
    """A handler for one task. Creates loggers, writes out task events, handle exit conditions"""

    def __init__(self):
        # Array of loggers that log allocation logs.
        self.loggers: List[Logger] = []
        # A set of message timestamp to know what has been printed.
        self.messages: Set[int] = set()
        self.exitcode: Optional[int] = None

    @staticmethod
    def _create_loggers(tk: TaskKey):
        global args_out
        ths: List[Logger] = []
        if args_out.stdout:
            ths.append(Logger(tk, False))
        if args_out.stderr:
            ths.append(Logger(tk, True))
        for th in ths:
            th.start()
        return ths

    def notify(self, tk: TaskKey, taskstate: nomadlib.AllocTaskState):
        """Receive notification that a task state has changed"""
        events = taskstate.Events
        global args_out
        if args_out.alloc:
            for e in events:
                msg = f"{e.Type} {e.DisplayMessage}"
                msgtime_ns = e.Time
                global args_lines_start_ns
                # Ignore message before ignore times.
                if (
                    msgtime_ns
                    and msg
                    and msgtime_ns not in self.messages
                    and (
                        not args_lines_start_ns
                        or msgtime_ns >= args_lines_start_ns
                        or len(self.messages) < args.lines
                    )
                ):
                    self.messages.add(msgtime_ns)
                    tk.log_alloc(ns2dt(msgtime_ns), msg)
        if (
            not self.loggers
            and taskstate.State in ["running", "dead"]
            and taskstate.was_started()
        ):
            self.loggers += self._create_loggers(tk)
            if taskstate.State == "dead":
                # If the task is already finished, give myself max 3 seconds to query all the logs.
                # This is to reduce the number of connections.
                threading.Timer(3, self.stop)
        if self.exitcode is None and taskstate.State == "dead":
            # Assigns None if Terminated event not found
            self.exitcode = (taskstate.find_event("Terminated") or {}).get("ExitCode")
            self.stop()

    def stop(self):
        for l in self.loggers:
            l.stop()


class AllocWorker:
    """Represents a worker that prints out and manages state related to one allocation"""

    def __init__(self):
        self.taskhandlers: Dict[TaskKey, TaskHandler] = {}

    def notify(self, alloc: nomadlib.Alloc):
        """Update the state with alloc"""
        for taskname, task in alloc.get_taskstates().items():
            if args.task and not args.task.search(taskname):
                continue
            tk = TaskKey(alloc.ID, alloc.NodeName, alloc.TaskGroup, taskname)
            self.taskhandlers.setdefault(tk, TaskHandler()).notify(tk, task)


class ExitCode:
    success = 0
    exception = 1
    interrupted = 2
    """This program execution flow was interrupted"""
    any_failed_tasks = 124
    all_failed_tasks = 125
    any_unfinished_tasks = 126
    no_allocations = 127


class AllocWorkers(Dict[str, AllocWorker]):
    """An containers for storing a map of allocation workers"""

    def notify(self, alloc: nomadlib.Alloc):
        self.setdefault(alloc.ID, AllocWorker()).notify(alloc)

    def stop(self):
        for w in self.values():
            for th in w.taskhandlers.values():
                th.stop()

    def join(self):
        # Logs stream outputs empty {} which allows to handle timeouts.
        threads: List[Tuple[str, threading.Thread]] = [
            (f"{tk.task}[{i}]", logger)
            for w in self.values()
            for tk, th in w.taskhandlers.items()
            for i, logger in enumerate(th.loggers)
        ]
        thcnt = sum(len(w.taskhandlers) for w in self.values())
        log.debug(
            f"Joining {len(self)} allocations with {thcnt} taskhandlers and {len(threads)} loggers"
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
            for w in self.values()
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


Event = nomadlib.Event
EventTopic = nomadlib.EventTopic
EventType = nomadlib.EventType


class Db:
    """Represents relevant state cache from Nomad database"""

    def __init__(
        self,
        topics: List[str],
        select_event_cb: Callable[[Event], bool],
        init_cb: Callable[[], List[Event]],
    ):
        """
        :param topic The topics to listen to, see Nomad event stream API documentation.
        :param select_event_cb: Filter only relevant events from Nomad event stream.
        :param init_cb Return a list of events to populate the database with and to poll. Has to be threadsafe!
        """
        self.topics: List[str] = topics
        self.init_cb: Callable[[], List[Event]] = init_cb
        self.select_event_cb: Callable[[Event], bool] = select_event_cb
        self.queue: queue.Queue[Optional[List[Event]]] = queue.Queue()
        """Queue where database thread puts the received events"""
        self.job: Optional[nomadlib.Job] = None
        """Watched job"""
        self.evaluations: Dict[str, nomadlib.Eval] = {}
        """Database of evaluations"""
        self.allocations: Dict[str, nomadlib.Alloc] = {}
        """Database of allocations"""
        self.initialized = threading.Event()
        """Was init_cb called to initiliaze the database"""
        self.stopevent = threading.Event()
        """If set, the database thread should exit"""
        self.thread = threading.Thread(
            target=self._thread_entry, name="db", daemon=True
        )
        assert self.topics
        assert not any(not x for x in topics)

    ###############################################################################

    def _thread_run_stream(self):
        log.debug(f"Starting listen Nomad stream with {' '.join(self.topics)}")
        with mynomad.stream(
            "event/stream",
            params={"topic": self.topics},
        ) as stream:
            for line in stream.iter_lines():
                if line:
                    data = json.loads(line)
                    events: List[Event] = [
                        Event(
                            EventTopic[event["Topic"]],
                            EventType[event["Type"]],
                            event["Payload"][event["Topic"]],
                            stream=True,
                        )
                        for event in data.get("Events", [])
                    ]
                    # log.debug(f"RECV EVENTS: {events}")
                    self.queue.put(events)
                if self.stopevent.is_set():
                    break

    def _thread_poll(self):
        """If listening to stream fails, we fallback to calling init_cb in a loop"""
        # Delay polling start up until init_cb has been called.
        self.initialized.wait()
        while not self.stopevent.wait(1):
            self.queue.put(self.init_cb())

    def _thread_entry(self):
        """Database thread entry"""
        try:
            try:
                if args.polling:
                    self._thread_poll()
                else:
                    try:
                        self._thread_run_stream()
                    except nomadlib.PermissionDenied as e:
                        log.warning(
                            f"Falling to polling method because stream API returned permission denied: {e}"
                        )
                        self._thread_poll()
            finally:
                log.debug("Nomad database thread exiting")
                self.queue.put(None)
        except requests.HTTPError as e:
            log.exception("http request failed")
            exit(ExitCode.exception)

    ###############################################################################

    def start(self):
        """Start the database thread"""
        assert (
            mynomad.namespace
        ), "Nomad namespace has to be set before starting to listen"
        self.thread.start()

    def select_is_in_db(self, e: Event):
        """Select events that are already in the database"""
        if e.topic == EventTopic.Evaluation:
            return e.data["ID"] in self.evaluations
        elif e.topic == EventTopic.Allocation:
            return e.data["ID"] in self.allocations
        return False

    def _add_event_to_db(self, e: Event):
        """Update database state to reflect received event"""
        if e.topic == EventTopic.Job:
            if e.type == EventType.JobDeregistered:
                self.job = None
            else:
                self.job = nomadlib.Job(e.data)
        elif e.topic == EventTopic.Evaluation:
            if e.type == EventType.JobDeregistered:
                self.job = None
            self.evaluations[e.data["ID"]] = nomadlib.Eval(e.data)
        elif e.topic == EventTopic.Allocation:
            self.allocations[e.data["ID"]] = nomadlib.Alloc(e.data)

    def handle_event(self, e: Event) -> bool:
        if self._select_new_event(e):
            if self.select_is_in_db(e) or self.select_event_cb(e):
                # log.debug(f"EVENT: {e}")
                self._add_event_to_db(e)
                return True
            else:
                # log.debug(f"USER FILTERED: {e}")
                pass
        else:
            # log.debug(f"OLD EVENT: {e}")
            pass
        return False

    def handle_events(self, events: List[Event]) -> List[Event]:
        """From a list of events, filter out ignored and add the rest to database"""
        return [e for e in events if self.handle_event(e)]

    @staticmethod
    def apply_selects(
        e: Event,
        job_select: Callable[[nomadlib.Job], bool],
        eval_select: Callable[[nomadlib.Eval], bool],
        alloc_select: Callable[[nomadlib.Alloc], bool],
    ) -> bool:
        """Apply specific selectors depending on event type"""
        return e.apply(job_select, eval_select, alloc_select)

    def _select_new_event(self, e: Event):
        """Select events which are newer than those in the database"""
        job_select: Callable[[nomadlib.Job], bool] = (
            lambda job: self.job is None or job.ModifyIndex > self.job.ModifyIndex
        )
        eval_select: Callable[[nomadlib.Eval], bool] = (
            lambda eval: eval.ID not in self.evaluations
            or eval.ModifyIndex > self.evaluations[eval.ID].ModifyIndex
        )
        alloc_select: Callable[[nomadlib.Alloc], bool] = (
            lambda alloc: alloc.ID not in self.allocations
            or alloc.ModifyIndex > self.allocations[alloc.ID].ModifyIndex
        )
        return e.data["Namespace"] == mynomad.namespace and self.apply_selects(
            e, job_select, eval_select, alloc_select
        )

    def stop(self):
        log.debug("Stopping listen Nomad stream")
        self.initialized.set()
        self.stopevent.set()

    def join(self):
        # Not joining - neither requests nor stream API allow for timeouts.
        # self.thread.join()
        pass

    def events(self) -> Iterable[List[Event]]:
        """Nomad stream returns Events array. Iterate over batches of events returned from Nomad stream"""
        assert self.thread.is_alive(), "Thread not alive"
        if not self.initialized.is_set():
            events = self.init_cb()
            events = self.handle_events(events)
            yield events
            self.initialized.set()
        log.debug("Starting getting events from thread")
        while not self.queue.empty() or (
            self.thread.is_alive() and not self.stopevent.is_set()
        ):
            events = self.queue.get()
            if events is None:
                break
            yield self.handle_events(events)
        log.debug("db exiting")


###############################################################################


def nomad_watch_eval(evalid: str):
    assert isinstance(evalid, str), f"not a string: {evalid}"
    db = Db(
        topics=[
            f"Evaluation:{evalid}",
        ],
        select_event_cb=lambda e: e.topic == EventTopic.Evaluation
        and e.data["ID"] == evalid,
        init_cb=lambda: [
            Event(
                EventTopic.Evaluation,
                EventType.EvaluationUpdated,
                mynomad.get(f"evaluation/{evalid}"),
            )
        ],
    )
    db.start()
    log.info(f"Waiting for evaluation {evalid}")
    eval_ = None
    for events in db.events():
        for event in events:
            assert event.topic == EventTopic.Evaluation
            eval_ = nomadlib.Eval(event.data)
            if eval_.Status != "pending":
                break
        if eval_ and eval_.Status != "pending":
            break
    db.stop()
    assert eval_ is not None
    assert (
        eval_.Status == "complete"
    ), f"Evaluation {evalid} did not complete: {eval_.get('StatusDescription')}"
    FailedTGAllocs = eval_.get("FailedTGAllocs")
    if FailedTGAllocs:
        groups = " ".join(list(FailedTGAllocs.keys()))
        log.warning(f"Evaluation {evalid} failed to place groups: {groups}")


def nomad_start_job_and_wait(input: str) -> nomadlib.Job:
    assert isinstance(input, str)
    evalid = nomad_smart_start_job(input, args.json)
    eval_: dict = mynomad.get(f"evaluation/{evalid}")
    mynomad.namespace = eval_["Namespace"]
    nomad_watch_eval(evalid)
    jobid = eval_["JobID"]
    return nomadlib.Job(mynomad.get(f"job/{jobid}"))


def nomad_find_job(jobid: str) -> nomadlib.Job:
    jobid = mynomad.find_job(jobid)
    return nomadlib.Job(mynomad.find_last_not_stopped_job(jobid))


###############################################################################


class NomadJobWatcher(ABC):
    """Watches over a job. Schedules watches over allocations. Spawns loggers."""

    def __init__(self, job: nomadlib.Job, untilstr: str):
        self.job = job
        self.untilstr = untilstr
        self.allocworkers = AllocWorkers()
        self.db = Db(
            topics=[
                f"Job:{self.job.ID}",
                f"Evaluation:{self.job.ID}",
                f"Allocation:{self.job.ID}",
            ],
            select_event_cb=self.db_select_event_job,
            init_cb=self.db_init_cb,
        )
        self.done = threading.Event()
        """I am using threading.Event because you can't handle KeyboardInterrupt while Thread.join()."""
        self.thread = threading.Thread(
            target=self.thread_run,
            name=f"NomadJobWatcher({self.job['ID']})",
            daemon=True,
        )
        # Start threads
        self.db.start()
        self.thread.start()

    def db_init_cb(self) -> List[Event]:
        """Db initialization callback"""
        job: dict = mynomad.get(f"job/{self.job.ID}")
        evaluations: List[dict] = mynomad.get(f"job/{self.job.ID}/evaluations")
        allocations: List[dict] = mynomad.get(f"job/{self.job.ID}/allocations")
        if not allocations:
            log.debug(f"Job {self.job.description()} has no allocations")
        return [
            *[
                Event(EventTopic.Evaluation, EventType.EvaluationUpdated, e)
                for e in evaluations
            ],
            *[
                Event(EventTopic.Allocation, EventType.AllocationUpdated, a)
                for a in allocations
            ],
            *[Event(EventTopic.Job, EventType.JobRegistered, job)],
        ]

    def db_select_event_jobid(self, e: Event):
        return Db.apply_selects(
            e,
            lambda job: job.ID == self.job.ID,
            lambda eval: eval.JobID == self.job.ID,
            lambda alloc: alloc["JobID"] == self.job.ID,
        )

    def db_select_event_job(self, e: Event):
        job_filter: Callable[[nomadlib.Job], bool] = lambda _: True
        eval_filter: Callable[[nomadlib.Eval], bool] = lambda eval: (
            # Either all, or the JobModifyIndex has to be greater.
            args.all
            or eval.get("JobModifyIndex", -1) >= self.job.JobModifyIndex
        )
        alloc_filter: Callable[[nomadlib.Alloc], bool] = lambda alloc: (
            args.all
            # If allocation has JobVersion, then it has to match the version in the job.
            or alloc.get("JobVersion", -1) >= self.job.Version
            # If the allocation has no JobVersion, find the maching evaluation.
            # The JobModifyIndex from the evalution has to match.
            or (
                self.db.evaluations.get(alloc.EvalID, {}).get("JobModifyIndex", -1)
                >= self.job.JobModifyIndex
            )
        )
        return self.db_select_event_jobid(e) and Db.apply_selects(
            e, job_filter, eval_filter, alloc_filter
        )

    @abstractmethod
    def finish_cb(self) -> bool:
        """Overloaded callback to call to determine if we should finish watching the job"""
        raise NotImplementedError()

    def thread_run(self):
        untilstr = (
            "forever"
            if args.all
            else f"for {args.shutdown_timeout} seconds"
            if args.no_follow
            else self.untilstr
        )
        log.info(f"Watching job {self.job.description()} {untilstr}")
        #
        no_follow_timeend = time.time() + args.shutdown_timeout
        for events in self.db.events():
            for event in events:
                if event.topic == EventTopic.Allocation:
                    alloc = nomadlib.Alloc(event.data)
                    # for alloc in self.db.allocations.values():
                    self.allocworkers.notify(alloc)
                elif event.topic == EventTopic.Job:
                    # self.job follows newest job definition.
                    if self.db.job and self.job.ModifyIndex < self.db.job.ModifyIndex:
                        self.job = self.db.job
            if (
                not self.done.is_set()
                and (not args.all and self.finish_cb())
                or (args.no_follow and time.time() > no_follow_timeend)
            ):
                self.done.set()
        log.debug("Watching job {self.job.description()} exiting")

    @abstractmethod
    def _get_exitcode(self) -> int:
        raise NotImplementedError()

    def get_exitcode(self) -> int:
        assert self.db.stopevent.is_set(), "stop not called"
        assert self.done.is_set(), "stop not called"
        return self._get_exitcode()

    def wait(self):
        self.done.wait()

    def stop(self):
        log.debug("stopping")
        self.allocworkers.stop()
        self.db.stop()
        self.allocworkers.join()
        self.db.join()
        mynomad.session.close()

    def run_and_exit(self):
        try:
            self.wait()
        finally:
            self.stop()
        exit(self.get_exitcode())

    def no_allocations_are_pending_or_running(self):
        return all(not x.is_pending_or_running() for x in self.db.allocations.values())


class NomadJobWatcherUntilFinished(NomadJobWatcher):
    """Watcher a job until the job is dead or purged"""

    def __init__(self, job: nomadlib.Job):
        super().__init__(job, "until it is finished")
        self.foundjob: bool = False
        """The job was found at least once."""
        self.purged: bool = False
        """If set, we are waiting until the job is there and it means that JobNotFound is not an error - the job was removed."""
        self.purgedlock: threading.Lock = threading.Lock()
        """A special lock to synchronize callback with changing finish conditions"""
        self.donemsg = ""
        """Message printed when done watching. It is not printed right away, because we might decide to wait for purging later."""

    def db_init_cb(self) -> List[Event]:
        try:
            return super().db_init_cb()
        except nomadlib.JobNotFound:
            # If the job was purged, this can possibly fail.
            with self.purgedlock:
                if self.purged:
                    return [
                        Event(
                            EventTopic.Job, EventType.JobRegistered, self.job.asdict()
                        )
                    ]
            raise

    def finish_cb(self) -> bool:
        # Protect against the situation when the job JSON is not in a database.
        # init_cb first queries allocations, then the job itself.
        if self.db.job:
            self.foundjob = True
        elif not self.foundjob:
            return False
        if self.no_allocations_are_pending_or_running():
            with self.purgedlock:
                # Depending on purge argument, we wait for the job to stop existing
                # or for the job to be dead.
                if self.purged:
                    if self.db.job is None:
                        self.donemsg = f"Job {self.job.description()} purged with no running or pending allocations. Exiting."
                        return True
                else:
                    if self.job.is_dead():
                        self.donemsg = f"Job {self.job.description()} is dead with no running or pending allocations. Exiting."
                        return True
        return False

    def stop(self):
        super().stop()
        if self.donemsg:
            log.info(self.donemsg)

    def _get_exitcode(self) -> int:
        exitcode: int = (
            (ExitCode.success if self.done.is_set() else ExitCode.interrupted)
            if args.no_preserve_status
            else self.allocworkers.exitcode()
        )
        log.debug(f"exitcode={exitcode}")
        return exitcode

    def stop_job(self, purge: bool):
        self.db.initialized.wait()
        if purge:
            with self.purgedlock:
                self.purged = True
                self.done.clear()
        mynomad.stop_job(self.job.ID, purge)

    def job_finished_successfully(self):
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
        if self.job.Status == "dead":
            return self.job_finished_successfully()
        s: nomadlib.JobSummarySummary = nomadlib.JobSummary(
            mynomad.get(f"job/{self.job.ID}/summary")
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

    def __init__(self, job: nomadlib.Job):
        super().__init__(job, "until it is started")
        self.hadalloc: bool = False
        """The job had allocations."""
        self.started: bool = False
        """The job finished because all allocations started, not because they failed."""

    def job_is_started(self):
        """Job is started if in current allocations all main tasks from all groups have started"""
        allocations = self.db.allocations.values()
        if not self.job.TaskGroups:
            # Job can't possibly be running with no taskgroups.
            return False
        groupmsgs: List[str] = []
        for group in self.job.TaskGroups:
            # Similar logic in db_filter_event_job()
            groupallocs: List[nomadlib.Alloc] = [
                alloc
                for alloc in allocations
                if alloc.TaskGroup == group.Name
                and (
                    alloc.get("JobVersion", -1) >= self.job.Version
                    or self.db.evaluations.get(alloc.EvalID, {}).get(
                        "JobModifyIndex", -1
                    )
                    >= self.job.JobModifyIndex
                )
            ]
            if not groupallocs:
                # This group has no allocations - yet!
                return False
            groupallocs.sort(key=lambda alloc: alloc.ModifyIndex, reverse=True)
            lastalloc: nomadlib.Alloc = groupallocs[0]
            # List of started tasks.
            startedtasks: Set[str] = set(
                name
                for name, taskstate in lastalloc.get_taskstates().items()
                if taskstate.was_started()
            )
            # Main tasks are tasks that:
            # - do not have lifecycle
            # - have lifecycle prestart with sidecar = true
            # - have lifecycle poststart
            # All these tasks have to be started.
            allmaintasks: Set[str] = set(
                t.Name
                for t in group.Tasks
                if "Lifecycle" not in t
                or t.Lifecycle is None
                or (
                    t.Lifecycle.Hook == "prestart" and t.Lifecycle.get_sidecar() == True
                )
                or t.Lifecycle.Hook == "poststart"
            )
            assert len(allmaintasks) > 0, f"{allmaintasks} {startedtasks}"
            notrunningmaintasks = allmaintasks.difference(startedtasks)
            if notrunningmaintasks:
                # There are main tasks that are not running.
                return False
            groupmsgs.append(
                f"allocation {lastalloc.ID:.6} with {len(allmaintasks)} running main tasks of group {group.Name!r}"
            )
        msg = f"Job {self.job.description()} started " + " and ".join(groupmsgs) + "."
        log.info(msg)
        self.started = True
        return True

    def finish_cb(self) -> bool:
        if self.job_is_started():
            return True
        if self.job.is_dead() and self.no_allocations_are_pending_or_running():
            log.info(
                f"Job {self.job.description()} is dead with no running or pending allocations. Bailing out."
            )
            return True
        return False

    def _get_exitcode(self) -> int:
        exitcode = ExitCode.success if self.started else ExitCode.interrupted
        log.debug(f"started={self.started} so exitcode={exitcode}")
        return exitcode


class NomadAllocationWatcher:
    """Watch an allocation until it is finished"""

    def __init__(self, alloc: nomadlib.Alloc):
        self.alloc = alloc
        self.allocworkers = AllocWorkers()
        self.db = Db(
            topics=[f"Allocation:{alloc.JobID}"],
            select_event_cb=lambda e: e.topic == EventTopic.Allocation
            and e.data["ID"] == alloc.ID,
            init_cb=lambda: [
                Event(
                    EventTopic.Allocation,
                    EventType.AllocationUpdated,
                    mynomad.get(f"allocation/{alloc.ID}"),
                )
            ],
        )
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
                        self.allocworkers.notify(alloc)
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
    Run a Nomad job in Nomad. Watch over the job and print all job
    allocation events and tasks stdouts and tasks stderrs logs. Depending
    on mode, wait for a specific event to happen to finish watching.
    The script is intended to help debugging issues with running jobs
    in Nomad and for synchronizing with execution of batch jobs in Nomad.

    \b
    If the option --no-preserve-exit is given, then exit with the following status:
        0    if operation was successful - the job was run or was purged on --purge
    Ohterwise, when mode is alloc, run, job, stop or stopped, exit with the following status:
        ?    when the job has one task, with that task exit status,
        0    if all tasks of the job exited with 0 exit status,
        {ExitCode.any_failed_tasks}  if any of the job tasks have failed,
        {ExitCode.all_failed_tasks}  if all job tasks have failed,
        {ExitCode.any_unfinished_tasks}  if any tasks are still running,
        {ExitCode.no_allocations}  if job has no started tasks.
    In any case, exit with the following status:
        1    if some error occured, like python exception.

    \b
    Examples:
        nomad-watch --namespace default run ./some-job.nomad.hcl
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
    help="""
        Do not exit after the current job version is finished.
        Instead, watch endlessly for any existing and new allocations of a job.
        """,
)
@click.option(
    "-o",
    "--out",
    type=click.Choice("all alloc A stdout out O 1 stderr err E 2 none".split()),
    default=["all"],
    multiple=True,
    show_default=True,
    help="Choose which stream of messages to print - allocation, stdout, stderr. This option is cummulative.",
)
@click.option("-v", "--verbose", count=True, help="Be more verbose.")
@click.option("-q", "--quiet", count=True, help="Be less verbose.")
@click.option(
    "--json",
    is_flag=True,
    help="Job input is in json form. Passed to nomad command line interface with -json.",
)
@click.option(
    "-d",
    "--detach",
    is_flag=True,
    help="Relevant in run mode only. Do not stop the job after it has finished or on interrupt.",
)
@click.option(
    "--purge-successful",
    is_flag=True,
    help="""
        Relevant in run and stop modes.
        When stopping the job, purge it when all job summary metrics are zero except nonzero complete metric.
        """,
)
@click.option(
    "--purge",
    is_flag=True,
    help="Relevant in run and stop modes. When stopping the job, purge it.",
)
@click.option(
    "-n",
    "--lines",
    default=-1,
    show_default=True,
    type=int,
    help="""
        Sets the tail location in best-efforted number of lines relative to the end of logs.
        Default prints all the logs.
        Set to 0 to try try best-efforted logs from the current log position.
        See also --lines-timeout.
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
    help="Shorthand for --all --lines=10 to act similar to tail -f.",
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
    global args
    args = argparse.Namespace(**kwargs)
    args.verbose -= args.quiet
    #
    if args.verbose > 1:
        http_client.HTTPConnection.debuglevel = 1
    global args_out
    args_out = Argsstream(
        stderr=any(s.lower() in "all stderr err e 2".split() for s in args.out),
        stdout=any(s.lower() in "all stdout out o 1".split() for s in args.out),
        alloc=any(s.lower() in "all alloc a".split() for s in args.out),
    )
    if args.follow:
        args.lines = 10
        args.all = True
    if args.namespace:
        os.environ["NOMAD_NAMESPACE"] = nomad_find_namespace(args.namespace)
    #
    if args.lines >= 0:
        global args_lines_start_ns
        args_lines_start_ns = time.time_ns()
    # init logging
    log_format_choose()


cli_jobid = click.argument(
    "jobid",
    shell_complete=complete_job(),
)
cli_jobfile_help = """
JOBFILE can be file with a HCL or JSON nomad job or
it can be a string containing a HCL or JSON nomad job.
"""
cli_jobfile = click.argument(
    "jobfile",
    shell_complete=click.File().shell_complete,
)

###############################################################################


@cli.command("alloc", help="Watch over specific allocation.")
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
    "run",
    help=f"""
Run a Nomad job and then watch over it until the job is dead and has no pending or running job allocations.
Stop the job on interrupt or when finished, unless --no-stop option is given.
{cli_jobfile_help}
""",
)
@cli_jobfile
@common_options()
def mode_run(jobfile):
    jobinit = nomad_start_job_and_wait(jobfile)
    do = NomadJobWatcherUntilFinished(jobinit)
    try:
        do.wait()
    finally:
        # On normal execution, the job is stopped.
        # On KeyboardException, job is still running.
        if not args.detach:
            purge: bool = args.purge or (
                args.purge_successful and do.job_finished_successfully()
            )
            do.stop_job(purge)
            do.wait()
        do.stop()
    exit(do.get_exitcode())


@cli.command(
    "job",
    help="Watch a Nomad job. Show the job allocation events and logs.",
)
@cli_jobid
@common_options()
def mode_job(jobid):
    jobinit = nomad_find_job(jobid)
    NomadJobWatcherUntilFinished(jobinit).run_and_exit()


@cli.command(
    "start", help=f"Start a Nomad Job. Then act like mode started. {cli_jobfile_help}"
)
@cli_jobfile
@common_options()
def mode_start(jobfile):
    jobinit = nomad_start_job_and_wait(jobfile)
    NomadJobWatcherUntilStarted(jobinit).run_and_exit()


@cli.command(
    "started",
    help="""
Watch a Nomad job until the jobs main tasks are running or have been run.
Main tasks are all tasks without lifetime or sidecar prestart tasks or poststart tasks.

\b
Exit with the following status:
    0    all tasks of the job have started running,
    2    the job was stopped before any of the tasks could start.
""",
)
@cli_jobid
@common_options()
def mode_started(jobid):
    jobinit = nomad_find_job(jobid)
    NomadJobWatcherUntilStarted(jobinit).run_and_exit()


@cli.command(
    "stop",
    help="Stop a Nomad job. Then watch the job until the job is dead and has no pending or running allocations.",
)
@cli_jobid
@common_options()
def mode_stop(jobid: str):
    jobinit = nomad_find_job(jobid)
    do = NomadJobWatcherUntilFinished(jobinit)
    purge: bool = args.purge or (
        args.purge_successful and do.job_running_successfully()
    )
    do.stop_job(purge)
    do.run_and_exit()


@cli.command(
    "stopped",
    help="Watch a Nomad job until the job is stopped - has not running allocation.",
)
@cli_jobid
@common_options()
def mode_stopped(jobid):
    jobinit = nomad_find_job(jobid)
    NomadJobWatcherUntilFinished(jobinit).run_and_exit()


###############################################################################

if __name__ == "__main__":
    try:
        cli.main()
    finally:
        mynomad.session.close()
