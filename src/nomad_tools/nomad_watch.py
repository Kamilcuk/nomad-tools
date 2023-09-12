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
import shlex
import shutil
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

###############################################################################

log = logging.getLogger(__name__)


def ns2dt(ns: int):
    return datetime.datetime.fromtimestamp(ns // 1000000000)


def run(cmd, *args, check=True, **kvargs):
    log.info(f"+ {' '.join(shlex.quote(x) for x in cmd)}")
    return subprocess.run(cmd, *args, check=check, text=True, **kvargs)


###############################################################################


class Test:
    """For running internal tests"""

    def __init__(self, mode: str):
        func = getattr(self, mode, None)
        assert func
        func()

    def short(self):
        spec = """
            job "test-nomad-watch" {
              type = "batch"
              reschedule { attempts = 0 }
              group "test-nomad-watch" {
                restart { attempts = 0 }
                task "test-nomad-watch" {
                  driver = "docker"
                  config {
                    image = "busybox"
                    args = ["sh", "-xc", <<EOF
                        for i in $(seq 10); do echo $i; sleep 1; done
                        exit 123
                        EOF
                    ]
                  }
                }
              }
            }
            """
        self._run(spec, stdout=None)

    def test(self):
        now = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        spec = f"""
        job "test-nomad-watch" {{
            datacenters = ["*"]
            type = "batch"
            reschedule {{
                attempts = 0
            }}
            """
        for i in range(2):
            spec += f"""
                group "{now}_group_{i}" {{
                    restart {{
                        attempts = 0
                    }}
                """
            for j in range(2):
                spec += f"""
                  task "{now}_task_{i}_{j}" {{
                    driver = "docker"
                    config {{
                      image = "busybox"
                      args = ["sh", "-xc", <<EOF
                        echo {now} group_{i} task_{i}_{j} START
                        sleep 1
                        echo {now} group_{i} task_{i}_{j} STOP
                        EOF
                      ]
                    }}
                  }}
                  """
            spec += f"""
                }}
                """
        spec += f"""
        }}
        """
        print(f"Running nomad job with {now}")
        output = self._run(spec)
        assert now in output

    def _run(self, spec: str, stdout: Any = subprocess.PIPE):
        spec = subprocess.check_output("nomad fmt -".split(), input=spec, text=True)
        rr = run(
            [
                sys.argv[0],
                *(["--purge"] if args.purge else []),
                *(["-v"] if args.verbose else []),
                *(["-v"] if args.verbose > 1 else []),
                "run",
                "-",
            ],
            input=spec,
            stdout=stdout,
            check=False,
        )
        if rr.returncode not in [0, 123]:
            exit(254)
        output = rr.stdout
        if output:
            outputbins: Dict[str, List[str]] = {}
            for line in output.splitlines():
                line = line.split(":", 1)
                if len(line) == 2:
                    outputbins.setdefault(line[0], []).append(line[1])
                else:
                    print(line)
            for k, v in outputbins.items():
                print(f"------ {k} -----")
                for l in v:
                    print(l)
        return output


###############################################################################

mynomad = nomadlib.Nomad()


def nomad_start_job(input: str) -> str:
    """Start a job from input file input."""
    if shutil.which("nomad"):
        jsonarg = "-json" if args.json else ""
        cmd = shlex.split(
            f"nomad job run -detach -verbose {jsonarg} {shlex.quote(input)}"
        )
        try:
            rr = run(
                cmd,
                stdin=None if input == "-" else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            # nomad will print its error, we can just exit
            exit(e.returncode)
        data = rr.stdout.strip()
        for line in data.splitlines():
            log.info(line)
        evalid = next(
            x.split(" ", 3)[-1] for x in data.splitlines() if "eval" in x.lower()
        ).strip()
    else:
        intxt = sys.stdin if input == "-" else open(input)
        txt: str = intxt.read()
        evalid = mynomad.start_job(txt)["EvalID"]
    return evalid


###############################################################################


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
    if not sys.stdout.isatty():
        return empty
    tputscript = "\n".join(tputdict.values()).replace("\n", "\nlongname\nlongname\n")
    try:
        longname = subprocess.check_output(f"tput longname".split(), text=True)
        ret = subprocess.run(
            "tput -S".split(), input=tputscript, stdout=subprocess.PIPE, text=True
        ).stdout
    except subprocess.CalledProcessError as e:
        return empty
    retarr = ret.split(f"{longname}{longname}")
    assert len(tputdict.keys()) == len(
        retarr
    ), f"Could not split tput -S output into parts"
    return {k: v for k, v in zip(tputdict.keys(), retarr)}


COLORS = _init_colors()


@dataclasses.dataclass(frozen=True)
class LogFormat:
    alloc: str
    stderr: str
    stdout: str
    app: str

    @classmethod
    def mk(cls, prefix: str):
        return cls(
            f"%(cyan)s{prefix}A %(now)s %(message)s%(reset)s",
            f"%(orange)s{prefix}E %(message)s%(reset)s",
            f"{prefix}O %(message)s",
            "%(blue)s%(module)s:%(lineno)03d: %(message)s%(reset)s",
        )

    def astuple(self):
        return dataclasses.astuple(self)


log_formats: Dict[str, LogFormat] = {
    "default": LogFormat.mk("%(allocid).6s:%(group)s:%(task)s:"),
    "long": LogFormat.mk("%(allocid)s:%(group)s:%(task)s:"),
    "short": LogFormat.mk("%(task)s:"),
    "onepart": LogFormat.mk(""),
}


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
        }

    def _log(self, fmt, **kvargs: Any):
        print(fmt % self._params(kvargs), flush=True)

    def log_alloc(self, now: datetime.datetime, message: str):
        self._log(args.log_format_alloc, now=now, message=message)

    def log_task(self, stderr: bool, message: str):
        self._log(
            args.log_format_stderr if stderr else args.log_format_stdout,
            message=message,
        )


###############################################################################


class Logger(threading.Thread):
    """Represents a single logging stream from Nomad. Such stream is created separately for stdout and stderr."""

    def __init__(self, tk: TaskKey, stderr: bool):
        super().__init__(name=f"{tk}{1 + int(stderr)}")
        self.tk = tk
        self.stderr: bool = stderr
        self.exitevent = threading.Event()
        self.ignoredlines = []
        self.first = True

    @staticmethod
    def _read_json_stream(stream: requests.Response):
        txt: str = ""
        for data in stream.iter_content(decode_unicode=True):
            for c in data:
                txt += c
                # Nomad is consistent, the jsons are flat.
                if c == "}":
                    try:
                        ret = json.loads(txt)
                        # log.debug(f"RECV: {ret}")
                        yield ret
                    except json.JSONDecodeError as e:
                        log.warn(f"error decoding json: {txt} {e}")
                    txt = ""

    def _taskout(self, lines: Optional[List[str]] = None):
        lines = lines or []
        if self.ignoretime and (self.first or time.time() < self.ignoretime):
            self.first = False
            self.ignoredlines += lines
            self.ignoredlines = self.ignoredlines[: args.lines]
        else:
            if self.ignoretime:
                lines = self.ignoredlines
                self.ignoredlines = []
                self.ignoretime = 0
            for line in lines:
                line = line.rstrip()
                self.tk.log_task(self.stderr, line)

    def run(self):
        self.ignoretime = 0 if args.lines < 0 else (time.time() + args.lines_timeout)
        with mynomad.stream(
            f"client/fs/logs/{self.tk.allocid}",
            params={
                "task": self.tk.task,
                "type": "stderr" if self.stderr else "stdout",
                "follow": True,
                "origin": "end" if self.ignoretime else "start",
                "offset": 50000 if self.ignoretime else 0,
            },
        ) as stream:
            for event in self._read_json_stream(stream):
                if event:
                    line64: Optional[str] = event.get("Data")
                    if line64:
                        lines = base64.b64decode(line64.encode()).decode().splitlines()
                        self._taskout(lines)
                else:
                    # Nomad json stream periodically sends empty {}.
                    # No idea why, but I can implement timeout.
                    self._taskout()
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
        ths: List[Logger] = []
        if args_stream.out:
            ths.append(Logger(tk, False))
        if args_stream.err:
            ths.append(Logger(tk, True))
        assert len(ths)
        for th in ths:
            th.start()
        return ths

    def notify(
        self, alloc: nomadlib.Alloc, tk: TaskKey, task: nomadlib.AllocTaskStates
    ):
        events = task.Events
        if args_stream.alloc:
            for e in events:
                msg = e.DisplayMessage
                time = e.Time
                if time and time not in self.messages and msg:
                    self.messages.add(time)
                    tk.log_alloc(ns2dt(time), msg)
        if (
            not self.loggers
            and task["State"] in ["running", "dead"]
            and task.find_event("Started")
        ):
            self.loggers += self._create_loggers(tk)
            if task.State == "dead":
                # If the task is already finished, give myself max 3 seconds to query all the logs.
                # This is to reduce the number of connections.
                threading.Timer(3, self.stop)
        if self.exitcode is None and task["State"] == "dead":
            # Assigns None if Terminated event not found
            self.exitcode = (task.find_event("Terminated") or {}).get("ExitCode")
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
        for taskname, task in alloc.taskstates().items():
            if args.task and not args.task.search(taskname):
                continue
            tk = TaskKey(alloc.ID, alloc.NodeName, alloc.TaskGroup, taskname)
            self.taskhandlers.setdefault(tk, TaskHandler()).notify(alloc, tk, task)


class AllocWorkers(Dict[str, AllocWorker]):
    """An containers for storing a map of allocation workers"""

    def notify(self, alloc: nomadlib.Alloc):
        self.setdefault(alloc.ID, AllocWorker()).notify(alloc)

    def stop(self):
        for w in self.values():
            for th in w.taskhandlers.values():
                th.stop()

    def join(self):
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
        anyunfinished = any(v == -1 for v in exitcodes)
        if anyunfinished or len(self) == 0:
            return 4
        onlyonetask = len(exitcodes) == 1
        if onlyonetask:
            return exitcodes[0]
        allfailed = all(v != 0 for v in exitcodes)
        if allfailed:
            return 3
        anyfailed = any(v != 0 for v in exitcodes)
        if anyfailed:
            return 2
        return 0


###############################################################################


Event = nomadlib.Event
Topic = nomadlib.Topic


class Db(threading.Thread):
    """Represents relevant state cache from Nomad database"""

    def __init__(
        self,
        topics: List[str],
        filter_event_cb: Callable[[Event], bool] = lambda _: True,
        init_cb: Callable[[], Iterable[Event]] = lambda: [],
    ):
        """
        :param topics: Passed to nomad evnet stream as topics.
        :param filter_event_cb: Filter the events from Nomad.
        :param init_cb: Get initial data from Nomad.
        """
        super().__init__(name="db", daemon=True)
        self.topics: List[str] = topics
        self.filter_event_cb: Callable[[Event], bool] = filter_event_cb
        self.init_cb: Optional[Callable[[], Iterable[Event]]] = init_cb
        self.job: Optional[nomadlib.Job] = None
        self.allocations: Dict[str, nomadlib.Alloc] = {}
        self.evaluations: Dict[str, nomadlib.Eval] = {}
        self._stopevent = threading.Event()
        self.queue: queue.Queue[Optional[Event]] = queue.Queue()
        assert self.topics
        assert not any(not x for x in topics)

    def start(self):
        assert (
            mynomad.namespace
        ), "Nomad namespace has to be set before starting to listen"
        super().start()

    def run(self):
        log.debug(f"Starting listen Nomad stream with {' '.join(self.topics)}")
        with mynomad.stream(
            "event/stream",
            params={"topic": self.topics},
        ) as stream:
            for line in stream.iter_lines():
                data = json.loads(line)
                events: List[dict] = data.get("Events", [])
                for event in events:
                    e = Event(
                        Topic[event["Topic"]],
                        event["Payload"][event["Topic"]],
                        stream=True,
                    )
                    # log.debug(f"RECV EVENT: {e}")
                    self.queue.put(e)
                if self._stopevent.is_set():
                    break
        log.debug("Listen Nomad stream exiting")
        self.queue.put(None)

    def handle_event(self, e: Event) -> bool:
        if self._filter_old_event(e):
            if self.filter_event_cb(e):
                log.debug(f"EVENT: {e}")
                self._add_event_to_db(e)
                return True
            else:
                # log.debug(f"USER FILTERED: {e}")
                pass
        else:
            # log.debug(f"OLD EVENT: {e}")
            pass
        return False

    def _add_event_to_db(self, e: Event):
        if e.topic == Topic.Job:
            self.job = nomadlib.Job(e.data)
        elif e.topic == Topic.Evaluation:
            self.evaluations[e.data["ID"]] = nomadlib.Eval(e.data)
        elif e.topic == Topic.Allocation:
            self.allocations[e.data["ID"]] = nomadlib.Alloc(e.data)

    @staticmethod
    def apply_filters(
        e: Event,
        job_filter: Callable[[nomadlib.Job], bool],
        eval_filter: Callable[[nomadlib.Eval], bool],
        alloc_filter: Callable[[nomadlib.Alloc], bool],
    ) -> bool:
        return e.apply(job_filter, eval_filter, alloc_filter)

    def _filter_old_event(self, e: Event):
        job_filter: Callable[[nomadlib.Job], bool] = (
            lambda job: self.job is None or job.ModifyIndex > self.job.ModifyIndex
        )
        eval_filter: Callable[[nomadlib.Eval], bool] = (
            lambda eval: eval.ID not in self.evaluations
            or eval.ModifyIndex > self.evaluations[eval.ID].ModifyIndex
        )
        alloc_filter: Callable[[nomadlib.Alloc], bool] = (
            lambda alloc: alloc.ID not in self.allocations
            or alloc.ModifyIndex > self.allocations[alloc.ID].ModifyIndex
        )
        return e.data["Namespace"] == mynomad.namespace and self.apply_filters(
            e, job_filter, eval_filter, alloc_filter
        )

    def stop(self):
        log.debug("Stopping listen Nomad stream")
        self._stopevent.set()

    def events(self) -> Iterable[Event]:
        assert self.is_alive(), "Thread not alive"
        if self.init_cb:
            for event in self.init_cb():
                if self.handle_event(event):
                    yield event
        self.init_cb = None
        log.debug("Starting getting events from thread")
        while not self.queue.empty() or (
            self.is_alive() and not self._stopevent.is_set()
        ):
            event = self.queue.get()
            if event is None:
                break
            if self.handle_event(event):
                yield event


###############################################################################


def nomad_watch_eval(evalid: str):
    assert isinstance(evalid, str), f"not a string: {evalid}"
    db = Db(
        topics=[
            f"Evaluation:{evalid}",
        ],
        filter_event_cb=lambda e: e.topic == Topic.Evaluation
        and e.data["ID"] == evalid,
        init_cb=lambda: [Event(Topic.Evaluation, mynomad.get(f"evaluation/{evalid}"))],
    )
    db.start()
    log.info(f"Waiting for evaluation {evalid}")
    eval_ = None
    for event in db.events():
        eval_ = event.data
        if eval_["Status"] != "pending":
            break
    db.stop()
    assert eval_ is not None
    assert (
        eval_["Status"] == "complete"
    ), f"Evaluation {evalid} did not complete: {eval_.get('StatusDescription')}"
    FailedTGAllocs = eval_.get("FailedTGAllocs")
    if FailedTGAllocs:
        groups = " ".join(list(FailedTGAllocs.keys()))
        log.info(f"Evaluation {evalid} failed to place groups: {groups}")


def nomad_start_job_and_wait(input: str) -> nomadlib.Job:
    assert isinstance(input, str)
    evalid = nomad_start_job(input)
    eval_: dict = mynomad.get(f"evaluation/{evalid}")
    mynomad.namespace = eval_["Namespace"]
    nomad_watch_eval(evalid)
    jobid = eval_["JobID"]
    return nomadlib.Job(mynomad.get(f"job/{jobid}"))


def nomad_find_job(jobid: str) -> nomadlib.Job:
    jobid = mynomad.find_job(jobid)
    return nomadlib.Job(mynomad.find_last_not_stopped_job(jobid))


###############################################################################


class NomadJobWatcher(ABC, threading.Thread):
    """Watches over a job. Schedules watches over allocations. Spawns loggers."""

    def __init__(self, job: nomadlib.Job):
        super().__init__(name=f"NomadJobWatcher({job['ID']})")
        self.job = job
        self.allocworkers = AllocWorkers()
        self.db = Db(
            topics=[
                f"Job:{self.job.ID}",
                f"Evaluation:{self.job.ID}",
                f"Allocation:{self.job.ID}",
            ],
            filter_event_cb=self.db_filter_event_job,
            init_cb=self.db_init_cb,
        )
        # I am using threading.Event because you can't handle KeyboardInterrupt while Thread.join().
        self.done = threading.Event()
        # If set to True, menas that JobNotFound is not an error - the job was removed.
        self.purged = threading.Event()

    def db_init_cb(self):
        """Db initialization callback"""

        job: dict = mynomad.get(f"job/{self.job.ID}")
        evaluations: List[dict] = mynomad.get(f"job/{self.job.ID}/evaluations")
        allocations: List[dict] = mynomad.get(f"job/{self.job.ID}/allocations")
        if not allocations:
            log.info(f"Job {self.job.description()} has no allocations")
        for e in evaluations:
            yield Event(Topic.Evaluation, e)
        for a in allocations:
            yield Event(Topic.Allocation, a)
        yield Event(Topic.Job, job)

    def db_filter_event_jobid(self, e: Event):
        return Db.apply_filters(
            e,
            lambda job: job.ID == self.job.ID,
            lambda eval: eval.JobID == self.job.ID,
            lambda alloc: alloc["JobID"] == self.job.ID,
        )

    def db_filter_event_job(self, e: Event):
        job_filter: Callable[[nomadlib.Job], bool] = lambda _: True
        eval_filter: Callable[[nomadlib.Eval], bool] = lambda eval: (
            # Either all, or the JobModifyIndex has to be greater.
            args.all
            or (
                "JobModifyIndex" in eval
                and eval.JobModifyIndex >= self.job.JobModifyIndex
            )
        )
        alloc_filter: Callable[[nomadlib.Alloc], bool] = lambda alloc: (
            args.all
            # If allocation has JobVersion, then it has to match the version in the job.
            or ("JobVersion" in alloc and alloc.JobVersion >= self.job.Version)
            or (
                # If the allocation has no JobVersion, find the maching evaluation.
                # The JobModifyIndex from the evalution has to match.
                alloc.EvalID in self.db.evaluations
                and self.db.evaluations[alloc.EvalID].JobModifyIndex
                >= self.job.JobModifyIndex
            )
        )
        return self.db_filter_event_jobid(e) and Db.apply_filters(
            e, job_filter, eval_filter, alloc_filter
        )

    @property
    def allocs(self):
        return list(self.db.allocations.values())

    @abstractmethod
    def until_cb(self) -> bool:
        """Overloaded callback to call to determine if we should finish watching the job"""
        raise NotImplementedError()

    def _watch_job(self):
        log.info(f"Watching job {self.job.description()}")
        no_follow_timeend = time.time() + args.shutdown_timeout
        for event in self.db.events():
            if event.topic == Topic.Allocation:
                alloc = nomadlib.Alloc(event.data)
                # for alloc in self.db.allocations.values():
                self.allocworkers.notify(alloc)
            if (not args.all and self.until_cb()) or (
                args.no_follow and time.time() > no_follow_timeend
            ):
                break

    def run(self):
        self.db.start()
        try:
            self._watch_job()
        except nomadlib.JobNotFound:
            # The job was purged and is not missing. All fine!
            pass
        finally:
            self.close()
            self.done.set()

    def close(self):
        log.debug("close()")
        self.db.stop()
        self.allocworkers.stop()
        # Logs stream outputs empty {} which allows to handle timeouts.
        self.allocworkers.join()
        # Not joining self.db - neither requests nor stream API allow for timeouts.
        # self.db.join()
        mynomad.session.close()

    def exitcode(self) -> int:
        assert self.done.is_set(), f"Watcher not finished"
        return self.allocworkers.exitcode()


class NomadJobWatcherUntilFinished(NomadJobWatcher):
    """Watcher a job until the job is dead"""

    def until_cb(self) -> bool:
        if len(self.allocworkers) == 0:
            jobjson = self.db.job
            if jobjson is not None:
                if jobjson.Version != self.job.Version:
                    log.info(f"New version of job {self.job.description()} posted")
                    return True
                if jobjson.Status == "dead":
                    log.info(
                        f"Job {self.job.description()} is dead with no allocations"
                    )
                    return True
        else:
            allallocsfinished = all(alloc.is_finished() for alloc in self.allocs)
            if allallocsfinished:
                log.info(
                    f"All {len(self.allocs)} allocations of {self.job.description()} are finished"
                )
                return True
        return False

    def stop_job(self, purge: bool):
        if purge:
            self.purged.set()
        mynomad.stop_job(self.job.ID, purge)

    def run_till_end(self):
        self.run()
        exit(self.exitcode())


class NomadJobWatcherUntilStarted(NomadJobWatcher):
    """Watches a job until the job is started"""

    def until_cb(self) -> bool:
        runningallocsids = list(self.allocworkers.keys())
        tasks = [
            task
            for allocid in runningallocsids
            for task in self.db.allocations[allocid].taskstates().values()
        ]
        alltasksstarted = all(task.find_event("Started") for task in tasks)
        if alltasksstarted and len(self.allocworkers) and len(tasks):
            allocsstr = " ".join(runningallocsids)
            log.info(
                f"Allocations {allocsstr} started {len(runningallocsids)} allocations with {len(tasks)} tasks"
            )
            return True
        return False

    def run_till_end(self):
        self.run()


###############################################################################


def nomad_find_namespace(prefix: str):
    """Finds a nomad namespace by prefix"""
    if prefix == "*":
        return prefix
    namespaces = mynomad.get("namespaces")
    names = [x["Name"] for x in namespaces]
    namesstr = " ".join(names)
    matchednames = [x for x in names if x.startswith(prefix)]
    matchednamesstr = " ".join(matchednames)
    assert (
        len(matchednames) > 0
    ), f"Couldn't find namespace maching prefix {prefix} from {namesstr}"
    assert (
        len(matchednames) < 2
    ), f"Prefix {prefix} matched multiple namespaces: {matchednamesstr}"
    return matchednames[0]


###############################################################################


def complete_set_namespace(ctx: click.Context):
    namespace = ctx.params.get("namespace")
    if namespace:
        try:
            os.environ["NOMAD_NAMESPACE"] = nomad_find_namespace(namespace)
        except Exception:
            pass


def completor(cb: Callable[[], Iterable[str]]):
    def completor_cb(ctx: click.Context, param: str, incomplete: str):
        complete_set_namespace(ctx)
        try:
            return [x for x in cb() if x.startswith(incomplete)]
        except Exception:
            pass

    return completor_cb


class JobPath:
    jobname: str
    group: Optional[Pattern]
    task: Optional[Pattern]

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
    def complete(ctx: click.Context, param: str, incomplete: str):
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
            except requests.HTTPError as e:
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
    Run a Nomad job in Nomad and then print logs to stdout and wait for the job to finish.
    Made for running batch commands and monitoring them until they are done.
    The exit code is 0 if the job was properly done and all tasks exited with 0 exit code.
    If the job has only one task, exit with the task exit status.
    Othwerwise, if all tasks exited failed, exit with 3.
    If there are failed tasks, exit with 2.
    If there was a python exception, standard exit code is 1.
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
@click.option(
    "-N",
    "--namespace",
    help="Finds Nomad namespace matching given prefix and sets NOMAD_NAMESPACE environment variable.",
    envvar="NOMAD_NAMESPACE",
    show_default=True,
    shell_complete=completor(lambda: (x["Name"] for x in mynomad.get("namespaces"))),
)
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
    "-s",
    "--stream",
    type=click.Choice("all alloc a stdout stderr out err o e 1 2".split()),
    default=["all"],
    multiple=True,
    help="Print only messages from allocation and stdout or stderr of the task. This option is cummulative.",
)
@click.option("-v", "--verbose", count=True, help="Be verbose")
@click.option(
    "--json",
    is_flag=True,
    help="job input is in json form, passed to nomad command with --json",
)
@click.option(
    "--stop",
    is_flag=True,
    help="In run mode, make sure to stop the job before exit.",
)
@click.option(
    "--purge",
    is_flag=True,
    help="In run mode, stop and purge the job before exiting.",
)
@click.option(
    "-n",
    "--lines",
    default=-1,
    show_default=True,
    type=int,
    help="Sets the tail location in best-efforted number of lines relative to the end of logs",
)
@click.option(
    "--lines-timeout",
    default=0.5,
    show_default=True,
    type=float,
    help="When using --lines the number of lines is best-efforted by ignoring lines for specific time",
)
@click.option(
    "--shutdown_timeout",
    default=2,
    show_default=True,
    type=float,
    help="Rather leave at 2 if you want all the logs.",
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
    "--log-format-alloc", default=log_formats["default"].alloc, show_default=True
)
@click.option(
    "--log-format-stderr", default=log_formats["default"].stderr, show_default=True
)
@click.option(
    "--log-format-stdout", default=log_formats["default"].stdout, show_default=True
)
@click.option("-l", "--log-long", is_flag=True, help="Log full allocation id")
@click.option(
    "-S",
    "--log-short",
    is_flag=True,
    help="Make the format short by logging only task name.",
)
@click.option(
    "-1",
    "--log-onepart",
    is_flag=True,
    help="Make the format short by logging only task name.",
)
@click.help_option("-h", "--help")
@click.pass_context
def cli(ctx, **kvargs):
    global args
    args = argparse.Namespace(**ctx.params)
    #
    if args.verbose > 1:
        http_client.HTTPConnection.debuglevel = 1
    global args_stream
    args_stream = argparse.Namespace(
        err=any(s in "all stderr err e 2".split() for s in args.stream),
        out=any(s in "all stdout out o 1".split() for s in args.stream),
        alloc=any(s in "all alloc a".split() for s in args.stream),
    )
    if args.follow:
        args.lines = 10
        args.all = True
    if args.namespace:
        os.environ["NOMAD_NAMESPACE"] = nomad_find_namespace(args.namespace)
    assert [args.log_long, args.log_short, args.log_onepart].count(
        True
    ) <= 1, f"Only one --log-* argument can be specified at a time"
    (
        args.log_format_alloc,
        args.log_format_stderr,
        args.log_format_stdout,
        logging_format,
    ) = log_formats[
        "long"
        if args.log_long
        else "short"
        if args.log_short
        else "onepart"
        if args.log_onepart
        else "default"
    ].astuple()
    logging.basicConfig(
        format=logging_format,
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    global log
    log = logging.LoggerAdapter(log, COLORS)


cli_jobid = click.argument(
    "jobid",
    shell_complete=completor(lambda: (x["ID"] for x in mynomad.get("jobs"))),
)
cli_jobfile = click.argument(
    "jobfile",
    type=click.File(lazy=True),
    callback=lambda _, __, x: "-" if x.name == "<stdin>" else x.name,
)

###############################################################################


@cli.command("alloc", help="Watch over specific allocation")
@click.argument(
    "allocid",
    shell_complete=completor(lambda: (x["ID"] for x in mynomad.get("allocations"))),
)
def mode_alloc(allocid):
    allocs = mynomad.get(f"allocations", params={"prefix": allocid})
    assert len(allocs) > 0, f"Allocation with id {allocid} not found"
    assert len(allocs) < 2, f"Multiple allocations found starting with id {allocid}"
    alloc = nomadlib.Alloc(allocs[0])
    mynomad.namespace = alloc.Namespace
    allocid = alloc.ID
    log.info(f"Watching allocation {allocid}")
    db = Db(
        topics=[f"Allocation:{alloc.JobID}"],
        filter_event_cb=lambda e: e.topic == Topic.Allocation
        and e.data["ID"] == allocid,
        init_cb=lambda: [Event(Topic.Allocation, mynomad.get(f"allocation/{allocid}"))],
    )
    allocworkers = AllocWorkers()
    db.start()
    try:
        for event in db.events():
            if event.topic == Topic.Allocation:
                alloc = nomadlib.Alloc(event.data)
                allocworkers.notify(alloc)
                if alloc.is_finished():
                    log.info(
                        f"Allocation {allocid} has status {alloc.ClientStatus}. Exiting."
                    )
                    break
            if args.no_follow:
                break
    finally:
        db.stop()
        allocworkers.stop()
        db.join()
        allocworkers.join()
        exit(allocworkers.exitcode())


@cli.command("run", help="Run a Nomad job and then watch over it until it is finished.")
@cli_jobfile
def mode_run(jobfile):
    jobinit = nomad_start_job_and_wait(jobfile)
    do = NomadJobWatcherUntilFinished(jobinit)
    do.start()
    try:
        do.done.wait()
    finally:
        # On exception, stop or purge the job if needed.
        if args.purge or args.stop:
            do.stop_job(args.purge)
        do.done.wait()
    exit(do.exitcode())


@cli.command("job", help="Watch a Nomad job, show its logs and events.")
@cli_jobid
def mode_job(jobid):
    jobinit = nomad_find_job(jobid)
    NomadJobWatcherUntilFinished(jobinit).run_till_end()


@cli.command(
    "start", help="Start a Nomad Job and watch it until all allocations are running"
)
@cli_jobfile
def mode_start(jobfile):
    jobinit = nomad_start_job_and_wait(jobfile)
    NomadJobWatcherUntilStarted(jobinit).run_till_end()


@cli.command("started", help="Watch a Nomad job until the job is started.")
@cli_jobid
def mode_started(jobid):
    jobinit = nomad_find_job(jobid)
    NomadJobWatcherUntilStarted(jobinit).run_till_end()


@cli.command(
    "stop",
    help="Stop a Nomad job and then watch the job until it is stopped - has no running allocations.",
)
@cli_jobid
def mode_stop(jobid: str):
    jobinit = nomad_find_job(jobid)
    do = NomadJobWatcherUntilFinished(jobinit)
    do.start()
    do.stop_job(args.purge)
    do.done.wait()


@cli.command(
    "stopped",
    help="Watch a Nomad job until the job is stopped - has not running allocation.",
)
@cli_jobid
def mode_stopped(jobid):
    jobinit = nomad_find_job(jobid)
    NomadJobWatcherUntilFinished(jobinit).run_till_end()


@cli.command("test", hidden=True)
@click.argument("mode")
def mode_test(mode):
    Test(mode)


###############################################################################

if __name__ == "__main__":
    try:
        cli.main()
    finally:
        mynomad.session.close()
