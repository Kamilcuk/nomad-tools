#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import base64
import dataclasses
import datetime
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from http import client as http_client
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import click
import requests

from . import nomadlib

###############################################################################

log = logging.getLogger("nomad-watch")


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
              group "cache" {
                restart { attempts = 0 }
                task "redis" {
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
        output = run(
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
        ).stdout
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

mynomad = nomadlib.Conn()


def nomad_start_job(input: str) -> str:
    """Start a job from input file input."""
    if shutil.which("nomad"):
        jsonarg = "-json" if args.json else ""
        cmd = shlex.split(
            f"nomad job run -detach -verbose {jsonarg} {shlex.quote(input)}"
        )
        try:
            rr = run(cmd, stdout=subprocess.PIPE if input == "-" else None)
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


@dataclasses.dataclass(frozen=True)
class TaskKey:
    """Represent data to unique identify a task"""

    allocid: str
    nodename: str
    group: str
    task: str

    def _param(self, param: Dict[str, Any] = {}, **kvargs: Any) -> Dict[str, Any]:
        ret = dict(param)
        ret.update(kvargs)
        ret.update(dataclasses.asdict(self))
        return ret

    def log_alloc(self, now: datetime.datetime, line: str):
        print("%(group)s:%(task)s:%(now)s %(line)s" % self._param(now=now, line=line))

    def log_task(self, stream: str, line: str):
        print(
            "%(group)s:%(task)s:%(stream)s: %(line)s"
            % self._param(stream=stream, line=line)
        )


###############################################################################


class Logger(threading.Thread):
    """Represents a single logging stream from Nomad. Such stream is created separately for stdout and stderr."""

    def __init__(self, tk: TaskKey, stderr: bool):
        super().__init__()
        self.tk = tk
        self.stderr: bool = stderr
        self.exitevent = threading.Event()
        self.ignoredlines = []
        self.first = True

    def _streamtype(self):
        return "err" if self.stderr else "out"

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
                        log.debug(f"RECV: {ret}")
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
                self.tk.log_task(self._streamtype(), line)

    def run(self):
        self.ignoretime = 0 if args.lines < 0 else (time.time() + 0.5)
        with mynomad.stream(
            f"client/fs/logs/{self.tk.allocid}",
            params={
                "task": self.tk.task,
                "type": f"std{self._streamtype()}",
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
            th.daemon = True
            th.start()
        return ths

    def notify(self, alloc: dict, tk: TaskKey, task: nomadlib.AllocTaskStates):
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
        for taskname, task in alloc.TaskStates.items():
            if args.task and not args.task.search(taskname):
                continue
            tk = TaskKey(alloc["ID"], alloc["NodeName"], alloc["TaskGroup"], taskname)
            self.taskhandlers.setdefault(tk, TaskHandler()).notify(alloc, tk, task)


class AllocWorkers(Dict[str, AllocWorker]):
    def notify(self, alloc: nomadlib.Alloc):
        self.setdefault(alloc["ID"], AllocWorker()).notify(alloc)

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
        timeend = time.time() + 2
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
            return -1
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


def nomad_start_job_and_wait(input: str):
    assert isinstance(input, str)
    evalid = nomad_start_job(input)
    eval_: dict = mynomad.get(f"evaluation/{evalid}")
    mynomad.namespace = eval_["Namespace"]
    nomad_watch_eval(evalid)
    jobid = eval_["JobID"]
    return mynomad.get(f"job/{jobid}")


def nomad_watch_eval(evalid: str):
    assert isinstance(evalid, str), f"not a string: {evalid}"
    while True:
        eval_ = mynomad.get(f"evaluation/{evalid}")
        if eval_["Status"] != "pending":
            break
        log.info(f"Waiting for evaluation {evalid}")
        time.sleep(1)
    assert (
        eval_["Status"] == "complete"
    ), f"Evaluation {evalid} did not complete: {eval_.get('StatusDescription')}"
    FailedTGAllocs = eval_.get("FailedTGAllocs")
    if FailedTGAllocs:
        groups = " ".join(list(FailedTGAllocs.keys()))
        log.info(f"Evaluation {evalid} failed to place groups: {groups}")
    return eval_


###############################################################################


class NomadWatchJob:
    """The main class of nomad-watch for watching over stuff"""

    def __init__(self, job: dict):
        self.job = nomadlib.Job(job)
        self.allocworkers = AllocWorkers()
        self.allocs: List[nomadlib.Alloc] = []

    def job_finished_sane(self) -> bool:
        if len(self.allocworkers) == 0:
            jobjson = nomadlib.Job(mynomad.get(f"job/{self.job.ID}"))
            if jobjson.Version != self.job.Version:
                log.info(f"New version of job {self.job.description()} posted")
                return True
            if jobjson.Status == "dead":
                log.info(f"Job {self.job.description()} is dead with no allocations")
                return True
            log.info(f"Job {self.job.description()} has no allocations")
        else:
            allallocsfinished = all(
                alloc.ClientStatus not in ["pending", "running"]
                for alloc in self.allocs
            )
            if allallocsfinished:
                log.info(
                    f"All {len(self.allocs)} allocations of {self.job.description()} are finished"
                )
                return True
        return False

    def job_started(self):
        for alloc in self.allocs:
            alltasksstarted = all(
                task.find_event("Started") for task in alloc.TaskStates.values()
            )
            if alltasksstarted:
                log.info(f"Allocation {alloc.ID} started {len(alloc.TaskStates)} tasks")
                return True
        return False

    def watch_job(self, finished: Optional[Callable[[], bool]] = None):
        log.info(f"Watching job {self.job.description()}")
        finished = finished or self.job_finished_sane
        while True:
            allocs: List[dict] = mynomad.get(f"job/{self.job.ID}/allocations")
            # Filter relevant allocations only.
            self.allocs = [nomadlib.Alloc(a) for a in allocs]
            self.allocs = [
                alloc
                for alloc in self.allocs
                if alloc.ID in self.allocworkers.keys()
                or alloc.JobVersion == self.job.Version
                or args.all
            ]
            for alloc in self.allocs:
                self.allocworkers.notify(alloc)
            if (not args.all and finished()) or args.no_follow:
                break
            time.sleep(1)

    def close(self) -> int:
        log.debug("close()")
        self.allocworkers.stop()
        self.allocworkers.join()
        mynomad.session.close()
        return self.allocworkers.exitcode()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()


def find_nomad_namespace(prefix: str):
    """Finds a nomad namespace by prefix"""
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


def completor(cb: Callable[[], Iterable[str]]):
    def completor_cb(ctx, param, incomplete):
        namespace = ctx.params.get("namespace")
        if namespace:
            try:
                os.environ["NOMAD_NAMESPACE"] = find_nomad_namespace(namespace)
            except Exception:
                pass
        try:
            return [x for x in cb() if x.startswith(incomplete)]
        except Exception:
            pass

    return completor_cb


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

    nomad-watch --all --task redis -N services job redis

    """,
    epilog="""
    Written by Kamil Cukrowski 2023. Jointly under Beerware and MIT Licenses.
    """,
)
@click.option(
    "-N",
    "--namespace",
    help="Finds Nomad namespace matching given prefix and sets NOMAD_NAMESPACE environment variable.",
    shell_complete=completor(lambda: (x["Name"] for x in mynomad.get("namespaces"))),
)
@click.option(
    "-a",
    "--all",
    is_flag=True,
    help="""
        Do not exit after the current job monitoring is done.
        Instead, watch endlessly for any existing and new allocations of a job.
        """,
)
@click.option(
    "-s",
    "--stream",
    type=click.Choice("all alloc stdout stderr out err 1 2".split()),
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
    type=int,
    help="Sets the tail location in best-efforted number of lines relative to the end of logs",
)
@click.option(
    "-f",
    "--follow",
    is_flag=True,
    help="Shorthand for --all --lines=10 to be like in tail -f.",
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
@click.help_option("-h", "--help")
@click.pass_context
def cli(ctx, **kvargs):
    global args
    args = argparse.Namespace(**ctx.params)
    #
    logging.basicConfig(
        format="%(module)s:%(lineno)03d: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    if args.verbose > 1:
        http_client.HTTPConnection.debuglevel = 1
    global args_stream
    args_stream = argparse.Namespace(
        err=any(s in "all stderr err 2".split() for s in args.stream),
        out=any(s in "all stdout out 1".split() for s in args.stream),
        alloc=any(s in "all alloc".split() for s in args.stream),
    )
    if args.follow:
        args.lines = 10
        args.all = True
    if args.namespace:
        os.environ["NOMAD_NAMESPACE"] = find_nomad_namespace(args.namespace)


cli_jobid = click.argument(
    "jobid", shell_complete=completor(lambda: (x["ID"] for x in mynomad.get("jobs")))
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
    allocid = allocs[0]["ID"]
    log.info(f"Watching allocation {allocid}")
    allocworkers = AllocWorkers()
    alloc = None
    try:
        while True:
            alloc = mynomad.get(f"allocation/{allocid}")
            allocworkers.notify(alloc)
            allocclientstatus = alloc["ClientStatus"]
            if allocclientstatus not in ["pending", "running"]:
                log.info(
                    f"Allocation {allocid} has status {allocclientstatus}. Exiting."
                )
                break
            if args.no_follow:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Received interrupt")
    allocworkers.stop()
    allocworkers.join()
    exit(allocworkers.exitcode())


@cli.command("run", help="Will run specified Nomad job and then watch over it.")
@cli_jobfile
def mode_run(jobfile):
    jobinit = nomad_start_job_and_wait(jobfile)
    #
    done = threading.Event()
    do = NomadWatchJob(jobinit)

    def waiter():
        do.watch_job()
        do.allocworkers.join()
        log.debug("Waiter exiting")
        done.set()

    waiterthread = threading.Thread(target=waiter, daemon=True)
    try:
        # First wait for the evaluation.
        waiterthread.start()
        # Waiting on Event, because thread.join does not handle interrupts well.
        done.wait()
    except KeyboardInterrupt:
        log.info("Received interrupt")
    finally:
        # On exception, stop or purge the job if needed.
        if args.purge or args.stop:
            mynomad.stop_job(jobinit["ID"], args.purge)
        if waiterthread and not args.all:
            waiterthread.join()
    exit(do.close())


@cli.command("job", help="Watch over a specific job. Will not stop the job.")
@cli_jobid
def mode_job(jobid):
    jobid = mynomad.find_job(jobid)
    jobinit = mynomad.find_last_not_stopped_job(jobid)
    with NomadWatchJob(jobinit) as do:
        do.watch_job()
    exit(do.close())


@cli.command(
    "start", help="Start a Nomad Job and monitor it until all allocations are running"
)
@cli_jobfile
def mode_start(jobfile):
    jobinit = nomad_start_job_and_wait(jobfile)
    with NomadWatchJob(jobinit) as do:
        do.watch_job(do.job_started)


@cli.command("starting", help="Wait until the job has started")
@cli_jobid
def mode_starting(jobid):
    jobid = mynomad.find_job(jobid)
    jobinit = mynomad.find_last_not_stopped_job(jobid)
    with NomadWatchJob(jobinit) as do:
        do.watch_job(do.job_started)


@cli.command("stop", help="Stop a Nomad job and then wait until it is dead")
@cli_jobid
def mode_stop(jobid: str):
    jobid = mynomad.find_job(jobid)
    resp = mynomad.stop_job(jobid, args.purge)
    nomad_watch_eval(resp["EvalID"])
    try:
        jobinit = mynomad.get(f"job/{jobid}")
        with NomadWatchJob(jobinit) as do:
            do.watch_job()
    except requests.exceptions.HTTPError as e:
        if args.purge and e.response.status_code == 404:
            log.info(f"Job {jobid} removed")
        else:
            raise e


@cli.command("stopping", help="Monitor a Nomad job until it is dead")
@cli_jobid
def mode_stopping(jobid):
    jobid = mynomad.find_job(jobid)
    jobinit = mynomad.get(f"job/{jobid}")
    with NomadWatchJob(jobinit) as do:
        do.watch_job()


@cli.command("test")
@click.argument("mode")
def mode_test(mode):
    Test(mode)


###############################################################################

if __name__ == "__main__":
    try:
        cli.main()
    finally:
        mynomad.session.close()
