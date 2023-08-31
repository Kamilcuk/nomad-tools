#!/usr/bin/env python3

import argparse
import base64
import dataclasses
import datetime
import enum
import json
import logging
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
from http import client as http_client
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import requests
from requests.auth import HTTPBasicAuth

###############################################################################

log = logging.getLogger("nomad-watch")


def run(cmd, *args, check=True, **kvargs):
    cmdtxt: str = " ".join(shlex.quote(x) for x in cmd)
    log.info(f"+ {cmdtxt}")
    return subprocess.run(cmd, *args, check=check, text=True, **kvargs)


def ns2dt(ns: int):
    """Convert nanoseconds to datetime"""
    return datetime.datetime.fromtimestamp(ns // 1000000000)


###############################################################################


class Test:
    """For running internal tests"""

    def __init__(self, mode: str):
        func = getattr(self, mode, None)
        assert func
        func()

    def short(self):
        spec = """
            job "test-nomad-do" {
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
        job "test-nomad-do" {{
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


@dataclasses.dataclass
class MyNomad:
    """Represents connection to Nomad"""

    namespace: Optional[str] = None
    session: requests.Session = dataclasses.field(default_factory=requests.Session)

    def request(
        self,
        method,
        url,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        *args,
        **kvargs,
    ):
        params = dict(params or {})
        assert "namespace" not in params
        params["namespace"] = self.namespace or os.environ.get("NOMAD_NAMESPACE", "*")
        return self.session.request(
            method,
            os.environ.get("NOMAD_ADDR", "http://127.0.0.1:4646") + "/v1/" + url,
            *args,
            auth=(
                HTTPBasicAuth(*os.environ["NOMAD_HTTP_AUTH"].split(":"))
                if "NOMAD_HTTP_AUTH" in os.environ
                else None
            ),
            headers=(
                {"X-Nomad-Token": os.environ["NOMAD_TOKEN"]}
                if "NOMAD_TOKEN" in os.environ
                else None
            ),
            params=params,
            json=json,
            **kvargs,
        )

    def _reqjson(self, mode, url, json: Optional[dict] = None, *args, **kvargs):
        rr = self.request(mode, url, *args, json=json, **kvargs)
        rr.raise_for_status()
        return rr.json()

    def get(self, url, json: Optional[dict] = None, *args, **kvargs):
        return self._reqjson("GET", url, json=json, *args, **kvargs)

    def put(self, *args, **kvargs):
        return self._reqjson("PUT", *args, **kvargs)

    def post(self, *args, **kvargs):
        return self._reqjson("POST", *args, **kvargs)

    def delete(self, *args, **kvargs):
        return self._reqjson("DELETE", *args, **kvargs)

    def stream(self, *args, **kvargs):
        stream = self.request("GET", *args, stream=True, **kvargs)
        stream.raise_for_status()
        return stream


mynomad = MyNomad()

###############################################################################


@dataclasses.dataclass(frozen=True)
class TaskKey:
    """Represent data to unique identify a task"""

    allocid: str
    group: str
    task: str

    def out(self, what: str):
        print(f"{self.group}:{self.task}:{what}")


###############################################################################


class Logger(threading.Thread):
    """Represents a single logging stream from Nomad. Such stream is created separately for stdout and stderr."""

    def __init__(self, tk: TaskKey, stderr: bool):
        super().__init__()
        self.tk = tk
        self.stderr: bool = stderr

    def streamtype(self):
        return "err" if self.stderr else "out"

    @staticmethod
    def _read_json_stream(stream: requests.Response):
        txt = ""
        for c in stream.iter_content(decode_unicode=True):
            txt += c
            # Nomad is consistent, the jsons are flat.
            if c == "}":
                try:
                    ret = json.loads(txt)
                    if ret:
                        log.debug(f"RECV: {ret}")
                        yield ret
                except Exception:
                    log.warn(f"error decoding json: {txt}")
                txt = ""

    def run(self):
        with mynomad.stream(
            f"client/fs/logs/{self.tk.allocid}",
            params={
                "task": self.tk.task,
                "type": f"std{self.streamtype()}",
                "follow": True,
            },
        ) as stream:
            for event in self._read_json_stream(stream):
                line64: Optional[str] = event.get("Data")
                if line64:
                    lines = base64.b64decode(line64.encode()).decode()
                    for line in lines.splitlines():
                        line = line.rstrip()
                        self.tk.out(f"{self.streamtype()}: {line}")


class TaskHandler:
    """A handler for one task. Creates loggers, writes out task events, handle exit conditions"""

    def __init__(self):
        self.loggers: List[Logger] = []
        self.messages: Set[int] = set()
        self.exitcode: Optional[int] = None

    @staticmethod
    def _create_loggers(tk: TaskKey):
        ths: List[Logger] = []
        if not args.only or args.only == "stdout":
            ths.append(Logger(tk, False))
        if not args.only or args.only == "stderr":
            ths.append(Logger(tk, True))
        assert len(ths)
        for th in ths:
            th.daemon = True
            th.start()
        return ths

    @staticmethod
    def _find_event(events: List[dict], type: str) -> dict:
        """Find event in TaskStates task Events. Return empty dict if not found"""
        return next((e for e in events if e["Type"] == type), {})

    def task_event(self, alloc: dict, tk: TaskKey, task: dict):
        events = task.get("Events") or []
        node = alloc["NodeName"]
        for e in events:
            msg = e.get("DisplayMessage")
            time = e.get("Time")
            if time and time not in self.messages and msg:
                self.messages.add(time)
                tk.out(f"{ns2dt(time)} {node} {msg}")
        if (
            not self.loggers
            and task["State"] in ["running", "dead"]
            and self._find_event(events, "Started")
        ):
            self.loggers += self._create_loggers(tk)
        if self.exitcode is None and task["State"] == "dead":
            # Assigns None if Terminated event not found
            self.exitcode = self._find_event(events, "Terminated").get("ExitCode")


class AllocWorker:
    """Represents a worker that prints out and manages state related to one allocation"""

    def __init__(self):
        self.taskhandlers: Dict[TaskKey, TaskHandler] = {}

    def alloc_event(self, alloc: dict):
        """Update the state with alloc"""
        allocid: str = alloc["ID"]
        taskstates: dict = alloc.get("TaskStates") or {}
        for taskname, task in taskstates.items():
            tk = TaskKey(alloc["ID"], alloc["TaskGroup"], taskname)
            self.taskhandlers.setdefault(tk, TaskHandler()).task_event(alloc, tk, task)


###############################################################################


class Topic(enum.StrEnum):
    Job = enum.auto()
    Evaluation = enum.auto()
    Allocation = enum.auto()


@dataclasses.dataclass
class Event:
    topic: Topic
    data: dict
    time: Optional[datetime.datetime] = None

    def __str__(self):
        return f"Event({self.topic.name} id={self.data.get('ID')} modify={self.data.get('ModifyIndex')} status={self.data.get('Status')})"


class Db:
    """Represents relevant state cache from Nomad database"""

    def __init__(self):
        self.allocations: Dict[str, dict] = {}
        self.evaluations: Dict[str, dict] = {}
        self.job: dict = {}
        self.exitevent = threading.Event()
        self._topics: List[str] = []
        self._queue: queue.Queue[Optional[Event]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    def _filter_event(self, e: Event):
        assert self.job
        ret = e.data["Namespace"] == self.job["Namespace"] and (
            (
                e.topic == Topic.Job
                and e.data["ID"] == self.job["ID"]
                and e.data["ModifyIndex"] >= self.job["ModifyIndex"]
            )
            or (
                e.topic == Topic.Evaluation
                and e.data["JobID"] == self.job["ID"]
                and (
                    e.data["ID"] not in self.evaluations
                    or e.data["ModifyIndex"]
                    >= self.evaluations[e.data["ID"]]["ModifyIndex"]
                )
            )
            or (
                e.topic == Topic.Allocation
                and e.data["JobID"] == self.job["ID"]
                and (
                    e.data["ID"] not in self.allocations
                    or e.data["ModifyIndex"]
                    >= self.allocations[e.data["ID"]]["ModifyIndex"]
                )
            )
        )
        return ret

    def _add_event_to_db(self, e: Event):
        if e.topic == Topic.Job:
            self.job = e.data
        elif e.topic == Topic.Evaluation:
            self.evaluations[e.data["ID"]] = e.data
        elif e.topic == Topic.Allocation:
            self.allocations[e.data["ID"]] = e.data
        else:
            assert 0, f"Unknown topic {e.topic}"

    def handle_event(self, e: Event):
        if self._filter_event(e):
            log.debug(f"adding {e}")
            self._add_event_to_db(e)
            return True
        return False

    def _listen_thread(self, *topics: str):
        with mynomad.stream(
            "event/stream",
            params={"topic": topics},
        ) as stream:
            for line in stream.iter_lines():
                data = json.loads(line)
                events: List[dict] = data.get("Events", [])
                for event in events:
                    e = Event(Topic[event["Topic"]], event["Payload"][event["Topic"]])
                    self._queue.put(e)
                if self.exitevent.is_set():
                    self._queue.put(None)
                    break
        log.debug("_listen_thread exit")

    def stop(self):
        log.debug("Stopping listen Nomad stream")
        self.exitevent.set()

    def start(self, *topics: str):
        log.debug(f"Starting listen Nomad stream with {' '.join(topics)}")
        assert mynomad.namespace is not None
        assert topics
        assert not any(not x for x in topics)
        self._thread = threading.Thread(
            target=self._listen_thread, args=topics, daemon=True
        )
        self._thread.start()

    def events(self, init_events: Callable[[], Iterable[Event]]) -> Iterable[Event]:
        assert self._thread is not None, "thread not started"
        for event in init_events():
            if self.handle_event(event):
                yield event
        log.debug("Starting getting events from thread")
        while self.running():
            event = self._queue.get()
            if event is None:
                break
            if self.handle_event(event):
                yield event

    def running(self):
        return (
            not self.exitevent.is_set()
            and self._thread is not None
            and self._thread.is_alive()
        )

    def join(self):
        log.debug(
            f"Joining running={self.running()} exitevent={self.exitevent.is_set()} stream thread"
        )
        if self.running():
            assert self._thread is not None
            self._thread.join()


class NomadWorkers:
    def __init__(self):
        # allocid -> AllocWorker
        self.workers: Dict[str, AllocWorker] = {}
        self.db = Db()

    def notify_alloc_worker(self, alloc: dict):
        self.workers.setdefault(alloc["ID"], AllocWorker()).alloc_event(alloc)

    def _join_workers(self):
        threads: List[Tuple[str, threading.Thread]] = [
            (f"{tk.task}[{i}]", logger)
            for w in self.workers.values()
            for tk, th in w.taskhandlers.items()
            for i, logger in enumerate(th.loggers)
        ]
        log.debug(
            f"joining {len(self.workers)} allocations with {len(threads)} loggers"
        )
        timeend = datetime.datetime.now() + datetime.timedelta(seconds=2)
        for desc, thread in threads:
            timeout = (timeend - datetime.datetime.now()).total_seconds()
            if timeout <= 0:
                log.debug("quitting")
                return
            log.debug(f"joining worker {desc} timeout={timeout}")
            thread.join(timeout=timeout)

    def close(self) -> int:
        self.db.join()
        self._join_workers()
        mynomad.session.close()
        exitcodes: List[int] = [
            # If thread did not return properly in time (?), exit with -1.
            -1 if th.exitcode is None else th.exitcode
            for w in self.workers.values()
            for th in w.taskhandlers.values()
        ]
        # If there is a only a single task, exit with the exit status of the task.
        if len(exitcodes) == 1:
            return exitcodes[0]
        # otherwise apply logic
        allfailed = all(v != 0 for v in exitcodes)
        if allfailed:
            return 3
        anyfailed = any(v != 0 for v in exitcodes)
        if anyfailed:
            return 2
        return 0


class NomadWatchJob(NomadWorkers):
    def _start_nomad_job(self, input: str) -> Tuple[str, str]:
        """Start a job from input file input."""
        if shutil.which("nomad"):
            jsonarg = "--json" if args.json else ""
            try:
                rr = run(
                    shlex.split(
                        f"nomad job run -detach -verbose {jsonarg} {shlex.quote(input)}"
                    ),
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
            eval = mynomad.get(f"evaluation/{evalid}")
            self.db.evaluations[eval["ID"]] = eval
            return eval["Namespace"], eval["JobID"]
        else:
            intxt = sys.stdin if input == "-" else open(input)
            txt: str = intxt.read()
            try:
                jobjson = json.loads(txt)
                isjson = True
            except Exception:
                jobjson = mynomad.post("jobs/parse", {"JobHCL": txt})
                isjson = False
            mynomad.post(
                "jobs",
                {
                    "Job": jobjson,
                    "Submission": txt if isjson else None,
                },
            )
            self.job = jobjson
            return jobjson["Namespace"], jobjson["ID"]

    def _stop_job(self, jobid):
        if args.purge:
            log.info(f"Purging job {jobid}")
        else:
            log.info(f"Stopping job {jobid}")
        assert args.mode in [
            "run",
            "job",
        ], f"Invalid args mode for stop or purge: {args.mode}"
        mynomad.delete(f"job/{jobid}", params={"purge": bool(args.purge)})

    def main_run(self, input: str):
        jobnamespace, jobid = self._start_nomad_job(input)
        mynomad.namespace = jobnamespace
        self.main_watch_job(jobid)

    def _db_init(self):
        jobid = self.db.job["ID"]
        for eval in mynomad.get(f"job/{jobid}/evaluations"):
            yield Event(Topic.Evaluation, eval)
        for alloc in mynomad.get(f"job/{jobid}/allocations"):
            yield Event(Topic.Allocation, alloc)
        yield Event(Topic.Job, mynomad.get(f"job/{jobid}"))

    def main_watch_job(self, jobid: str):
        self.db.job = mynomad.get(f"job/{jobid}")
        jobid = self.db.job["ID"]
        mynomad.namespace = self.db.job["Namespace"]
        try:
            self.db.start(
                # f"Job:{jobid}",
                "Job:*",
                "Evaluation:*",
                "Allocation:*",
            )
            for event in self.db.events(self._db_init):
                if event.topic == Topic.Allocation:
                    alloc = event.data
                    recentevalsids = set(
                        eval["ID"]
                        for eval in self.db.evaluations.values()
                        if eval.get("JobModifyIndex", -1)
                        >= self.db.job["JobModifyIndex"]
                    )
                    if alloc["EvalID"] in recentevalsids:
                        self.notify_alloc_worker(alloc)
                elif event.topic == Topic.Job:
                    if self.db.job["Status"] == "dead":
                        self.db.stop()
        finally:
            if args.purge and not args.detach:
                self._stop_job(jobid)
                self.db.join()
        return self.close()


class NomadWatchAlloc(NomadWorkers):
    def _db_init(self):
        allocid = list(self.db.allocations.values())[0]["ID"]
        yield Event(Topic.Allocation, mynomad.get(f"allocation/{allocid}"))

    def main_watch_alloc(self, allocid):
        allocs = mynomad.get(f"allocations", params={"prefix": allocid})
        assert len(allocs) > 0, f"Allocation with id {allocid} not found"
        assert len(allocs) < 2, f"Multiple allocations found starting with id {allocid}"
        self.db.allocations = allocs
        alloc = allocs[0]
        self.db.job = {"ID": alloc["JobID"]}
        mynomad.namespace = alloc["Namespace"]
        self.db.start(f"Allocation:{allocid}")
        for event in self.db.events(self._db_init):
            if event.topic == Topic.Allocation:
                if all(
                    alloc["ClientStatus"] not in ["pending", "running"]
                    for alloc in self.db.allocations.values()
                ):
                    self.db.stop()
        return self.close()


###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
Run a Nomad job in Nomad and then print logs to stdout and wait for the job to finish.
Made for running batch commands and monitoring them until they are done.
The exit code is 0 if the job was properly done and all tasks exited with 0 exit code.
If the job has only one task, exit with the task exit status.
Othwerwise, if all tasks exited failed, exit with 3.
If there are failed tasks, exit with 2.
If there was a python exception, exit code is 1.

Examples: nomad-watch --namespace default run ./some-job.nomad.hcl
          nomad-watch job some-job
          nomad-watch alloc af94b2
""",
        epilog="""
Written by Kamil Cukrowski 2023. Jointly under Beerware and MIT Licenses.
""",
    )
    parser.add_argument(
        "-n", "--namespace", help="Sets NOMAD_NAMESPACE environment variable"
    )
    parser.add_argument(
        "--only",
        choices=("stdout", "stderr"),
        default="",
        help="If set to stdout, will monitor only stdout, similar with stderr",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument(
        "--json", action="store_true", help="job input is in json form, passed to nomad"
    )
    parser.add_argument(
        "-d", "--detach", action="store_true", help="Do not stop the job on interrupt"
    )
    parser.add_argument(
        "--purge", action="store_true", help="Purge the nomad job before exiting"
    )
    parser.add_argument(
        "mode",
        choices=("run", "job", "alloc", "test"),
        help="""
        If mode is run, will run a job in Nomad and watch it.
        If mode is job, will watch a job by name.
        If mode is alloc, will watch a single allocation.
        Mode test is for internal testing.
        """,
    )
    parser.add_argument(
        "input",
        help="""
        If mode is run, this is a file or '-' to run in Nomad.
        If mode is job, this is job name to watch.
        If mode is alloc, this is allocation id to watch.
        """,
    )
    args = parser.parse_args()
    #
    logging.basicConfig(
        format="%(module)s:%(threadName)s:%(lineno)d: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    if args.verbose > 1:
        http_client.HTTPConnection.debuglevel = 1
    if args.namespace:
        os.environ["NOMAD_NAMESPACE"] = args.namespace
    if args.only:
        args.only = "stdout" if args.only in ["stdout", "out", "1"] else "stderr"
    if args.detach:
        assert not args.purge, f"purge is meaningless if detaching"
        assert args.mode in [
            "run",
            "job",
        ], "--detach is only meaningful in run or job modes"
    if args.purge:
        assert args.mode in [
            "run",
            "job",
            "test",
        ], "--purge is only meaningful in run or job modes"
    #
    return args


if __name__ == "__main__":
    global args
    args = parse_args()
    #
    if args.mode == "test":
        Test(args.input)
        exit()
    #
    try:
        try:
            if args.mode == "run":
                exit(NomadWatchJob().main_run(args.input))
            elif args.mode == "job":
                exit(NomadWatchJob().main_watch_job(args.input))
            elif args.mode == "alloc":
                exit(NomadWatchAlloc().main_watch_alloc(args.input))
            else:
                exit(f"Invalid mode: {args.mode}")
        finally:
            mynomad.session.close()
    except KeyboardInterrupt:
        exit(1)
