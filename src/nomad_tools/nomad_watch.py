#!/usr/bin/env python3

import argparse
import base64
import dataclasses
import datetime
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from http import client as http_client
from typing import Dict, List, Optional, Tuple

import requests
from requests.auth import HTTPBasicAuth

###############################################################################


class MyLogger(logging.Logger):
    def trace(self, msg, *args, **kwargs):
        self.log(5, msg, *args, **kwargs)


logging.setLoggerClass(MyLogger)
log = logging.getLogger("nomad-watch")

###############################################################################


def run(cmd, *args, check=True, **kvargs):
    log.info(f"+ {' '.join(shlex.quote(x) for x in cmd)}")
    return subprocess.run(cmd, *args, check=check, text=True, **kvargs)


def test():
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
    spec = subprocess.check_output("nomad fmt -".split(), input=spec, text=True)
    print("Running nomad job with {now}")
    output = run(
        [
            sys.argv[0],
            *(["-v"] if args.verbose else []),
            *(["-v"] if args.verbose > 1 else []),
            "-",
        ],
        input=spec,
        stdout=subprocess.PIPE,
    ).stdout
    assert now in output
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


mynomad = MyNomad()

###############################################################################


def ns2dt(ns: int):
    return datetime.datetime.fromtimestamp(1360287003083988472 // 1000000000)


class output:
    """Output templates in one place"""

    @staticmethod
    def prefix(allocid: str, group: str, task: str):
        # return f"{allocid:.6}:{group}:{task}:"
        return f"{group}:{task}:"

    @classmethod
    def task_start(
        cls,
        allocid: str,
        group: str,
        task: str,
        start: datetime.datetime,
        nodename: str,
    ):
        print(f"{cls.prefix(allocid, group, task)} at {start} started on {nodename}")

    @classmethod
    def log(cls, allocid: str, group: str, task: str, streamtype: str, line: str):
        print(f"{cls.prefix(allocid, group, task)}{streamtype}: {line}")

    @classmethod
    def task_end(
        cls, allocid: str, group: str, task: str, exitcode: int, stop: datetime.datetime
    ):
        print(f"{cls.prefix(allocid, group, task)} at {stop} exited with {exitcode}")


###############################################################################


class Logger(threading.Thread):
    """Represents a single logging stream from Nomad. Such stream is created separately for stdout and stderr."""

    def __init__(self, alloc: dict, taskname: str, stderr: bool):
        super().__init__()
        self.alloc: dict = alloc
        self.allocid: str = alloc["ID"]
        self.taskname: str = taskname
        self.stderr: bool = stderr
        self.stream: Optional[requests.Response] = None

    def streamtype(self):
        return "err" if self.stderr else "out"

    def _handle_lines(
        self, stream: requests.Response, fromoffset: Optional[int] = None
    ) -> Optional[int]:
        stream.raise_for_status()
        lastoffset = None
        for dataline in stream.iter_lines():
            if dataline:
                log.debug(f"RECV: {repr(dataline)}")
                try:
                    linejson: dict = json.loads(dataline)
                except Exception as e:
                    print(r"dataline={dataline}", file=sys.stderr)
                    raise Exception() from e
                lines = ""
                offset = linejson.get("Offset")
                if "Data" in linejson:
                    line64: str = linejson["Data"]
                    lines = base64.b64decode(line64.encode()).decode()
                    # Chop from offset if given.
                    if (
                        fromoffset
                        and offset
                        and offset < fromoffset < offset + len(lines)
                    ):
                        lines = lines[fromoffset - offset :]
                    for line in lines.splitlines():
                        line = line.rstrip()
                        output.log(
                            self.allocid,
                            self.alloc["TaskGroup"],
                            self.taskname,
                            self.streamtype(),
                            line,
                        )
                if offset:
                    lastoffset = offset + len(lines)
                if linejson.get("FileEvent") == "file deleted":
                    break
        return lastoffset

    def _stream(self, params={}):
        return mynomad.request(
            "GET",
            f"client/fs/logs/{self.allocid}",
            params={
                "task": self.taskname,
                "type": f"std{self.streamtype()}",
                **params,
            },
            stream=True,
        )

    def run(self):
        # To get all the logs, firstly stream is opened without follow.
        with self._stream() as self.stream:
            lastoffset = self._handle_lines(self.stream)
        # Then with follow to get all.
        with self._stream({"follow": True, "offset": lastoffset}) as self.stream:
            self._handle_lines(self.stream, lastoffset)


@dataclasses.dataclass
class AllocWorker:
    """Represents a worker that prints out and manages state related to one allocation"""

    loggers: Dict[str, List[Logger]] = dataclasses.field(default_factory=dict)
    exitcodes: Dict[str, int] = dataclasses.field(default_factory=dict)

    def _create_loggers(self, alloc: dict, taskname: str):
        ths: List[Logger] = []
        if not args.only or args.only == "stdout":
            ths.append(Logger(alloc, taskname, False))
        if not args.only or args.only == "stderr":
            ths.append(Logger(alloc, taskname, True))
        assert len(ths)
        for th in ths:
            th.daemon = True
            th.start()
        return ths

    def alloc_event(self, alloc: dict):
        """Update the state with alloc"""
        allocid = alloc["ID"]
        if alloc["TaskStates"] is None:
            return
        for taskname, task in alloc["TaskStates"].items():
            if taskname not in self.loggers and task["State"] in [
                "running",
                "dead",
            ]:
                startedevent: Optional[dict] = next(
                    (e for e in task["Events"] if e["Type"] == "Started"), None
                )
                if startedevent:
                    startedtime: int = startedevent["Time"]
                    output.task_start(
                        allocid,
                        alloc["TaskGroup"],
                        taskname,
                        ns2dt(startedtime),
                        alloc["NodeName"],
                    )
                    self.loggers[taskname] = self._create_loggers(alloc, taskname)
                else:
                    # No startedevent - means something went wrong.
                    # Get all messages and just display them
                    msgs = "\n".join(e["DisplayMessage"] for e in task["Events"])
                    for line in msgs.splitlines():
                        output.log(allocid, alloc["TaskGroup"], taskname, "!!!", line)
                    self.loggers[taskname] = []
            if taskname not in self.exitcodes and task["State"] == "dead":
                te = next(
                    (e for e in task["Events"] if e["Type"] == "Terminated"), None
                )
                exitcode = te["ExitCode"] if te else -1
                stoptime = te["Time"] if te else 0
                self.exitcodes[taskname] = exitcode
                output.task_end(
                    allocid, alloc["TaskGroup"], taskname, exitcode, ns2dt(stoptime)
                )


@dataclasses.dataclass
class NomadWatch:
    """The main class of nomad-watch for watching over stuff"""

    # allocid -> AllocWorker
    workers: Dict[str, AllocWorker] = dataclasses.field(default_factory=dict)

    def _start_job(self, input: str) -> Tuple[str, str]:
        """Start a job from input file input."""
        if shutil.which("nomad"):
            jsonarg = "--json" if args.json else ""
            data = run(
                shlex.split(
                    f"nomad job run -detach -verbose {jsonarg} {shlex.quote(input)}"
                ),
                stdout=subprocess.PIPE,
            ).stdout.strip()
            for line in data.splitlines():
                log.info(line)
            evalid = next(
                x.split(" ", 3)[-1] for x in data.splitlines() if "eval" in x.lower()
            ).strip()
            eval = mynomad.get(f"evaluation/{evalid}")
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
            return jobjson["Namespace"], jobjson["ID"]

    def _notify_alloc_worker(self, alloc: dict):
        self.workers.setdefault(alloc["ID"], AllocWorker()).alloc_event(alloc)

    def main_watch_job(self, jobid):
        while True:
            jobjson = mynomad.get(f"job/{jobid}")
            if not mynomad.namespace:
                mynomad.namespace = jobjson["Namespace"]
            evals = mynomad.get(f"job/{jobid}/evaluations")
            # Filter only evaluations created for newer version then the Job
            recentevalsids = set(
                eval["ID"]
                for eval in evals
                if eval.get("JobModifyIndex", -1) >= jobjson["JobModifyIndex"]
            )
            # Get all allocations of evaluations.
            allocs: List[dict] = mynomad.get(f"job/{jobid}/allocations")
            for alloc in allocs:
                if alloc["EvalID"] in recentevalsids:
                    self._notify_alloc_worker(alloc)
            if jobjson["Status"] == "dead":
                break
            if len(self.workers) == 0 and all(
                len(w.loggers) == 0 for w in self.workers.values()
            ):
                log.info("Waiting for allocation...")
            time.sleep(1)
        self._join_workers()

    def main_watch_alloc(self, allocid):
        allocs = mynomad.get(f"allocations", params={"prefix": allocid})
        assert len(allocs) > 0, f"Allocation with id {allocid} not found"
        assert len(allocs) < 2, f"Multiple allocations found starting with id {allocid}"
        allocid = allocs[0]["ID"]
        while True:
            alloc = mynomad.get(f"allocation/{allocid}")
            self._notify_alloc_worker(alloc)
            if alloc["ClientStatus"] not in ["pending", "running"]:
                break
            time.sleep(1)
        self._join_workers()

    def main(self, input: str):
        jobnamespace, jobid = self._start_job(input)
        mynomad.namespace = jobnamespace
        self.main_watch_job(jobid)

    def _join_workers(self):
        loggerscnt = sum(len(w.loggers) for w in self.workers.values())
        log.debug(f"joining {len(self.workers)} allocations with {loggerscnt} loggers")
        timeend = datetime.datetime.now() + datetime.timedelta(seconds=2)
        for allocid, worker in self.workers.items():
            for taskname, loggers in worker.loggers.items():
                for (
                    i,
                    logger,
                ) in enumerate(loggers):
                    timeout = (timeend - datetime.datetime.now()).total_seconds()
                    if timeout <= 0:
                        log.debug("quitting")
                        return
                    log.debug(f"joining worker {taskname}[{i}] timeout={timeout}")
                    logger.join(timeout=timeout)

    def close(self) -> int:
        mynomad.session.close()
        # If there is a only a single task, exit with the exit status of the task.
        if len(self.workers) == 1:
            worker = next(iter(self.workers.values()))
            if len(worker.exitcodes) == 1:
                exitcode = next(iter(worker.exitcodes.values()))
                return exitcode
        # otherwise apply logic
        allfailed = all(
            x != 0 for w in self.workers.values() for x in w.exitcodes.values()
        )
        if allfailed:
            return 3
        anyfailed = any(
            x != 0 for w in self.workers.values() for x in w.exitcodes.values()
        )
        if anyfailed:
            return 2
        return 0


if __name__ == "__main__":
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
        "mode",
        choices=("run", "job", "alloc"),
        help="""
        If mode is run, will run a job in Nomad and watch it.
        If mode is job, will watch a job by name.
        If mode is alloc, will watch a single allocation.
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
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    if args.verbose > 1:
        http_client.HTTPConnection.debuglevel = 1
    #
    if args.namespace:
        os.environ["NOMAD_NAMESPACE"] = args.namespace
    #
    if args.only:
        args.only = "stdout" if args.only in ["stdout", "out", "1"] else "stderr"
    #
    if args.input == "test":
        test()
        exit()
    #
    do = NomadWatch()
    try:
        try:
            if args.mode == "job":
                do.main_watch_job(args.input)
            elif args.mode == "alloc":
                do.main_watch_alloc(args.input)
            else:
                do.main(args.input)
            ret = do.close()
            exit(ret)
        finally:
            mynomad.session.close()
    except KeyboardInterrupt:
        exit(1)
