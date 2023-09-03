#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# PYTHON_ARGCOMPLETE_OK

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
from typing import Any, Dict, List, Optional, Set, Tuple

import requests.adapters
from requests.auth import HTTPBasicAuth

try:
    import argcomplete.completers
except ImportError:
    pass

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


@dataclasses.dataclass
class MyNomad:
    """Represents connection to Nomad"""

    namespace: Optional[str] = None
    session: requests.Session = dataclasses.field(default_factory=requests.Session)

    def __post_init__(self):
        a = requests.adapters.HTTPAdapter(
            pool_connections=1000, pool_maxsize=1000, max_retries=3
        )
        self.session.mount("http://", a)

    def request(
        self,
        method,
        url,
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
            **kvargs,
        )

    def _reqjson(self, mode, *args, **kvargs):
        rr = self.request(mode, *args, **kvargs)
        rr.raise_for_status()
        return rr.json()

    def get(self, *args, **kvargs):
        return self._reqjson("GET", *args, **kvargs)

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
        self.loggers: List[Logger] = []
        self.messages: Set[int] = set()
        self.exitcode: Optional[int] = None

    @staticmethod
    def _create_loggers(tk: TaskKey):
        ths: List[Logger] = []
        assert args.stream in [
            None,
            "",
            "err",
            "out",
        ], f"invalid args.stream = {args.stream}"
        if (args.stream or "out") == "out":
            ths.append(Logger(tk, False))
        if (args.stream or "err") == "err":
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
                tk.log_alloc(ns2dt(time), msg)
        if (
            not self.loggers
            and task["State"] in ["running", "dead"]
            and self._find_event(events, "Started")
        ):
            self.loggers += self._create_loggers(tk)
            if task["State"] == "dead":
                # If the task is already finished, give myself max 3 seconds to query all the logs.
                # This is to reduce the number of connections.
                threading.Timer(3, self.stop)
        if self.exitcode is None and task["State"] == "dead":
            # Assigns None if Terminated event not found
            self.exitcode = self._find_event(events, "Terminated").get("ExitCode")
            self.stop()

    def stop(self):
        for l in self.loggers:
            l.stop()


class AllocWorker:
    """Represents a worker that prints out and manages state related to one allocation"""

    def __init__(self):
        self.taskhandlers: Dict[TaskKey, TaskHandler] = {}

    def alloc_event(self, alloc: dict):
        """Update the state with alloc"""
        taskstates: dict = alloc.get("TaskStates") or {}
        for taskname, task in taskstates.items():
            if args.task and not args.task.search(taskname):
                continue
            tk = TaskKey(alloc["ID"], alloc["NodeName"], alloc["TaskGroup"], taskname)
            self.taskhandlers.setdefault(tk, TaskHandler()).task_event(alloc, tk, task)


###############################################################################


@dataclasses.dataclass
class NomadWatch:
    """The main class of nomad-watch for watching over stuff"""

    # allocid -> AllocWorker
    workers: Dict[str, AllocWorker] = dataclasses.field(default_factory=dict)

    def _start_job(self, input: str) -> str:
        """Start a job from input file input."""
        if shutil.which("nomad"):
            jsonarg = "-json" if args.json else ""
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
        else:
            intxt = sys.stdin if input == "-" else open(input)
            txt: str = intxt.read()
            try:
                jobjson = json.loads(txt)
            except json.JSONDecodeError:
                jobjson = mynomad.post("jobs/parse", json={"JobHCL": txt})
            evalid = mynomad.post("jobs", json={"Job": jobjson, "Submission": txt})[
                "EvalID"
            ]
        return evalid

    def _notify_alloc_worker(self, alloc: dict):
        self.workers.setdefault(alloc["ID"], AllocWorker()).alloc_event(alloc)

    def _stop_job(self, jobid):
        if args.purge:
            log.info(f"Purging job {jobid}")
        else:
            log.info(f"Stopping job {jobid}")
        resp = mynomad.delete(f"job/{jobid}", params={"purge": bool(args.purge)})
        assert resp["EvalID"], f"Stopping {jobid} did not trigger evaluation: {resp}"

    def _wait_for_eval(self, evalid: str):
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

    def _watch_job(self, jobinit: dict):
        jobid: str = jobinit["ID"]
        jobmodifyindex: int = jobinit["JobModifyIndex"]
        jobversion: int = jobinit["Version"]
        jobnamespace: str = jobinit["Namespace"]
        jobdesc = (
            f"{jobid}@{jobnamespace}"
            if args.all
            else f"{jobid}@{jobnamespace} v{jobversion}"
        )
        #
        log.info(f"Watching job {jobdesc}")
        while True:
            try:
                allocs: List[dict] = mynomad.get(f"job/{jobid}/allocations")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    log.info(f"Job {jobdesc} removed")
                    return
                raise e
            # Filter relevant allocations only.
            allocs = [
                alloc
                for alloc in allocs
                if alloc["ID"] in self.workers.keys()
                or alloc["JobVersion"] == jobversion
                or args.all
            ]
            for alloc in allocs:
                self._notify_alloc_worker(alloc)
            if not args.all:
                if len(self.workers) == 0:
                    try:
                        jobjson: dict = mynomad.get(f"job/{jobid}")
                        if jobjson["JobVersion"] != jobversion:
                            log.info(f"New version of job {jobdesc} posted")
                            break
                        if jobjson["Status"] == "dead":
                            log.info(f"Job {jobdesc} is dead with no allocations")
                            break
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 404:
                            log.info(f"Job {jobdesc} removed")
                            return
                        raise e
                    log.info(f"Job {jobdesc} has no allocations")
                else:
                    allallocsfinished = all(
                        alloc["ClientStatus"] not in ["pending", "running"]
                        for alloc in allocs
                    )
                    if allallocsfinished:
                        log.info(f"All allocations of {jobdesc} are finished")
                        break
            if args.no_follow:
                break
            time.sleep(1)

    def _join_workers(self):
        threads: List[Tuple[str, threading.Thread]] = [
            (f"{tk.task}[{i}]", logger)
            for w in self.workers.values()
            for tk, th in w.taskhandlers.items()
            for i, logger in enumerate(th.loggers)
        ]
        thcnt = sum(len(w.taskhandlers) for w in self.workers.values())
        log.debug(
            f"Joining {len(self.workers)} allocations with {thcnt} taskhandlers and {len(threads)} loggers"
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

    def _waiter(self, jobinit: dict, done: threading.Event):
        self._watch_job(jobinit)
        self._join_workers()
        log.debug("Waiter exiting")
        done.set()

    def main_watch_run(self, input: str):
        evalid = self._start_job(input)
        eval_ = mynomad.get(f"evaluation/{evalid}")
        mynomad.namespace = eval_["Namespace"]
        jobid = eval_["JobID"]
        jobinit = mynomad.get(f"job/{jobid}")
        #
        done = threading.Event()
        waiterthread = threading.Thread(
            target=self._waiter, args=(jobinit, done), daemon=True
        )
        try:
            # First wait for the evaluation.
            self._wait_for_eval(evalid)
            waiterthread.start()
            # Waiting on Event, because thread.join does not handle interrupts well.
            done.wait()
        except KeyboardInterrupt:
            log.info("Received interrupt")
        finally:
            # On exception, stop or purge the job if needed.
            if args.purge or args.stop:
                self._stop_job(jobid)
            if waiterthread and not args.all:
                waiterthread.join()

    def _find_job(self, jobid: str) -> str:
        jobs = mynomad.get("jobs", params={"prefix": jobid})
        assert len(jobs) > 0, f"No jobs found with prefix {jobid}"
        jobsnames = " ".join(f"{x['ID']}@{x['Namespace']}" for x in jobs)
        assert len(jobs) < 2, f"Multiple jobs found with name {jobid}: {jobsnames}"
        mynomad.namespace = jobs[0]["Namespace"]
        return jobs[0]["ID"]

    def _find_last_not_stopped_job(self, jobid: str) -> dict:
        jobinit = mynomad.get(f"job/{jobid}")
        if jobinit["Stop"]:
            # Find last job version that is not stopped.
            versions = mynomad.get(f"job/{jobid}/versions")
            notstopedjobs = [job for job in versions["Versions"] if not job["Stop"]]
            if notstopedjobs:
                notstopedjobs.sort(key=lambda job: -job["ModifyIndex"])
                return notstopedjobs[0]
        return jobinit

    def main_watch_job(self, input: str):
        jobid = self._find_job(input)
        jobinit = self._find_last_not_stopped_job(jobid)
        try:
            self._watch_job(jobinit)
        except KeyboardInterrupt:
            log.info("Received interrupt")
        finally:
            for w in self.workers.values():
                for th in w.taskhandlers.values():
                    th.stop()
            self._join_workers()

    def main_watch_alloc(self, allocid):
        allocs = mynomad.get(f"allocations", params={"prefix": allocid})
        assert len(allocs) > 0, f"Allocation with id {allocid} not found"
        assert len(allocs) < 2, f"Multiple allocations found starting with id {allocid}"
        allocid = allocs[0]["ID"]
        log.info(f"Watching allocation {allocid}")
        while True:
            alloc = mynomad.get(f"allocation/{allocid}")
            self._notify_alloc_worker(alloc)
            allocclientstatus = alloc["ClientStatus"]
            if allocclientstatus not in ["pending", "running"]:
                log.info(
                    f"Allocation {allocid} has status {allocclientstatus}. Exiting."
                )
                break
            if args.no_follow:
                break
            time.sleep(1)

    def close(self) -> int:
        log.debug("close()")
        mynomad.session.close()
        exitcodes: List[int] = [
            # If thread did not return, exit with -1.
            -1 if th.exitcode is None else th.exitcode
            for w in self.workers.values()
            for th in w.taskhandlers.values()
        ]
        anyunfinished = any(v == -1 for v in exitcodes)
        if anyunfinished:
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


def bash_completion():
    return """
_nomad_watch_api() {
    env NOMAD_NAMESPACE="$NOMAD_NAMESPACE" nomad operator api /v1/"$1"
}
_nomad_watch_jq() {
    /usr/bin/env python3 -c 'import json,sys; [print(x[sys.argv[1]]) for x in json.load(sys.stdin)]' "$2"
}
_nomad_watch_get() {
    _nomad_watch_api "$1" | _nomad_watch_jq "$2"
}
_nomad_watch() {
    local arg next NOMAD_NAMESPACE="${NOMAD_NAMESPACE:-}" job
    set -- "${COMP_WORDS[@]}"
    while (($#)); do
        case "$arg" in
        --namespace=*) NOMAD_NAMESPACE=${1##--namespace=}; ;;
        -*N*) NOMAD_NAMESPACE=${1%*N}; ;;
        --namespace|-*N|-N) shift; NOMAD_NAMESPACE=$1; ;;
        job) shift; job=$1; ;;
        run) shift; job=${1%%.*}; ;;
        esac
        shift
    done
    #
    local api jq
    #
    local cur prev tmp
	cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    case "$cur" in
    --namespace=*) COMPREPLY=($(compgen -W "$("${api[@]}" /v1/namespaces | "${jq[@]}" Name)" -- "$cur")); ;;
    --task=*)
    *)
        case "$prev" in
        run) COMPREPLY=($(compgen -o filenames -A file -- "$cur")); ;;
        job) COMPREPLY=($(compgen -W "$(_nomad_watch_get jobs ID)" -- "$cur")); ;;
        alloc) COMPREPLY=($(compgen -W "$(_nomad_watch_get allocations ID)" -- "$cur")); ;;
        -*N|--namespace) COMPREPLY=($(compgen -W "$(_nomad_watch_get namespaces Name)" -- "$cur")); ;;
        -*t|--task)
            if [[ -n "${job:-}" ]] && tmp=$("${api[@]}" "$job" 2>/dev/null); then
                COMPREPLY=($(compgen -W "$(<<<"$tmp" 
            fi
            ;;
        *) COMPREPLY=($(compgen -W "run job alloc" -- "$cur")); ;;
        ;;
    esac
    COMP_WORDS
}
complete -F _nomad_watch nomad-watch
    """


###############################################################################

def complete_input(parsed_args, **kwargs):
    if parsed_args.namespace:
        os.environ["NOMAD_NAMESPACE"] = find_nomad_namespace(parsed_args.namespace)
    return (
        [x["ID"] for x in mynomad.get("allocations")]
        if parsed_args.mode == "alloc"
        else [x["ID"] for x in mynomad.get("jobs")]
        if parsed_args.mode == "job"
        else argcomplete.completers.FilesCompleter([".hcl", ".nomad"], directories=False)(parsed_args=parsed_args, **kwargs)
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="""
            Run a Nomad job in Nomad and then print logs to stdout and wait for the job to finish.
            Made for running batch commands and monitoring them until they are done.
            The exit code is 0 if the job was properly done and all tasks exited with 0 exit code.
            If the job has only one task, exit with the task exit status.
            Othwerwise, if all tasks exited failed, exit with 3.
            If there are failed tasks, exit with 2.
            If there was a python exception, standard exit code is 1.
            Examples: nomad-watch --namespace default run ./some-job.nomad.hcl ;
                      nomad-watch job some-job ;
                      nomad-watch alloc af94b2 ;
                      nomad-watch --all --task redis -N services job redis .
            """,
        epilog="""
            Written by Kamil Cukrowski 2023. Jointly under Beerware and MIT Licenses.
            """,
    )
    parser.add_argument(
        "-N",
        "--namespace",
        help="Finds Nomad namespace matching given prefix and sets NOMAD_NAMESPACE environment variable.",
    ).completer = lambda prefix, parsed_args, **kwargs: [x["Name"] for x in mynomad.get("namespaces")]
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="""
            Do not exit after the current job monitoring is done.
            Instead, watch endlessly for any existing and new allocations of a job.
            """,
    )
    parser.add_argument(
        "-s",
        "--stream",
        choices=("stdout", "stderr", "out", "err", "1", "2"),
        default="",
        help="Only monitor stdout or stderr, not both stream.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="job input is in json form, passed to nomad command with --json",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="In run mode, make sure to stop the job before exit.",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="In run mode, stop and purge the job before exiting.",
    )
    parser.add_argument(
        "-n",
        "--lines",
        default=-1,
        type=int,
        help="Sets the tail location in best-efforted number of lines relative to the end of logs",
    )
    parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Shorthand for --all --lines=10 to be like in tail -f.",
    )
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Just run once, get the logs in a best-effort style and exit.",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=re.compile,
        help="Only watch tasks names matching this regex.",
    )
    parser.add_argument(
        "mode",
        choices=("run", "start", "job", "alloc", "test"),
        help="""
        If mode is run, will run specified Nomad job and then watch over it.
        If mode is start, will run specific Nomad job and wait until it has at least one allocation."
        If mode is job, will watch a job by name.
        If mode is alloc, will watch tasks of a single allocation.
        Mode test is for internal testing.
        """,
    )
    parser.add_argument(
        "input",
        help="""
        If mode is run or start, this is a file path or '-' if using stdin of a job to run in Nomad.
        If mode is job, this is the job name to watch.
        If mode is alloc, this is the allocation id to watch.
        """,
    ).completer = complete_input
    if "argcomplete" in sys.modules:
        argcomplete.autocomplete(parser)
    args = parser.parse_args()
    #
    logging.basicConfig(
        format="%(module)s:%(lineno)03d: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    if args.verbose > 1:
        http_client.HTTPConnection.debuglevel = 1
    if args.stream:
        args.stream = "out" if args.stream in ("1", "stdout", "out") else "err"
    if args.stop:
        assert args.mode in ["run", "test"], "--detach is only meaningful in run mode"
    if args.purge:
        assert args.mode in ["run", "test"], "--purge is only meaningful in run mode"
    if args.all:
        assert args.mode in ["job", "run"], "--all only relevant in job or run modes"
    if args.follow:
        args.lines = 10
        args.all = True
    if args.namespace:
        os.environ["NOMAD_NAMESPACE"] = find_nomad_namespace(args.namespace)
    #
    return args


if __name__ == "__main__":
    args = parse_args()
    #
    if args.mode == "test":
        Test(args.input)
        exit()
    #
    do = NomadWatch()
    if args.mode == "run" or args.mode == "start":
        do.main_watch_run(args.input)
    elif args.mode == "job":
        do.main_watch_job(args.input)
    elif args.mode == "alloc":
        do.main_watch_alloc(args.input)
    else:
        assert 0, f"unhandled mode: {args.mode}"
    exit(do.close())
