#!/usr/bin/env python3

import argparse
import dataclasses
import datetime
import functools
import json
import logging
import os
import shlex
import socket
import subprocess
import sys
import time
from typing import List, Set

import tomli

###############################################################################

log = logging.getLogger("nomad-gitlab-runner")


def quotearr(cmd: List[str]):
    return " ".join(shlex.quote(x) for x in cmd)


def run(cmdstr: str, *args, check=True, quiet=False, **kvargs):
    cmd = shlex.split(cmdstr)
    if not quiet:
        log.info(f"+ {quotearr(cmd)}")
    try:
        return subprocess.run(cmd, *args, check=check, text=True, **kvargs)
    except subprocess.CalledProcessError as e:
        exit(e.returncode)


def ns2dt(ns: int):
    return datetime.datetime.fromtimestamp(ns // 1000000000)


###############################################################################


@functools.lru_cache(maxsize=0)
def get_env():
    """Get all gitlab exported environment variables and remove CUSTOM_ENV_ from front"""
    ret = {
        k.replace("CUSTOM_ENV_", ""): v
        for k, v in os.environ.items()
        if k.startswith("CUSTOM_ENV_")
    }
    ret.update({f"CUSTOM_ENV_{k}": v for k, v in ret.items()})
    return ret


@functools.lru_cache(maxsize=0)
def get_jobname():
    """Get the name of the nomad job"""
    return "gitlabrunner.%(CI_PROJECT_PATH_SLUG)s.%(CI_CONCURRENT_ID)s" % get_env()


def gen_nomadjob(jobname: str):
    task_exec = {
        "Name": jobname,
        "Driver": "exec",
        "User": "gitlab-runner:gitlab-runner",
        "Config": {
            "command": "sleep",
            "args": [
                str(env["CI_JOB_TIMEOUT"]),
            ],
        },
        "Resources": {
            "CPU": config.CPU,
            "MemoryMB": config.MemoryMB,
        },
    }
    task = {
        "exec": task_exec,
        "raw_exec": {
            **task_exec,
            "Driver": "raw_exec",
            # https://github.com/hashicorp/nomad/issues/5397
            "User": "gitlab-runner",
        },
    }
    chosentask = task.get(config.mode)
    assert (
        chosentask
    ), f"Invalid mode in configuration, available: {' '.join(task.keys())}"
    nomadjob = {
        "Job": {
            "ID": jobname,
            "Type": "batch",
            "TaskGroups": [
                {
                    "Name": jobname,
                    "ReschedulePolicy": {"Attempts": 0},
                    "RestartPolicy": {"Attempts": 0},
                    "Tasks": [
                        chosentask,
                    ],
                }
            ],
        }
    }
    return nomadjob


def purge_previous_nomad_job(jobname: str):
    rr = run(
        f"nomad job inspect {jobname}", check=False, stdout=subprocess.PIPE, quiet=True
    )
    assert rr.returncode in [
        0,
        1,
    ], f"Invalid nomad job inspect {jobname} output - it should be either 0 or 1"
    if rr.returncode == 0:
        assert (
            json.loads(rr.stdout)["Job"]["Stop"] == True
        ), f"Job with the name {jobname} already exists and is not stopped. Bailing out"
        run(f"nomad job stop -purge {jobname}")


def wait_for_nomad_job_to_start(name: str):
    printedevents: Set[int] = set()
    while True:
        time.sleep(2)
        res = run(
            f"nomad operator api /v1/job/{name}/allocations",
            stdout=subprocess.PIPE,
            quiet=True,
        ).stdout
        allocs = json.loads(res)
        # Wait for job to start allocation.
        msg = f"Waiting for job {name} to start an allocation"
        if len(allocs) == 0:
            log.info(msg)
            continue
        assert (
            len(allocs) == 1
        ), f"Job should only have one allocation: {len(allocs)} {'   '.join(str(a) for a in allocs)}"
        # Get allocation data.
        alloc = allocs[0]
        allocid = alloc["ID"][:6]
        allocstatus = alloc["ClientStatus"]
        ts: dict = alloc["TaskStates"]
        assert len(ts) == 1
        taskname: str = list(ts.keys())[0]
        task: dict = list(ts.values())[0]
        events: List[dict] = task.get("Events") or []
        nodename = alloc["NodeName"]
        # Pring events.
        for e in events:
            if e["Time"] not in printedevents:
                printedevents.add(e["Time"])
                timestamp = ns2dt(e["Time"])
                dmsg = e["DisplayMessage"]
                log.info(f"{allocid} {timestamp} {nodename} {dmsg}")
        # Check allocation status.
        assert allocstatus in [
            "pending",
            "running",
        ], f"Allocation {allocid} failed with status {allocstatus}"
        msg = f"Waiting for job {name} allocation {allocid} with status {allocstatus} to start"
        if allocstatus != "running":
            log.info(msg)
            continue
        # Make sure the task is running and started.
        msg = f"Waiting for job {name} allocation {allocid} with status {allocstatus} to start task {taskname}"
        taskstate = task["State"]
        if taskstate != "running":
            log.info(msg)
            continue
        if not any(event["Type"] == "Started" for event in events):
            log.info(msg)
            continue
        break


###############################################################################


def mode_config():
    config = {
        "builds_dir": "/local",
        "cache_dir": "/local",
        "builds_dir_is_shared": False,
        "hostname": socket.gethostname(),
        "driver": {
            "name": "nomad-gitlab-runner",
            "version": "v0.0.1",
        },
    }
    cfg = json.dumps(config)
    log.debug(cfg)
    print(cfg)


def mode_prepare():
    jobname = get_jobname()
    nomadjob = gen_nomadjob(jobname)
    purge_previous_nomad_job(jobname)
    run("nomad job run --json -", input=json.dumps(nomadjob))
    wait_for_nomad_job_to_start(jobname)


def mode_run():
    jobname = get_jobname()
    run(
        f"nomad alloc exec -job {jobname} bash -s",
        stdin=open(args.script),
        quiet=True,
    )


def mode_cleanup():
    jobname = get_jobname()
    purge = "-purge" if config.purge else ""
    run(f"nomad job stop {purge} {jobname}")


###############################################################################


@dataclasses.dataclass
class Config:
    """Configuration of this program"""

    NOMAD_NAMESPACE: str = ""
    NOMAD_TOKEN: str = ""
    purge: bool = True
    CPU: int = 1024
    MemoryMB: int = 1024
    mode: str = "raw_exec"

    def update_env(self):
        """Update environment variables from configuration - to set NOMAD_TOKEN variable"""
        for k, v in dataclasses.asdict(self).items():
            if k.startswith("NOMAD_") and v:
                os.environ[k] = v


###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
            This is a script to execute Nomad job from custom gitlab executor.
            """,
        epilog="""
            Written by Kamil Cukrowski 2023. Licensed under GNU GPL version or later.
            """,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("rb"),
        default="/etc/gitlab-runner/nomad.toml",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("mode")
    parser.add_argument("script", nargs="?")
    parser.add_argument("stage", nargs="?")
    return parser.parse_args()


def cli():
    global args
    args = parse_args()
    logging.basicConfig(
        format="%(module)s:%(lineno)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    #
    config = Config(**tomli.load(args.config))
    config.update_env()
    env = get_env()
    #
    log.debug(f"{sys.argv}")
    if args.mode == "config":
        mode_config()
    elif args.mode == "prepare":
        mode_prepare()
    elif args.mode == "run":
        mode_run()
    elif args.mode == "cleanup":
        mode_cleanup()
    else:
        assert 0, f"Unknown mode: {args.mode}"


if __name__ == "__main__":
    cli()
