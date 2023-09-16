import functools
import inspect
import json
import os
import shlex
import subprocess
import sys
from typing import List, TextIO, Union


def caller(up=0):
    return inspect.stack()[1 + up][3]


@functools.lru_cache(maxsize=0)
def nomad_has_docker():
    nodes = json.loads(subprocess.check_output("nomad node status -json".split()))
    for n in nodes:
        if n["Drivers"].get("docker", {}).get("Healthy"):
            return True
    return False


def gen_job(script=""" echo hello world """):
    os.environ.setdefault("NOMAD_NAMESPACE", "default")
    jobname = f"test-nomad-utils-{caller(1)}"
    docker_task = {
        "Driver": "docker",
        "Config": {
            "image": "busybox",
            "command": "sh",
            "args": ["-xc", script],
            "init": True,
        },
    }
    raw_exec_task = {
        "Driver": "raw_exec",
        "Config": {
            "command": "sh",
            "args": ["-xc", script],
        },
    }
    task = docker_task if nomad_has_docker() else raw_exec_task
    job = {
        "ID": jobname,
        "Type": "batch",
        "TaskGroups": [
            {
                "Name": jobname,
                "ReschedulePolicy": {"Attempts": 0},
                "RestartPolicy": {"Attempts": 0},
                "Tasks": [
                    {
                        **task,
                        "Name": jobname,
                    }
                ],
            }
        ],
    }
    return {"Job": job}


def quotearr(cmd: List[str]):
    return " ".join(shlex.quote(x) for x in cmd)


def run(
    cmd: str, check=True, text=True, stdout: Union[int, TextIO] = sys.stderr, **kwargs
) -> subprocess.CompletedProcess:
    cmda = shlex.split(cmd)
    print(" ", file=sys.stderr, flush=True)
    print(f"+ {quotearr(cmda)}", file=sys.stderr, flush=True)
    r = subprocess.run(cmda, check=check, text=text, stdout=stdout, **kwargs)
    return r


def check_output(cmd: str, **kwargs) -> str:
    ret = run(cmd, stdout=subprocess.PIPE, **kwargs).stdout
    assert "exception" not in ret
    return ret


def run_nomad_cp(cmd: str, **kwargs):
    return run(f"python -m nomad_tools.nomad_cp -v {cmd}", **kwargs)


def run_nomad_watch(cmd: str, **kwargs):
    return run(f"python -m nomad_tools.nomad_watch -v {cmd}", **kwargs)


def check_output_nomad_watch(cmd: str, **kwargs):
    return check_output(f"python -m nomad_tools.nomad_watch -v {cmd}", **kwargs)
