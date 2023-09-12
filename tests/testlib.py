import functools
import inspect
import json
import shlex
import subprocess
from typing import List


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


def run(cmd: str, check=True, text=True, **kvargs):
    cmda = shlex.split(cmd)
    print(f"+ {quotearr(cmda)}")
    return subprocess.run(cmda, check=check, text=text, **kvargs)


def check_output(cmd: str, **kvargs):
    ret = run(cmd, stdout=subprocess.PIPE, **kvargs).stdout
    assert "exception" not in ret
    return ret
