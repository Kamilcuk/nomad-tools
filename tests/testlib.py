import functools
import inspect
import json
import os
import shlex
import subprocess
import sys
from typing import List, Optional, Union

os.environ.setdefault("NOMAD_NAMESPACE", "default")


def caller(up=0):
    return inspect.stack()[1 + up][3]


@functools.lru_cache(maxsize=0)
def nomad_has_docker():
    nodes = json.loads(subprocess.check_output("nomad node status -json".split()))
    for n in nodes:
        if n["Drivers"].get("docker", {}).get("Healthy"):
            return True
    return False


def gen_job(script=""" echo hello world """, name: Optional[str] = None):
    assert name is None or " " not in name
    name = (
        name
        or os.environ.get("PYTEST_CURRENT_TEST", caller(1)).split(":")[-1].split(" ")[0]
    )
    jobname = f"test-nomad-utils-{name}"
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
    cmd: str,
    check: Union[bool, int, List[int]] = True,
    text=True,
    stdout: Union[bool, int] = False,
    **kwargs,
) -> subprocess.CompletedProcess:
    cmda = shlex.split(cmd)
    print(" ", file=sys.stderr, flush=True)
    print(f"+ {quotearr(cmda)}", file=sys.stderr, flush=True)
    rr = subprocess.run(
        cmda,
        text=text,
        stdout=subprocess.PIPE if stdout else sys.stderr,
        **kwargs,
    )
    if stdout:
        print(rr.stdout)
    if isinstance(check, bool):
        if check:
            rr.check_returncode()
    elif isinstance(check, int):
        assert rr.returncode == check, f"{rr.returncode} {check}"
    elif isinstance(check, list):
        assert rr.returncode in check, f"{rr.returncode} {check}"
    return rr


def run_nomad_cp(cmd: str, **kwargs):
    return run(f"python -m nomad_tools.nomad_cp -v {cmd}", **kwargs)


def run_nomad_watch(cmd: str, **kwargs):
    return run(f"python -m nomad_tools.nomad_watch -v {cmd}", **kwargs)


def run_nomad_vardir(cmd: str, **kwargs):
    return run(f"python -m nomad_tools.nomad_vardir -v {cmd}", **kwargs)
