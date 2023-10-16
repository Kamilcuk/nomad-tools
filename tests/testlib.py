import dataclasses
import functools
import inspect
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import NoneType
from typing import IO, Dict, List, Optional, Union

from nomad_tools import nomadlib

os.environ.setdefault("NOMAD_NAMESPACE", "default")


def caller(up=0):
    return inspect.stack()[1 + up][3]


def job_exists(jobname):
    try:
        nomadlib.NomadConn().get(f"job/{jobname}")
        return True
    except nomadlib.JobNotFound:
        return False


@functools.lru_cache(maxsize=0)
def nomad_has_docker():
    nodes = json.loads(subprocess.check_output("nomad node status -json".split()))
    for n in nodes:
        if n["Drivers"].get("docker", {}).get("Healthy"):
            return True
    return False


@dataclasses.dataclass
class NamedTemporaryFileContent:
    content: str
    suffix: Optional[str] = None
    file: Optional[IO] = None

    def __enter__(self):
        self.file = tempfile.NamedTemporaryFile("w", suffix=self.suffix)
        self.file.write(self.content)
        self.file.flush()
        return self.file.name

    def __exit__(self, *args):
        if self.file:
            self.file.close()


def gen_job(
    script=""" echo hello world """, name: Optional[str] = None, mode: str = "raw_exec"
):
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
    task = raw_exec_task if mode == "raw_exec" else docker_task
    job = {
        "ID": jobname,
        "Type": "batch",
        "Meta": {
            "TIME": f"{time.time_ns()}",
        },
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


def get_testjobs() -> Dict[str, Path]:
    files = list(Path(f"./jobs/").glob("*.nomad.hcl"))
    assert len(files) > 3
    ret = {f.name[: -len(".nomad.hcl")]: f for f in files}
    print(ret)
    return ret


def quotearr(cmd: List[str]):
    return " ".join(shlex.quote(x) for x in cmd)


def run(
    cmd: str,
    check: Union[bool, int, List[int], NoneType] = True,
    text=True,
    stdout: Union[bool, int] = False,
    output: Union[List[str], List[Union[str, re.Pattern]], List[re.Pattern]] = [],
    input: Optional[str] = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    cmda: List[str] = []
    for i in shlex.split(cmd):
        if i in "nomad-cp nomad-watch nomad-port".split():
            cmda.extend(f"python3 -m nomad_tools.{i.replace('-', '_')} -v".split())
        else:
            cmda.append(i)
    print(" ", file=sys.stderr, flush=True)
    print(f"+ {quotearr(cmda)}", file=sys.stderr, flush=True)
    stdout = stdout or bool(output)
    pp = subprocess.Popen(
        cmda,
        text=text,
        stdout=subprocess.PIPE if stdout else sys.stderr,
        stdin=subprocess.PIPE if input else subprocess.DEVNULL,
        **kwargs,
    )
    captured_stdout = None
    try:
        if input:
            assert pp.stdin
            pp.stdin.write(input)
            pp.stdin.close()
        if stdout:
            captured_stdout = ""
            assert pp.stdout
            for line in pp.stdout:
                line = line.rstrip()
                captured_stdout += line + "\n"
                print(line)
        pp.wait()
    finally:
        pp.terminate()
        pp.wait()
    rr = subprocess.CompletedProcess(cmda, pp.returncode, captured_stdout, None)
    #
    if check is True:
        check = [0]
    if check is None:
        check = None
    elif check is False:
        assert (
            rr.returncode != 0
        ), f"Command {rr} died with zero {rr.returncode}, which is wrong"
        check = None
    if isinstance(check, int):
        check = [check]
    if isinstance(check, list):
        assert (
            rr.returncode in check
        ), f"Command {rr} died with {rr.returncode} not in {check}"
    #
    if output:
        assert rr.stdout is not None
        for pat in output:
            if isinstance(pat, str):
                assert pat in rr.stdout, f"String {pat} not found"
            elif isinstance(pat, re.Pattern):
                assert pat.findall(rr.stdout), f"Pattern {pat} not found"
            else:
                assert 0
    #
    return rr


def run_nomad_cp(cmd: str, **kwargs):
    return run(f"python3 -m nomad_tools.nomad_cp -v {cmd}", **kwargs)


def run_nomad_watch(
    cmd: str,
    pre: str = "",
    check: Union[bool, int, List[int]] = True,
    text=True,
    stdout: Union[bool, int] = False,
    output: Union[List[str], List[Union[str, re.Pattern]], List[re.Pattern]] = [],
    input: Optional[str] = None,
    **kwargs,
):
    return run(
        f"{pre} python3 -m nomad_tools.nomad_watch -v {cmd}",
        check=check,
        text=text,
        stdout=stdout,
        output=output,
        input=input,
        **kwargs,
    )


def run_nomad_vardir(cmd: str, **kwargs):
    return run(f"python3 -m nomad_tools.nomad_vardir -v {cmd}", **kwargs)
