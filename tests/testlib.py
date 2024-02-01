import contextlib
import dataclasses
import functools
import inspect
import json
import os
import re
import shlex
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import IO, Dict, List, Optional, Union

from nomad_tools import nomadlib
from nomad_tools.common import quotearr

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


def get_testname() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", caller(1)).split(":")[-1].split(" ")[0]


def gen_job(
    script=""" echo hello world """, name: Optional[str] = None, mode: str = "raw_exec"
):
    assert name is None or " " not in name
    name = name or get_testname()
    jobname = f"test-nomad-utils-{name}"
    docker_task = {
        "Driver": "docker",
        "Config": {
            "image": "busybox:stable",
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
    files = list(Path("./jobs/").glob("*.nomad.hcl"))
    assert len(files) > 3
    ret = {f.name[: -len(".nomad.hcl")]: f for f in files}
    # print(ret)
    return ret


@dataclasses.dataclass
class JobHcl:
    id: str
    hcl: str


def get_templatejob(name: str = "", script: str = "") -> JobHcl:
    name = name or get_testname()
    return JobHcl(
        f"test-{name}",
        string.Template(get_testjobs()["test-template"].read_text()).safe_substitute(
            dict(
                NAME=name,
                SCRIPT=script.replace("${", "$${").replace("%{", "%%{"),  # }}}}
            )
        ),
    )


def run(
    cmd: str,
    check: Optional[Union[bool, int, List[int]]] = True,
    text=True,
    stdout: Union[bool, int] = False,
    output: Union[List[str], List[Union[str, re.Pattern]], List[re.Pattern]] = [],
    input: Optional[str] = None,
    timeout: int = 120,
    **kwargs,
) -> subprocess.CompletedProcess:
    """
    Run a command and process its output and exit status for testing.

    :param cmd: command to run
    :param check: if null, don't perform any checks, if true, exit status has to be zero, if false, exit status
    :param text: passed to subprocess.run
    :param input: pass input to subprocess.run
    :param output: an array of strings or patterns to check output of command against
    :param stdout: capture stdout. tru if output is given
    """
    cmd = f"timeout {timeout} {cmd}"
    cmda: List[str] = []
    for i in shlex.split(cmd):
        if i in "nomad-cp nomad-watch nomad-port".split():
            cmda.extend(f"python3 -m nomad_tools.{i.replace('-', '_')} -v".split())
        else:
            cmda.append(i)
    print(" ", file=sys.stderr, flush=True)
    print(f"+ {quotearr(cmda)}", file=sys.stderr, flush=True)
    stdout = stdout or bool(output)
    # Run subprocess.Popen, input stdin and output stdout.
    with subprocess.Popen(
        cmda,
        text=text,
        stdout=subprocess.PIPE if stdout else sys.stderr,
        stdin=subprocess.PIPE if input else subprocess.DEVNULL,
        **kwargs,
    ) as pp:
        if input:
            assert pp.stdin
            pp.stdin.write(input)
            pp.stdin.close()
        captured_stdout = None
        if stdout:
            captured_stdout = ""
            assert pp.stdout
            for line in pp.stdout:
                line = line.rstrip()
                captured_stdout += line + "\n"
                print(line)
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
        print("+ all fine - command is expected to fail")
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
                assert pat in rr.stdout, f"String not found: {pat!r}"
            elif isinstance(pat, re.Pattern):
                assert pat.findall(rr.stdout.strip()), f"Pattern not found: {pat}"
            else:
                assert 0
    #
    return rr


def prefixed_run(prefix: str):
    @functools.wraps(run)
    def inner(cmd: str, *args, **kwargs):
        return run(f"{prefix} {cmd}", *args, **kwargs)

    return inner


run_nomad_cp = prefixed_run("python3 -m nomad_tools.nomad_cp -vv")
run_nomad_watch = prefixed_run("python3 -m nomad_tools.nomad_watch -v")
run_nomad_vardir = prefixed_run("python3 -m nomad_tools.nomad_vardir -v")
run_nomad_dockers = prefixed_run("python3 -m nomad_tools.nomad_dockers -v")
run_downloadrelease = prefixed_run("python3 -m nomad_tools.nomad_downloadrelease")
run_nomadt = prefixed_run("python3 -m nomad_tools.nomadt --verbose")


def run_bash(script: str, **kwargs):
    return run(f"bash -o pipefail -euxc {shlex.quote(script)}", **kwargs)


###############################################################################


@dataclasses.dataclass
class TestNomadVardir:
    prefix: str

    def run(
        self,
        cmds: str,
        check: Optional[Union[bool, int, List[int]]] = True,
        text=True,
        stdout: Union[bool, int] = False,
        output: Union[List[str], List[Union[str, re.Pattern]], List[re.Pattern]] = [],
        input: Optional[str] = None,
        **kwargs,
    ):
        return run_nomad_vardir(
            f"{self.prefix} {cmds}",
            check=check,
            text=text,
            stdout=stdout,
            output=output,
            input=input,
        )


@contextlib.contextmanager
def Chdir(where: Union[Path, str]):
    old_cwd = Path.cwd()
    os.chdir(where)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@contextlib.contextmanager
def nomad_vardir_test():
    with tempfile.TemporaryDirectory() as d:
        with Chdir(d):
            with tempfile.NamedTemporaryFile() as testf:
                t = TestNomadVardir(
                    f"--test {shlex.quote(testf.name)} {shlex.quote(testf.name)}"
                )
                yield t
