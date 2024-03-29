import os
import tempfile
import time
from pathlib import Path
from shlex import split
from typing import List

from nomad_tools import taskexec
from nomad_tools.common import mynomad
from nomad_tools.nomad_cp import NomadOrHostMyPath
from tests.testlib import get_templatejob, run, run_nomad_cp, run_nomad_watch


class NomadTempdir:
    def __init__(self, jobid: str):
        self.jobid = jobid
        self.allocid, self.task = taskexec.find_job(self.jobid)

    def __enter__(self):
        return taskexec.check_output(
            self.allocid,
            self.task,
            split("sh -xeuc 'echo $NOMAD_TASK_DIR'"),
            text=True,
        ).strip()

    def __exit__(self, type, value, traceback):
        return taskexec.run(
            self.allocid, self.task, split("sh -xeuc 'rm -rv $NOMAD_TASK_DIR'")
        )


def run_temp_job():
    hcl = get_templatejob(script="exec sleep 60")
    jobid = hcl.id
    run_nomad_watch(f"-x purge {jobid}")
    try:
        run_nomad_watch("start -", input=hcl.hcl)
        with NomadTempdir(jobid) as nomaddir:
            with tempfile.TemporaryDirectory() as hostdir:
                yield jobid, nomaddir, hostdir
    finally:
        run_nomad_watch(f"-x purge {jobid}")


def g(incomplete: str) -> List[str]:
    """Wrapper around gen_shell_complete which converts ShellComplete objects to string for easier comparison"""
    os.environ["COMP_DEBUG"] = "1"
    ret = NomadOrHostMyPath.gen_shell_complete(incomplete)
    arr = [x.value for x in ret if x.type == "plain"]
    print(f"gen_shell_complete {incomplete} -> {arr}")
    return arr


def test_nomad_cp_complete():
    hcl = get_templatejob(script="exec sleep 60")
    jobid = hcl.id
    run_nomad_watch(f"-x purge {jobid}")
    try:
        run_nomad_watch("start -", input=hcl.hcl)
        allocid = mynomad.get(f"job/{jobid}/allocations")[0]["ID"]
        task = group = jobid.replace("test-", "")
        tests = "/t:/tmp/ ./h:./home/ h:home/ /usr/bi:/usr/bin/"
        for src, dst in (x.split(":") for x in tests.split()):
            assert g(f"{jobid}:{src}") == [dst], f"{src} {dst}"
            assert g(f"{jobid}:{group}:{src}") == [dst], f"{src} {dst}"
            assert g(f"{jobid}:{group}:{task}:{src}") == [dst], f"{src} {dst}"
            assert g(f":{allocid}:{src}") == [dst], f"{src} {dst}"
            assert g(f":{allocid}:{group}:{src}") == [dst], f"{src} {dst}"
            assert g(f":{allocid}:{group}:{task}:{src}") == [dst], f"{src} {dst}"
            assert g(f":{allocid[:6]}:{src}") == [dst], f"{src} {dst}"
            assert g(f":{allocid[:6]}:{group}:{src}") == [dst], f"{src} {dst}"
            assert g(f":{allocid[:6]}:{group}:{task}:{src}") == [dst], f"{src} {dst}"
    finally:
        run_nomad_watch(f"-x purge {jobid}")


def test_nomad_cp_dir():
    for jobname, nomaddir, hostdir in run_temp_job():
        taskexec.run(
            *taskexec.find_job(jobname),
            split(f"sh -xeuc 'cd {nomaddir} && mkdir -p dir && touch dir/1 dir/2'"),
        )
        run_nomad_cp(f"{jobname}:{nomaddir}/dir {hostdir}/dir")
        run_nomad_cp(f"{hostdir}/dir {jobname}:{nomaddir}/dir2")
        run_nomad_cp(f"{jobname}:{nomaddir}/dir2 {hostdir}/dir2")
        run(f"diff -r {hostdir}/dir {hostdir}/dir2")


def test_nomad_cp_file():
    for jobname, nomaddir, hostdir in run_temp_job():
        txt = f"{time.time()}"
        with Path(f"{hostdir}/file").open("w") as f:
            f.write(txt)
        run_nomad_cp(f"{hostdir}/file {jobname}:{nomaddir}/file")
        run_nomad_cp(f"{jobname}:{nomaddir}/file {hostdir}/file2")
        with Path(f"{hostdir}/file2").open() as f:
            assert f.read() == txt
