import contextlib
import os
import tempfile
import time
from pathlib import Path
from shlex import split
from typing import List

from nomad_tools import taskexec
from nomad_tools.common_nomad import mynomad
from nomad_tools.entry_cp import ArgPath
from tests.testlib import get_templatejob, run, run_entry_cp, run_entry_watch


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


@contextlib.contextmanager
def run_temp_job():
    hcl = get_templatejob(script="exec sleep 60")
    jobid = hcl.id
    run_entry_watch(f"-x purge {jobid}")
    try:
        run_entry_watch("start -", input=hcl.hcl)
        with NomadTempdir(jobid) as nomaddir:
            with tempfile.TemporaryDirectory() as hostdir:
                yield jobid, nomaddir, hostdir
    finally:
        run_entry_watch(f"-x purge {jobid}")


def g(incomplete: str) -> List[str]:
    """Wrapper around gen_shell_complete which converts ShellComplete objects to string for easier comparison"""
    os.environ["COMP_DEBUG"] = "1"
    ret = ArgPath.mk(incomplete).gen_shell_complete()
    arr = [x.value for x in ret if x.type == "plain"]
    print(f"gen_shell_complete {incomplete} -> {arr}")
    return arr


def test_entry_cp_complete():
    with run_temp_job() as (jobid, nomaddir, hostdir):
        allocid = mynomad.get(f"job/{jobid}/allocations")[0]["ID"]
        task = group = jobid.replace("test-", "")
        tests = "/t:/tmp/ ./h:./home/ h:home/ /usr/bi:/usr/bin/"
        for src, dst in (x.split(":") for x in tests.split()):
            assert g(f"{jobid}:{src}") == [dst], f"{src} {dst}"
            assert g(f"{jobid}:{group}:{src}") == [dst], f"{src} {dst}"
            assert g(f"{jobid}:{group}::{src}") == [dst], f"{src} {dst}"
            assert g(f"{jobid}:{group}:{task}:{src}") == [dst], f"{src} {dst}"
            assert g(f"{jobid}::{task}:{src}") == [dst], f"{src} {dst}"
            assert g(f":{allocid}:{src}") == [dst], f"{src} {dst}"
            assert g(f":{allocid}:{task}:{src}") == [dst], f"{src} {dst}"
            assert g(f":{allocid[:6]}:{src}") == [dst], f"{src} {dst}"
            assert g(f":{allocid[:6]}:{task}:{src}") == [dst], f"{src} {dst}"
            assert g(f":{allocid[:7]}:{task}:{src}") == [dst], f"{src} {dst}"
            assert g(f":{allocid[:9]}:{task}:{src}") == [dst], f"{src} {dst}"
            assert g(f":{allocid[:14]}:{task}:{src}") == [dst], f"{src} {dst}"
            assert g(f":{allocid[:15]}:{task}:{src}") == [dst], f"{src} {dst}"


def test_entry_cp_dir():
    with run_temp_job() as (jobname, nomaddir, hostdir):
        taskexec.run(
            *taskexec.find_job(jobname),
            split(f"sh -xeuc 'cd {nomaddir} && mkdir -p dir && touch dir/1 dir/2'"),
        )
        run_entry_cp(f"{jobname}:{nomaddir}/dir {hostdir}/dir")
        run_entry_cp(f"{hostdir}/dir {jobname}:{nomaddir}/dir2")
        run_entry_cp(f"{jobname}:{nomaddir}/dir2 {hostdir}/dir2")
        run(f"diff -r {hostdir}/dir {hostdir}/dir2")


def test_entry_cp_file():
    with run_temp_job() as (jobname, nomaddir, hostdir):
        txt = f"{time.time()}"
        Path(f"{hostdir}/file").write_text(txt)
        run_entry_cp(f"{hostdir}/file {jobname}:{nomaddir}/file")
        run_entry_cp(f"{jobname}:{nomaddir}/file {hostdir}/file2")
        with Path(f"{hostdir}/file2").open() as f:
            assert f.read() == txt
