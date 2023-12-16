import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from nomad_tools.common import mynomad
from nomad_tools.nomad_cp import NomadOrHostMyPath
from tests.testlib import get_templatejob, run, run_nomad_cp, run_nomad_watch

alloc_exec = "nomad alloc exec -i=false -t=false -job"


@dataclass
class NomadTempdir:
    jobid: str

    def __enter__(self):
        return run(
            f"{alloc_exec} {self.jobid} sh -xeuc 'echo $NOMAD_TASK_DIR'",
            stdout=1,
        ).stdout.strip()

    def __exit__(self, type, value, traceback):
        return run(f"{alloc_exec} {self.jobid} sh -xeuc 'rm -rv $NOMAD_TASK_DIR'")


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
    arr = [x.value if x.type == "plain" else x.type for x in ret]
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
        run(
            f"{alloc_exec} {jobname} sh -xeuc 'cd {nomaddir} && mkdir -p dir && touch dir/1 dir/2'"
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
