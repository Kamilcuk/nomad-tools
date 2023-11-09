import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tests.testlib import gen_job, run, run_nomad_cp, run_nomad_watch

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
    jobjson = gen_job(script="exec sleep 60")
    job = jobjson["Job"]
    jobname = job["ID"]
    run_nomad_watch(f"-x purge {jobname}")
    try:
        run_nomad_watch("start -json -", input=json.dumps(jobjson))
        with NomadTempdir(jobname) as nomaddir:
            with tempfile.TemporaryDirectory() as hostdir:
                yield jobname, nomaddir, hostdir
    finally:
        run_nomad_watch(f"-x purge {jobname}")


def test_nomad_cp_dir():
    for jobname, nomaddir, hostdir in run_temp_job():
        run(
            f"{alloc_exec} {jobname} sh -xeuc 'cd {nomaddir} && mkdir -p dir && touch dir/1 dir/2'"
        )
        run_nomad_cp(f"-job {jobname}:{nomaddir}/dir {hostdir}/dir")
        run_nomad_cp(f"-job {hostdir}/dir {jobname}:{nomaddir}/dir2")
        run_nomad_cp(f"-job {jobname}:{nomaddir}/dir2 {hostdir}/dir2")
        run(f"diff -r {hostdir}/dir {hostdir}/dir2")


def test_nomad_cp_file():
    for jobname, nomaddir, hostdir in run_temp_job():
        txt = f"{time.time()}"
        with Path(f"{hostdir}/file").open("w") as f:
            f.write(txt)
        run_nomad_cp(f"-job {hostdir}/file {jobname}:{nomaddir}/file")
        run_nomad_cp(f"-job {jobname}:{nomaddir}/file {hostdir}/file2")
        with Path(f"{hostdir}/file2").open() as f:
            assert f.read() == txt
