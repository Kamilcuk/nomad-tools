import json
import subprocess

from tests.testlib import gen_job, run


def test_nomad_watch_run():
    job = gen_job()
    rr = run(
        "nomad-watch --purge --json run -",
        input=json.dumps(job),
        check=False,
        stdout=subprocess.PIPE,
    )
    print(rr.stdout)
    assert "exit 123" in rr.stdout
    assert rr.returncode == 123, f"{rr.returncode}"


def test_nomad_watch_start():
    job = gen_job()
    jobid = job["Job"]["ID"]
    try:
        run("nomad-watch --json start -", input=json.dumps(job))
        run(f"nomad-watch starting {jobid}")
        run(f"nomad-watch stop {jobid}")
        run(f"nomad-watch stopping {jobid}")
    finally:
        run(f"nomad-watch --purge stop {jobid}")
