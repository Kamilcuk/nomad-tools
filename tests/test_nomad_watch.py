import json
import subprocess

from tests.testlib import gen_job, run


def test_nomad_watch_run():
    mark = "5581c3a0-cd72-4b84-8b95-799d1aebe1cd"
    job = gen_job(script=f"echo {mark}; exit 123")
    rr = run(
        "nomad-watch --purge --json run -",
        input=json.dumps(job),
        check=False,
        stdout=subprocess.PIPE,
    )
    print(rr.stdout)
    assert mark in rr.stdout
    assert "exit 123" in rr.stdout
    assert rr.returncode == 123, f"{rr.returncode}"


def test_nomad_watch_start():
    job = gen_job()
    jobid = job["Job"]["ID"]
    try:
        print()
        run("nomad-watch --json start -", input=json.dumps(job))
        print()
        run(f"nomad-watch starting {jobid}")
        print()
        run(f"nomad-watch stop {jobid}")
        print()
        run(f"nomad-watch stopping {jobid}")
        print()
    finally:
        print()
        run(f"nomad-watch --purge stop {jobid}")
