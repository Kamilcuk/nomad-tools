import json
import subprocess

from tests.testlib import (
    check_output,
    check_output_nomad_watch,
    gen_job,
    run,
    run_nomad_watch,
)


def test_nomad_watch_run_0():
    job = gen_job(script=f"echo hello world")
    run_nomad_watch("--purge --json -", input=json.dumps(job))


def test_nomad_watch_run():
    mark = "5581c3a0-cd72-4b84-8b95-799d1aebe1cd"
    job = gen_job(script=f"echo {mark}; exit 123")
    jobid = job["Job"]["ID"]
    run(f"nomad stop --purge {jobid}", check=False)
    rr = run_nomad_watch(
        "--json run -",
        input=json.dumps(job),
        check=False,
        stdout=subprocess.PIPE,
    )
    print(rr.stdout)
    assert mark in rr.stdout
    assert "exit 123" in rr.stdout
    assert rr.returncode == 123, f"{rr.returncode}"
    run_nomad_watch(f"--purge stop {jobid}")


def test_nomad_watch_start():
    mark = "7bc8413c-8619-48bf-a46d-f42727724632"
    exitstatus = 234
    job = gen_job(f"echo {mark} ; exit {exitstatus}")
    jobid = job["Job"]["ID"]
    try:
        print()
        run_nomad_watch("--json start -", input=json.dumps(job))
        print()
        ret = check_output_nomad_watch(f"started {jobid}")
        print(ret)
        assert mark in ret
        print()
        ret = run_nomad_watch(f"stop {jobid}", check=False, stdout=subprocess.PIPE)
        print(ret)
        assert mark in ret.stdout
        assert ret.returncode == exitstatus
        print()
        ret = run_nomad_watch(f"stopped {jobid}", check=False, stdout=subprocess.PIPE)
        print(ret)
        assert mark in ret.stdout
        assert ret.returncode == exitstatus
        print()
        ret = check_output_nomad_watch(f"--no-follow job {jobid}")
        print(ret)
        assert mark in ret
        ret = check_output_nomad_watch(f"--no-follow -s out job {jobid}")
        print(ret)
        assert mark in ret
        ret = check_output_nomad_watch(f"--no-follow -s err job {jobid}")
        print(ret)
        assert mark in ret
        ret = check_output_nomad_watch(f"--no-follow -s out -s err job {jobid}")
        print(ret)
        assert mark in ret
        ret = check_output_nomad_watch(f"--no-follow -s all job {jobid}")
        print(ret)
        assert mark in ret
    finally:
        print()
        ret = run_nomad_watch(f"stop {jobid}", check=False)
        assert ret.returncode == exitstatus
    run_nomad_watch(f"--purge stop {jobid}")
