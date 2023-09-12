import json
import subprocess

from tests.testlib import check_output, gen_job, run


def test_noamd_watch_run_0():
    job = gen_job(script=f"echo hello world")
    run("nomad-watch --purge --json -", input=json.dumps(job))


def test_nomad_watch_run():
    mark = "5581c3a0-cd72-4b84-8b95-799d1aebe1cd"
    job = gen_job(script=f"echo {mark}; exit 123")
    jobid = job["Job"]["ID"]
    run(f"nomad stop --purge {jobid}", check=False)
    rr = run(
        "nomad-watch -v --json run -",
        input=json.dumps(job),
        check=False,
        stdout=subprocess.PIPE,
    )
    print(rr.stdout)
    assert mark in rr.stdout
    assert "exit 123" in rr.stdout
    assert rr.returncode == 123, f"{rr.returncode}"
    run(f"nomad-watch --purge stop {jobid}")


def test_nomad_watch_start():
    mark = "7bc8413c-8619-48bf-a46d-f42727724632"
    exitstatus = 234
    job = gen_job(f"echo {mark} ; exit {exitstatus}")
    jobid = job["Job"]["ID"]
    try:
        print()
        check_output("nomad-watch --json start -", input=json.dumps(job))
        print()
        ret = check_output(f"nomad-watch started {jobid}")
        print(ret)
        assert mark in ret
        print()
        ret = run(f"nomad-watch stop {jobid}", check=False, stdout=subprocess.PIPE)
        print(ret)
        assert mark in ret.stdout
        assert ret.returncode == exitstatus
        print()
        ret = run(f"nomad-watch stopped {jobid}", check=False, stdout=subprocess.PIPE)
        print(ret)
        assert mark in ret.stdout
        assert ret.returncode == exitstatus
        print()
        ret = check_output(f"nomad-watch --no-follow job {jobid}")
        print(ret)
        assert mark in ret
        ret = check_output(f"nomad-watch --no-follow -s out job {jobid}")
        print(ret)
        assert mark in ret
        ret = check_output(f"nomad-watch --no-follow -s err job {jobid}")
        print(ret)
        assert mark in ret
        ret = check_output(f"nomad-watch --no-follow -s out -s err job {jobid}")
        print(ret)
        assert mark in ret
        ret = check_output(f"nomad-watch --no-follow -s all job {jobid}")
        print(ret)
        assert mark in ret
    finally:
        print()
        ret = run(f"nomad-watch stop {jobid}", check=False)
        assert ret.returncode == exitstatus
    run(f"nomad-watch --purge stop {jobid}")
