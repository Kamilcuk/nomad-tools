import json
import subprocess

from tests.testlib import gen_job, run, caller, check_output


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
    job = gen_job(f"echo {mark}")
    jobid = job["Job"]["ID"]
    try:
        print()
        check_output("nomad-watch --json start -", input=json.dumps(job))
        print()
        ret = check_output(f"nomad-watch started {jobid}")
        print(ret)
        assert mark in ret
        print()
        ret = check_output(f"nomad-watch stop {jobid}")
        print(ret)
        assert mark in ret
        print()
        ret = check_output(f"nomad-watch stopped {jobid}")
        print(ret)
        assert mark in ret
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
        run(f"nomad-watch stop {jobid}")
    run(f"nomad-watch --purge stop {jobid}")
