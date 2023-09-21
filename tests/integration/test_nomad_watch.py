import datetime
import json
from typing import List

from tests.testlib import caller, gen_job, nomad_has_docker, run, run_nomad_watch


def test_nomad_watch_run_0():
    job = gen_job(script=f"echo hello world")
    run_nomad_watch("--purge --json -", input=json.dumps(job))


def test_nomad_watch_run():
    mark = "5581c3a0-cd72-4b84-8b95-799d1aebe1cd"
    exitstatus = 123
    job = gen_job(script=f"echo {mark}; exit {exitstatus}")
    jobid = job["Job"]["ID"]
    run(f"nomad stop --purge {jobid}", check=False)
    output = run_nomad_watch(
        "--json run -",
        input=json.dumps(job),
        check=exitstatus,
        stdout=1,
    ).stdout
    assert mark in output
    run_nomad_watch(f"--purge stop {jobid}", check=exitstatus)


def test_nomad_watch_start():
    mark = "7bc8413c-8619-48bf-a46d-f42727724632"
    exitstatus = 234
    job = gen_job(script=f"echo {mark} ; exit {exitstatus}")
    jobid = job["Job"]["ID"]
    try:
        run_nomad_watch("--json start -", input=json.dumps(job))
        assert mark in run_nomad_watch(f"started {jobid}", stdout=1).stdout
        cmds = [
            f"stop {jobid}",
            f"stopped {jobid}",
            f"--no-follow job {jobid}",
            f"--no-follow -s out job {jobid}",
            f"--no-follow -s err job {jobid}",
            f"--no-follow -s out -s err job {jobid}",
            f"--no-follow -s all job {jobid}",
        ]
        for cmd in cmds:
            assert mark in run_nomad_watch(cmd, check=exitstatus, stdout=1).stdout
    finally:
        run_nomad_watch(f"stop {jobid}", check=exitstatus)
    run_nomad_watch(f"--purge stop {jobid}", check=exitstatus)


def test_nomad_watch_run_short():
    name = caller()
    spec = f"""
        job "{name}" {{
          type = "batch"
          reschedule {{ attempts = 0 }}
          group "{name}" {{
            restart {{ attempts = 0 }}
            task "{name}" {{
                driver = "{'docker' if nomad_has_docker() else 'raw_exec'}"
                config {{
                    {'image = "busybox"' if nomad_has_docker() else ''}
                    command = "sh"
                    args = ["-xc", <<EOF
                        for i in $(seq 5); do echo MARK $i; sleep 0.123; done
                        EOF
                    ]
              }}
            }}
          }}
        }}
        """
    print(spec)
    output = run_nomad_watch("--purge run -", input=spec, stdout=1).stdout
    assert output.count("sleep 0.123") == 5
    assert output.count("MARK ") == 10


def test_nomad_watch_run_multiple():
    name = caller()
    now = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    hastohave: List[str] = []
    spec = f"""
    job "{name}" {{
        datacenters = ["*"]
        type = "batch"
        reschedule {{
            attempts = 0
        }}
        """
    for i in range(2):
        spec += f"""
            group "{name}_group_{i}" {{
                restart {{
                    attempts = 0
                }}
            """
        for j in range(2):
            txt = f"{now} group_{i} task_{i}_{j}"
            start = f"{txt} START"
            stop = f"{txt} STOP"
            spec += f"""
              task "{name}_task_{i}_{j}" {{
                driver = "{'docker' if nomad_has_docker() else 'raw_exec'}"
                config {{
                  {'image = "busybox"' if nomad_has_docker() else ''}
                  command = "sh"
                  args = ["-xc", <<EOF
                    echo {start}
                    sleep 0.1
                    echo {stop}
                    EOF
                  ]
                }}
              }}
              """
            hastohave += [start, stop]
        spec += f"""
            }}
            """
    spec += f"""
    }}
    """
    print(spec)
    output = run_nomad_watch("--purge run -", input=spec, stdout=1).stdout
    for i in hastohave:
        assert output.count(i) == 2
