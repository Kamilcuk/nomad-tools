import datetime
import json
import time
from typing import Dict, List

from nomad_tools import nomadlib
from tests.testlib import caller, gen_job, job_exists, nomad_has_docker, run_entry_watch


def test_entry_watch_run_0():
    """Watch a simple jow that outputs hello world"""
    job = gen_job(script="echo hello world")
    run_entry_watch(
        "--purge run -json -", input=json.dumps(job), output=["hello world"]
    )


def test_entry_watch_run():
    """Run a simple job, then purge it. Check if we have our uuid on output"""
    mark = "5581c3a0-cd72-4b84-8b95-799d1aebe1cd"
    exitstatus = 123
    job = gen_job(script=f"echo {mark}; exit {exitstatus}")
    jobid = job["Job"]["ID"]
    run_entry_watch(f"-x purge {jobid}")
    run_entry_watch(
        "run -json -",
        input=json.dumps(job),
        check=exitstatus,
        output=[mark],
    )
    run_entry_watch(f"purge {jobid}", check=exitstatus)


def test_entry_watch_start():
    mark = "7bc8413c-8619-48bf-a46d-f42727724632"
    exitstatus = 234
    job = gen_job(script=f"sleep 2; echo {mark} ; exit {exitstatus}")
    jobid = job["Job"]["ID"]
    try:
        run_entry_watch("start -json -", input=json.dumps(job))
        assert mark in run_entry_watch(f"started {jobid}", stdout=1).stdout
        cmds = [
            f"stop {jobid}",
            f"stopped {jobid}",
            f"--no-follow job {jobid}",
            f"--no-follow -o stdout job {jobid}",
            f"--no-follow -o stderr job {jobid}",
            f"--no-follow -o stdout,stderr job {jobid}",
            f"--no-follow -o all job {jobid}",
        ]
        for cmd in cmds:
            assert mark in run_entry_watch(cmd, check=exitstatus, stdout=1).stdout
    finally:
        run_entry_watch(f"-o none stop {jobid}", check=exitstatus)
    run_entry_watch(f"-o none --purge stop {jobid}", check=exitstatus)


def test_entry_watch_run_short():
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
                    {'image = "busybox:stable"' if nomad_has_docker() else ''}
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
    try:
        output = run_entry_watch("-n -1 run -", input=spec, stdout=1).stdout
        assert output.count("sleep 0.123") == 5
        assert output.count("MARK ") == 10
    finally:
        run_entry_watch(f"-x purge {name}")


def test_entry_watch_run_multiple():
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
                  {'image = "busybox:stable"' if nomad_has_docker() else ''}
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
        spec += """
            }
            """
    spec += """
    }
    """
    print(spec)
    output = run_entry_watch(
        "--shutdown-timeout 3 --purge run -", input=spec, stdout=1
    ).stdout
    for i in hastohave:
        assert output.count(i) == 2, f"{output}.count({i}) != 2"


def test_entry_watch_purge_successful_0():
    job = gen_job("exit 0")
    jobid = job["Job"]["ID"]
    run_entry_watch("--purge-successful run -json -", input=json.dumps(job), check=[0])
    assert not job_exists(jobid)


def test_entry_watch_purge_successful_123():
    job = gen_job("exit 123")
    jobid = job["Job"]["ID"]
    try:
        run_entry_watch(
            "--purge-successful run -json -", input=json.dumps(job), check=[123]
        )
        assert job_exists(jobid)
    finally:
        run_entry_watch(f"-x --purge stop {jobid}")
        assert not job_exists(jobid)


def gen_task(name: str, script: str, add: dict = {}):
    return {
        "Name": name,
        "Driver": "raw_exec",
        "config": {"command": "sh", "args": ["-xc", nomadlib.escape(script)]},
        **add,
    }


def test_entry_watch_starting_with_preinit_tasks():
    jobid = caller()
    job = {
        "Job": {
            "ID": jobid,
            "Type": "batch",
            "Meta": {
                "TIME": f"{time.time_ns()}",
            },
            "TaskGroups": [
                {
                    "Name": jobid,
                    "Tasks": [
                        gen_task("main", "sleep 60"),
                        gen_task(
                            "prestart",
                            "echo prestart",
                            {"Lifecycle": {"Hook": "prestart"}},
                        ),
                        gen_task(
                            "prestart_sidecar",
                            "sleep 60",
                            {
                                "Lifecycle": {
                                    "Hook": "prestart",
                                    "Sidecar": True,
                                }
                            },
                        ),
                        gen_task(
                            "poststart",
                            "sleep 60",
                            {"Lifecycle": {"Hook": "poststart"}},
                        ),
                        gen_task(
                            "poststop", "sleep 1", {"Lifecycle": {"Hook": "poststop"}}
                        ),
                    ],
                }
            ],
        }
    }
    try:
        run_entry_watch("start -json -", input=json.dumps(job))
        assert job_exists(jobid)
        allocs = [
            nomadlib.Alloc(x)
            for x in nomadlib.NomadConn().get(f"job/{jobid}/allocations")
        ]
        allocs = [x for x in allocs if x.ClientStatus == "running"]
        assert allocs
        allocs.sort(key=lambda x: x.ModifyIndex, reverse=True)
        lastalloc: nomadlib.Alloc = allocs[0]
        states: Dict[str, nomadlib.AllocTaskState] = lastalloc.get_taskstates()
        assert states["main"].was_started()
        assert not states["main"].FinishedAt
        assert states["prestart"].was_started()
        assert states["prestart"].FinishedAt
        assert states["prestart_sidecar"].was_started()
        assert not states["prestart_sidecar"].FinishedAt
        assert states["poststart"].was_started()
        assert not states["poststart"].FinishedAt
        assert not states["poststop"].was_started()
        #
    finally:
        run_entry_watch(f"-x stop {jobid}")
    assert job_exists(jobid)
    allocs = [
        nomadlib.Alloc(x) for x in nomadlib.NomadConn().get(f"job/{jobid}/allocations")
    ]
    assert allocs
    allocs.sort(key=lambda x: x.ModifyIndex, reverse=True)
    lastalloc = allocs[0]
    states = lastalloc.get_taskstates()
    #
    tmp = [f"name={n} finishedat={s.FinishedAt}" for n, s in states.items()]
    assert all([s.FinishedAt for s in states.values()]), f"{states} | {tmp}"
    #
    run_entry_watch(f"-xn0 --purge stop {jobid}")
    assert not job_exists(jobid)
