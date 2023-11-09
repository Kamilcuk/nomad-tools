import re
from shlex import quote
from typing import List, Union

from nomad_tools.common import mynomad
from tests.testlib import get_testjobs, run, run_nomad_watch

testjobs = get_testjobs()


def test_nomad_watch2_start():
    job = "test-start"
    mark = "7bc8413c-8619-48bf-a46d-f42727724632"
    exitstatus = 234
    script = f"sleep 1; echo {mark} ; exit {exitstatus}"
    try:
        run_nomad_watch(f"start -var script={quote(script)} {testjobs[job]}")
        run_nomad_watch(f"started {job}", output=[mark])
        cmds = [
            f"stop {job}",
            f"stopped {job}",
            f"--no-follow job {job}",
            f"--no-follow -o out job {job}",
            f"--no-follow -o err job {job}",
            f"--no-follow -o out -o err job {job}",
            f"--no-follow -o all job {job}",
        ]
        for cmd in cmds:
            run_nomad_watch(cmd, check=exitstatus, output=[mark])
    finally:
        run_nomad_watch(f"stop {job}", check=exitstatus)
    run_nomad_watch(f"--purge stop {job}", check=exitstatus)


def test_nomad_watch2_canary():
    """
    Run a service that listens on a port.
    Check if nomad-port works.
    Then try to upgrade that service so that it fails. Deployment should fail.
    Then try to upgrade that service but with success. Deployment should succeed.
    """
    job = "test-listen"
    output = ["+ hostname", "+ exec httpd", "Deployment completed successfully"]
    try:
        # Run nice deployment.
        run_nomad_watch(f"start -var ok=true  {testjobs[job]}", output=output)
        run(f"nomad-port {job}", output=[re.compile(r"[0-9\.]+:[0-9]+")])
        run(
            f"nomad-port -l {job}",
            output=[re.compile(r"[0-9\.]+:[0-9]+ http [^ ]* [^ ]*")],
        )
        # This should fail and revert deployment.
        run_nomad_watch(
            f"start -var ok=false {testjobs[job]}",
            output=[
                "Failed due to unhealthy allocations - rolling back to job",
            ],
            check=False,
        )
        run_nomad_watch(f"start -var ok=true  {testjobs[job]}", output=output)
        run(f"nomad-port {job}", output=[re.compile(r"[0-9\.]+:[0-9]+")])
        run(
            f"nomad-port -l {job}",
            output=[re.compile(r"[0-9\.]+:[0-9]+ http [^ ]* [^ ]*")],
        )
        run_nomad_watch(f"-x stop {job}")
    finally:
        run_nomad_watch(f"-x purge {job}")


def test_nomad_watch2_blocked():
    """
    Run a job that blocks because not enough memory.
    Then unblock it and run
    """
    job = "test-blocked"
    for i in "start run".split():
        try:
            for j in range(2):
                run_nomad_watch(
                    f"{i} -var block=true {testjobs[job]}",
                    pre="timeout 2",
                    check=124,
                    output=[
                        "Placement Failures",
                        "Resources exhaused",
                        "Dimension memory exhaused",
                    ],
                )
                run_nomad_watch(
                    f"{i} -var block=false {testjobs[job]}", output=["+ true"]
                )
        finally:
            run_nomad_watch(f"-x purge {job}")


def test_nomad_watch2_maintask():
    """
    Run a job that has all combinations of lifetime property.
    Check if proper tasks are started.
    """
    job = "test-maintask"
    try:
        run_nomad_watch(f"purge {job}", check=None)
        out = run_nomad_watch(
            f"start {testjobs[job]}",
            output=[
                "+ echo main",
                "+ echo prestart",
                "+ echo prestart_sidecar",
                "+ echo poststart",
            ],
        ).stdout
        assert "+ echo poststop" not in out
        assert len(mynomad.get(f"job/{job}/allocations")) == 2
    finally:
        run_nomad_watch(f"-x purge {job}")


def test_nomad_watch2_multiple():
    job = "test-multiple"
    uuid = "876f767f-7dbb-4e1f-8625-4dcd39f1adaa"
    output: List[Union[str, re.Pattern]] = []
    for i in range(1, 2):
        for j in range(1, 2):
            for k in range(3):
                pre = f"{uuid} alloc{k} group{i} task{j}"
                output += [
                    f"{pre} START",
                    f"{pre} STOP",
                    re.compile(rf"{pre} START[\S\n ]*{pre} STOP"),
                ]
    run_nomad_watch(f"--purge run {testjobs[job]}", output=output)


def test_nomad_watch2_invalidconfig():
    job = "test-invalidconfig"
    run_nomad_watch(
        f"--purge run {testjobs[job]}",
        check=126,
        output=["Failed Validation 3 errors occurred"],
    )


def test_nomad_watch2_deploymulti():
    job = "test-deploymulti"
    try:
        pp = run_nomad_watch(f"purge {job}", check=None)
        run_nomad_watch(
            f"start {testjobs[job]}",
            output=[
                re.compile(r"( Allocation .* started.*){2}", re.MULTILINE | re.DOTALL),
                re.compile(
                    "Canaries=0/0 Placed=2 Desired=2 Healthy=2 Unhealthy=0 Deployment completed successfully"
                ),
                re.compile("E>[0-9a-fA-F]*>#0>web> .*response"),
            ],
        )
        run_nomad_watch(
            f"start {testjobs[job]}",
            output=[
                re.compile(r"( Allocation .* started.*){2}", re.MULTILINE | re.DOTALL),
                re.compile(
                    r"( Allocation .* finished.*){2}",
                    re.MULTILINE | re.DOTALL,
                ),
                re.compile(
                    "Canaries=1/1 Placed=2 Desired=2 Healthy=2 Unhealthy=0 Deployment completed successfully"
                ),
                re.compile("E>[0-9a-fA-F]*>#1>web> .*response"),
            ],
        )
        run_nomad_watch(
            f"start -var ok=false {testjobs[job]}",
            output=[
                "Failed due to unhealthy allocations - rolling back to job",
                re.compile("E>[0-9a-fA-F]*>#2>web> .*response"),
            ],
            check=False,
        )
    finally:
        pass
        # run_nomad_watch(f"-x purge {job}")
