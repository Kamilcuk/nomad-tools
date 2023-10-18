import re
from typing import List, Union

from nomad_tools.common import mynomad
from tests.testlib import get_testjobs, run, run_nomad_watch

testjobs = get_testjobs()


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
        run_nomad_watch(f"-x stop {job}", output=output)
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
            run_nomad_watch(
                f"{i} -var block=true {testjobs[job]}",
                pre="timeout 1",
                check=124,
                output=[
                    "Placement Failures",
                    "Resources exhaused",
                    "Dimension memory exhaused",
                ],
            )
            run_nomad_watch(f"{i} -var block=false {testjobs[job]}", output=["+ true"])
        finally:
            run_nomad_watch(f"-x purge {job}")


def test_nomad_watch2_maintask():
    """
    Run a job that has all combinations of lifetime property.
    Check if proper tasks are started.
    """
    job = "test-maintask"
    try:
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
        assert len(mynomad.get(f"job/{job}/allocations")) == 1
    except:
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
