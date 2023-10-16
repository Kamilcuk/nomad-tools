from nomad_tools.common import mynomad
from tests.testlib import get_testjobs, run, run_nomad_watch

testjobs = get_testjobs()


def test_nomad_watch2_canary():
    job = "test-listen"
    output = ["+ hostname", "+ exec httpd"]
    try:
        # Run nice deployment.
        run_nomad_watch(f"start -var ok=true  {testjobs[job]}", output=output)
        out = run(f"nomad-port {job}", stdout=True).stdout
        assert len(out.splitlines()) == 1, f"{out}"
        # This should fail and revert deployment.
        run_nomad_watch(
            f"start -var ok=false {testjobs[job]}",
            output=[
                "Failed due to unhealthy allocations - rolling back to job",
                *output,
            ],
            check=False,
        )
        run_nomad_watch(f"-x stop {job}", output=output)
    finally:
        run_nomad_watch(f"-x purge {job}")


def test_nomad_watch2_blocked():
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
