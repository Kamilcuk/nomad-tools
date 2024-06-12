from tests.testlib import get_testjobs, run_entry_dockers


def test_entry_dockers():
    # This is an integration test, because nomad-vardir calls nomad executable to parse HCL.
    run_entry_dockers("", check=False)
    jobf = get_testjobs()["test-listen"]
    run_entry_dockers(f"{jobf}", output=["busybox:stable"])
    jobf = get_testjobs()["test-upgrade1"]
    run_entry_dockers(f"{jobf}", output=["busybox:stable"])
