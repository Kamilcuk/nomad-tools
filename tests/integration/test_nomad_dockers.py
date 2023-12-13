from tests.testlib import get_testjobs, run_nomad_dockers


def test_nomad_dockers():
    jobf = get_testjobs()["test-listen"]
    run_nomad_dockers(f"{jobf}", output=["busybox:stable"])
    jobf = get_testjobs()["test-upgrade1"]
    run_nomad_dockers(f"{jobf}", output=["busybox:stable"])
