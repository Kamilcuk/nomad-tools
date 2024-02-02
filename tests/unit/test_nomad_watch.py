from tests.testlib import run_nomad_watch


def test_nomad_watch_multiple_log():
    rr = run_nomad_watch("--log-long --log-short job -", check=False)
    assert rr.returncode != 0
