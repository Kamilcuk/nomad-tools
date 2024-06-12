from tests.testlib import run_entry_watch


def test_entry_watch_multiple_log():
    rr = run_entry_watch("--log-long --log-short job -", check=False)
    assert rr.returncode != 0
