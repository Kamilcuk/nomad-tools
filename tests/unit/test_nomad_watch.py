import subprocess


def test_nomad_watch_multiple_log():
    rr = subprocess.run("nomad-watch --log-long --log-short job -".split())
    assert rr.returncode != 0
