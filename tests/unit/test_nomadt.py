import pytest
import yaml

from tests.testlib import run_nomadt


def test_nomadt_help():
    run_nomadt("-h")
    run_nomadt("--help")


@pytest.mark.parametrize(
    "app",
    "constrainteval cp dockers downloadrelease gitlab-runner".split()
    + "githubrunner go port vardir watch".split(),
)
def test_nomadt_app_help(app):
    run_nomadt(f"{app} --help")


def test_githubrunner_dumpconfig():
    rr = run_nomadt("githubrunner -c '---\nrepos: [test]\nloop: 123' dumpconfig", stdout=True)
    assert yaml.safe_load(rr.stdout)
