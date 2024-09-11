import pytest
import yaml

from tests.testlib import run_nomadt


def test_githubrunner_dumpconfig():
    rr = run_nomadt(
        "githubrunner -c '---\nrepos: [test]\nloop: 123' dumpconfig", stdout=True
    )
    assert yaml.safe_load(rr.stdout)
