import re
import uuid

import pytest

from nomad_tools.entry_vardir import human_size
from tests.testlib import entry_vardir_test, run_bash


def test_entry_vardir_test():
    with entry_vardir_test() as t:
        val = str(uuid.uuid4())
        # putting works?
        t.run("ls", output=[re.compile("^$")])
        run_bash(f"echo {val} > a")
        t.run("put a")
        t.run("cat a", output=[f"{val}"])
        t.run("diff a")
        t.run("ls", output=[re.compile(r"a +[0-9]+$")])
        # getting works?
        run_bash("rm a")
        t.run("cat a", output=[f"{val}"])
        t.run("diff a", check=False)
        t.run("get a")
        t.run("diff a")


def test_entry_vardir_test2():
    with entry_vardir_test() as t:
        t.run("put a", check=False)
        t.run("diff", check=False)
        t.run("get a", check=False)


def test_units():
    assert human_size("1") == 1
    assert human_size("1K") == 1024
    assert human_size("1M") == 1024 * 1024
    assert human_size(" 1 M") == 1024 * 1024
    assert human_size(" 0.5 M") == 0.5 * 1024 * 1024
    with pytest.raises(ValueError):
        human_size("1X")
    with pytest.raises(ValueError):
        human_size("1KB")
    with pytest.raises(ValueError):
        human_size("1KM")
