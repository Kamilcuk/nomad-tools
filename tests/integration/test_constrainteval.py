import json
import subprocess

from tests.testlib import run_nomadt


def test_constrainteval_1():
    pp = run_nomadt(
        "constrainteval --json node.unique.name is_set", stdout=subprocess.PIPE
    )
    assert pp.stdout
    assert len(json.loads(pp.stdout)) >= 1, f"{pp}"


def test_constrainteval_2():
    pp = run_nomadt("constrainteval --json node.unique.name", stdout=subprocess.PIPE)
    assert pp.stdout
    assert len(json.loads(pp.stdout)) >= 1, f"{pp}"


def test_constrainteval_3():
    pp = run_nomadt(
        "constrainteval --json nfdsafsafdasfd is_set",
        stdout=subprocess.PIPE,
        check=2,
    )
    assert pp.stdout is not None
    assert pp.stdout == ""


def test_constrainteval_prefix():
    pp = run_nomadt("constrainteval --json --prefix attr.", stdout=subprocess.PIPE)
    assert pp.stdout
    assert len(json.loads(pp.stdout)) >= 10, f"{pp}"
