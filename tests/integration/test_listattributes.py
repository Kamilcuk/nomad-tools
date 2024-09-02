import json
import subprocess

from tests.testlib import run_nomadt


def test_listattributes_1():
    pp = run_nomadt("listattributes --json", stdout=subprocess.PIPE)
    assert pp.stdout
    name = json.loads(pp.stdout)[0]["node.unique.name"]
    pp = run_nomadt(f"listattributes --json {name}", stdout=subprocess.PIPE)
    assert pp.stdout
    assert json.loads(pp.stdout)[0]["node.unique.name"] == name
