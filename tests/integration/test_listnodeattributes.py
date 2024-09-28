import json
import subprocess

from tests.testlib import run_nomadt


def test_listnodeattributes_1():
    pp = run_nomadt("listnodeattributes --json", stdout=subprocess.PIPE)
    assert pp.stdout
    name = json.loads(pp.stdout)[0]["node.unique.name"]
    pp = run_nomadt(f"listnodeattributes --json {name}", stdout=subprocess.PIPE)
    assert pp.stdout
    assert json.loads(pp.stdout)[0]["node.unique.name"] == name
    pp = run_nomadt("listnodeattributes", stdout=subprocess.PIPE)
    assert pp.stdout
    pp = run_nomadt(f"listnodeattributes {name}", stdout=subprocess.PIPE)
    assert pp.stdout
