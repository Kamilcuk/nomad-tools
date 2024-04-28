import json
import shlex

from tests.testlib import run_nomadt


def test_go_json():
    json.loads(run_nomadt("go -O --driver raw_exec command", stdout=True).stdout)


def test_go_auth():
    ret = json.loads(
        run_nomadt(
            "go -O --auth 'username=a password=b' --driver raw_exec command",
            stdout=True,
        ).stdout
    )
    assert ret["TaskGroups"][0]["Tasks"][0]["Config"]["auth"] == {
        "username": "a",
        "password": "b",
    }
    ret = json.loads(
        run_nomadt(
            r"""
            go -O --auth 'username="!@#$%^&*" password="\"'\''\""' --driver raw_exec command
            """,
            stdout=True,
        ).stdout
    )
    assert ret["TaskGroups"][0]["Tasks"][0]["Config"]["auth"] == {
        "username": "!@#$%^&*",
        "password": '"\'"',
    }
    ret = json.loads(
        run_nomadt(
            """go -O --auth '{"username": "a", "password": "b"}' --driver raw_exec command""",
            stdout=True,
        ).stdout
    )
    assert ret["TaskGroups"][0]["Tasks"][0]["Config"]["auth"] == {
        "username": "a",
        "password": "b",
    }


def test_go_raw_exec_cmd():
    ret = json.loads(
        run_nomadt("""go -O --driver raw_exec cmd arg1 arg2""", stdout=True).stdout
    )
    config = ret["TaskGroups"][0]["Tasks"][0]["Config"]
    assert config["command"] == "cmd", config
    assert config["args"] == ["arg1", "arg2"]


def test_go_docker_cmd():
    ret = json.loads(run_nomadt("""go -O image cmd arg1 arg2""", stdout=True).stdout)
    config = ret["TaskGroups"][0]["Tasks"][0]["Config"]
    assert config["image"] == "image"
    assert config["command"] == "cmd"
    assert config["args"] == ["arg1", "arg2"]
    ret = json.loads(
        run_nomadt(
            """go -O --driver docker --image image cmd arg1 arg2""", stdout=True
        ).stdout
    )
    config = ret["TaskGroups"][0]["Tasks"][0]["Config"]
    assert config["image"] == "image"
    assert config["command"] == "cmd"
    assert config["args"] == ["arg1", "arg2"]


def test_go_template():
    txt = "".join(chr(i) for i in range(1, 127))
    arg = f"destination=local/script.sh data={shlex.quote(txt)}"
    ret = json.loads(
        run_nomadt(f"""go -O --template {shlex.quote(arg)} cmd""", stdout=True).stdout
    )
    config1 = ret["TaskGroups"][0]["Tasks"][0]["Config"]
    arg = json.dumps({"DestPath": "local/script.sh", "EmbeddedTmpl": txt})
    ret = json.loads(
        run_nomadt(f"""go -O --template {shlex.quote(arg)} cmd""", stdout=True).stdout
    )
    config2 = ret["TaskGroups"][0]["Tasks"][0]["Config"]
    assert config1 == config2
