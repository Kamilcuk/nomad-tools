import shlex
import subprocess
import json
import inspect

def caller(up=0):
    return inspect.stack()[1 + up][3]

def gen_job():
    script = """
        seq 10
        exit 123
    """
    jobname = f"test-nomad-utils-{caller(1)}"
    return {
        "Job": {
            "ID": jobname,
            "Type": "batch",
            "TaskGroups": [
                {
                    "Name": jobname,
                    "ReschedulePolicy": {"Attempts": 0},
                    "RestartPolicy": {"Attempts": 0},
                    "Tasks": [
                        {
                            "Name": jobname,
                            "Driver": "docker",
                            "Config": {
                                "image": "busybox",
                                "command": "sh",
                                "args": ["-xc", script],
                            },
                        }
                    ],
                }
            ],
        }
    }

def run(cmd: str, check=True, text=True, **kvargs):
    cmda = shlex.split(cmd)
    print(f"+ {shlex.join(cmda)}")
    return subprocess.run(cmda, check=check, text=text, **kvargs)


def check_output(cmd: str, **kvargs):
    return run(cmd, stdout=subprocess.STDOUT, **kvargs).stdout
