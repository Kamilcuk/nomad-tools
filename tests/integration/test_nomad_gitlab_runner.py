import dataclasses
import json
import os
import time
from shlex import quote
from textwrap import dedent
from typing import Dict, List

from tests.testlib import NamedTemporaryFileContent, get_testname, run


@dataclasses.dataclass
class GitlabState:
    script: str
    configfile: str
    env: Dict[str, str]
    build_failure: int = 33
    system_failure: int = 34

    def __post_init__(self):
        self.script = dedent(self.script)
        self.env.update(os.environ)
        defaults: Dict[str, str] = dict(
            CUSTOM_ENV_CI_JOB_ID=str(hash(time.time_ns())),
            CUSTOM_ENV_CI_RUNNER_ID=str(hash(time.time_ns())),
            CUSTOM_ENV_CI_PROJECT_PATH_SLUG=get_testname(),
            CUSTOM_ENV_CI_CONCURRENT_ID="ID",
            CUSTOM_ENV_CI_JOB_TIMEOUT="60",
            BUILD_FAILURE_EXIT_CODE=str(self.build_failure),
            SYSTEM_FAILURE_EXIT_CODE=str(self.system_failure),
        )
        for k, v in defaults.items():
            self.env.setdefault(k, v)

    def nomad_gitlab_runner(self, cmd: str, check: List[int], **kwargs):
        # [0, self.build_failure, self.system_failure],
        return run(
            f"python3 -m nomad_tools.nomad_gitlab_runner -v -c {quote(self.configfile)} {cmd}",
            env=self.env,
            check=check,
            **kwargs,
        )

    def stage_config(self):
        driverconfig = json.loads(
            self.nomad_gitlab_runner("config", check=[0], stdout=1).stdout
        )
        assert isinstance(driverconfig["builds_dir"], str)
        assert isinstance(driverconfig["cache_dir"], str)
        driverenv = driverconfig["job_env"]
        assert all(isinstance(key, str) for key in driverenv.keys())
        assert all(isinstance(val, str) for val in driverenv.values())
        self.env.update(driverenv)

    def stage_prepare(self):
        return self.nomad_gitlab_runner("prepare", check=[0])

    def stage_script(self, scriptfile: str, stage: str):
        return self.nomad_gitlab_runner(
            f"run {quote(scriptfile)} {quote(stage)}", check=[0]
        )

    def stage_cleanup(self):
        return self.nomad_gitlab_runner("cleanup", check=[0])


def cycle(config: dict, script: str, env: Dict[str, str] = {}):
    for k, v in env.items():
        assert not k.startswith("CUSTOM_ENV_")
    env = {f"CUSTOM_ENV_{k}": v for k, v in env.items()}
    with NamedTemporaryFileContent(json.dumps(config), suffix=".json") as configfile:
        gl = GitlabState(script, configfile, dict(env))
        gl.stage_config()
        try:
            gl.stage_prepare()
            with NamedTemporaryFileContent(
                "#!/bin/bash\nset -xeuo pipefail\n" + script,
                suffix=".sh",
            ) as scriptfile:
                gl.stage_script(scriptfile, "build_stage")
        finally:
            gl.stage_cleanup()


raw_exec_config = {"default": {"mode": "raw_exec", "raw_exec": {"user": ""}}}
docker_config = {
    "default": {"mode": "docker", "docker": {"image": "docker:stable", "privileged": True}}
}


def test_nomad_gitlab_runner_raw_exec():
    cycle(raw_exec_config, "echo hello world")


def test_nomad_gitlab_runner_docker():
    cycle(docker_config, "echo hello world ")


docker_test_script = (
    "env | grep DOCKER_ && docker info && docker run --rm alpine echo hello world"
)


def test_nomad_gitlab_runner_dockerd_tls():
    cycle(
        docker_config,
        docker_test_script,
        dict(
            DOCKER_HOST="tcp://docker:2376",
            DOCKER_TLS_CERTDIR="/alloc",
            DOCKER_CERT_PATH="/alloc/client",
            DOCKER_TLS_VERIFY="1",
            CI_JOB_SERVICES=json.dumps(["docker:dind"]),
        ),
    )


def test_nomad_gitlab_runner_dockerd_notls():
    cycle(
        docker_config,
        docker_test_script,
        dict(
            DOCKER_HOST="tcp://docker:2375",
            DOCKER_TLS_CERTDIR="",
            CI_JOB_SERVICES=json.dumps(["docker:dind"]),
        ),
    )


def test_nomad_gitlab_runner_alias():
    cycle(
        {
            "default": {
                "mode": "docker",
                "docker": {"image": "nginxdemos/hello:0.3"},
            }
        },
        """
        curl http://helloalias
        """,
        dict(
            CI_JOB_SERVICES=json.dumps(
                [{"name": "nginxdemos/hello:0.3", "alias": "helloalias"}]
            ),
        ),
    )
