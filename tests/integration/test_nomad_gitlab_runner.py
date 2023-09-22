import json
import os
import tempfile
from shlex import quote
from textwrap import dedent
from typing import IO, Dict, Optional

from tests.testlib import run


def nomad_gitlab_runner(
    configfile: IO, cmd: str, env: Optional[Dict[str, str]] = None, **kwargs
):
    env = dict(env or {})
    env.update(os.environ)
    defaults: Dict[str, str] = dict(
        CUSTOM_ENV_CI_PROJECT_PATH_SLUG="test",
        CUSTOM_ENV_CI_CONCURRENT_ID="123",
        CUSTOM_ENV_CI_JOB_TIMEOUT="10",
    )
    for k, v in defaults.items():
        env.setdefault(k, v)
    #
    return run(
        f"python -m nomad_tools.nomad_gitlab_runner -vvc {quote(configfile.name)} {cmd}",
        env=env,
        **kwargs,
    )


def cycle(script: str, config: str = ""):
    with tempfile.NamedTemporaryFile("w") as configfile:
        configfile.write(dedent(config))
        configfile.flush()
        driverconfig = json.loads(nomad_gitlab_runner(configfile, "config", stdout=1).stdout)
        assert isinstance(driverconfig["builds_dir"], str)
        assert isinstance(driverconfig["cache_dir"], str)
        driverenv = driverconfig["job_env"]
        assert all(isinstance(key, str) for key in driverenv.keys())
        assert all(isinstance(val, str) for val in driverenv.values())
        nomad_gitlab_runner(configfile, "prepare", env=driverenv)
        try:
            with tempfile.NamedTemporaryFile("w") as scriptfile:
                scriptfile.write(dedent(script))
                scriptfile.flush()
                nomad_gitlab_runner(
                    configfile,
                    f"run {quote(scriptfile.name)} build_script",
                    env=driverenv,
                )
        finally:
            nomad_gitlab_runner(configfile, "cleanup", env=driverenv)


def test_nomad_gitlab_runner_1():
    cycle(
        """
        #!/bin/bash
        echo hello world
        """,
        """
        [default]
        mode = "raw_exec"
        """,
    )
