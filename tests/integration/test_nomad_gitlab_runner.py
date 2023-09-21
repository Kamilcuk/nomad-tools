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
        nomad_gitlab_runner(configfile, "config")
        nomad_gitlab_runner(configfile, "prepare")
        try:
            with tempfile.NamedTemporaryFile("w") as scriptfile:
                scriptfile.write(dedent(script))
                scriptfile.flush()
                nomad_gitlab_runner(
                    configfile, f"run {quote(scriptfile.name)} build_script"
                )
        finally:
            nomad_gitlab_runner(configfile, "cleanup")


def test_1():
    cycle(
        """
        #!/bin/bash
        echo hello world
        """,
        """
        mode = "raw_exec"
        """,
    )
