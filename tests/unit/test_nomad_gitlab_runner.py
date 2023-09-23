import tempfile
from textwrap import dedent

from nomad_tools import nomad_gitlab_runner
from tests.testlib import run


def test_nomad_gitlab_runner_showconfig():
    config = """
[default]
mode = "docker"
"""
    with tempfile.NamedTemporaryFile("w") as configfile:
        configfile.write(dedent(config))
        configfile.flush()
        run(
            f"python3 -m nomad_tools.nomad_gitlab_runner -vvc {configfile.name} showconfig",
        )


def test_nomad_gitlab_runner_templater():
    txt = "$ ${ ${} ${fdsfa} $$"
    assert nomad_gitlab_runner.OnlyBracedCustomEnvTemplate(txt).substitute() == txt
    txt = "${CUSTOM_ENV}"
    assert nomad_gitlab_runner.OnlyBracedCustomEnvTemplate(txt).substitute() == txt
    txt = "${CUSTOM_ENV_ABC}"
    assert (
        nomad_gitlab_runner.OnlyBracedCustomEnvTemplate(txt).substitute(
            {"CUSTOM_ENV_ABC": 123}
        )
        == "123"
    )
