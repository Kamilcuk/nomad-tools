from textwrap import dedent

from nomad_tools import nomad_gitlab_runner
from tests.testlib import NamedTemporaryFileContent, run_nomadt


def test_nomad_gitlab_runner_showconfig():
    config = """
default:
    mode: docker
"""
    with NamedTemporaryFileContent(dedent(config)) as configfile:
        run_nomadt(f"gitlab-runner -vvc {configfile} showconfig")


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
