import subprocess

import pytest
import nomad_tools.entry

scripts = nomad_tools.entry.cli.commands.keys()


@pytest.mark.parametrize("cli", scripts)
def test_cli_help(cli):
    subprocess.check_call(["nomadtools", cli, "--help"])


@pytest.mark.parametrize("cli", scripts)
def test_cli_h(cli):
    subprocess.check_call(["nomadtools", cli, "-h"])
