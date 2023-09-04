import subprocess
from pathlib import Path

import pytest
import tomli

rootdir = Path(".")
pyproject = rootdir / "pyproject.toml"
scripts = list(tomli.load(pyproject.open("rb"))["project"]["scripts"].keys())


@pytest.mark.parametrize("cli", scripts)
def test_cli_help(cli):
    subprocess.check_call([cli, "-h"])


@pytest.mark.parametrize("cli", scripts)
def test_cli_h(cli):
    subprocess.check_call([cli, "-h"])
