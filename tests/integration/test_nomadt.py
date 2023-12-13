import shutil
import subprocess

import pytest

from tests.testlib import run_nomadt


@pytest.mark.skipif(not shutil.which("nomad"), reason="requires nomad executable")
def test_nomadt_vs_nomad():
    assert run_nomadt("var --help", stdout=1).stdout == subprocess.check_output(
        "nomad var --help".split(), text=True
    )
