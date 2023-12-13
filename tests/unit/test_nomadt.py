import shutil
import subprocess

import pytest

from tests.testlib import run_nomadt


@pytest.mark.skipif(not shutil.which("nomad-watch"), reason="requires nomad-watch exec")
def test_nomadt_vs_nomad_watch():
    assert run_nomadt("watch --help", stdout=1).stdout == subprocess.check_output(
        "nomad-watch --help".split(), text=True
    )


@pytest.mark.skipif(not shutil.which("nomad-watch"), reason="requires nomad-watch exec")
def test_nomadt_vs_nomad_watch_alloc():
    assert run_nomadt("watch alloc --help", stdout=1).stdout == subprocess.check_output(
        "nomad-watch alloc --help".split(), text=True
    )


@pytest.mark.skipif(not shutil.which("nomad-port"), reason="requires nomad-watch exec")
def test_nomadt_vs_nomad_port():
    assert run_nomadt("port --help", stdout=1).stdout == subprocess.check_output(
        "nomad-port --help".split(), text=True
    )
