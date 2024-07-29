import os
import tempfile

import pytest

from tests.testlib import run_downloadrelease


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


@pytest.mark.skip()  # too slow
@pytest.mark.parametrize("tool", "nomad consul".split())
def test_downloadrelease(tool):
    with tempfile.TemporaryDirectory() as d:
        run_downloadrelease(f"{tool} {d}")
        toolf = f"{d}/{tool}"
        assert os.path.exists(toolf)
        os.remove(toolf)


@pytest.mark.parametrize("tool", "nomad consul".split())
def test_downloadrelease_version(tool):
    rr = run_downloadrelease(f"--showversion {tool}", stdout=True)
    assert len(versiontuple(rr.stdout)) == 3
    assert versiontuple(rr.stdout) > versiontuple("1.8.0")
