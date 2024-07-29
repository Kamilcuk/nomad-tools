import os
import tempfile

import pytest

from tests.testlib import run_downloadrelease


@pytest.mark.skip()
@pytest.mark.parametrize("tool", "nomad consul".split())
def test_downloadrelease(tool):
    with tempfile.TemporaryDirectory() as d:
        run_downloadrelease(f"{tool} {d}")
        toolf = f"{d}/{tool}"
        assert os.path.exists(toolf)
        os.remove(toolf)
