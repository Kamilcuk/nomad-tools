import subprocess
from pathlib import Path


def test_entry_cp_test():
    script = Path(__file__).parent / "test_nomad_cp.sh"
    assert script.exists()
    subprocess.check_call(["bash", str(script)])
