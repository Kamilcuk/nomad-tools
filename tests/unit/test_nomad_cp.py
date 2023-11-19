from tests.testlib import run_nomad_cp


def test_nomad_cp_test():
    run_nomad_cp("--test")
