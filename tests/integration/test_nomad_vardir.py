import tempfile

from nomad_tools.common import mynomad
from tests.testlib import caller, run, run_nomad_vardir


def test_nomad_vardir_1():
    name = caller()
    try:
        with tempfile.TemporaryDirectory() as d:
            #
            run(
                "bash -c 'mkdir -p 1 && cd 1 && mkdir -p a && echo 123 > a/b && echo 234 > c'",
                cwd=d,
            )
            rr = run_nomad_vardir(f"-p {name} diff .", cwd=f"{d}/1", check=False)
            assert rr.returncode != 0
            run_nomad_vardir(f"-p {name} put .", cwd=f"{d}/1")
            run_nomad_vardir(f"-p {name} diff .", cwd=f"{d}/1")
            back = mynomad.variables.read(name).Items
            assert back == {"a/b": "123\n", "c": "234\n"}
            #
            run_nomad_vardir(f"-p {name} get 1b", cwd=d)
            run("diff -r 1 1b", cwd=d)
            #
            run(
                "bash -c 'mkdir -p 2 && cd 2 && mkdir -p a && echo 234 > a/b && echo 345 > c'",
                cwd=d,
            )
            run_nomad_vardir(f"-p {name} put .", cwd=f"{d}/2")
            #
            back = mynomad.variables.read(name).Items
            assert back == {"a/b": "234\n", "c": "345\n"}
            #
            run_nomad_vardir(f"-p {name} get 2b", cwd=d)
            run("diff -r 2 2b", cwd=d)
    finally:
        run(f"nomad var purge {name}")
