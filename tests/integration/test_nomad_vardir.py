import tempfile

from nomad_tools.common import mynomad
from tests.testlib import caller, run, run_nomad_vardir


class VardirCaller:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, cmd, **kvargs):
        return run_nomad_vardir(f"{self.name} {cmd}", **kvargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        run(f" nomad var purge {self.name}", check=None)


def test_nomad_vardir_1():
    with VardirCaller(caller()) as vardir:
        with tempfile.TemporaryDirectory() as d:
            #
            run(
                "bash -c 'mkdir -p 1 && cd 1 && mkdir -p a && echo 123 > a/b && echo 234 > c'",
                cwd=d,
            )
            vardir("diff .", cwd=f"{d}/1", check=False)
            vardir("put -r .", cwd=f"{d}/1")
            vardir("diff .", cwd=f"{d}/1")
            back = mynomad.variables.read(vardir.name).Items
            assert back == {"a/b": "123\n", "c": "234\n"}
            #
            vardir("get c", cwd=d)
            #
            run(
                "bash -c 'mkdir -p 2 && cd 2 && mkdir -p a && echo 234 > a/b && echo 345 > c'",
                cwd=d,
            )
            vardir("put -r .", cwd=f"{d}/2")
            #
            back = mynomad.variables.read(vardir.name).Items
            assert back == {"a/b": "234\n", "c": "345\n"}
            #
            vardir("get c", cwd=d)
            vardir("rm c", cwd=d)


def test_nomad_vardir_setrm():
    with VardirCaller(caller()) as vardir:
        vardir("set a avalue")
        vardir("cat a", output="avalue")
        vardir("set b bvalue")
        vardir("cat a", output="avalue")
        vardir("cat b", output="bvalue")
        vardir("rm b")
        vardir("cat a", output="avalue")
        vardir("cat b", check=False)
        vardir("rm a")
        vardir("cat a", check=False)
        vardir("cat b", check=False)
