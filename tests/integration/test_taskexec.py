import contextlib
import logging
import shutil

from nomad_tools import taskexec
from tests.testlib import get_templatejob, run_entry_watch


@contextlib.contextmanager
def run_temp_job():
    hcl = get_templatejob(script="exec sleep 60")
    jobid = hcl.id
    run_entry_watch(f"-q -o none -x purge {jobid}")
    try:
        run_entry_watch("-q -o none start -", input=hcl.hcl)
        taskexec.log.setLevel(level=logging.DEBUG)
        yield jobid
    finally:
        taskexec.log.setLevel(level=logging.INFO)
        run_entry_watch(f"-q -o none -x purge {jobid}")


def test_taskexec_noutf():
    with run_temp_job() as jobid:
        alloc, task = taskexec.find_job(jobid)
        buf = taskexec.check_output(
            alloc,
            task,
            [
                "sh",
                "-xc",
                """
                printf "stdout 0xc0 byte: \\300\\n"
                printf "stderr 0xc0 byte: \\300\\n" >&2
                """,
            ],
        )
        assert buf == b"stdout 0xc0 byte: \300\n"


def test_taskexec_1():
    with run_temp_job() as jobid:
        alloc, task = taskexec.find_job(jobid)
        if 1:
            buf = taskexec.check_output(
                alloc, task, ["sh", "-xc", "echo STDOUT"]
            ).strip()
            assert buf == b"STDOUT"
        if 1:
            buf = taskexec.check_output(
                alloc, task, ["sh", "-xc", "echo STDOUT"], text=True
            ).strip()
            assert buf == "STDOUT"
        if 1:
            with taskexec.NomadPopen(
                alloc,
                task,
                ["sed", "s/^/INPUT: /"],
                stdin=taskexec.PIPE,
                stdout=taskexec.PIPE,
                text=True,
            ) as p:
                output = p.communicate("1\n2\n3\n")[0]
            assert output == "INPUT: 1\nINPUT: 2\nINPUT: 3\n"
        if 1:
            with taskexec.NomadPopen(
                alloc,
                task,
                ["sh", "-c", "echo 123; echo 234"],
                stdout=taskexec.PIPE,
            ) as inp:
                with taskexec.NomadPopen(
                    alloc,
                    task,
                    ["sed", "s/^/RECV:/"],
                    stdin=taskexec.PIPE,
                    stdout=taskexec.PIPE,
                ) as outp:
                    assert inp.stdout
                    assert outp.stdin
                    shutil.copyfileobj(inp.stdout, outp.stdin)
                    outp.stdin.close()
                    assert outp.stdout
                    output = outp.stdout.read()
            assert output == b"RECV:123\nRECV:234\n"
