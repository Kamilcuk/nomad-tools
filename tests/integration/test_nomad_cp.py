import json
import os

from tests.testlib import gen_job, run


def test_nomad_cp():
    jobjson = gen_job(script="sleep 60")
    job = jobjson["Job"]
    jobname = job["ID"]
    ns = os.environ.get('NOMAD_NAMESPACE', 'default')
    try:
        run("nomad-watch --json start -", input=json.dumps(jobjson))
        return  # TODO
        run(
            f"nomad-cp -v -N {ns} -job {jobname}:/root /tmp/root"
        )
        run("ls -la /tmp/root")
        run("rm -vr /tmp/root")
    finally:
        run(f"nomad-watch --purge stop {jobname}")
