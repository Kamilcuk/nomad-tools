import json

from tests.testlib import gen_job, run_entry_watch


def test_nomad_watch_plan():
    job1 = gen_job(script="sleep 10", count=0)
    job2 = gen_job(script="sleep 10", count=0)
    jobid = job1["Job"]["ID"]
    run_entry_watch(f"-x purge {jobid}")
    run_entry_watch("run -json -", input=json.dumps(job1), check=127)
    run_entry_watch("plan -json -", input=json.dumps(job1))
    rr = run_entry_watch("-q -q plan -json -", input=json.dumps(job1), output=[""])
    assert not rr.stdout, (
        f"watch -q plan returned changes but the job did not change at all. The output is:\n{rr.stdout}"
    )
    run_entry_watch(
        "-q -q plan -json -",
        input=json.dumps(job2),
        output="To submit the job with version verification run:",
    )
    run_entry_watch(f"-x purge {jobid}")
