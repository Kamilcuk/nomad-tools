import json
import subprocess
import sys
from typing import List

import click.shell_completion

from . import common, nomadlib


def complete_job(
    ctx: click.Context, param: click.Parameter, incomplete: str
) -> List[click.shell_completion.CompletionItem]:
    if "job" in ctx.params:
        try:
            jobs = common.mynomad.get("jobs", params=dict(prefix=incomplete))
        except:
            return []
        print(jobs, file=sys.stderr)
        return []
    else:
        return click.File().shell_complete(ctx, param, incomplete)


@click.command(
    help="""
    List all docker images referenced by the service file.
    Typically used to download or test the images like
    nomad-dockers ./file.nomad.hcl | xargs docker pull.
    """
)
@common.common_options()
@click.option("-l", "--long", is_flag=True)
@click.option(
    "-j",
    "--job",
    "isjob",
    is_flag=True,
    help="The argument is not a file, but a job name",
)
@click.argument("job")
def cli(long: bool, isjob: bool, job: str, **kwargs):
    if isjob:
        njob = nomadlib.Job(common.mynomad.get(f"job/{job}"))
    else:
        if job == "-":
            content = sys.stdin.read()
            sys.stdin.close()
        else:
            with open(job, "r") as f:
                content = f.read()
        try:
            jobdata = json.loads(content)
        except json.JSONDecodeError:
            content = subprocess.check_output(
                "nomad job run -output -".split(), input=content, text=True
            )
            jobdata = json.loads(content)
        njob = nomadlib.Job(jobdata.get("Job", jobdata))
    out: List[str] = []
    for group in njob.TaskGroups or []:
        for task in group.Tasks or []:
            image = task.Config.get("image")
            if image and task.Driver == "docker":
                out.append(f"{group.Name!r} {task.Name!r} {image}" if long else image)
    out = sorted(list(set(out)))
    for line in out:
        print(line)


if __name__ == "__main__":
    cli()
