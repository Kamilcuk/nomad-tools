import json
import logging
import shlex
import subprocess
import sys
from typing import IO, Any, Dict, List

import click.shell_completion

from . import common, nomadlib
from .common_click import alias_option

log = logging.getLogger(__name__)


def complete_job(
    ctx: click.Context, param: click.Parameter, incomplete: str
) -> List[click.shell_completion.CompletionItem]:
    if "job" in ctx.params:
        try:
            jobs = common.mynomad.get("jobs", params=dict(prefix=incomplete))
        except Exception:
            return []
        print(jobs, file=sys.stderr)
        return []
    else:
        return click.File().shell_complete(ctx, param, incomplete)


def load_job_file(file: IO) -> nomadlib.Job:
    content = file.read()
    seekable = file.seekable()
    file.close()
    try:
        jobdata = json.loads(content)
    except json.JSONDecodeError:
        if seekable:
            cmd = f"nomad job run -output {shlex.quote(file.name)}"
            log.debug(f"+ {cmd}")
            content = subprocess.check_output(shlex.split(cmd), text=True)
        else:
            cmd = "nomad job run -output -"
            log.debug(f"+ {cmd}")
            content = subprocess.check_output(cmd.split(), input=content, text=True)
        jobdata = json.loads(content)
    return nomadlib.Job(jobdata.get("Job", jobdata))


@click.command(
    help="""
    List all docker images referenced by a Nomad job.
    Typically used to download or test the images like
    nomad-dockers ./file.nomad.hcl | xargs docker pull.
    """
)
@common.common_options()
@click.option("-v", "--verbose", count=True, help="Be verbose")
@click.option(
    "-f",
    "--format",
    help="Passed to python .format()",
    show_default=True,
    default="{image}",
)
@alias_option(
    "-l",
    "--long",
    aliased=dict(format="{groupName} {taskName} {image}"),
)
@click.option(
    "-j",
    "--job",
    multiple=True,
    help="List images referenced by a Nomad job name",
)
@click.option(
    "-a",
    "--all",
    multiple=True,
    help="NextList docker images referenced by all job versions",
)
@click.argument("files", nargs=-1, type=click.File("r"))
def cli(
    all: List[str], files: List[IO], job: List[str], format: str, verbose: int, **kwargs
):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    if not (all or files or job):
        raise click.ClickException(
            "One of --all --job or positional arguments have to given"
        )
    njobs: List[nomadlib.Job] = []
    njobs += [
        nomadlib.Job(njobversion)
        for job in all
        for njobversion in common.mynomad.get(
            f"job/{common.nomad_find_job(job)}/versions"
        )["Versions"]
    ]
    njobs += [
        nomadlib.Job(common.mynomad.get(f"job/{common.nomad_find_job(job)}"))
        for job in job
    ]
    njobs += [load_job_file(file) for file in files]
    for njob in njobs:
        out: List[str] = []
        for group in njob.TaskGroups or []:
            for task in group.Tasks or []:
                image = task.Config.get("image")
                if image and task.Driver == "docker":
                    params: Dict[str, Any] = dict(
                        groupName=group.Name,
                        taskName=task.Name,
                        jobVersion=njob.Version,
                        image=image,
                    )
                    log.debug(f"{format!r} {params}")
                    out.append(format.format(**params))
        out = sorted(list(set(out)))
        for line in out:
            print(line)


if __name__ == "__main__":
    cli()
