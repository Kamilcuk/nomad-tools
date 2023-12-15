import os

import click

from . import nomadlib
from .common_base import NOMAD_NAMESPACE
from .common_click import completor

mynomad = nomadlib.NomadConn()


def nomad_find_job(id: str) -> str:
    """Find job named jobprefix if namespace is *."""
    jobs = mynomad.get("jobs", params={"prefix": id})
    matches = [job for job in jobs if job["ID"] == id]
    if len(matches) == 0:
        raise nomadlib.JobNotFound(f"Job named {id} not found")
    assert len(matches) < 2, f"Found multiple jobs named {id}"
    found = matches[0]
    os.environ[NOMAD_NAMESPACE] = mynomad.namespace = found["Namespace"]
    return found["ID"]


def namespace_option():
    return click.option(
        "-N",
        "--namespace",
        help="Set NOMAD_NAMESPACE environment variable.",
        envvar=NOMAD_NAMESPACE,
        show_default=True,
        default="default",
        shell_complete=completor(
            lambda: (x["Name"] for x in mynomad.get("namespaces"))
        ),
        callback=lambda ctx, param, value: os.environ.__setitem__(
            NOMAD_NAMESPACE, value
        ),
    )


def complete_job():
    return completor(lambda: (x["ID"] for x in mynomad.get("jobs")))
