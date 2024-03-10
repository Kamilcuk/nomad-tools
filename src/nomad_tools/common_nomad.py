import os
from typing import Optional

import click

from . import nomadlib
from .common_click import completor
from .nomadlib.connection import NOMAD_NAMESPACE
from .nomadlib.types import JobsJob

mynomad = nomadlib.NomadConn()


class NoJobFound(Exception):
    """
    Means that nomad_find_job did not find any matching jobs.
    This is different from JobNotFound, as the latter is an error from Nomad API.
    """

    pass


def nomad_find_job(id: str, namespace: Optional[str] = None):
    """Set nomad namespace to the given id"""
    namespace = namespace or os.environ.get(NOMAD_NAMESPACE, "*")
    if namespace == "*":
        mynomad.namespace = namespace
        jobs = [JobsJob(x) for x in mynomad.get("jobs", params={"prefix": id})]
        jobs = [job for job in jobs if job.ID == id]
        if not jobs:
            raise NoJobFound(f"Job named {id!r} not found in any namespace")
        jobsstr = " ".join(f"{j.ID}@{j.Namespace}" for j in jobs)
        assert len(jobs) < 2, f"Found multiple jobs named {id}: {jobsstr}"
        found = jobs[0]
        assert (
            found.Namespace
        ), "Internal error: Nomad returned NULL for found job namespace"
        os.environ[NOMAD_NAMESPACE] = mynomad.namespace = found.Namespace
    return id


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
