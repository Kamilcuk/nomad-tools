#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import datetime
import enum
import functools
import json
import logging
import os
import pkgutil
import socket
import string
import subprocess
from pathlib import Path
from shlex import quote, split
from textwrap import dedent
from typing import Dict, List, Optional

import click
import tomli

from . import nomad_watch
from .nomadlib.datadict import DataDict
from .nomadlib.types import Job, JobTask, JobTaskConfig

###############################################################################

log = logging.getLogger("nomad-gitlab-runner")


def quotearr(cmd: List[str]):
    return " ".join(quote(x) for x in cmd)


def run(cmdstr: str, *args, check=True, quiet=False, **kvargs):
    cmd = split(cmdstr)
    if not quiet:
        log.info(f"+ {quotearr(cmd)}")
    try:
        return subprocess.run(cmd, *args, check=check, text=True, **kvargs)
    except subprocess.CalledProcessError as e:
        exit(e.returncode)


def ns2dt(ns: int):
    return datetime.datetime.fromtimestamp(ns // 1000000000)


###############################################################################


@functools.lru_cache(maxsize=0)
def get_gitlab_env():
    """Get all gitlab exported environment variables and remove CUSTOM_ENV_ from front"""
    ret = {
        k.replace("CUSTOM_ENV_", ""): v
        for k, v in os.environ.items()
        if k.startswith("CUSTOM_ENV_")
    }
    ret.update({f"CUSTOM_ENV_{k}": v for k, v in ret.items()})
    log.debug(f"env={ret}")
    return ret


@functools.lru_cache(maxsize=0)
def get_package_file(file: str) -> str:
    """Get a file relative to current package"""
    res = pkgutil.get_data(__package__, file)
    assert res is not None, f"Could not find {file}"
    return res.decode()


###############################################################################


default_jobname_template = (
    "gitlabrunner.${CUSTOM_ENV_CI_PROJECT_PATH_SLUG}.${CUSTOM_ENV_CI_CONCURRENT_ID}"
)


class ConfigOverride(DataDict):
    # Add additional overrides to the specific keys.
    job: Job = Job()
    task: JobTask = JobTask()
    task_config: JobTaskConfig = JobTaskConfig()


class ConfigCustom(DataDict):
    # The script to execute when running gitlab generated scripts.
    script: str = ""
    # The task specificatino to execute.
    task: JobTask = JobTask()


class ConfigDockerService(DataDict):
    docker_image: str = "docker"


class ConfigDocker(ConfigCustom):
    script: str = get_package_file("nomad_gitlab_runner/docker.sh")
    task: JobTask = JobTask(
        {
            "Name": default_jobname_template,
            "Driver": "docker",
            "Config": {
                "image": "${CUSTOM_ENV_CI_JOB_IMAGE}",
                "command": "sleep",
                "args": [
                    "${CUSTOM_ENV_CI_JOB_TIMEOUT}",
                ],
            },
        }
    )
    service: ConfigDockerService = ConfigDockerService()


class ConfigExec(ConfigCustom):
    script: str = get_package_file("nomad_gitlab_runner/docker.sh")
    task: JobTask = JobTask(
        {
            "Name": default_jobname_template,
            "Driver": "exec",
            "User": "gitlab-runner:gitlab-runner",
            "Config": {
                "command": "sleep",
                "args": [
                    "${CUSTOM_ENV_CI_JOB_TIMEOUT}",
                ],
            },
        }
    )


class ConfigRawExec(ConfigCustom):
    script: str = get_package_file("nomad_gitlab_runner/exec.sh")
    task: JobTask = JobTask(
        {
            "Name": default_jobname_template,
            "Driver": "raw_exec",
            # https://github.com/hashicorp/nomad/issues/5397
            # "User": "0",
            "Config": {
                "command": "${NOMAD_TASK_DIR}/command.sh",
                "args": [
                    "${CUSTOM_ENV_CI_JOB_TIMEOUT}",
                ],
            },
            "Templates": [
                {
                    "ChangeMode": "noop",
                    "DestPath": "local/command.sh",
                    "EmbeddedTmpl": dedent(
                        """\
                        #!/bin/sh
                        exec sleep "$@"
                        """
                    ),
                    "Perms": "777",
                },
            ],
        }
    )


class ConfigMode(enum.Enum):
    raw_exec = enum.auto()
    docker = enum.auto()
    exec = enum.auto()
    custom = enum.auto()


class Config(DataDict):
    """Configuration of this program"""

    NOMAD_ADDR: Optional[str] = os.environ.get("NOMAD_ADDR")
    NOMAD_REGION: Optional[str] = os.environ.get("NOMAD_REGION")
    NOMAD_NAMESPACE: Optional[str] = os.environ.get("NOMAD_NAMESPACE")
    NOMAD_HTTP_AUTH: Optional[str] = os.environ.get("NOMAD_HTTP_AUTH")
    NOMAD_TOKEN: Optional[str] = os.environ.get("NOMAD_TOKEN")
    NOMAD_CLIENT_CERT: Optional[str] = os.environ.get("NOMAD_CLIENT_CERT")
    NOMAD_CLIENT_KEY: Optional[str] = os.environ.get("NOMAD_CLIENT_KEY")
    NOMAD_CACERT: Optional[str] = os.environ.get("NOMAD_CACERT")
    NOMAD_CAPATH: Optional[str] = os.environ.get("NOMAD_CAPATH")
    NOMAD_SKIP_VERIFY: Optional[str] = os.environ.get("NOMAD_SKIP_VERIFY")
    NOMAD_TLS_SERVER_NAME: Optional[str] = os.environ.get("NOMAD_TLS_SERVER_NAME")
    NOMAD_LICENSE_PATH: Optional[str] = os.environ.get("NOMAD_LICENSE_PATH")
    NOMAD_LICENSE: Optional[str] = os.environ.get("NOMAD_LICENSE")

    # Should the job be purged after we are done?
    purge: bool = True
    # The job name
    jobname: str = default_jobname_template
    # Mode to execute with.
    mode: str = str(ConfigMode.raw_exec)
    # The defualt job constraints.
    CPU: int = 1024
    MemoryMB: int = 1024

    override: ConfigOverride = ConfigOverride()
    custom: ConfigCustom = ConfigCustom()
    exec: ConfigExec = ConfigExec()
    raw_exec: ConfigRawExec = ConfigRawExec()
    docker: ConfigDocker = ConfigDocker()

    def __post_init__(self):
        """Update environment variables from configuration - to set NOMAD_TOKEN variable"""
        for k, v in self.asdict().items():
            if k.startswith("NOMAD_") and v:
                assert isinstance(v, str)
                os.environ[k] = v

    def get_jobname(self) -> str:
        return string.Template(self.jobname).substitute(env)

    @functools.lru_cache(maxsize=0)
    def _get_configcustom(self) -> ConfigCustom:
        modes: Dict[ConfigMode, ConfigCustom] = {
            ConfigMode.raw_exec: self.raw_exec,
            ConfigMode.exec: self.exec,
            ConfigMode.custom: self.custom,
            ConfigMode.docker: self.docker,
        }
        try:
            cd = ConfigMode[self.mode]
        except KeyError:
            raise Exception(f"Not a valid driver: {self.mode}")
        cc: ConfigCustom = modes[cd]
        return cc

    def _get_task(self) -> JobTask:
        """Get the task to run, apply transformations and configuration as needed"""
        task: JobTask = self._get_configcustom().task
        print(task)
        assert task, f"is invalid: {task}"
        assert "Name" in task
        assert "Config" in task
        task = JobTask(
            {
                **task.asdict(),
                "RestartPolicy": {"Attempts": 0},
                "Resources": {
                    "CPU": self.CPU,
                    "MemoryMB": self.MemoryMB,
                },
            }
        )
        # Apply overrides
        task = JobTask({**task.asdict(), **self.override.task})
        task.Config = JobTaskConfig(
            {**task.Config.asdict(), **self.override.task_config}
        )
        return task

    def get_script(self) -> str:
        return self._get_configcustom().script

    def get_nomad_job(self) -> Job:
        job = Job(
            {
                "ID": self.jobname,
                "Type": "batch",
                "TaskGroups": [
                    {
                        "Name": self.jobname,
                        "ReschedulePolicy": {"Attempts": 0},
                        "RestartPolicy": {"Attempts": 0},
                        "Tasks": [self._get_task()],
                    }
                ],
            }
        )
        # Apply overrides
        job = Job({**job, **self.override.job.asdict()})
        return job


###############################################################################


class ServiceSpec(DataDict):
    """
    Specification of services as given to use by Gitlab
    https://docs.gitlab.com/ee/ci/yaml/#services
    """

    name: str
    alias: Optional[str] = None
    entrypoint: Optional[List[str]] = None
    command: Optional[List[str]] = None

    @staticmethod
    def get() -> List[ServiceSpec]:
        """Read the Gitlab environment variable to extract the service"""
        CI_JOB_SERVICES = os.environ.get("CUSTOM_ENV_CI_JOB_SERVICES")
        data = json.loads(CI_JOB_SERVICES) if CI_JOB_SERVICES else {}
        ret: List[ServiceSpec] = []
        for x in data:
            if isinstance(data, str):
                s = ServiceSpec(name=data)
                ret += [s]
            elif isinstance(data, dict):
                s = ServiceSpec(data)
                ret += [s]
            else:
                assert 0
        return ret

    def get_alias(self):
        """Alias defaults to name"""
        return self.alias if self.alias is not None else self.name


@dataclasses.dataclass
class DockerServices:
    services: List[ServiceSpec]

    @staticmethod
    def get_network():
        """Get the name of the docker network we will be creating"""
        return default_jobname_template

    def docker_task(self, cmd: str, lifecycle: Dict[str, Dict[str, str]]):
        """A task to manipulate docker on the host. Currently mounting docker.sock"""
        return JobTask(
            {
                "Name": "remove_docker_network",
                "Driver": "docker",
                "Config": {
                    "image": config.docker.service.docker_image,
                    "args": cmd.split(),
                    "mount": [
                        {
                            "type": "bind",
                            "source": "/var/run/docker.sock",
                            "target": "/var/run/docker.sock",
                            "readonly": True,
                        }
                    ],
                },
                "Resources": {
                    "CPU": config.CPU,
                    "MemoryMB": config.MemoryMB,
                },
                **lifecycle,
            }
        )

    def create_network_task(self):
        """Task to create the docker network. Will be run before all other tasks"""
        return self.docker_task(
            f"docker network create {self.get_network}",
            {
                "Lifecycle": {
                    "Hook": "prestart",
                },
            },
        )

    def service_task(self, s: ServiceSpec):
        """Task that will run specific service. Generated from specification given to us by Gitlab"""
        return JobTask(
            {
                "Name": s.name,
                "Driver": "docker",
                "Config": {
                    "image": s.name,
                    **({} if s.entrypoint is None else {"entrypoint": s.entrypoint}),
                    **(
                        {}
                        if s.command is None
                        else {
                            **({} if not s.command else {"command": s.command[0]}),
                            **({} if len(s.command) <= 1 else {"args": s.command[1:]}),
                        }
                    ),
                    "network_mode": self.get_network(),
                    "network_aliases": [s.get_alias()],
                    "privileged": True,
                },
                "Resources": {
                    "CPU": config.CPU,
                    "MemoryMB": config.MemoryMB,
                },
            }
        )

    def remove_network_task(self):
        """Task to remove docker network - cleanup"""
        return self.docker_task(
            f"docker network rm {self.get_network}",
            {
                "Lifecycle": {
                    "Hook": "poststop",
                },
            },
        )

    def apply(self, task: JobTask) -> List[JobTask]:
        """Apply the services to the task. Return list of tasks to run"""
        assert task.Driver == "docker", f"Task has to be docker: {task}"
        # Put the task also in the same network.
        task.Config.network_mode = self.get_network()
        # The task should start after services have started.
        # TODO: start after healthcheck is healthy.
        task["Lifecycle"] = {"Hook": "poststart"}
        # The list of tasks to run:
        tasks: List[JobTask] = [
            self.create_network_task(),
            task,
            *[self.service_task(x) for x in self.services],
            self.remove_network_task(),
        ]
        return tasks


def apply_services(nomadjob: Job):
    """Apply services from gitlab spec unto nomad job specification"""
    services = ServiceSpec.get()
    if not services:
        return nomadjob
    assert (
        config.mode == ConfigMode.docker
    ), "services are only implemented in docker mode"
    ds = DockerServices(services)
    nomadjob.TaskGroups[0].Tasks = ds.apply(nomadjob.TaskGroups[0].Tasks[0])


###############################################################################


def purge_previous_nomad_job(jobname: str):
    rr = run(
        f"nomad job inspect {jobname}", check=False, stdout=subprocess.PIPE, quiet=True
    )
    assert rr.returncode in [
        0,
        1,
    ], f"Invalid nomad job inspect {jobname} output - it should be either 0 or 1"
    if rr.returncode == 0:
        job = Job(json.loads(rr.stdout)["Job"])
        assert (
            job.Stop == True or job.Status == "dead"
        ), f"Job {job.description()} already exists and is not stopped or not dead. Bailing out"
        run(f"nomad job stop -purge {jobname}")


###############################################################################


@click.group(
    help="""
        This is a script to execute Nomad job from custom gitlab executor.
        """,
    epilog="Written by Kamil Cukrowski 2023. Licensed under GNU GPL version or later.",
)
@click.option("-v", "--verbose", count=True)
@click.option(
    "-c",
    "--config",
    "configpath",
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    default=Path("/etc/gitlab-runner/nomad.toml"),
)
@click.help_option("-h", "--help")
def cli(verbose: int, configpath: Path):
    logging.basicConfig(
        format="%(module)s:%(lineno)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )
    with configpath.open("rb") as f:
        data = tomli.load(f)
    #
    global config
    config = Config(data)
    global env
    env = get_gitlab_env()


@cli.command(
    "config", help="https://docs.gitlab.com/runner/executors/custom.html#config"
)
def mode_config():
    config = {
        "builds_dir": "/local",
        "cache_dir": "/local",
        "builds_dir_is_shared": False,
        "hostname": socket.gethostname(),
        "driver": {
            "name": "nomad-gitlab-runner",
            "version": "v0.0.1",
        },
    }
    cfg = json.dumps(config)
    log.debug(f"config={cfg}")
    print(cfg)


@cli.command(
    "prepare", help="https://docs.gitlab.com/runner/executors/custom.html#prepare"
)
def mode_prepare():
    jobname = config.get_jobname()
    nomadjob = config.get_nomad_job()
    apply_services(nomadjob)
    #
    jobjson = json.dumps({"Job": nomadjob.asdict()})
    jobjson = string.Template(jobjson).safe_substitute(env)
    log.debug(json.loads(jobjson))
    #
    purge_previous_nomad_job(jobname)
    nomad_watch.cli.main(["start", jobjson])


@cli.command("run", help="https://docs.gitlab.com/runner/executors/custom.html#run")
@click.argument("script")
@click.argument("stage")
def mode_run(script: str, stage: str):
    assert stage
    jobname = config.get_jobname()
    run(
        f"nomad alloc exec -task {jobname} -job {jobname} sh -c {quote(config.get_script())}",
        stdin=open(script),
        quiet=True,
    )


@cli.command(
    "cleanup", help="https://docs.gitlab.com/runner/executors/custom.html#cleanup"
)
def mode_cleanup():
    jobname = config.get_jobname()
    nomad_watch.cli((["--purge"] if config.purge else []) + ["-xn0", "stop", jobname])


@cli.command("showconfig", help="Show current configuration")
def mode_showconfig():
    print(json.dumps(config.asdict(), indent=2))


###############################################################################

if __name__ == "__main__":
    cli.main()
