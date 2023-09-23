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
import sys
from collections import defaultdict
from pathlib import Path
from shlex import quote, split
from textwrap import dedent
from typing import Dict, List, Optional, Union

import click
import tomli

from . import nomad_watch, nomadlib
from .common import common_options, get_version, mynomad
from .nomadlib.datadict import DataDict
from .nomadlib.types import Job, JobTask, JobTaskConfig

###############################################################################

log = logging.getLogger("nomad-gitlab-runner")


def quotearr(cmd: List[str]):
    return " ".join(quote(x) for x in cmd)


def run(cmdstr: str, *args, check=True, quiet=False, **kvargs):
    cmd = split(cmdstr)
    msg = f"+ {quotearr(cmd)}"
    if quiet:
        log.debug(msg)
    else:
        log.info(msg)
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
    # log.debug(f"env={ret}")
    return ret


@functools.lru_cache(maxsize=0)
def get_package_file(file: str) -> str:
    """Get a file relative to current package"""
    res = pkgutil.get_data(__package__, file)
    assert res is not None, f"Could not find {file}"
    return res.decode()


###############################################################################


# Default job name template name
default_jobname_template = "gitlabrunner.${CUSTOM_ENV_CI_RUNNER_ID}.${CUSTOM_ENV_CI_PROJECT_PATH_SLUG}.${CUSTOM_ENV_CI_CONCURRENT_ID}"


class ConfigOverride(DataDict):
    """Add additional overrides to the specific keys. The keys are merged with others"""

    job: Job = Job()
    task: JobTask = JobTask()
    task_config: JobTaskConfig = JobTaskConfig()


class ConfigDockerService(DataDict):
    """Configuration of docker service extract from gitlab job"""

    # A image capabable of executing docker commands used for creating docker network for intercommunication.
    docker_image: str = "docker"


class ConfigDriver(DataDict):
    """Configuration for custom executor"""

    # https://docs.gitlab.com/runner/executors/custom.html#config
    builds_dir: str
    # https://docs.gitlab.com/runner/executors/custom.html#config
    cache_dir: str
    # https://docs.gitlab.com/runner/executors/custom.html#config
    builds_dir_is_shared: bool
    # The script to execute when running gitlab generated scripts.
    script: str = ""
    # The task specification to execute.
    task: JobTask = JobTask()


class ConfigDocker(ConfigDriver):
    builds_dir: str = "/alloc"
    cache_dir: str = "/alloc"
    builds_dir_is_shared: bool = False
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
    #
    clone_task: JobTask = JobTask(
        {
            "Name": default_jobname_template + "-clone",
            "Driver": "docker",
            "Config": {
                "image": "gitlab/gitlab-runner:alpine-v16.3.1",
                "entrypoint": [],
                "command": "sleep",
                "args": [
                    "${CUSTOM_ENV_CI_JOB_TIMEOUT}",
                ],
            },
        }
    )
    # defualt docker image
    image: str = "alpine"
    # docker service configuration
    service: ConfigDockerService = ConfigDockerService()


class ConfigExec(ConfigDriver):
    builds_dir: str = "/local"
    cache_dir: str = "/local"
    builds_dir_is_shared: bool = ConfigDocker.builds_dir_is_shared
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


class ConfigRawExec(ConfigDriver):
    builds_dir: str = "/var/lib/gitlab-runner/builds"
    cache_dir: str = "/var/lib/gitlab-runner/cache"
    builds_dir_is_shared: bool = True
    script: str = get_package_file("nomad_gitlab_runner/exec.sh")
    task: JobTask = JobTask(
        {
            "Name": default_jobname_template,
            "Driver": "raw_exec",
            # https://github.com/hashicorp/nomad/issues/5397
            # "User": "0",
            "Config": {
                "command": "${NOMAD_TASK_DIR}/command.sh",
            },
            "Templates": [
                {
                    "ChangeMode": "noop",
                    "DestPath": "local/command.sh",
                    "EmbeddedTmpl": dedent(
                        """\
                        #!/bin/sh
                        exec sleep ${CUSTOM_ENV_CI_JOB_TIMEOUT}
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

    # Mode to execute with. Has to be set.
    mode: str = "docker"
    # Enable debugging?
    verbose: int = 0
    # Should the job be purged after we are done?
    purge: bool = True
    # The job name
    jobname: str = default_jobname_template
    # The defualt job constraints.
    CPU: int = 1024
    MemoryMB: int = 1024

    override: ConfigOverride = ConfigOverride()
    custom: ConfigDriver = ConfigDriver()
    exec: ConfigExec = ConfigExec()
    raw_exec: ConfigRawExec = ConfigRawExec()
    docker: ConfigDocker = ConfigDocker()

    def __post_init__(self):
        """Update environment variables from configuration - to set NOMAD_TOKEN variable"""
        for k, v in self.asdict().items():
            if k.startswith("NOMAD_") and v:
                assert isinstance(v, str)
                os.environ[k] = v

    @functools.lru_cache(maxsize=0)
    def get_driverconfig(self) -> ConfigDriver:
        modes: Dict[ConfigMode, ConfigDriver] = {
            ConfigMode.raw_exec: self.raw_exec,
            ConfigMode.exec: self.exec,
            ConfigMode.custom: self.custom,
            ConfigMode.docker: self.docker,
        }
        try:
            cd = ConfigMode[self.mode]
        except KeyError:
            raise Exception(f"Not a valid driver: {self.mode}")
        cc: ConfigDriver = modes[cd]
        return cc

    def _get_task(self) -> JobTask:
        """Get the task to run, apply transformations and configuration as needed"""
        task: JobTask = self.get_driverconfig().task
        assert task, f"is invalid: {task}"
        assert "Name" in task
        assert "Config" in task
        task = JobTask(
            {
                "RestartPolicy": {"Attempts": 0},
                "Resources": {
                    "CPU": self.CPU,
                    "MemoryMB": self.MemoryMB,
                },
                **task.asdict(),
                # Apply override on task.
                **self.override.task,
            }
        )
        # Apply override on config
        task.Config = JobTaskConfig(
            {**task.Config.asdict(), **self.override.task_config}
        )
        return task

    def get_script(self) -> str:
        return self.get_driverconfig().script

    def get_nomad_job(self) -> Job:
        return Job(
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
                # Apply overrides
                **self.override.job.asdict(),
            }
        )


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
        data: List[Union[str, Dict[str, str]]] = (
            json.loads(CI_JOB_SERVICES) if CI_JOB_SERVICES else []
        )
        ret: List[ServiceSpec] = []
        for x in data:
            if isinstance(x, str):
                s = ServiceSpec(name=x)
                ret += [s]
            elif isinstance(x, dict):
                s = ServiceSpec(x)
                ret += [s]
            else:
                raise Exception(f"Invalid service specification element: {x}")
        return ret

    def get_alias(self):
        """Alias defaults to name"""
        return self.alias if self.alias is not None else self.name


@dataclasses.dataclass
class DockerServices:
    """Execute services in docker and manage"""

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
    if services:
        assert (
            config.mode == ConfigMode.docker
        ), "services are only implemented in docker mode"
        ds = DockerServices(services)
        nomadjob.TaskGroups[0].Tasks = ds.apply(nomadjob.TaskGroups[0].Tasks[0])
    dc = config.get_driverconfig()
    if isinstance(dc, ConfigDocker):
        # Docker has additional cloning job.
        nomadjob.TaskGroups[0].Tasks += [
            JobTask(
                {
                    "RestartPolicy": {"Attempts": 0},
                    "Resources": {
                        "CPU": config.CPU,
                        "MemoryMB": config.MemoryMB,
                    },
                    "Lifecycle": {
                        "Hook": "prestart",
                        "Sidecar": True,
                    },
                    **dc.clone_task.asdict(),
                }
            )
        ]


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


# The job "template" has environment variables substituted with gitlab exported env.
class NotCustomEnvIsFine(defaultdict):
    """
    If key starts with CUSTOM_ENV_ substitute it.
    Otherwise preserve the key value for the purpouse of string.Template() operation.
    """

    def __init__(self):
        super().__init__()
        self.update(
            {k: v for k, v in os.environ.items() if k.startswith("CUSTOM_ENV_")}
        )

    def __missing__(self, key):
        if key.startswith("CUSTOM_ENV_"):
            raise Exception(
                f"environment variable {key} is not set but requested by template"
            )
        return "${" + key + "}"


class OnlyBracedCustomEnvTemplate(string.Template):
    pattern: str = r"""
            # match ${CUSTOM_ENV_*} only
            \${(?P<braced>CUSTOM_ENV_[_A-Za-z0-9]*)} |
            (?P<escaped>\x01)  |  # match nothing
            (?P<named>\x01)    |  # match nothing
            (?P<invalid>\x01)     # match nothing
            """


class Jobenv:
    """
    The job specification is stored in job_env inside config section returned on config stage.
    Manage that job spefication - serialize and deserialize it.
    """

    def __init__(self):
        NOMAD_GITLAB_RUNNER_JOB = os.environ.get("NOMAD_GITLAB_RUNNER_JOB")
        assert (
            NOMAD_GITLAB_RUNNER_JOB is not None
        ), f"Env variable set in config NOMAD_GITLAB_RUNNER_JOB is missing"
        jobjson = json.loads(NOMAD_GITLAB_RUNNER_JOB)
        self.job = Job(jobjson)
        self.assert_job(self.job)

    @classmethod
    def create_job_env(cls) -> Dict[str, str]:
        global config
        nomadjob: Job = config.get_nomad_job()
        apply_services(nomadjob)
        # Template the job - expand all ${CUSTOM_ENV_*} references.
        jobjson = json.dumps(nomadjob.asdict())
        try:
            jobjson = OnlyBracedCustomEnvTemplate(jobjson).substitute(
                NotCustomEnvIsFine()
            )
        except ValueError:
            log.exception(f"{jobjson}")
            raise
        nomadjob = Job(json.loads(jobjson))
        cls.assert_job(nomadjob)
        return {
            "NOMAD_GITLAB_RUNNER_JOB": jobjson,
        }

    @staticmethod
    def assert_job(job: Job):
        assert "ID" in job, f"ID is missing in {job}"
        assert "Name" in job["TaskGroups"][0]["Tasks"][0], f"{job}"

    @property
    def jobname(self):
        return self.job.ID

    def get_clone_task_name(self) -> str:
        # log.info(f"names = {[x.Name for x in self.job.TaskGroups[0].Tasks]}")
        return next(
            (
                x.Name
                for x in self.job.TaskGroups[0].Tasks
                # The container has to have specific name. Synchronized with ConfigDocker
                if x.Name == self.jobname + "-clone" and x.Driver == "docker"
            ),
            self.jobname,
        )


###############################################################################


class BuildFailure(Exception):
    code: int = 76
    """Quite a special exit status returned by script.sh when the build script has failed."""


@click.group(
    help="""
This is a script to execute Nomad job from custom gitlab executor.

\b
Example /etc/gitlab-runner/config.toml configuration file:
  [[runners]]
  ...
  id = 27898742
  ...
  executor = "custom"
  [runners.custom]
    config_exec = "nomad-gitlab-runner"
    config_args = ["config"]
    prepare_exec = "nomad-gitlab-runner"
    prepare_args = ["prepare"]
    run_exec = "nomad-gitlab-runner"
    run_args = ["run"]
    cleanup_exec = "nomad-gitlab-runner"
    cleanup_args = ["cleanup"]

\b
Example /etc/gitlab-runner/nomad.toml configuration file:
    [default]
    # You can use NOMAD_* variables here
    NOMAD_TOKEN = 1234567
\b
    # Id of the runner from config.toml file allows overriding the values for speicfic runner.
    [27898742]
    # Mode to use - "raw_exec", "exec", "docker" or "custom"
    mode = "raw_exec"
    verbose = 0
    CPU = 2048
    MemoryMB = 2048
    # If it possible to override some things. This is TOML syntax.
    [27898742.override.job]
    [27898742.override.task]
    [27898742.override.task_config]
    # for example https://developer.hashicorp.com/nomad/docs/drivers/docker#logging
    [27898742.override.task_config.logging]
    type = "fluentd"
    [27898742.override.task_config.logging.config]
    fluentd-address = "localhost:24224"
    tag = "your_tag"

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
    help="""Path to configuration file.""",
    show_default=True,
)
@click.option(
    "-s",
    "--section",
    help="""
An additional section read from configuration file to merge with defaults.
The value defaults to CUSTOM_ENV_CI_RUNNER_ID which is set to the unique ID of the runner being used.
""",
    envvar="CUSTOM_ENV_CI_RUNNER_ID",
    show_default=True,
)
@common_options()
def main(verbose: int, configpath: Path, section: str):
    # Read configuration
    with configpath.open("rb") as f:
        data = tomli.load(f)
    for key, val in data.items():
        assert isinstance(
            val, dict
        ), f"All items in config have to be in a section. {key} is not"
    configs = {key: val for key, val in data.items()}
    global config
    config = Config(
        {**configs.get("default", {}), **configs.get(section, {})}
    ).remove_none()
    if verbose:
        config.verbose = 1
    #
    logging.basicConfig(
        format="%(module)s:%(lineno)s: %(message)s",
        level=logging.DEBUG if config.verbose else logging.INFO,
    )
    log.debug(f"+ {sys.argv}")
    #
    dc = config.get_driverconfig()
    if dc.get("image"):
        os.environ.setdefault("CUSTOM_ENV_CI_JOB_IMAGE", dc["image"])


@main.command(
    "config", help="https://docs.gitlab.com/runner/executors/custom.html#config"
)
def mode_config():
    dc = config.get_driverconfig()
    driver_config = {
        "builds_dir": dc.builds_dir,
        "cache_dir": dc.cache_dir,
        "builds_dir_is_shared": dc.builds_dir_is_shared,
        "hostname": socket.gethostname(),
        "driver": {
            "name": "nomad-gitlab-runner",
            "version": get_version(),
        },
        "job_env": Jobenv.create_job_env(),
    }
    driver_config_json = json.dumps(driver_config)
    log.debug(f"driver_config={driver_config_json}")
    click.echo(driver_config_json)


@main.command(
    "prepare", help="https://docs.gitlab.com/runner/executors/custom.html#prepare"
)
def mode_prepare():
    je = Jobenv()
    purge_previous_nomad_job(je.jobname)
    nomad_watch.cli.main(["-G", "start", json.dumps({"Job": je.job.asdict()})])


@main.command("run", help="https://docs.gitlab.com/runner/executors/custom.html#run")
@click.argument("script")
@click.argument("stage")
def mode_run(script: str, stage: str):
    assert stage
    je = Jobenv()
    set_x = "-x" if config.verbose > 1 else ""
    # Execute all except step_ and build_ in that special gitlab docker container.
    taskname = (
        je.jobname
        if stage.startswith("step_") or stage.startswith("build_")
        else je.get_clone_task_name()
    )
    run(
        f"nomad alloc exec -task {taskname} -job {je.jobname} sh -c {quote(config.get_script())} gitlabrunner {set_x}",
        stdin=open(script),
        quiet=True,
    )


@main.command(
    "cleanup", help="https://docs.gitlab.com/runner/executors/custom.html#cleanup"
)
def mode_cleanup():
    je = Jobenv()
    nomad_watch.cli.main(
        (["--purge"] if config.purge else []) + ["-xn0", "stop", je.jobname]
    )


@main.command("showconfig", help="Show current configuration")
def mode_showconfig():
    print(json.dumps(config.asdict(), indent=2))
    print()
    arr = [
        "CI_PROJECT_PATH_SLUG",
        "CI_CONCURRENT_ID",
        "CI_JOB_TIMEOUT",
        "CI_RUNNER_ID",
    ]
    os.environ.update({f"CUSTOM_ENV_{i}": f"CUSTOM_ENV_{i}_VAL" for i in arr})
    print(Jobenv.create_job_env())


###############################################################################


def cli(*args, **kvargs) -> int:
    try:
        main.main(*args, **kvargs)
    except BuildFailure:
        log.debug(f"build failure")
        return int(os.environ.get("BUILD_FAILURE_EXIT_CODE", BuildFailure.code))
    except Exception as e:
        log.exception(e)
        return int(os.environ.get("SYSTEM_FAILURE_EXIT_CODE", 111))
    return 0


if __name__ == "__main__":
    exit(cli())
