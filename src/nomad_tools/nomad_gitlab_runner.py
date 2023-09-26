#!/usr/bin/env python3

from __future__ import annotations

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
import yaml

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


@functools.lru_cache(maxsize=0)
def get_package_file(file: str) -> str:
    """Get a file relative to current package"""
    res = pkgutil.get_data(__package__, "nomad_gitlab_runner/" + file)
    assert res is not None, f"Could not find {file}"
    return res.decode()


def get_CUSTOM_ENV() -> Dict[str, str]:
    return {
        k[len("CUSTOM_ENV_") :]: v
        for k, v in os.environ.items()
        if k.startswith("CUSTOM_ENV_")
    }


def print_env():
    log.fatal("\n".join(f"{k}={quote(v)}" for k, v in os.environ.items()))


###############################################################################


class ServiceSpec(DataDict):
    """
    Specification of services as given to use by Gitlab
    https://docs.gitlab.com/ee/ci/yaml/#services
    """

    name: str
    alias: str = ""
    entrypoint: Optional[List[str]] = None
    command: Optional[List[str]] = None

    def __post_init__(self):
        if not self.alias:
            self.alias = self.name.split("/")[-1].split(":")[0]

    @staticmethod
    def get() -> List[ServiceSpec]:
        """Read the Gitlab environment variable to extract the service"""
        CI_JOB_SERVICES = os.environ.get("CUSTOM_ENV_CI_JOB_SERVICES")
        data: List[Union[str, Dict[str, str]]] = (
            json.loads(CI_JOB_SERVICES) if CI_JOB_SERVICES else []
        )
        assert isinstance(
            data, list
        ), f"CUSTOM_ENV_CI_JOB_SERVICES is not a list: {CI_JOB_SERVICES}"
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


###############################################################################


main_task_name: str = "ci-task"


class ConfigOverride(DataDict):
    """Add additional overrides to the specific keys. The keys are merged with others"""

    job: Job = Job()
    task: JobTask = JobTask()
    task_config: JobTaskConfig = JobTaskConfig()


class ConfigCustom(DataDict):
    """Configuration for custom executor"""

    builds_dir: str = ""
    """https://docs.gitlab.com/runner/executors/custom.html#config"""
    cache_dir: str = ""
    """https://docs.gitlab.com/runner/executors/custom.html#config"""
    builds_dir_is_shared: Optional[bool] = None
    """https://docs.gitlab.com/runner/executors/custom.html#config"""
    task: JobTask = JobTask()
    """The task specification to execute."""
    user: str = ""
    """User to execute the task."""

    def apply(self, nomadjob: Job):
        assert not ServiceSpec.get(), f"Services only handled in Docker mode"
        assert nomadjob

    def get_task_for_stage(self, stage: str) -> str:
        assert stage
        return main_task_name


class ConfigDockerService(DataDict):
    """Configuration of docker service extract from gitlab job"""

    privileged: bool = True
    """Are the services running as privileged. As usuall, needed for docker:dind"""
    waiter_image: str = "docker:24.0.6-cli"
    """A image with POSIX sh and docker that waits for services to have open ports"""

    def service_waiter(self, services: List[ServiceSpec]) -> JobTask:
        """https://docs.gitlab.com/ee/ci/services/#how-the-health-check-of-services-works"""
        return JobTask(
            {
                "Name": "ci-wait",
                "Driver": "docker",
                "Config": {
                    "image": self.waiter_image,
                    "command": "sh",
                    "args": [
                        "-c",
                        nomadlib.escape(get_package_file("waiter.sh")),
                        "waiter",
                        "300",
                        *[s.alias for s in services],
                    ],
                    "mount": [
                        {
                            "type": "bind",
                            "source": "/var/run/docker.sock",
                            "target": "/var/run/docker.sock",
                            "readonly": True,
                        }
                    ],
                },
                "Lifecycle": {
                    "Hook": "prestart",
                },
            }
        )

    def service_task(self, s: ServiceSpec):
        """Task that will run specific service. Generated from specification given to us by Gitlab"""
        return JobTask(
            {
                "Name": s.alias,
                "Driver": "docker",
                "Config": {
                    "image": s.name,
                    **({"entrypoint": s.entrypoint} if s.entrypoint else {}),
                    **({"args": s.command} if s.command else {}),
                    "network_aliases": [s.alias],
                    "privileged": True,
                },
                "Env": get_CUSTOM_ENV(),
                "Resources": {
                    "CPU": config.CPU,
                    "MemoryMB": config.MemoryMB,
                },
                "Lifecycle": {
                    "Hook": "prestart",
                    "Sidecar": True,
                },
            }
        )


class ConfigDocker(ConfigCustom):
    builds_dir: str = "/alloc"
    cache_dir: str = "/alloc"
    builds_dir_is_shared: bool = False
    task: JobTask = JobTask(
        {
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
    clone_task: JobTask = JobTask(
        {
            "Name": "ci-sync",
            "Driver": "docker",
            "Config": {
                "image": "gitlab/gitlab-runner:alpine-v16.3.1",
                "entrypoint": [],
                "command": "sleep",
                "args": [
                    "${CUSTOM_ENV_CI_JOB_TIMEOUT}",
                ],
            },
            "RestartPolicy": {"Attempts": 0},
            "Lifecycle": {
                "Hook": "prestart",
                "Sidecar": True,
            },
            "Resources": {
                "CPU": 100,
                "MemoryMB": 200,
            },
        }
    )
    """The task speficiation used to clone git and send artifacts. Has to have git, git-lfs and gitlab-runner."""
    image: str = "alpine"
    """defualt docker image"""
    service: ConfigDockerService = ConfigDockerService()
    """docker service configuration"""

    def apply(self, nomadjob: Job):
        # Docker has additional cloning job.
        nomadjob.TaskGroups[0].Tasks += [self.clone_task]
        # Handle services.
        services = ServiceSpec.get()
        if not services:
            return
        # Using bridge network!
        nomadjob.TaskGroups[0]["Networks"] = [{"Mode": "bridge"}]
        # The list of tasks to run:
        nomadjob.TaskGroups[0].Tasks += [
            self.service.service_waiter(services),
            *[self.service.service_task(s) for s in services],
        ]
        # Apply extra_hosts of services to every task.
        extra_hosts = [f"{s.alias}:127.0.0.1" for s in services]
        for task in nomadjob.TaskGroups[0].Tasks:
            task.Config.extra_hosts = extra_hosts

    def get_task_for_stage(self, stage: str) -> str:
        return (
            main_task_name
            if stage.startswith("step_") or stage.startswith("build_")
            else self.clone_task.Name
        )


class ConfigExec(ConfigCustom):
    builds_dir: str = "/local"
    cache_dir: str = "/local"
    builds_dir_is_shared: bool = ConfigDocker.builds_dir_is_shared
    user: str = "gitlab-runner"
    task: JobTask = JobTask(
        {
            "Driver": "exec",
            "Config": {
                "command": "sleep",
                "args": [
                    "${CUSTOM_ENV_CI_JOB_TIMEOUT}",
                ],
            },
        }
    )


class ConfigRawExec(ConfigCustom):
    builds_dir: str = "/var/lib/gitlab-runner/builds"
    cache_dir: str = "/var/lib/gitlab-runner/cache"
    builds_dir_is_shared: bool = True
    user: str = "gitlab-runner"
    task: JobTask = JobTask(
        {
            "Driver": "raw_exec",
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
    docker = enum.auto()
    exec = enum.auto()
    raw_exec = enum.auto()
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

    mode: str = "docker"
    """Mode to execute with. Has to be set."""
    verbose: int = 0
    """Enable debugging?"""
    purge: bool = False
    """Should the job be purged after we are done?"""
    purge_successful: bool = True
    """Should the successful Nomad jobs be purged after we are done?"""
    jobname: str = "gitlabrunner.${CUSTOM_ENV_CI_RUNNER_ID}.${CUSTOM_ENV_CI_PROJECT_PATH_SLUG}.${CUSTOM_ENV_CI_JOB_ID}"
    """The job name"""
    CPU: Optional[int] = None
    """The defualt job constraints."""
    MemoryMB: Optional[int] = None
    """The defualt job constraints."""
    script: str = get_package_file("script.sh")
    """See https://docs.gitlab.com/runner/executors/custom.html#config"""
    override: ConfigOverride = ConfigOverride()
    """Override Nomad job specification contents"""
    docker: ConfigDocker = ConfigDocker()
    """Relevant in docker mode """
    exec: ConfigExec = ConfigExec()
    """Relevant in exec mode """
    raw_exec: ConfigRawExec = ConfigRawExec()
    """Relevant in raw_exec mode """
    custom: ConfigCustom = ConfigCustom()
    """Relevant in custom mode """

    def __post_init__(self):
        """Update environment variables from configuration - to set NOMAD_TOKEN variable"""
        for k, v in self.asdict().items():
            if k.startswith("NOMAD_") and v:
                assert isinstance(v, str)
                os.environ[k] = v

    @functools.lru_cache(maxsize=0)
    def get_driverconfig(self) -> ConfigCustom:
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

    def _add_meta(self, job: Job):
        dc = self.get_driverconfig()
        task = job.TaskGroups[0].Tasks[0]
        job.Meta = job.Meta if "Meta" in job and job.Meta else {}
        job.Meta.update(
            {
                "CI_JOB_URL": os.environ.get("CUSTOM_ENV_CI_JOB_URL", ""),
                "CI_JOB_NAME": os.environ.get("CUSTOM_ENV_CI_JOB_NAME", ""),
                "CI_PROJECT_URL": os.environ.get("CUSTOM_ENV_CI_PROJECT_URL", ""),
                "CI_DRIVER": task.Driver,
                "CI_RUNUSER": dc.user,
            }
        )

    def _gen_main_task(self) -> JobTask:
        """Get the task to run, apply transformations and configuration as needed"""
        task: JobTask = self.get_driverconfig().task
        assert task, f"is invalid: {task}"
        assert "Config" in task
        task = JobTask(
            {
                "Name": main_task_name,
                "RestartPolicy": {"Attempts": 0},
                "Resources": {
                    "CPU": self.CPU,
                    "MemoryMB": self.MemoryMB,
                },
                "Env": get_CUSTOM_ENV(),
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

    def get_nomad_job(self) -> Job:
        nomadjob = Job(
            {
                "ID": self.jobname,
                "Type": "batch",
                "TaskGroups": [
                    {
                        "Name": "g",
                        "ReschedulePolicy": {"Attempts": 0},
                        "RestartPolicy": {"Attempts": 0},
                        "Tasks": [self._gen_main_task()],
                    }
                ],
                # Apply overrides
                **self.override.job.asdict(),
            }
        )
        self._add_meta(nomadjob)
        # Apply driver specific configuration. I.e. docker.
        self.get_driverconfig().apply(nomadjob)
        return nomadjob


###############################################################################


def run_nomad_watch(cmd: str):
    cmdarr = ["-TG", *split(cmd)]
    cmd = quotearr(cmdarr)
    log.debug(f"+ nomad-watch {cmd}")
    try:
        return nomad_watch.cli.main(cmdarr, standalone_mode=False)
    except SystemExit as e:
        if not (e.code is None or e.code == 0):
            raise


def purge_previous_nomad_job(jobname: str):
    try:
        jobdata = mynomad.get("job/{jobname}")
    except nomadlib.JobNotFound:
        return
    job = Job(jobdata["Job"])
    assert (
        job.Stop == True or job.Status == "dead"
    ), f"Job {job.description()} already exists and is not stopped or not dead. Bailing out"
    run_nomad_watch(f"--purge -xn0 stop {quote(jobname)}")


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
    def create_job(cls) -> Job:
        global config
        nomadjob: Job = config.get_nomad_job()
        # Template the job - expand all ${CUSTOM_ENV_*} references.
        jobjson = json.dumps(nomadjob.asdict())
        try:
            jobjson = OnlyBracedCustomEnvTemplate(jobjson).substitute(os.environ)
        except ValueError:
            log.exception(f"{jobjson}")
            raise
        nomadjob = Job(json.loads(jobjson))
        cls.assert_job(nomadjob)
        return nomadjob

    @classmethod
    def create_job_env(cls) -> Dict[str, str]:
        return {
            "NOMAD_GITLAB_RUNNER_JOB": json.dumps(cls.create_job().asdict()),
        }

    @staticmethod
    def assert_job(job: Job):
        assert "ID" in job, f"ID is missing in {job}"
        assert "Name" in job["TaskGroups"][0]["Tasks"][0], f"{job}"

    @property
    def jobname(self):
        return self.job.ID


###############################################################################


class BuildFailure(Exception):
    code: int = 76
    """Quite a special exit status returned by script.sh when the build script has failed."""


@click.group(
    help="""
This is a script implemeting custom gitlab-runner executor to run jobs in Nomad job from custom gitlab executor.

\b
The /etc/gitlab-runner/config.yaml configuration file should look like:
  [[runners]]
  id = 27898742
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
Example /etc/gitlab-runner/nomad-gitlab-runner.yaml configuration file:
    ---
    default:
        # You can use NOMAD_* variables here
        NOMAD_TOKEN: "1234567"
        NOMAD_ADDR: "http://127.0.0.1:4646"
    # Id of the runner from config.yaml file allows overriding the values for specific runner.
    27898742:
        # Mode to use - "raw_exec", "exec", "docker" or "custom"
        mode: "docker"
        purge: false
        verbose: 0
        CPU: 2048
        MemoryMB: 2048
        docker:
            image: "alpine"
            privileged: false
            services:
                privileged: true
        # If it possible to override some things.
        override:
            task_config:
                cpuset_cpus: "1-3"

\b
Example .gitlab-ci.yml with dockerd service:
    ---
    docker_dind_tls:
        image: docker:24.0.5
        services:
            - docker:24.0.5-dind
        variables:
            DOCKER_HOST: tcp://docker:2376
            DOCKER_TLS_CERTDIR: "/alloc"
            DOCKER_TLS_VERIFY: 1
        script;
            - docker info
    docker_dind_notls:
        image: docker:24.0.5
        services:
            - docker:24.0.5-dind
        variables:
            DOCKER_HOST: tcp://docker:2375
        script;
            - docker info

        """,
    epilog="Written by Kamil Cukrowski 2023. Licensed under GNU GPL version or later.",
)
@click.option("-v", "--verbose", count=True)
@click.option(
    "-c",
    "--config",
    "configpath",
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    default=Path("/etc/gitlab-runner/nomad-gitlab-runner.yaml"),
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
def cli(verbose: int, configpath: Path, section: str):
    # Read configuration
    configcontent = configpath.read_text()
    data = yaml.safe_load(configcontent)
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
        format="%(asctime)s:%(module)s:%(lineno)s: %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        level=logging.DEBUG if config.verbose else logging.INFO,
    )
    log.debug(f"+ {sys.argv}")
    #
    dc = config.get_driverconfig()
    if dc.get("image"):
        os.environ.setdefault("CUSTOM_ENV_CI_JOB_IMAGE", dc["image"])


@cli.command(
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


@cli.command(
    "prepare", help="https://docs.gitlab.com/runner/executors/custom.html#prepare"
)
def mode_prepare():
    # print_env()
    je = Jobenv()
    purge_previous_nomad_job(je.jobname)
    jobjson = json.dumps({"Job": je.job.asdict()})
    run_nomad_watch(f"start {quote(jobjson)}")


@cli.command("run", help="https://docs.gitlab.com/runner/executors/custom.html#run")
@click.argument("script")
@click.argument("stage")
def mode_run(script: str, stage: str):
    je = Jobenv()
    set_x = "-x" if config.verbose > 1 else ""
    # Execute all except step_ and build_ in that special gitlab docker container.
    taskname = config.get_driverconfig().get_task_for_stage(stage)
    rr = run(
        f"nomad alloc exec -t=false -task {quote(taskname)} -job {quote(je.jobname)}"
        f" sh -c {quote(config.script)} gitlabrunner {set_x} -- {quote(stage)}",
        stdin=open(script),
        quiet=True,
        check=False,
    )
    if rr.returncode == BuildFailure.code:
        raise BuildFailure()
    else:
        rr.check_returncode()


@cli.command(
    "cleanup", help="https://docs.gitlab.com/runner/executors/custom.html#cleanup"
)
def mode_cleanup():
    je = Jobenv()
    run_nomad_watch(
        f" {'--purge' if config.purge else ''}"
        f" {'--purge-successful' if config.purge_successful else ''}"
        f" -xn0 stop {quote(je.jobname)}"
    )


@cli.command("showconfig", help="Show current configuration")
def mode_showconfig():
    print(json.dumps(config.asdict(), indent=2))
    print()
    arr = [
        "CI_PROJECT_PATH_SLUG",
        "CI_CONCURRENT_ID",
        "CI_JOB_TIMEOUT",
        "CI_RUNNER_ID",
        "CI_JOB_ID",
    ]
    os.environ.update({f"CUSTOM_ENV_{i}": f"CUSTOM_ENV_{i}_VAL" for i in arr})
    print(json.dumps(Jobenv.create_job().asdict(), indent=2))


###############################################################################


def main(*args, **kvargs) -> int:
    try:
        cli.main(*args, **kvargs)
    except BuildFailure:
        log.debug(f"build failure")
        return int(os.environ.get("BUILD_FAILURE_EXIT_CODE", BuildFailure.code))
    except Exception as e:
        log.exception(e)
        return int(os.environ.get("SYSTEM_FAILURE_EXIT_CODE", 111))
    return 0


if __name__ == "__main__":
    exit(main())
