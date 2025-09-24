#!/usr/bin/env python3

from __future__ import annotations

import enum
import json
import logging
import os
import socket
import string
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from shlex import quote, split
from textwrap import dedent
from typing import Callable, Dict, List, Optional, Set, Union

import click
import yaml

from . import entry_watch, nomadlib, taskexec
from .aliasedgroup import AliasedGroup
from .common import (
    cached_property,
    get_package_file,
    get_version,
    help_h_option,
    mynomad,
    quotearr,
)
from .nomadlib.datadict import DataDict
from .nomadlib.types import Job, JobTask, JobTaskConfig

###############################################################################

NAME = "nomad-gitlab-runner"
log = logging.getLogger(NAME)


def get_gitlab_runner_package_script(file: str):
    return get_package_file(f"entry_gitlab_runner/{file}")


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
    entrypoint: Optional[List[str]] = None
    command: Optional[List[str]] = None
    alias: str = ""
    variables: Dict[str, str] = {}

    def __post_init__(self):
        if not self.alias:
            self.alias = self.name.split("/")[-1].split(":")[0]

    @staticmethod
    def get_servicespecs() -> List[ServiceSpec]:
        """Read the Gitlab environment variable to extract the service"""
        CI_JOB_SERVICES = os.environ.get("CUSTOM_ENV_CI_JOB_SERVICES")
        data: List[Union[str, Dict[str, str]]] = (
            json.loads(CI_JOB_SERVICES) if CI_JOB_SERVICES else []
        )
        assert isinstance(data, list), (
            f"CUSTOM_ENV_CI_JOB_SERVICES is not a list: {CI_JOB_SERVICES}"
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
        assert not ServiceSpec.get_servicespecs(), (
            "Services only handled in Docker mode"
        )
        assert nomadjob

    def get_task_for_stage(self, stage: str) -> str:
        assert stage
        return main_task_name


class ConfigDocker(ConfigCustom):
    builds_dir: str = "/alloc/builds"
    cache_dir: str = "/alloc/cache"
    builds_dir_is_shared: Optional[bool] = False
    image: str = "alpine:latest"
    """The image to run jobs with."""
    volumes: List[str] = []
    """https://developer.hashicorp.com/nomad/docs/drivers/docker#volumes
    The default mounts ${NOMAD_ALLOD_DIR} to /certs to have it available for docker dind sevice just like in gitlab-runner."""
    wait_for_services_timeout: int = 30
    """How long to wait for Docker services. Set to -1 to disable. Default is 30."""
    privileged: bool = False
    """Make the container run in privileged mode. Insecure."""
    services_privileged: Optional[bool] = None
    """Allow services to run in privileged mode. If unset (default) privileged value is used instead.
    Use with the Docker executor. Insecure."""
    force_pull: bool = True
    """https://developer.hashicorp.com/nomad/docs/drivers/docker#force_pull"""
    waiter_image: str = "docker:25.0.3-cli"
    """A image with POSIX sh and docker that waits for services to have open ports
    Url: https://hub.docker.com/r/gitlab/gitlab-runner/tags"""
    # helper_image: str = "gitlab/gitlab-runner:alpine-v16.9.1"
    helper_image: str = "gitlab/gitlab-runner:bleeding"
    """(Advanced) The default helper image used to clone repositories and upload artifacts."""
    auto_fix_docker_dind: bool = True
    """
    If there is a service aliased "docker" and DOCKER_TLS_CERTDIR is set to exactly "/certs"
    and DOCKER_CERT_PATH, DOCKER_HOST nor DOCKER_TLS_VERIFY are not set, then:
    automatically mount volume "${NOMAD_ALLOC_DIR}:/certs"
    and add following environment variables:
        DOCKER_CERT_PATH="/certs/client"
        DOCKER_HOST=tcp://docker:2376
        DOCKER_TLS_CERTDIR=/certs
        DOCKER_TLS_VERIFY=1
    This automatically properly handles the environment variables with docker like in plain gitlab-runner.
    See also https://docs.gitlab.com/ee/ci/docker/using_docker_build.html#docker-in-docker-with-tls-enabled-in-kubernetes
    """
    task: JobTask = JobTask(
        {
            "Driver": "docker",
            "Config": {
                "image": "${CUSTOM_ENV_CI_JOB_IMAGE}",
                "entrypoint": ["sleep", "${CUSTOM_ENV_CI_JOB_TIMEOUT}"],
            },
        }
    )
    """The main task specification"""

    def get_helper_task(self):
        """The task specification used to clone git and send artifacts. Has to have git, git-lfs and gitlab-runner."""
        return JobTask(
            {
                "Name": "ci-help",
                "Driver": "docker",
                "Config": {
                    "image": self.helper_image,
                    "entrypoint": ["sleep", "${CUSTOM_ENV_CI_JOB_TIMEOUT}"],
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

    def get_waiter_task(self, services: List[ServiceSpec]) -> JobTask:
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
                        nomadlib.escape(get_gitlab_runner_package_script("waiter.sh")),
                        "waiter",
                        str(self.wait_for_services_timeout),
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

    def get_service_task(self, s: ServiceSpec):
        """Task that will run specific service. Generated from specification given to us by Gitlab"""
        return JobTask(
            {
                "Name": s.alias,
                "Driver": "docker",
                "Config": {
                    "image": s.name,
                    **({"entrypoint": s.entrypoint} if s.entrypoint else {}),
                    **({"args": s.command} if s.command else {}),
                    "privileged": (
                        self.services_privileged
                        if self.services_privileged is not None
                        else self.privileged
                    ),
                },
                "Env": get_CUSTOM_ENV(),
                "RestartPolicy": {"Attempts": 0},
                **config.resources(),
                "force_pull": self.force_pull,
                "Lifecycle": {
                    "Hook": "prestart",
                    "Sidecar": True,
                },
            }
        )

    def apply(self, nomadjob: Job):
        services = ServiceSpec.get_servicespecs()
        # Apply configuration to main task.
        assert len(nomadjob.TaskGroups) == 1
        taskgroup = nomadjob.TaskGroups[0]
        assert len(taskgroup.Tasks) == 1
        maintask: nomadlib.JobTask = taskgroup.Tasks[0]
        maintask["force_pull"] = self.force_pull
        maintask["privileged"] = self.privileged
        # Docker has additional cloning job.
        taskgroup.Tasks += [self.get_helper_task()]
        # Handle services.
        if services:
            # Using bridge network!
            taskgroup["Networks"] = [{"Mode": "bridge"}]
            # The list of tasks to run:
            taskgroup.Tasks += [
                self.get_waiter_task(services),
                *[self.get_service_task(s) for s in services],
            ]
            # Apply extra_hosts of services to every task.
            extra_hosts = [f"{s.alias}:127.0.0.1" for s in services]
            for task in taskgroup.Tasks:
                assert task["Driver"] == "docker"
                task.Config["extra_hosts"] = extra_hosts
        # Apply auto_fix_docker_ding
        if (
            self.auto_fix_docker_dind
            and os.environ.get("CUSTOM_ENV_DOCKER_TLS_CERTDIR") == "/certs"
            and all(
                f"CUSTOM_ENV_{i}" not in os.environ
                for i in "DOCKER_HOST DOCKER_CERT_PATH DOCKER_TLS_VERIFY".split()
            )
            and services
            and any(s.alias == "docker" for s in services)
        ):
            for task in taskgroup.Tasks:
                if "Env" in task and task.Env:
                    assert task.Env["DOCKER_TLS_CERTDIR"] == "/certs"
                    assert "DOCKER_CERT_PATH" not in task.Env
                    assert "DOCKER_HOST" not in task.Env
                    assert "DOCKER_TLS_VERIFY" not in task.Env
                    task.Env.update(
                        {
                            "DOCKER_CERT_PATH": "/certs/client",
                            "DOCKER_HOST": "tcp://docker:2376",
                            "DOCKER_TLS_VERIFY": "1",
                        }
                    )
            # Volumes handled below.
            self.volumes += ["${NOMAD_ALLOC_DIR}:/certs"]
        for task in taskgroup.Tasks:
            taskconfig = task["Config"]
            # Apply cpuset_cpus
            if config.cpuset_cpus:
                taskconfig["cpuset_cpus"] = config.cpuset_cpus
            # Apply volumes.
            if self.volumes:
                taskconfig.setdefault("volumes", []).extend(self.volumes)

    def get_task_for_stage(self, stage: str) -> str:
        return (
            main_task_name
            if stage.startswith("step_") or stage.startswith("build_")
            else self.get_helper_task().Name
        )


class ConfigExec(ConfigCustom):
    builds_dir: str = "/local/builds"
    cache_dir: str = "/local/cache"
    builds_dir_is_shared: Optional[bool] = False
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
    builds_dir_is_shared: Optional[bool] = True
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


def min_none(*data: Optional[int]):
    return min((x for x in data if x), default=None)


def int_list_to_ints(data: str) -> Set[int]:
    ret: Set[int] = set()
    for i in data.split(","):
        if i:
            if "-" in i:
                a, b = i.split("-", 1)
                for k in range(int(a), int(b)):
                    ret.add(k)
            else:
                ret.add(int(i))
    return ret


class Config(DataDict):
    """Configuration of this program"""

    NOMAD_ADDR: Optional[str] = os.environ.get("NOMAD_ADDR")
    NOMAD_REGION: Optional[str] = os.environ.get("NOMAD_REGION")
    NOMAD_NAMESPACE: Optional[str] = os.environ.get("NOMAD_NAMESPACE", "gitlabrunner")
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
    verbose: Union[bool, int] = 0
    """Enable debugging?"""
    purge: Optional[bool] = None
    """Should the job be purged after we are done?"""
    purge_successful: bool = True
    """Should the successful Nomad jobs be purged after we are done? Only relevant when purge=none."""
    jobname: str = "gitlabrunner.${CUSTOM_ENV_CI_RUNNER_ID}.${CUSTOM_ENV_CI_PROJECT_PATH_SLUG}.${CUSTOM_ENV_CI_JOB_ID}"
    """The job name"""
    CPU: Optional[int] = None
    """The default job constraints."""
    cores: Optional[int] = None
    """The default job constraints."""
    MemoryMB: Optional[int] = None
    """The default job constraints."""
    MemoryMaxMB: Optional[int] = None
    """The default job constraints."""
    script: str = get_gitlab_runner_package_script("script.sh")
    """See https://docs.gitlab.com/runner/executors/custom.html#config"""
    oom_score_adjust: int = 10
    """OOM score adjustment. Positive means kill earlier."""
    cpuset_cpus: str = ""
    """Set with taskset -c or with Nomad cpuset_cpus. The control group’s CpusetCpus. A string."""
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
        # If cpuset_cpus and NOMADRUNNER_CPUSET_CPUS are given, restrict the number of cpus only to cpuset_cpus.
        tmp = os.environ.get("CUSTOM_ENV_NOMADRUNNER_CPUSET_CPUS", "")
        self.cpuset_cpus: str = (
            ",".join(
                str(x)
                for x in sorted(
                    int_list_to_ints(self.cpuset_cpus) - int_list_to_ints(tmp)
                )
            )
            if self.cpuset_cpus
            else tmp
        )

    @cached_property
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
        cenv = get_CUSTOM_ENV()
        dc = self.get_driverconfig
        task = job.TaskGroups[0].Tasks[0]
        job.Meta = job.Meta if "Meta" in job and job.Meta else {}
        job.Meta.update(
            {
                # These variables are for easy navigation documenation in the Nomad gui.
                "CI_JOB_URL": os.environ.get("CUSTOM_ENV_CI_JOB_URL", ""),
                "CI_JOB_NAME": os.environ.get("CUSTOM_ENV_CI_JOB_NAME", ""),
                "CI_PROJECT_URL": os.environ.get("CUSTOM_ENV_CI_PROJECT_URL", ""),
                "CI_DRIVER": task.Driver,
                "CI_RUNUSER": dc.user,
                "CI_OOM_SCORE_ADJUST": str(self.oom_score_adjust),
                "CI_CPUSET_CPUS": self.cpuset_cpus,
            }
        )

    def resources(self):
        cenv = get_CUSTOM_ENV()
        return {
            "Resources": {
                "CPU": min_none(int(cenv.get("NOMADRUNNER_CPU", 0)), self.CPU),
                "Cores": min_none(int(cenv.get("NOMADRUNNER_CORES", 0)), self.cores),
                "MemoryMB": min_none(
                    int(cenv.get("NOMADRUNNER_MEMORY_MB", 0)), self.MemoryMB
                ),
                "MemoryMaxMB": min_none(
                    int(cenv.get("NOMADRUNNER_MEMORY_MAX_MB", 0)),
                    self.MemoryMaxMB,
                ),
            }
        }

    def __gen_main_task(self) -> JobTask:
        """Get the task to run, apply transformations and configuration as needed"""
        task: JobTask = self.get_driverconfig.task
        assert task, f"is invalid: {task}"
        assert "Config" in task
        task = JobTask(
            {
                "Name": main_task_name,
                "RestartPolicy": {"Attempts": 0},
                "Env": get_CUSTOM_ENV(),
                **self.resources(),
                **task.asdict(),
                # Apply override on task.
                **self.override.task,
            }
        )
        # Apply override on config
        task.Config = JobTaskConfig({**task.Config, **self.override.task_config})
        return task

    def get_nomad_job(self) -> Job:
        nomadjob = Job(
            {
                "ID": self.jobname,
                "Type": "batch",
                "TaskGroups": [
                    {
                        "Name": "R",
                        "ReschedulePolicy": {"Attempts": 0},
                        "RestartPolicy": {"Attempts": 0},
                        "Tasks": [self.__gen_main_task()],
                    }
                ],
                # Apply overrides
                **self.override.job.asdict(),
            }
        )
        self._add_meta(nomadjob)
        # Apply driver specific configuration. I.e. docker.
        self.get_driverconfig.apply(nomadjob)
        # Force apply 0 restart policy to every task.
        for t in nomadjob.TaskGroups[0].Tasks:
            t.setdefault("RestartPolicy", {"Attempts": 0})
        return nomadjob


###############################################################################


def run_entry_watch(cmd: str):
    cmdarr = ["-T", *split(cmd)]
    cmd = quotearr(cmdarr)
    log.debug(f"+ nomad-watch {cmd}")
    try:
        return entry_watch.cli.main(cmdarr, standalone_mode=False)
    except SystemExit as e:
        if not (e.code is None or e.code == 0):
            raise


def purge_previous_nomad_job(jobname: str):
    try:
        jobdata = mynomad.get(f"job/{jobname}")
    except nomadlib.JobNotFound:
        return
    try:
        job = Job(jobdata.get("Job", jobdata))
    except KeyError:
        log.exception(
            "Something wrong with purging the nomad job. Feel free to fill an issue"
        )
        return
    assert job.Stop is True or job.Status == "dead", (
        f"Job {job.description()} already exists and is not stopped or not dead. Bailing out"
    )
    run_entry_watch(f"--purge -xn0 stop {quote(jobname)}")


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
    pattern: str = str(  # pyright: ignore [reportIncompatibleVariableOverride]
        r"""
        # match ${CUSTOM_ENV_*} only
        \${(?P<braced>CUSTOM_ENV_[_A-Za-z0-9]*)} |
        (?P<escaped>\x01)  |  # match nothing
        (?P<named>\x01)    |  # match nothing
        (?P<invalid>\x01)     # match nothing
        """
    )

    @staticmethod
    def run(template: str) -> str:
        try:
            return OnlyBracedCustomEnvTemplate(template).substitute(os.environ)
        except ValueError:
            log.exception(f"{template}")
            raise


class Jobenv:
    """
    The job specification is stored in job_env inside config section returned on config stage.
    Manage that job spefication - serialize and deserialize it.
    """

    def __init__(self):
        NOMAD_GITLAB_RUNNER_JOB = os.environ.get("NOMAD_GITLAB_RUNNER_JOB")
        assert NOMAD_GITLAB_RUNNER_JOB is not None, (
            "Env variable set in config NOMAD_GITLAB_RUNNER_JOB is missing"
        )
        jobjson = json.loads(NOMAD_GITLAB_RUNNER_JOB)
        self.job = Job(jobjson)
        self.assert_job(self.job)

    @classmethod
    def create_job(cls) -> Job:
        global config
        nomadjob: Job = config.get_nomad_job()
        # Template the job - expand all ${CUSTOM_ENV_*} references.
        jobjson = json.dumps(nomadjob.asdict())
        jobjson = OnlyBracedCustomEnvTemplate.run(jobjson)
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


@dataclass
class BuildFailure(Exception):
    """Raising this exception makes the code exit with BUILD_FAILURE_EXIT_CODE"""

    code: int
    """Special chosen exit status returned by script.sh that the build script has failed."""


@click.group(
    "gitlab-runner",
    cls=AliasedGroup,
    help=""" Custom gitlab-runner executor to run gitlab-ci jobs in Nomad. """,
    epilog="Written by Kamil Cukrowski 2023. Licensed under GNU GPL version or later.",
)
@click.option("-v", "--verbose", count=True)
@click.option(
    "-c",
    "--config",
    "configpath",
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    default=Path("/etc/gitlab-runner/nomad-gitlab-runner.yml"),
    help="""Path to configuration file.""",
    show_default=True,
)
@click.option(
    "-r",
    "--runner-id",
    help="""
An additional section read from configuration file to merge with defaults.
The value defaults to CUSTOM_ENV_CI_RUNNER_ID which is set to the unique ID of the runner being used.
""",
    type=int,
    envvar="CUSTOM_ENV_CI_RUNNER_ID",
    show_default=True,
)
@help_h_option()
def cli(verbose: int, configpath: Path, runner_id: int):
    # Read configuration
    configcontent = configpath.read_text()
    data = yaml.safe_load(configcontent)
    configs = {key: val for key, val in (data or {}).items()}
    global config
    config = Config(
        {**configs.get("default", {}), **configs.get(runner_id, {})}
    ).remove_none()
    if verbose:
        config.verbose = 1
    #
    logging.basicConfig(
        format=f"%(asctime)s:{NAME}:%(lineno)s: %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        level=logging.DEBUG if config.verbose else logging.INFO,
    )
    log.debug(f"+ {sys.argv}")
    #
    dc = config.get_driverconfig
    if dc.get("image"):
        os.environ.setdefault("CUSTOM_ENV_CI_JOB_IMAGE", dc["image"])


def executor_exit(f: Callable) -> Callable:
    """
    Custom executor should always exit with specific exit codes
    https://docs.gitlab.com/runner/executors/custom/#build-failure-exit-code
    """

    def executor_exiting(*args, **kvargs):
        try:
            f(*args, **kvargs)
        except BuildFailure as e:
            log.debug(f"build failure {e}")
            BUILD_EXIT_CODE_FILE = os.environ.get("BUILD_EXIT_CODE_FILE")
            if BUILD_EXIT_CODE_FILE:
                Path(BUILD_EXIT_CODE_FILE).write_text(str(e.code))
            exit(int(os.environ.get("BUILD_FAILURE_EXIT_CODE", e.code)))
        except Exception as e:
            log.exception(e)
            exit(int(os.environ.get("SYSTEM_FAILURE_EXIT_CODE", 111)))
        return 0

    return executor_exiting


@cli.command(
    "config", help="https://docs.gitlab.com/runner/executors/custom.html#config"
)
@executor_exit
def mode_config():
    dc = config.get_driverconfig
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
    driver_config_json = OnlyBracedCustomEnvTemplate.run(driver_config_json)
    log.debug(f"driver_config={driver_config_json}")
    click.echo(driver_config_json)


@cli.command(
    "prepare", help="https://docs.gitlab.com/runner/executors/custom.html#prepare"
)
@executor_exit
def mode_prepare():
    # print_env()
    je = Jobenv()
    purge_previous_nomad_job(je.jobname)
    jobjson = json.dumps({"Job": je.job.asdict()})
    log.debug(f"JOBJSON: {jobjson}")
    with tempfile.NamedTemporaryFile("w+") as f:
        f.write(jobjson)
        f.flush()
        run_entry_watch(f"--json start {f.name}")


FILE_ISSUE: str = "File an issue at https://github.com/Kamilcuk/nomadtools/issues/new"


@cli.command("run", help="https://docs.gitlab.com/runner/executors/custom.html#run")
@click.argument("script")
@click.argument("stage")
@executor_exit
def mode_run(script: str, stage: str):
    je = Jobenv()
    set_x = "-x" if config.verbose > 1 else ""
    # Execute all except step_ and build_ in that special gitlab docker container.
    taskname = config.get_driverconfig.get_task_for_stage(stage)
    # Find our running allocation.
    allocid = taskexec.find_job_alloc(je.jobname, taskname)
    scriptcontent = Path(script).read_text()
    # Run our script with positional arguments with the gitlab script passed as first argument.
    log.debug(f"executing {allocid}/{taskname} state={stage}")
    rr = taskexec.run(
        allocid,
        taskname,
        split(
            f"sh {set_x} -c {quote(config.script)} sh {quote(scriptcontent)} {quote(stage)}"
        ),
    )
    # 155 is hardcoded in script.sh
    if rr.returncode == 155:
        cmd = f"nomad alloc fs {quote(allocid)} {quote(taskname)}/local/code.txt"
        log.debug(f"Getting exit status from {cmd}")
        try:
            code = int(subprocess.check_output(split(cmd)))
        except (subprocess.CalledProcessError, ValueError):
            log.exception(
                f"nomad gitlab script failed to store exit code. This might be an internal error. {FILE_ISSUE}"
            )
            raise
        raise BuildFailure(code)
    rr.check_returncode()


@cli.command(
    "cleanup", help="https://docs.gitlab.com/runner/executors/custom.html#cleanup"
)
@executor_exit
def mode_cleanup():
    je = Jobenv()
    run_entry_watch(
        f" {'--purge' if config.purge else ''}"
        f" {'--purge-successful' if config.purge_successful and config.purge is None else ''}"
        f" -x -n0 stop {quote(je.jobname)}"
    )


@cli.command(
    "showconfig",
    help="Can be run manually. Check and show current configuration.",
)
def mode_showconfig():
    arr = [
        "CI_CONCURRENT_ID",
        "CI_JOB_ID",
        "CI_JOB_TIMEOUT",
        "CI_PROJECT_PATH_SLUG",
        "CI_RUNNER_ID",
    ]
    for i in arr:
        os.environ.setdefault(f"CUSTOM_ENV_{i}", f"CUSTOM_ENV_{i}_VAL")
    os.environ.setdefault("CUSTOM_ENV_CI_JOB_SERVICES", json.dumps(["docker:dind"]))
    os.environ.setdefault("CUSTOM_ENV_DOCKER_TLS_CERTDIR", "/certs")
    #
    print("--- generated example job configuration ---")
    print(json.dumps(Jobenv.create_job().asdict(), indent=2))
    print()
    print("--- program configuration: ---")
    print(yaml.dump(config.asdict()))


###############################################################################

if __name__ == "__main__":
    cli()
