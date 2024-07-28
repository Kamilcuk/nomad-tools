#!/usr/bin/env python3
from __future__ import annotations

import collections
import concurrent.futures
import datetime
import functools
import itertools
import json
import logging
import os
import random
import re
import shlex
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

import click
import clickdc
import jinja2
import requests
import yaml

from . import nomadlib
from .common import mynomad
from .nomadlib.connection import urlquote
from .nomadlib.datadict import DataDict

log = logging.getLogger(__name__)

###############################################################################
# config

DEFAULT_RUNNER: str = """
job "{{ run.JOB_NAME }}" {
  type = "batch"
  meta {
    INFO = <<EOF
This is a runner based on {{ image }} image.
{% if opts.docker == "dind" %}
It also starts a docker daemon and is running as privileged
{% elif opts.docker == "host" %}
It also mounts a docker daemon from the host it is running on
{% endif %}
EOF
  }
  group "{{ run.JOB_NAME }}" {
    reschedule {
      attempts  = 0
      unlimited = false
    }
    restart {
      attempts = 0
      mode     = "fail"
    }
    task "{{ run.JOB_NAME }}" {
      driver = "docker"
      config {
        image = "{{ arg.image|default("myoung34/github-runner:latest") }}"
        {% if arg.debug %}
        # for debugging
        entrypoint = ["bash", "-x", "/entrypoint.sh", "./bin/Runner.Listener", "run", "--startuptype", "service"]
        {% endif %}
        {% if opts.docker == "dind" %}
        privileged = true
        {% elif opts.docker == "host" %}
        mount {
            type   = "bind"
            target = "/var/run/docker.sock"
            source = "/var/run/docker.sock"
        }
        {% endif %}
        {{ opts.config }}
      }
      env {
        ACCESS_TOKEN        = "{{ run.ACCESS_TOKEN }}"
        REPO_URL            = "{{ run.REPO_URL }}"
        RUNNER_NAME         = "{{ run.JOB_NAME }}"
        RUNNER_SCOPE        = "repo"
        LABELS              = "{{ run.LABELS }}"
        # RUN_AS_ROOT         = "false"
        {% if opts.ephemeral == "true" %}
        EPHEMERAL           = "true"
        {% endif %}
        DISABLE_AUTO_UPDATE = "true"
        {% if opts.docker == "dind" %}
        START_DOCKER_SERVICE = "true"
        {% endif %}
      }
      resources {
        cpu = {{ arg.cpu|default(300) }}
        {% set mem = arg.mem|default(2000) %}
        memory = {{ mem}}
        memory_max = {{ arg.maxmem|default(mem) }}
      }
      {{ opts.task }}
    }
    {{ opts.group }}
  }
  {{ opts.job }}
}
        """


class NomadConfig(DataDict):
    """Nomad configuration"""

    namespace: str = "github"
    """The namespace to set on the job."""
    token: Optional[str] = os.environ.get("NOMAD_TOKEN")
    jobprefix: str = "NTGithubRunner"
    """The string to prefix run jobs with.
    Preferably something short, there is a limit on Github runner name."""
    meta: str = "NT"
    """The prefix applied to metadata keys set by this program.
    Default: "NT" like Nomad Tools."""
    purge: bool = True
    """Purge dead jobs"""


class GithubConfig(DataDict):
    url: str = "https://api.github.com"
    """The url to github api"""
    token: Optional[str] = os.environ.get("GH_TOKEN", os.environ.get("GITHUB_TOKEN"))
    """The token to access github api"""
    access_token: Optional[str] = None
    """The github access token consul-template template code."""
    cachefile: str = "~/.cache/nomadtools/githubcache.json"
    """The location of cachefile. os.path.expanduser is used to expand."""


class LimitConfig(DataDict):
    repo: str = ".*"
    """Match repositories full name with this regex"""
    labels: str = ".*"
    """Match runners with this regex"""
    max: int = -1
    """Maximum number of running runners"""


class Config(DataDict):
    """Configuration"""

    nomad: NomadConfig = NomadConfig()
    """Nomad related configuration"""

    github: GithubConfig = GithubConfig()
    """Configuration related to github with source repositories list"""

    repos: List[str] = []
    """List of repositories to watch.
    This is either a organization is user, in which case API is called
    to get all repositories of this organization or user.
    Or this is a single repository full name in the form 'user/repo'.
    A star '*' causes to get all the repositories available.
    """

    loop: int = 10
    """How many seconds will the loop run"""

    runners: dict[str, str] = {
        "nomadtools.*": DEFAULT_RUNNER,
    }
    """The runners configuration.
    Each field should be either a path to a file containing a HCL or JSON
    job specification, or it should be a HCL or JSON Nomad job specification
    in string form.
    The job should contain a job meta field called NT_LABELS with comma
    separated list of labels of the job.
    The job will be started with additional NT_* metadata fields
    with values from the Spec object below.
    You can use for example ".*self-hosted.*" for all-catch.
    """

    runner_inactivity_timeout: str = "1h"
    """How much time a runner will be inactive for it to be removed?"""

    limits: List[LimitConfig] = []
    """Add limitation on the number of runners"""

    opts: Any = {
        "job": "",
        "group": "",
        "task": "",
        "ephemeral": False,
        "docker": "dind",
    }
    """Additional template variable passed as 'opts' global variable."""

    def __post_init__(self):
        assert self.loop >= 0
        assert self.repos
        # check if throws
        self.get_runner_inactivity_timeout()

    def get_runner_inactivity_timeout(self) -> Optional[datetime.timedelta]:
        return (
            parse_time(self.runner_inactivity_timeout)
            if self.runner_inactivity_timeout
            else None
        )


CONFIG: Config

###############################################################################
# counters


@dataclass
class IncAtomicInt:
    v: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, v: int):
        with self.lock:
            self.v = v

    def inc(self):
        with self.lock:
            pre = self.v
            self.v += 1
            return pre

    def get(self):
        with self.lock:
            return self.v

    def __str__(self):
        return str(self.get())

    def __add__(self, o: IncAtomicInt):
        with self.lock:
            return self.v + o.get()


@dataclass
class AtomicIncIntSet:
    data: set[int] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_free(self) -> int:
        with self.lock:
            for i in range(sorted(list(self.data or [0]))[-1] + 1):
                if i not in self.data:
                    self.data.add(i)
                    return i
        assert False, "This is not possible"

    def add(self, x: int):
        with self.lock:
            assert x not in self.data
            self.data.add(x)

    def clear(self):
        self.data.clear()


class Counters:
    def __init__(self):
        self.github_cached = IncAtomicInt()
        self.github_miss = IncAtomicInt()
        self.nomad_get = IncAtomicInt()
        self.nomad_run = IncAtomicInt()
        self.nomad_stop = IncAtomicInt()
        self.nomad_purge = IncAtomicInt()
        # not reset, global counter
        self.cnt = IncAtomicInt(random.randint(0, 1000))

    def print(self):
        stats: str = (
            "stats: Github("
            + " ".join(
                [
                    f"requests={self.github_cached + self.github_miss}",
                    f"cached={self.github_cached}",
                    f"miss={self.github_miss})",
                ]
            )
            + " Nomad("
            + " ".join(
                [
                    f"get={self.nomad_get}",
                    f"run={self.nomad_run}",
                    f"stop={self.nomad_stop}",
                    f"purge={self.nomad_purge}",
                ]
            )
            + ")"
        )
        log.info(stats)
        for i in dir(self):
            if (
                not i.startswith("_")
                and isinstance(getattr(self, i), IncAtomicInt)
                and i != "cnt"
            ):
                getattr(self, i).set(0)


COUNTERS = Counters()
"""Small object to accumulate what is happening in a single loop"""

###############################################################################
# helpers

T = TypeVar("T")
R = TypeVar("R")


def parallelmap(
    func: Callable[[T], R], arr: Iterable[T], nproc: Optional[int] = None
) -> Iterable[R]:
    """Execute lambda over array in parellel and return an array of it"""
    if ARGS.noparallel:
        return map(func, arr)
    else:
        with concurrent.futures.ThreadPoolExecutor(nproc) as p:
            futures = [p.submit(func, x) for x in arr]
            return (x.result() for x in concurrent.futures.as_completed(futures))
            # return list(p.map(func, arr))


def flatten(xss: Iterable[Iterable[T]]) -> Iterable[T]:
    return (x for xs in xss for x in xs)


PARSE_TIME_REGEX = re.compile(
    r"((?P<hours>\d+?)h)?((?P<minutes>\d+?)m)?((?P<seconds>\d+?)s)?"
)


def parse_time(time_str) -> Optional[datetime.timedelta]:
    # https://stackoverflow.com/a/4628148/9072753
    parts = PARSE_TIME_REGEX.match(time_str)
    if not parts:
        return
    parts = parts.groupdict()
    time_params = {}
    for name, param in parts.items():
        if param:
            time_params[name] = int(param)
    return datetime.timedelta(**time_params)


@functools.lru_cache()
def nomad_job_to_json(job: str) -> dict:
    """Convert nomad specification in string form to a json"""
    data = None
    # try json
    try:
        data = json.loads(job)
    except json.JSONDecodeError:
        pass
    # try hcl
    if not data:
        try:
            data = json.loads(
                subprocess.check_output(
                    "nomad job run -output -".split(), text=True, input=job
                )
            )
        except subprocess.CalledProcessError:
            pass
    # try filename containing json
    if not data:
        try:
            with Path(job).open() as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    # try filename containing HCL
    if not data:
        try:
            data = json.loads(
                subprocess.check_output(
                    "nomad job run -output".split() + [job], text=True
                )
            )
        except subprocess.CalledProcessError:
            pass
    if not data:
        raise Exception(f"Could not read {job}")
    data = data.get("Job", data)
    try:
        subprocess.run(
            "nomad job validate -json -".split(),
            text=True,
            input=json.dumps(data),
            stdout=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError:
        raise Exception(f"Validating runner Nomad job specification failed: {data}")
    return data


###############################################################################
# github


@functools.lru_cache()
def github_cachefile():
    return Path(os.path.expanduser(CONFIG.github.cachefile))


@dataclass
class GithubCache:
    """Stores and manages github cache
    See:
    https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api?apiVersion=2022-11-28#use-conditional-requests-if-appropriate
    """

    Url = str

    @dataclass
    class Value:
        """The cache may be etag or last-modified see github docs"""

        is_etag: bool
        etag_or_last_modified: str
        response: dict
        timestamp: float = field(default_factory=lambda: time.time())

    data: dict[Url, Value] = field(default_factory=dict)
    """The data stored by cache is keyed with URL"""
    version: int = 1

    @classmethod
    def load(cls):
        """Load the GithubCache from file"""
        try:
            with open(github_cachefile()) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return GithubCache()
        if data["version"] != GithubCache.version:
            return GithubCache()
        gh = GithubCache({k: cls.Value(**v) for k, v in data["data"].items()})
        log.info(
            f"Github cache loaded {len(gh.data)} entries datafrom {github_cachefile()}"
        )
        return gh

    def save(self):
        """Save the cache to file. Run each loop"""
        data = {
            "version": self.version,
            "data": {k: asdict(v) for k, v in self.data.items()},
        }
        os.makedirs(os.path.dirname(github_cachefile()), exist_ok=True)
        with open(github_cachefile(), "w") as f:
            json.dump(data, f)
        log.debug(
            f"Github cache saved {len(self.data)} entires to {github_cachefile()}"
        )

    def prepare(self, url: str, headers: dict[str, str]):
        cached = self.data.get(url)
        if cached:
            if cached.is_etag:
                headers["if-none-match"] = cached.etag_or_last_modified
            else:
                headers["if-modified-since"] = cached.etag_or_last_modified

    def handle(self, response: requests.Response) -> Optional[dict]:
        if response.status_code == 304:
            COUNTERS.github_cached.inc()
            return self.data[response.url].response
        else:
            COUNTERS.github_miss.inc()
        etag = response.headers.get("etag")
        if etag:
            self.data[response.url] = self.Value(True, etag, response.json())
        else:
            last_modified = response.headers.get("last-modified")
            if last_modified:
                self.data[response.url] = self.Value(
                    False, last_modified, response.json()
                )
        return None


GITHUB_CACHE: GithubCache
"""Stores and manages the github cache"""


def gh_get(url: str, key: str = "") -> Any:
    """Execute query to github

    @param key if set, means the output is paginated
    """
    headers = {
        "Accept": "application/vnd.github+json",
        **(
            {"Authorization": "Bearer " + CONFIG.github.token}
            if CONFIG.github.token
            else {}
        ),
    }
    ret: list[dict] = []
    while True:
        # Prepare the request adding headers from Github cache.
        GITHUB_CACHE.prepare(url, headers)
        response = requests.get(url, headers=headers)
        log.debug(f"{url} {headers} {response}")
        response.raise_for_status()
        # If the result is found in github cache, use it, otherwise extract it.
        data = GITHUB_CACHE.handle(response)
        if not data:
            data = response.json()
        # If no key, this is not paged url.
        if not key:
            return data
        # If key, this is paged url - append it and continue.
        try:
            ret.extend(data[key])
        except KeyError:
            log.exception(f"key={key} data={data}")
            raise
        next = response.links.get("next")
        if not next:
            break
        url = next["url"]
    return ret


@functools.lru_cache()
@functools.wraps(gh_get)
def gh_get_cached(url: str, key: str = ""):
    return gh_get(url, key)


###############################################################################
# github high level

GithubRepo = str


@dataclass(frozen=True)
class GithubJob:
    repo: GithubRepo
    run: dict
    job: dict

    def labelsstr(self):
        return ",".join(sorted(list(set(self.job["labels"]))))

    def job_url(self):
        return self.job["html_url"]

    def repo_url(self):
        return self.run["repository"]["html_url"]


def get_gh_repos_one_to_many(repo: str) -> List[GithubRepo]:
    """If github repo is a organization or user, expand to
    all repositories of that organization or user"""
    assert repo.count("/") <= 1, f"{repo} has too many /"
    if repo == "*":
        assert "api.github.com" not in CONFIG.github.url
        users = gh_get(f"{CONFIG.github.url}/users")
        organizations = gh_get(f"{CONFIG.github.url}/organizations")
        return list(
            parallelmap(
                lambda x: gh_get(x["repos_url"])["full_name"], users + organizations
            )
        )
    elif repo.count("/") == 0:
        d = gh_get_cached(f"{CONFIG.github.url}/users/{repo}")
        d = gh_get(d["repos_url"])
        return [x["full_name"] for x in d]
    else:
        return [repo]


def get_gh_repos() -> Set[GithubRepo]:
    """Get list of configured repositoires with organizations
    and users repositories expanded"""
    return set(flatten(parallelmap(get_gh_repos_one_to_many, set(CONFIG.repos))))


def get_gh_run_jobs(repo: GithubRepo, run: dict) -> List[GithubJob]:
    interesting = "queued in_progress".split()
    ret: List[GithubJob] = []
    if run["status"] in interesting:
        jobs = gh_get(
            f"{CONFIG.github.url}/repos/{repo}/actions/runs/{run['id']}/jobs",
            "jobs",
        )
        for job in jobs:
            if job["status"] in interesting:
                ret.append(GithubJob(repo, run, job))
    return ret


def get_gh_repo_jobs(repo: GithubRepo) -> Iterable[GithubJob]:
    runs = flatten(
        parallelmap(
            lambda status: gh_get(
                f"{CONFIG.github.url}/repos/{repo}/actions/runs?status={status}",
                "workflow_runs",
            ),
            ["in_progress", "queued"],
        )
    )
    return flatten(parallelmap(lambda x: get_gh_run_jobs(repo, x), runs))


def get_gh_state(repos: Set[GithubRepo]) -> list[GithubJob]:
    reqstate: list[GithubJob] = list(flatten(parallelmap(get_gh_repo_jobs, repos)))
    # desc: str = ", ".join(s.labelsstr() + " for " + s.job_url() for s in reqstate)
    # log.info(f"Found {len(reqstate)} required runners to run: {desc}")
    for idx, s in enumerate(reqstate):
        logging.info(f"GHJOB: {idx} {s.job_url()} {s.labelsstr()} {s.run['status']}")
    return reqstate


###############################################################################
# nomad


class NomadJobCommon(DataDict):
    def labelsstr(self) -> str:
        return self["Meta"][CONFIG.nomad.meta + "_LABELS"]

    def repo(self) -> str:
        return self["Meta"][CONFIG.nomad.meta + "_REPO"]

    def repo_url(self) -> str:
        return self["Meta"][CONFIG.nomad.meta + "_REPO_URL"]

    def job_url(self) -> str:
        return self["Meta"][CONFIG.nomad.meta + "_JOB_URL"]


class NomadJob(nomadlib.Job, NomadJobCommon):
    """/v1/job/ID return value of the runner"""

    pass


@dataclass
class GithubRunnerState:
    inactive_since: Optional[datetime.datetime] = None


class NomadRunner(nomadlib.JobsJob, NomadJobCommon):
    """/v1/jobs return value of the runner"""

    def tostr(self):
        return f"{self.ID} {self.repo_url()} {self.Status}"

    def get_github_runner_state(self) -> Optional[GithubRunnerState]:
        """Check if this runner is _right now_
        executing a github job. This is best efforted
        by parsing stdout logs of the runner"""
        if not self.is_running():
            return None
        COUNTERS.nomad_get.inc()
        allocs = [
            nomadlib.Alloc(x)
            for x in mynomad.get(
                f"job/{urlquote(self.ID)}/allocations",
                params=dict(
                    namespace=CONFIG.nomad.namespace,
                    filter='ClientStats == "running"',
                ),
            )
        ]
        alloc = next((alloc for alloc in allocs if alloc.is_running_started()), None)
        if not alloc:
            return None
        COUNTERS.nomad_get.inc()
        logs = mynomad.request(
            "GET",
            f"/client/fs/logs/{alloc.ID}",
            params=dict(
                namespace=CONFIG.nomad.namespace,
                task=alloc.get_tasknames()[0],
                type="stdout",
                origin="end",
                plain=True,
            ),
        ).text
        """
        2024-07-14 12:14:17Z: Listening for Jobs
        2024-07-14 12:14:20Z: Running job: build
        2024-07-14 12:14:29Z: Job build completed with result: Succeeded
        See: https://github.com/actions/runner/blob/main/src/Runner.Listener/JobDispatcher.cs
        curl -sS 'https://raw.githubusercontent.com/actions/runner/main/src/Runner.Listener/JobDispatcher.cs' | grep term.WriteLine
        """
        for line in reversed(logs.strip().splitlines()):
            parts = line.split(": ", 1)
            if len(parts) == 2:
                try:
                    time = datetime.datetime.fromisoformat(parts[0]).astimezone()
                except ValueError as e:
                    log.debug(f"{line} -> {e}")
                    continue
                msg = parts[1]
                if (
                    "Listening for jobs".lower() in msg.lower()
                    or "completed with result".lower() in msg.lower()
                ):
                    return GithubRunnerState(time)
                if "Running job".lower() in msg.lower():
                    # is not inactive - return now
                    return GithubRunnerState()
        return None


def get_nomad_state() -> list[NomadRunner]:
    COUNTERS.nomad_get.inc()
    jobsjobs = mynomad.get(
        "jobs",
        params=dict(
            prefix=CONFIG.nomad.jobprefix + "-",
            meta=True,
            namespace=CONFIG.nomad.namespace,
        ),
    )
    #
    curstate: list[NomadRunner] = []
    for jobsjob in jobsjobs:
        nj = NomadRunner(jobsjob)
        try:
            # Get only valid jobsjobs
            nj.repo_url()
            nj.labelsstr()
        except (KeyError, AttributeError):
            continue
        curstate.append(nj)
    #
    # log.info(f"Found {len(curstate)} runners:")
    for i, s in enumerate(curstate):
        log.info(f"RUNNER: {i} {s.tostr()}")
    return curstate


###############################################################################
# runners


@dataclass
class Runners:
    dict: dict[re.Pattern, jinja2.Template] = field(default_factory=dict)
    env: jinja2.Environment = field(
        default_factory=lambda: jinja2.Environment(loader=jinja2.BaseLoader())
    )

    @staticmethod
    def load(specs: dict[str, str]) -> Runners:
        ret = Runners()
        for k, v in specs.items():
            ret.dict[re.compile(k)] = ret.env.from_string(v)
        return ret

    def find(self, labelsstr: str) -> Optional[jinja2.Template]:
        template = next((v for k, v in self.dict.items() if k.match(labelsstr)), None)
        if not template:
            return None
        return template


RUNNERS: Runners


@dataclass
class TemplateContext:
    JOB_NAME: str
    ACCESS_TOKEN: str
    REPO: str
    REPO_URL: str
    JOB_URL: str
    LABELS: str

    @staticmethod
    def make_example(labelsstr: str):
        return TemplateContext(
            JOB_NAME="Example_job_name",
            ACCESS_TOKEN="example_access_token",
            REPO="user/repo",
            REPO_URL="http://example.repo.url/user/repo",
            JOB_URL="http://example.job.url/user/repo/run/12312/job/1231",
            LABELS=labelsstr,
        )

    @staticmethod
    def make_from_github_job(gj: GithubJob):
        return TemplateContext(
            JOB_NAME=re.sub(
                r"[^a-zA-Z0-9_.-]",
                "",
                "-".join(
                    str(x)
                    for x in [
                        CONFIG.nomad.jobprefix,
                        COUNTERS.cnt.inc(),
                        gj.repo,
                        gj.labelsstr(),
                    ]
                ),
            )[:64],
            ACCESS_TOKEN=CONFIG.github.token or "",
            REPO=gj.repo,
            REPO_URL=gj.repo_url(),
            JOB_URL=gj.job_url(),
            LABELS=gj.labelsstr(),
        )

    def to_template_args(self) -> dict:
        arg = {}
        for label in self.LABELS.split(","):
            for part in shlex.split(label):
                split = part.split("=", 1)
                arg[split[0]] = split[1] if len(split) == 2 else ""
        return dict(
            run=asdict(self),
            arg=arg,
            opts=CONFIG.opts,
            CONFIG=CONFIG,
            RUNNERS=RUNNERS,
            ARGS=ARGS,
        )


def get_runner_jobspec(gj: GithubJob) -> Optional[NomadJob]:
    template = RUNNERS.find(gj.labelsstr())
    if not template:
        return None
    tc = TemplateContext.make_from_github_job(gj)
    jobtext = template.render(tc.to_template_args())
    if not jobtext:
        return None
    #
    try:
        jobspec = json.loads(jobtext)
    except json.JSONDecodeError:
        try:
            jobspec = json.loads(
                subprocess.check_output(
                    "nomad job run -output -".split(), input=jobtext, text=True
                )
            )
        except subprocess.CalledProcessError:
            log.exception(f"Could not decode template for {gj.labelsstr()}: {template}")
            raise
    jobspec = jobspec.get("Job", jobspec)
    # Apply default transformations.
    for key in ["ID", "Name"]:
        if key in jobspec:
            # Github runner name can be max 64 characters and no special chars.
            # Still try to be as verbose as possible.
            # The hash generated from job_url should be unique enough.
            jobspec[key] = tc.JOB_NAME
    jobspec["Namespace"] = CONFIG.nomad.namespace
    jobspec["Meta"] = {
        **(jobspec.get("Meta", {}) or {}),
        **{
            CONFIG.nomad.meta + "_" + k: str(v)
            for k, v in asdict(tc).items()
            if k.lower() != "token"
        },
    }
    jobspec = NomadJob(jobspec)
    return jobspec


###############################################################################
# scheduler


@dataclass
class Todo:
    """Represents actions to take"""

    tostart: List[NomadJob] = field(default_factory=list)
    tostop: List[str] = field(default_factory=list)
    topurge: List[str] = field(default_factory=list)

    def __add__(self, o: Todo):
        return Todo(
            self.tostart + o.tostart,
            self.tostop + o.tostop,
            self.topurge + o.topurge,
        )

    def execute(self):
        # Sanity checks.
        tostartids: List[str] = [x["ID"] for x in self.tostart]
        ID: str
        for ID in tostartids:
            assert ID not in self.tostop, f"{ID} is tostart and tostop"
            assert ID not in self.topurge, f"{ID} is tostart and topurge"
        for ID in self.tostop:
            assert ID not in self.topurge, f"{ID} is tostrop and topurge"
        for ID, cnt in collections.Counter(tostartids).items():
            assert cnt == 1, f"Repeated id in tostart: {ID}"
        for ID, cnt in collections.Counter(self.tostop).items():
            assert cnt == 1, f"Repeated id in tostop: {ID}"
        for ID, cnt in collections.Counter(self.topurge).items():
            assert cnt == 1, f"Repeated id in tostop: {ID}"
        # Print them.
        if self.tostart:
            log.info(f"tostart: {' '.join(tostartids)}")
        if self.tostop:
            log.info(f"tostop: {' '.join(self.tostop)}")
        if self.topurge:
            log.info(f"topurge: {' '.join(self.topurge)}")
        # Execute them.
        if ARGS.dryrun:
            log.error("DRYRUN")
        else:
            for spec in self.tostart:
                COUNTERS.nomad_run.inc()
                try:
                    resp = mynomad.start_job(spec.asdict())
                except Exception:
                    log.exception(f"Could not start: {json.dumps(spec)}")
                    raise
                log.info(resp)
            for name in self.tostop:
                COUNTERS.nomad_stop.inc()
                resp = mynomad.stop_job(name)
                log.info(resp)
            if CONFIG.nomad.purge:
                for name in self.topurge:
                    COUNTERS.nomad_purge.inc()
                    resp = mynomad.stop_job(name, purge=True)
                    log.info(resp)


@dataclass(frozen=True)
class RepoState:
    """Represents state of one repository.
    Currently this program supports only github-runners bound to one repo"""

    repo_url: str
    githubjobs: List[GithubJob] = field(default_factory=list)
    nomadrunners: List[NomadRunner] = field(default_factory=list)

    def validate(self):
        # Check that all data are for one repo.
        for gj in self.githubjobs:
            assert gj.repo_url() == self.repo_url
        for nr in self.nomadrunners:
            assert nr.repo_url() == self.repo_url
        for k, v in collections.Counter(nr.ID for nr in self.nomadrunners).items():
            assert v == 1, f"Repeated runner: {k}"

    def __find_free_githubjob(self, labelsstr: str, todo: Todo) -> GithubJob:
        used_job_urls = set(
            itertools.chain(
                (
                    (x.job_url() for x in todo.tostart if x.labelsstr() == labelsstr),
                    (
                        nr.job_url()
                        for nr in self.nomadrunners
                        if nr.labelsstr() == labelsstr
                    ),
                )
            )
        )
        gjobs = [x for x in self.githubjobs if x.labelsstr() == labelsstr]
        picked = next((x for x in gjobs if x.job_url() not in used_job_urls), None)
        if picked:
            return picked
        return random.choice(tuple(gjobs))

    def gen_runners_to_stop(self, labelsstr: str) -> Generator[NomadRunner]:
        """Generate runners to stop."""
        # To stop runner it can't be alredy stopped
        notdeadnotstop = [
            nr
            for nr in self.nomadrunners
            if nr.labelsstr() == labelsstr and not nr.is_dead() and not nr.Stop
        ]
        # First stop pending ones.
        for nr in notdeadnotstop:
            if nr.is_pending():
                yield nr
        # Then stop ones inactive for longer thatn the configured timeout.
        # Getting runner state is costly.
        timeout = CONFIG.get_runner_inactivity_timeout()
        if timeout:
            for nr in notdeadnotstop:
                if nr.is_running():
                    rstate = nr.get_github_runner_state()
                    if rstate and rstate.inactive_since:
                        inactivefor = (
                            datetime.datetime.now().astimezone() - rstate.inactive_since
                        )
                        if inactivefor > timeout:
                            log.warning(
                                f"INACTIVE: Runner is inactive for {inactivefor} > {CONFIG.runner_inactivity_timeout}: {nr.tostr()}"
                            )
                            yield nr
                        else:
                            log.info(
                                f"INACTIVE: Runner continues inactive for {inactivefor} < {CONFIG.runner_inactivity_timeout}: {nr.tostr()}"
                            )

    def scheduler(self) -> Todo:
        """Decide which jobs to run or stop"""
        self.validate()
        todo = Todo()
        neededlabelsstrs: dict[str, int] = collections.Counter(
            gj.labelsstr() for gj in self.githubjobs
        )
        runninglabelsstrs: dict[str, int] = collections.Counter(
            nr.labelsstr() for nr in self.nomadrunners if not nr.is_dead()
        )
        alllabelsstrs: set[str] = set(
            itertools.chain(neededlabelsstrs.keys(), runninglabelsstrs.keys())
        )
        for labelsstr in alllabelsstrs:
            needed = neededlabelsstrs.get(labelsstr, 0)
            running = runninglabelsstrs.get(labelsstr, 0)
            diff = needed - running
            if diff > 0:
                for _ in range(diff):
                    jobspec = get_runner_jobspec(
                        self.__find_free_githubjob(labelsstr, todo)
                    )
                    if jobspec is not None:
                        log.info(
                            f"Running job {jobspec.ID} with {jobspec.labelsstr()} for {jobspec.job_url()}"
                        )
                        log.debug(f"Running {jobspec}")
                        todo.tostart.append(jobspec)
                    else:
                        log.error(
                            f"Runner for {labelsstr} in {self.repo_url} not found"
                        )
            elif diff < 0:
                # Only remove those ones that do not run anything.
                # Generator to evaluate only those that needed.
                # Diff id negative.
                todo.tostop.extend(
                    x.ID
                    for x in itertools.islice(
                        self.gen_runners_to_stop(labelsstr), -diff
                    )
                )
        #
        if CONFIG.nomad.purge:
            for nr in self.nomadrunners:
                if nr.is_dead() and nr.JobSummary.get_sum_summary().only_completed():
                    # Check that there are no running evaluations to be sure.
                    evals = mynomad.get(f"job/{nr.ID}/evaluations")
                    if len(evals) == 0 or all(
                        eval["Status"] == "complete" for eval in evals
                    ):
                        log.info(
                            f"Purging {nr.ID} for {nr.repo_url()} with no running deployments"
                        )
                        todo.topurge.append(nr.ID)
        #
        return todo


def loop():
    """The main loop of this program"""
    # Get the github state.
    repos = get_gh_repos()
    reqstate: list[GithubJob] = get_gh_state(repos)
    # Get the nomad state.
    curstate: list[NomadRunner] = get_nomad_state()
    # Construct current state with repo as the key.
    repostates: dict[str, RepoState] = {}
    for gj in reqstate:
        repostates.setdefault(
            gj.repo_url(), RepoState(gj.repo_url())
        ).githubjobs.append(gj)
    for nj in curstate:
        repostates.setdefault(
            nj.repo_url(), RepoState(nj.repo_url())
        ).nomadrunners.append(nj)
    # For each repo, run the scheduler and collect results.
    todo: Todo = functools.reduce(
        Todo.__add__, parallelmap(RepoState.scheduler, repostates.values()), Todo()
    )
    # Apply limits
    for ll in CONFIG.limits:
        if ll.max >= 0:
            count = 0
            tmp = []
            for tostart in todo.tostart:
                if re.match(ll.repo, tostart.repo()) and re.match(
                    ll.labels, tostart.labelsstr()
                ):
                    count += 1
                    if count <= ll.max:
                        tmp.append(tostart)
                else:
                    tmp.append(tostart)
            todo.tostart = tmp
    # Execute what we need to execute.
    todo.execute()
    # Ending.
    GITHUB_CACHE.save()
    COUNTERS.print()


###############################################################################
# command line


@dataclass
class Args:
    dryrun: bool = clickdc.option("-n")
    verbose: bool = clickdc.option("-v")
    noparallel: bool = clickdc.option()
    config: str = clickdc.option(
        "-c",
        shell_complete=click.Path(
            exists=True, dir_okay=False, path_type=Path
        ).shell_complete,
        help="""
            If the arguments contains a newline, the configuration in YAML format.
            Otherwise the configuration file location with is read as a YAML.
            """,
    )


ARGS: Args


@click.group("githubrunner")
@clickdc.adddc("args", Args)
def cli(args: Args):
    global ARGS
    ARGS = args
    logging.basicConfig(
        format="%(asctime)s:githubrunner:%(lineno)04d: %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    #
    if "\n" in args.config:
        configstr = args.config
    else:
        with open(args.config) as f:
            configstr = f.read()
    global CONFIG
    tmp = yaml.safe_load(configstr)
    CONFIG = Config(tmp)
    # Load runners
    global RUNNERS
    RUNNERS = Runners.load(CONFIG.runners)
    #
    mynomad.namespace = os.environ["NOMAD_NAMESPACE"] = CONFIG.nomad.namespace
    #
    global GITHUB_CACHE
    GITHUB_CACHE = GithubCache.load()


@cli.command(
    help="""Stop all nomad containers associated with this runner confguration.
             Use with care."""
)
def stopall():
    for x in get_nomad_state():
        log.error(f"Stopping {x.ID}")
        if not ARGS.dryrun:
            mynomad.stop_job(x.ID)


@cli.command(
    help="""Purge all nomad containers associated with this runner confguration.
             Use with care."""
)
def purgeall():
    for x in get_nomad_state():
        log.error(f"Purging {x.ID}")
        if not ARGS.dryrun:
            mynomad.stop_job(x.ID, purge=True)


@cli.command(
    help="""Purge all nomad containers associated with this runner confguration that are dead.
             Use with care."""
)
def purgealldead():
    for x in get_nomad_state():
        if x.is_dead():
            log.error(f"Purging {x.ID}")
            if not ARGS.dryrun:
                mynomad.stop_job(x.ID, purge=True)


@cli.command(help="Dump the configuration")
def dumpconfig():
    print(yaml.safe_dump(CONFIG.asdict()))


@cli.command(help="List configured runners")
def listrunners():
    for rgx, tmpl in RUNNERS.dict.items():
        print(f"{rgx}")
        print(f"{tmpl}")
        print()


@cli.command(
    help="Prints out github cache. If given argument URL, print the cached response"
)
@click.argument("url", required=False)
def showgithubcache(url: str):
    if not url:
        for url, val in GITHUB_CACHE.data.items():
            print(f"{url}")
            print(
                f"  isetag={val.is_etag} {val.etag_or_last_modified} {hash(str(val.response))}"
            )
    else:
        print(json.dumps(GITHUB_CACHE.data[url].response))


@cli.command(help="Execute the loop once")
def once():
    loop()


@cli.command()
@click.argument("label", required=True, nargs=-1)
def rendertemplate(label: Tuple[str, ...]):
    labelsstr: str = ",".join(sorted(list(label)))
    template = RUNNERS.find(labelsstr)
    if not template:
        raise Exception(f"Template not found for {labelsstr}")
    res = template.render(TemplateContext.make_example(labelsstr).to_template_args())
    print(res)


@cli.command(help="Main entrypoint - run the loop periodically")
def run():
    while True:
        loop()
        log.info(f"Sleeping for {CONFIG.loop} seconds")
        time.sleep(CONFIG.loop)


if __name__ == "__main__":
    cli()
