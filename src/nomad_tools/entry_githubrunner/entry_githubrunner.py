#!/usr/bin/env python3
# vim: foldmethod=marker
from __future__ import annotations

import concurrent.futures
import datetime
import enum
import functools
import json
import logging
import os
import random
import re
import shlex
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from http.client import RemoteDisconnected
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import click
import clickdc
import dateutil.parser
import jinja2
import pydantic
import requests
import yaml
from typing_extensions import override

from .. import common_click, nomadlib
from ..aliasedgroup import AliasedGroup
from ..common import get_package_file, mynomad
from ..common_base import cached_property, dict_remove_none
from ..common_click import help_h_option
from ..mytabulate import mytabulate
from ..nomadlib.connection import urlquote

log = logging.getLogger(__name__)

###############################################################################
# {{{1 helpers

T = TypeVar("T")
R = TypeVar("R")


def parallelmap(func: Callable[[T], R], arr: Iterable[T]) -> Iterable[R]:
    """Execute lambda over array in parellel and return an array of it"""
    if ARGS.parallel == 1:
        return map(func, arr)
    else:
        nproc = ARGS.parallel if ARGS.parallel > 0 else None
        with concurrent.futures.ThreadPoolExecutor(nproc) as p:
            futures = [p.submit(func, x) for x in arr]
            return (x.result() for x in concurrent.futures.as_completed(futures))
            # return list(p.map(func, arr))


def flatten(xss: Iterable[Iterable[T]]) -> Iterable[T]:
    return (x for xs in xss for x in xs)


def list_split(
    condition: Callable[[T], bool], data: List[T]
) -> Tuple[List[T], List[T]]:
    ret: Tuple[List[T], List[T]] = ([], [])
    for v in data:
        ret[condition(v)].append(v)
    return ret


def list_group_by(data: List[T], key: Callable[[T], R]) -> Dict[R, List[T]]:
    ret: Dict[R, List[T]] = {}
    for v in data:
        ret.setdefault(key(v), []).append(v)
    return ret


PARSE_TIME_REGEX = re.compile(
    r"((?P<weeks>\d+?)w)?"
    r"((?P<days>\d+?)d)?"
    r"((?P<hours>\d+?)h)?"
    r"((?P<minutes>\d+?)m)?"
    r"((?P<seconds>\d+?)s)?"
)


def parse_time(time_str: str) -> datetime.timedelta:
    # https://stackoverflow.com/a/4628148/9072753
    parts = PARSE_TIME_REGEX.match(time_str)
    assert (
        parts
    ), f"{time_str} is not a valid time interval and does not match {PARSE_TIME_REGEX}"
    parts = parts.groupdict()
    time_params = {}
    for name, param in parts.items():
        if param:
            time_params[name] = int(param)
    return datetime.timedelta(**time_params)


class WrapParseTime:
    def __init__(self, txt: str):
        self.v: datetime.timedelta = parse_time(txt)


class WrapRePattern:
    def __init__(self, txt: str):
        self.v = re.compile(txt)


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


def github_repo_from_url(repo: str):
    user, repo = repo.split("/")[3:5]
    return user.capitalize() + "/" + repo.capitalize()


def labels_to_kv(labels: Iterable[str]) -> Dict[str, str]:
    arg = {}
    for label in labels:
        for part in shlex.split(label):
            split = part.split("=", 1)
            arg[split[0]] = split[1] if len(split) == 2 else ""
    return arg


def nomad_job_text_to_json(jobtext: str) -> dict:
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
            log.exception(f"Could not decode Nomad job to json:\n{jobtext}")
            raise
    jobspec = jobspec.get("Job", jobspec)
    jobspec = dict_remove_none(jobspec)
    return jobspec


###############################################################################
# {{{1 config


class NomadConfig(pydantic.BaseModel, frozen=True, extra=pydantic.Extra.forbid):
    """Nomad configuration"""

    namespace: str = "github"
    """The namespace to set on the job."""
    jobprefix: str = "NTGithubRunner"
    """The string to prefix run jobs with.
    Preferably something short, there is a limit on Github runner name."""
    meta: str = "NOMADTOOLS"
    """The metadata where to put important arguments"""

    def __post_init__(self):
        assert self.namespace
        assert (
            self.namespace != "*"
        ), "NOMAD_NAMESPACE is set to '*', something is wrong"
        assert self.jobprefix
        assert self.meta


class GithubConfig(pydantic.BaseModel, frozen=True, extra=pydantic.Extra.forbid):
    url: str = "https://api.github.com"
    """The url to github api"""
    token: Optional[str] = os.environ.get("GH_TOKEN", os.environ.get("GITHUB_TOKEN"))
    """The token to access github api"""
    access_token: Optional[str] = None
    """The github access token consul-template template code."""
    cachefile: str = "~/.cache/nomadtools/githubcache.json"
    """The location of cachefile. os.path.frozen=True, expanduser is used to expand."""


class SchedulerConfig(pydantic.BaseModel, frozen=True, extra=pydantic.Extra.forbid):
    runner_inactivity_timeout: str = ""
    """How much time a runner will be inactive for it to be removed?"""
    purge_successfull_timeout: str = "10m"
    """Purge dead successfull jobs after this time"""
    purge_failure_timeout: str = "1w"
    """Purge dead problematic jobs after this time"""
    max_runners: int = 0
    """The maximum number of runners"""
    pending_to_kill_listening: str = "1m"
    """If a Nomad job is pending for this time, start killing other jobs
    in listening state to make room for it"""
    strict: bool = False
    """Remove a runner once it is unneeded. It's better to run the runner in ephemeral mode"""


class Config(pydantic.BaseModel, frozen=True, extra=pydantic.Extra.forbid):
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

    runson_match: str = "nomadtools *(.*)"
    """ Only execute if github action workflow job runs-on labels matching this regex. """

    loop: int = 60
    """ How many seconds will the loop run. """

    template: str = get_package_file("entry_githubrunner/default.nomad.hcl")
    """Jinja2 template for the runner"""

    template_default_settings: Dict[str, Any] = {
        "extra_config": "",
        "extra_group": "",
        "extra_task": "",
        "extra_job": "",
        "docker": "none",
        "ephemeral": True,
        "run_as_root": False,
        "cache": "",
        "entrypoint": get_package_file("entry_githubrunner/entrypoint.sh"),
    }
    """Default values for SETTINGS template variable"""

    template_settings: Dict[str, Any] = {}
    """ Overwrite settings according to your whim """

    scheduler: SchedulerConfig = SchedulerConfig()

    def __post_init__(self):
        assert self.loop >= 0
        assert self.repos

    def get_template(self):
        try:
            with Path(self.template).open() as f:
                return f.read()
        except OSError:
            return self.template


class ParsedConfig:
    def __init__(self, config: Config):
        self.label_match = re.compile(config.runson_match)
        self.runner_inactivity_timeout = parse_time(
            config.scheduler.runner_inactivity_timeout
        )
        self.purge_successfull_timeout = parse_time(
            config.scheduler.purge_successfull_timeout
        )
        self.purge_failure_timeout = parse_time(config.scheduler.purge_failure_timeout)
        self.pending_to_kill_listening = parse_time(
            config.scheduler.pending_to_kill_listening
        )


###############################################################################
# {{{1 counters


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
# {{{1 abstract description


@pydantic.dataclasses.dataclass(frozen=True)
class Desc:
    labelsstr: str
    repo_url: str
    version: int = 0

    @classmethod
    def mk(cls, labels: Iterable[str], repo_url: str):
        tmp = sorted(list(labels))
        assert not any("," in x for x in tmp), f", in {tmp}"
        return cls(",".join(tmp), repo_url, VERSION)

    def __post_init__(self):
        assert self.labelsstr
        assert self.repo_url
        assert "://" in self.repo_url
        assert self.repo_url.count("/") == 4
        assert any(PARSEDCONFIG.label_match.match(x) for x in self.labels)

    @property
    def labels(self):
        return self.labelsstr.split(",")

    @property
    def repo(self):
        return github_repo_from_url(self.repo_url)

    def tostr(self):
        return f"labels={self.labelsstr} url={self.repo_url}"


class DescProtocol(ABC):
    @abstractmethod
    def get_desc(self) -> Desc: ...

    def try_get_desc(self) -> Optional[Desc]:
        try:
            return self.get_desc()
        except Exception as e:
            log.warning(f"Could not create description from {self}: {e}")
            return None


###############################################################################
# {{{1 github


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
        json: dict
        links: dict
        timestamp: float = field(default_factory=lambda: time.time())

    data: Dict[Url, Value] = field(default_factory=dict)
    """The data stored by cache is keyed with URL"""
    lock: threading.Lock = field(default_factory=threading.Lock)
    """Lock against concurrent execution"""
    VERSION: ClassVar[int] = 2
    """The version of self.data and value class"""

    @classmethod
    def load(cls):
        """Load the GithubCache from file"""
        try:
            with open(github_cachefile()) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.warning(f"Github cache loading error: {e}")
            return GithubCache()
        if data["version"] != GithubCache.VERSION:
            log.warning("Github cache version mismatch, zeroing")
            return GithubCache()
        gh = GithubCache({k: cls.Value(**v) for k, v in data["data"].items()})
        log.info(
            f"Github cache loaded {len(gh.data)} entries datafrom {github_cachefile()}"
        )
        return gh

    def tojson(self):
        return {
            "version": self.VERSION,
            "data": {k: asdict(v) for k, v in self.data.items()},
        }

    def save(self):
        """Save the cache to file. Run each loop"""
        data = self.tojson()
        os.makedirs(os.path.dirname(github_cachefile()), exist_ok=True)
        with open(github_cachefile(), "w") as f:
            json.dump(data, f)
        log.debug(
            f"Github cache saved {len(self.data)} entires to {github_cachefile()}"
        )

    def prepare(self, url: str) -> Dict[str, str]:
        headers = {}
        with self.lock:
            cached = self.data.get(url)
        if cached:
            if cached.is_etag:
                headers["if-none-match"] = cached.etag_or_last_modified
            else:
                headers["if-modified-since"] = cached.etag_or_last_modified
        return headers

    def handle(self, response: requests.Response) -> Optional[Value]:
        with self.lock:
            if response.status_code == 304:
                COUNTERS.github_cached.inc()
                return self.data[response.url]
            else:
                COUNTERS.github_miss.inc()
            etag = response.headers.get("etag")
            if etag:
                self.data[response.url] = self.Value(
                    True, etag, response.json(), response.links
                )
                self.save()
            else:
                last_modified = response.headers.get("last-modified")
                if last_modified:
                    self.data[response.url] = self.Value(
                        False, last_modified, response.json(), response.links
                    )
                    self.save()
        return None


class GithubConnection:
    def __init__(self, config: Config):
        self.s = requests.Session()
        self.s.headers.update({"Accept": "application/vnd.github+json"})
        if config.github.token:
            self.s.headers.update({"Authorization": "Bearer " + config.github.token})

    def get(self, url: str, key: str = ""):
        """Execute query to github

        @param key if set, means the output is paginated
        """
        ret: list[dict] = []
        trynum = 0
        while True:
            # Prepare the request adding headers from Github cache.
            headers = GITHUB_CACHE.prepare(url)
            response = self.s.get(url, headers=headers)
            log.debug(f"{url} {response}")
            try:
                try:
                    response.raise_for_status()
                except RemoteDisconnected:
                    trynum += 1
                    if trynum >= 3:
                        raise
                    continue
            except Exception:
                raise Exception(f"{response.url}\n{response.text}")
            # If the result is found in github cache, use it, otherwise extract it.
            cached: Optional[GithubCache.Value] = GITHUB_CACHE.handle(response)
            data: Any = cached.json if cached else response.json()
            links: dict = cached.links if cached else response.links
            # if there are no links, this is not a paginated url.
            if not links:
                return data[key] if key else data
            # Append the data to ret. Use key if given.
            add = data if isinstance(data, list) else data[key]
            try:
                assert isinstance(add, list)
                ret.extend(add)
            except KeyError:
                log.exception(f"key={key} data={data}")
                raise
            # if no next, this is the end.
            next = links.get("next")
            if not next:
                break
            url = next["url"]
        return ret


def gh_get(url: str, key: str = "") -> Any:
    return GH.get(url, key)


@functools.lru_cache()
@functools.wraps(gh_get)
def gh_get_cached(url: str, key: str = ""):
    return gh_get(url, key)


###############################################################################
# {{{1 github high level

GithubRepo = str


@dataclass(frozen=True)
class GithubJob(DescProtocol):
    repo: GithubRepo
    run: dict
    job: dict

    @property
    def labels(self) -> List[str]:
        return self.job["labels"]

    @override
    def get_desc(self):
        return Desc.mk(self.labels, self.run["repository"]["html_url"])


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
    try:
        runstatus = run["status"]
    except TypeError:
        log.exception(f"AAA {run}")
        raise
    if runstatus in interesting:
        jobs = gh_get(
            f"{CONFIG.github.url}/repos/{repo}/actions/runs/{run['id']}/jobs",
            "jobs",
        )
        for job in jobs:
            if job["status"] in interesting:
                gj = GithubJob(repo, run, job)
                if any("," in x for x in gj.labels):
                    log.warning("Job has comma in labels {desc}. IGNORING")
                else:
                    if gj.try_get_desc():
                        ret.append(gj)

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
    log.debug(f"Found {len(reqstate)} github jobs")
    for idx, s in enumerate(reqstate):
        logging.info(f"GHJOB={idx} {s.get_desc().tostr()} {s.run['status']}")
    return reqstate


###############################################################################
# {{{1 nomad


def get_desc_from_nomadjob(job):
    txt = (job["Meta"] or {})[CONFIG.nomad.meta]
    return Desc(**json.loads(txt))


@dataclass
class NomadJobToRun(DescProtocol):
    nj: nomadlib.Job
    hcl: str

    @override
    def get_desc(self):
        return get_desc_from_nomadjob(self.nj)

    def tostr(self):
        return f"id={self.nj.ID} {self.get_desc().tostr()}"

    def start(self):
        COUNTERS.nomad_run.inc()
        mynomad.start_job(
            self.nj.asdict(),
            nomadlib.JobSubmission.mk_hcl(self.hcl),
        )


class RunnerState(enum.IntEnum):
    r"""
                        /->----------/->--------/-> dead_oom
           /->---------/->----------/->--------/--> dead_stopped
    pending -> starting -> listening -> running --> dead_success
                           ^---------------<-/ \--> dead_failure
    """

    pending = enum.auto()
    starting = enum.auto()
    listening = enum.auto()
    running = enum.auto()
    stopping = enum.auto()
    dead_oom = enum.auto()
    dead_failure = enum.auto()
    dead_stopped = enum.auto()
    dead_success = enum.auto()


@dataclass(frozen=True)
class RunnerStateSince:
    state: RunnerState
    since: Optional[datetime.datetime] = None

    def is_dead(self):
        return self.state in [
            RunnerState.dead_stopped,
            RunnerState.dead_oom,
            RunnerState.dead_failure,
            RunnerState.dead_success,
        ]

    def passed(self):
        assert self.since
        return datetime.datetime.now().astimezone() - self.since

    def cmp(
        self,
        o: Union[RunnerState, List[RunnerState]],
        timeout: Optional[datetime.timedelta] = None,
    ):
        return (
            self.state == o if isinstance(o, RunnerState) else self.state in o
        ) and (timeout is None or self.since is None or self.passed() > timeout)

    def tostr(self):
        return f"state={self.state.name} since={self.since.isoformat() if self.since else None}"


@dataclass
class NomadRunner(DescProtocol):
    nj: nomadlib.JobsJob

    def __post_init__(self):
        assert isinstance(self.nj, nomadlib.JobsJob)

    @property
    def ID(self):
        return self.nj.ID

    def is_dead(self):
        return self.nj.is_dead()

    @override
    def get_desc(self):
        return get_desc_from_nomadjob(self.nj)

    @cached_property
    def state(self) -> RunnerStateSince:
        if self.nj.is_dead():
            allocs = self._allocs
            if allocs:
                alloc = allocs[0]
                if alloc.any_oom_killed():
                    since = self.get_since(allocs)
                    return RunnerStateSince(RunnerState.dead_oom, since)
        if self.nj.Stop:
            since = self.get_since(self._evals)
            if self.nj.is_dead():
                return RunnerStateSince(RunnerState.dead_stopped, since)
            else:
                return RunnerStateSince(RunnerState.stopping, since)
        elif self.nj.is_pending():
            since = self.get_since(self._evals)
            return RunnerStateSince(RunnerState.pending, since)
        elif self.nj.is_running():
            logs = self.get_logs()
            if logs is None:
                since = self.get_since(self._evals)
                return RunnerStateSince(RunnerState.starting, since)
            return self._parse_logs(logs)
        else:  # dead
            since = self.get_since(self._allocs) or self.get_since(self._evals)
            if self.nj.JobSummary.get_sum_summary().only_completed():
                return RunnerStateSince(RunnerState.dead_success, since)
            return RunnerStateSince(RunnerState.dead_failure, since)

    def tostr(self):
        return f"id={self.nj.ID} {self.state.tostr()} {self.get_desc().tostr()}"

    @staticmethod
    def get_since(arr: Union[List[nomadlib.Alloc], List[nomadlib.Eval]]):
        # Get since from evaluations.
        return arr[0].ModifyTime_dt() if arr else None

    @cached_property
    def _evals(self):
        COUNTERS.nomad_get.inc()
        evals = [
            nomadlib.Eval(x)
            for x in mynomad.get(
                f"job/{urlquote(self.nj.ID)}/evaluations",
                params=dict(namespace=self.nj.Namespace),
            )
        ]
        # Newest first
        evals.sort(key=lambda a: -a.ModifyTime)
        return evals

    @cached_property
    def _allocs(self):
        COUNTERS.nomad_get.inc()
        allocs = [
            nomadlib.Alloc(x)
            for x in mynomad.get(
                f"job/{urlquote(self.nj.ID)}/allocations",
                params=dict(namespace=self.nj.Namespace),
            )
        ]
        # Newest first
        allocs.sort(key=lambda a: -a.ModifyTime)
        return allocs

    def get_logs(self, running: bool = True) -> Optional[str]:
        allocs = self._allocs
        allocs = [alloc for alloc in allocs if alloc.any_was_started()]
        if not allocs:
            return None
        alloc = allocs[0]
        COUNTERS.nomad_get.inc()
        try:
            return mynomad.request(
                "GET",
                f"/client/fs/logs/{alloc.ID}",
                params=dict(
                    namespace=self.nj.Namespace,
                    task=alloc.get_tasknames()[0],
                    type="stdout",
                    origin="end",
                    plain=True,
                ),
            ).text
        except nomadlib.LogNotFound:
            return None

    def _parse_logs(self, logs: str) -> RunnerStateSince:
        """Check if this runner is _right now_
        executing a github job. This is best efforted
        by parsing stdout logs of the runner"""
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
                    since = dateutil.parser.parse(parts[0]).astimezone()
                except Exception:
                    # log.exception(f"ERR")
                    continue
                msg = parts[1].lower()
                if "Listening for jobs".lower() in msg:
                    return RunnerStateSince(RunnerState.listening, since)
                if "completed with result".lower() in msg:
                    return RunnerStateSince(RunnerState.listening, since)
                if "Running job".lower() in msg:
                    return RunnerStateSince(RunnerState.running, since)
        return RunnerStateSince(RunnerState.starting, self.get_since(self._allocs))

    def stop(self):
        COUNTERS.nomad_stop.inc()
        mynomad.stop_job(self.nj.ID)

    def purge(self):
        COUNTERS.nomad_purge.inc()
        mynomad.stop_job(self.nj.ID, purge=True)


def get_nomad_state() -> List[NomadRunner]:
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
        nr = NomadRunner(nomadlib.JobsJob(jobsjob))
        if nr.try_get_desc():
            curstate.append(nr)
        else:
            nr.stop()
    #
    log.debug(f"Found {len(curstate)} nomad runners:")
    for i, s in enumerate(curstate):
        log.info(f"RUNNER={i} {s.tostr()}")
    return curstate


###############################################################################
# {{{1 runners


def range_forever():
    i = 1
    while True:
        yield i
        i += 1


class TakenRunnerNames(Set[str]):
    pass


@dataclass
class RunnerGenerator:
    """Stores parsed runners ready to template"""

    desc: Desc
    info: str
    takenrunnernames: TakenRunnerNames

    def _generate_job_name(self):
        # Github runner name can be max 64 characters and no special chars.
        # Still try to be as verbose as possible.
        for i in range_forever():
            parts = [
                CONFIG.nomad.jobprefix,
                self.desc.repo,
                i,
                self.desc.labelsstr.replace("nomadtools", ""),
                self.info,
            ]
            name = re.sub(r"[^a-zA-Z0-9_.-]", "", "-".join(str(x) for x in parts))
            name = name[:64].rstrip("-")
            if name not in self.takenrunnernames:
                self.takenrunnernames.add(name)
                return name

    @staticmethod
    def make_example(labels: List[str]):
        return RunnerGenerator(
            Desc.mk(labels=labels, repo_url="http://example.repo.url/user/repo"),
            "This is an information",
            TakenRunnerNames(),
        )

    def _to_template_args(self) -> dict:
        return dict(
            SETTINGS={
                **CONFIG.template_default_settings,
                **CONFIG.template_settings,
            },
            RUN=dict(
                REPO_URL=self.desc.repo_url,
                RUNNER_NAME=self._generate_job_name(),
                LABELS=self.desc.labelsstr.replace('"', r"\""),
                REPO=self.desc.repo,
                INFO=self.info,
            ),
            RUNSON=labels_to_kv(self.desc.labels),
            CONFIG=CONFIG,
            nomadlib=nomadlib,
        )

    def get_runnertorun(self) -> NomadJobToRun:
        tc: dict = self._to_template_args()
        jobtext: str = TEMPLATE.render(tc)
        jobspec: dict = nomad_job_text_to_json(jobtext)
        # Apply default transformations.
        jobspec["Namespace"] = CONFIG.nomad.namespace
        jobspec.setdefault("Meta", {})[CONFIG.nomad.meta] = json.dumps(
            asdict(self.desc)
        )
        #
        nomadjobtorun = NomadJobToRun(nomadlib.Job(jobspec), jobtext)
        assert (
            nomadjobtorun.get_desc() == self.desc
        ), f"{nomadjobtorun.get_desc()} != {self.desc}"
        return nomadjobtorun


###############################################################################
# {{{1 scheduler


def get_runners_to_trim(runners: List[NomadRunner]) -> List[NomadRunner]:
    """Generate runners to stop."""
    order = [
        RunnerState.pending,
        RunnerState.starting,
        RunnerState.listening,
        RunnerState.running,
    ]
    runners = [runner for runner in runners if runner.state.state in order]
    runners.sort(
        key=lambda nr: (
            order.index(nr.state.state),
            nr.state.since,
            -nr.nj.ModifyIndex,
        )
    )
    return runners


@dataclass
class Scheduler:
    desc: Desc
    ghjobs: List[GithubJob]
    runners: List[NomadRunner]
    takenrunnernames: TakenRunnerNames

    def cleanup_timeouted(self):
        for runner in self.runners:
            if CONFIG.scheduler.runner_inactivity_timeout and runner.state.cmp(
                RunnerState.listening, PARSEDCONFIG.runner_inactivity_timeout
            ):
                log.info(
                    f"Stopping inactive runner {runner.tostr()} over timeout {CONFIG.scheduler.runner_inactivity_timeout}"
                )
                runner.stop()
            if CONFIG.scheduler.purge_successfull_timeout and runner.state.cmp(
                [RunnerState.dead_success, RunnerState.dead_stopped],
                PARSEDCONFIG.purge_successfull_timeout,
            ):
                log.info(
                    f"Purging runner {runner.tostr()} over timeout {CONFIG.scheduler.purge_successfull_timeout}"
                )
                runner.purge()
            if CONFIG.scheduler.purge_failure_timeout and runner.state.cmp(
                [RunnerState.dead_oom, RunnerState.dead_failure],
                PARSEDCONFIG.purge_failure_timeout,
            ):
                log.info(
                    f"Purging runner {runner.tostr()} over timeout {CONFIG.scheduler.purge_failure_timeout}"
                )
                runner.purge()

    def schedule(self) -> List[NomadRunner]:
        """
        Returns the runners that can be stopped without repercusions
        """
        not_dead_runners = [x for x in self.runners if not x.state.is_dead()]
        diff = len(self.ghjobs) - len(not_dead_runners)
        unneededrunners: List[NomadRunner] = []
        #
        if diff > 0:
            # If there are more github jobs associated with this repository and labels, start them.
            torun = diff
            for _ in range(torun):
                if (
                    CONFIG.scheduler.max_runners > 0
                    and len(self.takenrunnernames) >= CONFIG.scheduler.max_runners
                ):
                    log.info(
                        f"The number of runners {len(self.takenrunnernames)} is greater than {CONFIG.scheduler.max_runners}, so ignoring starting runner for {self.desc}"
                    )
                    break
                new_runner = RunnerGenerator(
                    self.desc, "", self.takenrunnernames
                ).get_runnertorun()
                log.info(f"Starting runner {new_runner.tostr()}")
                new_runner.start()
        elif diff < 0:
            # There are runners that are not needed right now - there are less github jobs.
            unneededcnt = -diff
            unneededrunners = get_runners_to_trim(self.runners)[:unneededcnt]
            ret = []
            for runner in unneededrunners:
                # Pending runners can be just stopped right away.
                if (
                    runner.state.cmp(RunnerState.pending)
                    or CONFIG.scheduler.strict
                    or runner.get_desc().version != VERSION
                ):
                    log.info(f"Stopping runner {runner.tostr()}")
                    runner.stop()
                ret.append(runner)
            unneededrunners = ret
        self.cleanup_timeouted()
        return unneededrunners


MergedState = Dict[Desc, Tuple[List[GithubJob], List[NomadRunner]]]


def merge_states(ghjobs: List[GithubJob], runners: List[NomadRunner]) -> MergedState:
    state: MergedState = {}
    for i in ghjobs:
        state.setdefault(i.get_desc(), ([], []))[0].append(i)
    for i in runners:
        state.setdefault(i.get_desc(), ([], []))[1].append(i)
    return state


def make_place_for_pending_scheduler(
    runners: List[NomadRunner], unneededrunners: List[NomadRunner]
):
    # Remove listening runners if there are long pending jobs.
    pending_want_kill = [
        runner
        for runner in runners
        if runner.state.cmp(RunnerState.pending, PARSEDCONFIG.pending_to_kill_listening)
    ]
    if not pending_want_kill:
        return
    pending_want_kill_str = (
        "["
        + ",".join(f"NomadRunner({runner.tostr()})" for runner in pending_want_kill)
        + "]"
    )
    runners_can_die = get_runners_to_trim(unneededrunners)[: len(pending_want_kill)]
    for runner in runners_can_die:
        log.info(
            f"Stopping runner {runner.tostr()} to make room for {pending_want_kill_str} that are waiting for more than {CONFIG.scheduler.pending_to_kill_listening}"
        )
        runner.stop()


def loop():
    """The main loop of this program"""
    # Get the github state.
    repos = get_gh_repos()
    ghjobs: List[GithubJob] = get_gh_state(repos)
    # Get the nomad state.
    runners: List[NomadRunner] = get_nomad_state()
    takenrunnernames = TakenRunnerNames(x.ID for x in runners)
    # Merge states on label and repo.
    unneededrunners: List[NomadRunner] = []
    state: MergedState = merge_states(ghjobs, runners)
    for k, v in state.items():
        unneededrunners += Scheduler(k, v[0], v[1], takenrunnernames).schedule()
    if CONFIG.scheduler.pending_to_kill_listening:
        make_place_for_pending_scheduler(runners, unneededrunners)
    # Ending.
    COUNTERS.print()


###############################################################################
# {{{1 command line


@dataclass
class Args:
    dryrun: bool = clickdc.option("-n")
    parallel: int = clickdc.option("-P", default=4)
    config: Optional[str] = clickdc.option(
        "-c",
        shell_complete=click.Path(
            exists=True, dir_okay=False, path_type=Path
        ).shell_complete,
        help="""
            If the arguments contains a newline, the configuration in YAML format.
            Otherwise the configuration file location with is read as a YAML.
            """,
    )


@click.command(
    "githubrunner",
    cls=AliasedGroup,
    help="""
        Execute Nomad job to run github self-hosted runner for user repositories.

        See documentation in doc/githubrunner.md on github.
        """,
)
@clickdc.adddc("args", Args)
@help_h_option()
@common_click.quiet_option()
@common_click.verbose_option()
def cli(args: Args):
    global ARGS
    ARGS = args
    logging.root.level -= 10
    logging.basicConfig(
        format="%(asctime)s:githubrunner:%(lineno)04d: %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    #
    global CONFIG
    if args.config:
        if "\n" in args.config:
            configstr = args.config
        else:
            with open(args.config) as f:
                configstr = f.read()
        tmp = yaml.safe_load(configstr)
        CONFIG = Config(**tmp)
    else:
        CONFIG = Config()
    #
    mynomad.namespace = os.environ["NOMAD_NAMESPACE"] = CONFIG.nomad.namespace
    #
    global VERSION
    VERSION = hash(json.dumps(CONFIG.dict()))
    global PARSEDCONFIG
    PARSEDCONFIG = ParsedConfig(CONFIG)
    global GH
    GH = GithubConnection(CONFIG)
    #
    global TEMPLATE
    TEMPLATE = jinja2.Environment(
        loader=jinja2.BaseLoader(), lstrip_blocks=True, trim_blocks=True
    ).from_string(CONFIG.get_template())
    global GITHUB_CACHE
    GITHUB_CACHE = GithubCache.load()


@cli.command(
    help="""Stop all nomad containers associated with this runner confguration.
             Use with care."""
)
def stopallrunners():
    for x in get_nomad_state():
        log.warning(f"Stopping {x.ID}")
        if not ARGS.dryrun:
            x.stop()


@cli.command(
    help="""Purge all nomad containers associated with this runner confguration.
             Use with care."""
)
def purgeallrunners():
    for x in get_nomad_state():
        log.warning(f"Purging {x.ID}")
        if not ARGS.dryrun:
            x.purge()


@cli.command(
    help="""Purge all nomad containers associated with this runner confguration that are dead.
             Use with care."""
)
def purgedeadrunners():
    for x in get_nomad_state():
        if x.is_dead():
            log.warning(f"Purging {x.ID}")
            if not ARGS.dryrun:
                x.purge()


@cli.command(help="Dump the configuration")
@click.option("usejson", "-j", "--json", is_flag=True)
def dumpconfig(usejson: bool):
    tmp = CONFIG.dict()
    if usejson:
        print(json.dumps(tmp))
    else:
        print(yaml.safe_dump(tmp))


@cli.command(
    help="Prints out github cache. If given argument URL, print the cached response"
)
@click.argument("url", required=False)
def showgithubcache(url: str):
    if not url:
        for url, val in GITHUB_CACHE.data.items():
            print(f"{url}")
            print(
                f"  isetag={val.is_etag} {val.etag_or_last_modified} {hash(str(val.json))}"
            )
    else:
        print(json.dumps(asdict(GITHUB_CACHE.data[url])))


@cli.command(help="Execute the loop once")
def once():
    loop()


@cli.command(
    help="""
    Render the template from configuration given the labels given on command line.
    Usefull for testing the job.
    """
)
@click.argument("labels", required=True, nargs=-1)
def rendertemplate(labels: Tuple[str, ...]):
    res = RunnerGenerator.make_example(list(labels)).get_runnertorun()
    print(res.hcl)


@cli.command(help="Main entrypoint - run the loop periodically")
def run():
    while True:
        loop()
        log.info(f"Sleeping for {CONFIG.loop} seconds")
        time.sleep(CONFIG.loop)


@cli.command(help="List cache directories used by runners")
def listrunners():
    data: List[List[str]] = []
    for x in sorted(get_nomad_state(), key=lambda x: x.state.state):
        logs = x.get_logs(False)
        if logs is None:
            dir = "no_logs"
        elif not logs:
            dir = "empty_logs"
        else:
            m = re.search("Using ([^\n]*) as cache directory", logs)
            if m:
                dir = m[1]
            else:
                dir = "no_match"
        data.append(
            [
                dir,
                x.ID,
                x.state.state.name,
                x.state.since.isoformat() if x.state.since else "None",
                x.get_desc().labelsstr,
                x.get_desc().repo,
            ]
        )
    log.info("DONE")
    data.sort(key=lambda x: (RunnerState[x[2]].value, x[0], x[1]))
    data.insert(0, ["dir", "ID", "state", "since", "labels", "repo"])
    print(mytabulate(data, True))


if __name__ == "__main__":
    cli()
