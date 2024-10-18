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
import requests
import yaml

from nomad_tools.nomadlib.types import MyStrEnum

from .. import nomadlib
from ..aliasedgroup import AliasedGroup
from ..common import get_package_file, mynomad
from ..common_base import cached_property
from ..common_click import help_h_option
from ..nomadlib.connection import urlquote
from ..nomadlib.datadict import DataDict

log = logging.getLogger(__name__)

###############################################################################
# {{{1 config


class NomadConfig(DataDict):
    """Nomad configuration"""

    namespace: str = os.environ.get("NOMAD_NAMESPACE", "default")
    """The namespace to set on the job."""
    token: Optional[str] = os.environ.get("NOMAD_TOKEN")
    jobprefix: str = "NTGithubRunner"
    """The string to prefix run jobs with.
    Preferably something short, there is a limit on Github runner name."""
    meta: str = "NOMADTOOLS"
    """The metadata where to put important arguments"""
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

    loop: int = 60
    """How many seconds will the loop run"""

    template: str = get_package_file("entry_githubrunner/default.nomad.hcl")
    """Jinja2 template for the runner"""

    label_match: str = "nomadtools (.*)"

    runner_inactivity_timeout: str = "5m"
    """How much time a runner will be inactive for it to be removed?"""

    default_template_context: Any = {
        "extra_config": "",
        "extra_group": "",
        "extra_task": "",
        "extra_job": "",
        "ephemeral": "true",
        "startscript": get_package_file("entry_githubrunner/startscript.sh"),
    }
    """Additional template variables"""

    template_context: Any = {}
    """Additional template variables"""

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


def github_repo_from_url(repo: str):
    return "/".join(repo.split("/")[3:4])


def labelstr_to_kv(labelstr: str) -> Dict[str, str]:
    arg = {}
    for part in shlex.split(labelstr):
        split = part.split("=", 1)
        arg[split[0]] = split[1] if len(split) == 2 else ""
    return arg


def nomad_job_text_to_json(jobtext: str):
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
    return jobspec


###############################################################################
# {{{1 abstract description


@dataclass
class Desc:
    labels: str
    repo_url: str

    def __post_init__(self):
        assert self.labels
        assert self.repo_url
        assert "://" in self.repo_url
        assert re.match(CONFIG.label_match, self.labels)

    @property
    def repo(self):
        return github_repo_from_url(self.repo_url)

    def tostr(self):
        return f"labels={self.labels} url={self.repo_url}"


class DescProtocol(ABC):
    @abstractmethod
    def get_desc(self) -> Desc: ...

    def try_get_desc(self) -> Optional[Desc]:
        try:
            return self.get_desc()
        except Exception:
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

    def prepare(self, url: str, headers: Dict[str, str]):
        with self.lock:
            cached = self.data.get(url)
        if cached:
            if cached.is_etag:
                headers["if-none-match"] = cached.etag_or_last_modified
            else:
                headers["if-modified-since"] = cached.etag_or_last_modified

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
        headerslog = {**headers, "Authorization": "***"}
        log.debug(f"{url} {headerslog} {response}")
        try:
            response.raise_for_status()
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

    def labelsstr(self):
        return "\n".join(sorted(list(set(self.job["labels"]))))

    def job_url(self):
        return self.job["html_url"]

    def repo_url(self):
        return self.run["repository"]["html_url"]

    def get_desc(self):
        return Desc(self.labelsstr(), self.repo_url())


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
    # desc: str = ", ".join(s.labelsstr() + " for " + s.job_url() for s in reqstate)
    # log.info(f"Found {len(reqstate)} required runners to run: {desc}")
    for idx, s in enumerate(reqstate):
        logging.info(f"GHJOB={idx} {s.job_url()} {s.labelsstr()} {s.run['status']}")
    return reqstate


###############################################################################
# {{{1 nomad


def get_desc_from_nomadjob(job):
    return Desc(**json.loads((job["Meta"] or {})[CONFIG.nomad.meta]))


@dataclass
class NomadJobToRun:
    nj: nomadlib.Job
    hcl: str

    def get_desc(self):
        return get_desc_from_nomadjob(self.nj)

    def tostr(self):
        return f"id={self.nj.ID} {self.get_desc().tostr()}"


@dataclass
class GithubRunnerState:
    inactive_since: Optional[datetime.datetime] = None


class RunnerState(MyStrEnum):
    pending = enum.auto()
    listening = enum.auto()
    running = enum.auto()
    unknown = enum.auto()
    dead_success = enum.auto()
    dead_failure = enum.auto()


@dataclass(frozen=True)
class RunnerStateSince:
    state: RunnerState
    since: datetime.datetime = datetime.datetime.min

    def is_dead(self):
        return self.state in [RunnerState.dead_failure, RunnerState.dead_success]

    def is_pending(self):
        return self.state == RunnerState.pending

    def is_listening(self):
        return self.state == RunnerState.listening

    def is_unknown(self):
        return self.state == RunnerState.unknown

    def passed(self):
        return datetime.datetime.now() - self.since

    def __eq__(self, o: Union[RunnerStateSince, RunnerState]) -> bool:
        if isinstance(o, RunnerState):
            return self.state == o
        else:
            return self.state == o.state and self.since == o.since

    def tostr(self):
        return f"state={self.state} since={self.since.isoformat()}"


@dataclass(frozen=True)
class NomadRunner(DescProtocol):
    nj: nomadlib.JobsJob

    @property
    def ID(self):
        return self.nj.ID

    def is_dead(self):
        return self.nj.is_dead()

    def get_desc(self):
       return get_desc_from_nomadjob(self.nj)

    @cached_property
    def state(self) -> RunnerStateSince:
        if self.nj.is_pending():
            return RunnerStateSince(RunnerState.pending)
        if self.nj.is_dead():
            if self.nj.JobSummary.get_sum_summary().only_completed():
                return RunnerStateSince(RunnerState.dead_success)
            return RunnerStateSince(RunnerState.dead_failure)
        if self.nj.is_running():
            logs = self._get_logs()
            if logs is None:
                return RunnerStateSince(RunnerState.pending)
            return self._parse_logs(logs)
        return RunnerStateSince(RunnerState.unknown)

    def tostr(self):
        return f"id={self.nj.ID} {self.state.tostr()} {self.get_desc().tostr()}"

    @cached_property
    def _allocs(self):
        COUNTERS.nomad_get.inc()
        return [
            nomadlib.Alloc(x)
            for x in mynomad.get(
                f"job/{urlquote(self.nj.ID)}/allocations",
                params=dict(
                    namespace=CONFIG.nomad.namespace,
                    filter='ClientStats == "running"',
                ),
            )
        ]

    def _get_logs(self) -> Optional[str]:
        allocs = self._allocs
        allocs = [alloc for alloc in allocs if alloc.is_running_started()]
        if not allocs:
            return None
        alloc = allocs[0]
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
        return logs

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
                    time = dateutil.parser.parse(parts[0])
                except Exception as e:
                    print(f"ERR {e}")
                    continue
                msg = parts[1]
                if "Listening for jobs".lower() in msg.lower():
                    return RunnerStateSince(RunnerState.listening, time)
                if "completed with result".lower() in msg.lower():
                    return RunnerStateSince(RunnerState.listening, time)
                if "Running job".lower() in msg.lower():
                    # is not inactive - return now
                    return RunnerStateSince(RunnerState.running, time)
        return RunnerStateSince(RunnerState.unknown)


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
        nr = NomadRunner(jobsjob)
        try:
            # Get only valid jobsjobs
            nr.get_desc()
        except (KeyError, AttributeError):
            continue
        curstate.append(nr)
    #
    log.debug(f"Found {len(curstate)} runners:")
    for i, s in enumerate(curstate):
        log.info(f"RUNNER={i} {s.tostr()}")
    return curstate


###############################################################################
# {{{1 runners


@dataclass
class RunnerGenerator:
    """Stores parsed runners ready to template"""

    desc: Desc
    info: str

    def _generate_job_name(self):
        # Github runner name can be max 64 characters and no special chars.
        # Still try to be as verbose as possible.
        parts = [
            CONFIG.nomad.jobprefix,
            COUNTERS.cnt.inc(),
            self.desc.repo,
            self.desc.labels.replace("nomadtools ", ""),
        ]
        return (re.sub(r"[^a-zA-Z0-9_.-]", "", "-".join(str(x) for x in parts))[:64],)

    @staticmethod
    def make_example(labelsstr: str):
        return RunnerGenerator(
            Desc(labels=labelsstr, repo_url="http://example.repo.url/user/repo"),
            info="This is an information",
        )

    def _to_template_args(self) -> dict:
        arg = labelstr_to_kv(self.desc.labels)
        name = self._generate_job_name()
        return dict(
            param={
                **arg,
                **CONFIG.template_context,
                **CONFIG.default_template_context,
                **dict(
                    REPO_URL=self.desc.repo_url,
                    RUNNER_NAME=name,
                    JOB_NAME=name,
                    LABELS=self.desc.labels,
                ),
            },
            run=asdict(self),
            arg=arg,
            CONFIG=CONFIG,
            TEMPLATE=TEMPLATE,
            ARGS=ARGS,
            escape=nomadlib.escape,
            META=json.dumps(asdict(self.desc)),
        )

    def get_runnertorun(self) -> NomadJobToRun:
        tc = self._to_template_args()
        jobtext = TEMPLATE.render(tc)
        jobspec = nomad_job_text_to_json(jobtext)
        # Apply default transformations.
        jobspec["Namespace"] = CONFIG.nomad.namespace
        for key in ["ID", "Name"]:
            if key in jobspec:
                jobspec[key] = tc["param"]["JOB_NAME"]
        jobspec["Meta"][CONFIG.nomad.meta] = tc["META"]
        #
        nomadjobtorun = NomadJobToRun(jobspec, jobtext)
        assert nomadjobtorun.get_desc() == self.desc
        return nomadjobtorun


###############################################################################
# {{{1 scheduler


@dataclass
class Scheduler:
    desc: Desc
    gjs: List[GithubJob]
    nrs: List[NomadRunner]

    def gen_runners_to_trim(self) -> List[NomadRunner]:
        """Generate runners to stop."""
        order = [
            RunnerState.unknown,
            RunnerState.pending,
            RunnerState.listening,
        ]
        nrs = [nr for nr in self.nrs if nr.state.state in order]
        nrs.sort(
            key=lambda nr: (
                order.index(nr.state.state),
                nr.state.since,
                -nr.nj.ModifyIndex,
            )
        )
        return nrs

    def schedule(self):
        nrs_not_dead, nrs_dead = list_split(lambda x: x.state.is_dead(), self.nrs)
        torun = len(self.gjs) - len(nrs_not_dead)
        if torun > 0:
            for _ in range(torun):
                nomadjobtorun = RunnerGenerator(self.desc, "").get_runnertorun()
                log.info(f"Starting runner {nomadjobtorun.tostr()}")
                COUNTERS.nomad_run.inc()
                mynomad.start_job(
                    nomadjobtorun.nj.asdict(),
                    nomadlib.JobSubmission.mk_hcl(nomadjobtorun.hcl),
                )
        elif torun < 0:
            tostop = -torun
            nomadjobstostop = self.gen_runners_to_trim()[:tostop]
            for nr in nomadjobstostop:
                log.info(f"Stopping runner {nr.tostr()}")
                COUNTERS.nomad_purge.inc()
                mynomad.stop_job(nr.nj.ID, purge=True)
        if CONFIG.nomad.purge:
            for nr in nrs_dead:
                log.info(f"Purging runner {nr.tostr()}")
                COUNTERS.nomad_purge.inc()
                mynomad.stop_job(nr.nj.ID, purge=True)


def merge_states(gjs: List[GithubJob], nrs: List[NomadRunner]):
    state: Dict[Desc, Tuple[List[GithubJob], List[NomadRunner]]] = {}
    for i in gjs:
        state.setdefault(i.get_desc(), ([], []))[0].append(i)
    for i in nrs:
        state.setdefault(i.get_desc(), ([], []))[1].append(i)
    return state


def loop():
    """The main loop of this program"""
    # Get the github state.
    repos = get_gh_repos()
    reqstate: list[GithubJob] = get_gh_state(repos)
    # Get the nomad state.
    curstate: list[NomadRunner] = get_nomad_state()
    # Merge states on label and repo.
    state = merge_states(reqstate, curstate)
    for k, v in state.items():
        Scheduler(k, v[0], v[1]).schedule()
    # Ending.
    COUNTERS.print()


###############################################################################
# {{{1 command line


@dataclass
class Args:
    dryrun: bool = clickdc.option("-n")
    verbose: bool = clickdc.option("-v")
    parallel: int = clickdc.option("-P", default=0)
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


@click.command("githubrunner", cls=AliasedGroup)
@clickdc.adddc("args", Args)
@help_h_option()
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
    tmp = yaml.safe_load(configstr)
    global CONFIG
    CONFIG = Config(tmp)
    global TEMPLATE
    TEMPLATE = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(
        CONFIG.template
    )
    mynomad.namespace = os.environ["NOMAD_NAMESPACE"] = CONFIG.nomad.namespace
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


@cli.command()
@click.argument("label", required=True, nargs=-1)
def rendertemplate(label: Tuple[str, ...]):
    labelsstr: str = ",".join(sorted(list(label)))
    res = RunnerGenerator.make_example(labelsstr).get_runnertorun()
    print(res)


@cli.command(help="Main entrypoint - run the loop periodically")
def run():
    while True:
        loop()
        log.info(f"Sleeping for {CONFIG.loop} seconds")
        time.sleep(CONFIG.loop)


if __name__ == "__main__":
    cli()
