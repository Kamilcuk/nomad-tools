#!/usr/bin/env python3
from __future__ import annotations

import collections
import concurrent.futures
import copy
import functools
import glob
import json
import logging
import os
import pkgutil
import subprocess
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import click
import clickdc
import requests
import yaml

from .common import mynomad
from .nomadlib.datadict import DataDict

log = logging.getLogger(__name__)

###############################################################################


class NomadConfig(DataDict):
    """Nomad configuration"""

    namespace: str = "github"
    """The namespace to set on the job."""
    token: Optional[str] = os.environ.get("NOMAD_TOKEN")
    jobprefix: str = "NTGHR"
    """The string to prefix run jobs with.
    Default: "NTGHR" like Nomad Tools GitHub Runner."""
    meta: str = "NT"
    """The metadata prefix to prefix jobs.
    Default: "NT" like Nomad Tools."""
    purge: bool = True
    """Purge dead jobs"""


class GithubConfig(DataDict):
    url: str = "https://api.github.com"
    """The url to github api"""
    token: Optional[str] = os.environ.get("GH_TOKEN", os.environ.get("GITHUB_TOKEN"))
    access_token: Optional[str] = None
    repos: List[str] = []
    """List of repositories to watch.
    This is either a organization is user, in which case API is called
    to get all repositories of this organization or user.
    Or this is a single repository name."""


class Config(DataDict):
    """Configuration"""

    nomad: NomadConfig = NomadConfig()
    """Nomad related configuration"""
    github: List[GithubConfig] = []
    """Configuration related to github with source repositories list"""
    load_default_runners: bool = True

    runners: List[str] = []
    """The runners configuration.
    Each field should be either a path to a file containing a HCL or JSON
    job specification, or it should be a HCL or JSON Nomad job specification
    in string form.
    The job should contain a job meta field called NT_LABELS with comma
    separated list of labels of the job.
    The job will be started with additional NT_* metadata fields
    with values from the Spec object below."""
    loop: int = 10
    """How many seconds will the loop run"""


###############################################################################


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


def gh_get(url: str, token: Optional[str], key: str = "") -> Any:
    """Execute query to github

    @param key if set, means the output is paginated
    """
    headers = {
        "Accept": "application/vnd.github+json",
        **({"Authorization": "Bearer " + token} if token else {}),
    }
    ret: list = []
    while True:
        response = requests.get(url, headers=headers)
        data = response.json()
        if not key:
            return data
        ret.extend(data[key])
        next = response.links.get("next")
        if not next:
            break
        url = next["url"]
    return ret


@functools.lru_cache()
@functools.wraps(gh_get)
def gh_get_cached(url: str, token: Optional[str], key: str = ""):
    return gh_get(url, token, key)


@dataclass
class Args:
    dryrun: bool = clickdc.option("-n")
    verbose: bool = clickdc.option("-v")
    config: Path = clickdc.option(
        "-c", type=click.Path(exists=True, dir_okay=False, path_type=Path)
    )


###############################################################################


@dataclass
class GithubRepo:
    github: GithubConfig
    repo: str


def get_repos() -> List[GithubRepo]:
    repos: List[GithubRepo] = []
    for github in CONFIG.github:
        for repo in github.repos:
            if repo.count("/") == 0:
                d = gh_get_cached(f"{github.url}/users/{repo}", github.token)
                d = gh_get(d["repos_url"], github.token)
                repos.extend([GithubRepo(github, x["full_name"]) for x in d])
            else:
                repos.append(GithubRepo(github, repo))
    return repos


class NomadJob(dict):
    def job_url(self) -> Optional[str]:
        return (self["Meta"] or {}).get(CONFIG.nomad.meta + "_job_url")

    def labels(self) -> Optional[str]:
        return (self["Meta"] or {}).get(CONFIG.nomad.meta + "_labels")


def get_curstate() -> list[NomadJob]:
    jobsjobs = mynomad.get(
        "jobs",
        params=dict(
            prefix=CONFIG.nomad.jobprefix + "-",
            meta=True,
            namespace=CONFIG.nomad.namespace,
        ),
    )
    curstate: list[NomadJob] = []
    for jobsjob in jobsjobs:
        nj = NomadJob(jobsjob)
        if nj.job_url():
            curstate.append(nj)
    desc = ", ".join(f"({x['Status']} {x.job_url()})" for x in curstate)
    log.info(f"Found {len(curstate)} runners: {desc}")
    for s in curstate:
        logging.debug(f"CUR: {s}")
    return curstate


JobUrl = str


@dataclass
class GithubJob(GithubRepo):
    run: dict
    job: dict

    def labels(self):
        return ",".join(sorted(list(set(self.job["labels"]))))

    def job_url(self):
        return self.job["html_url"]


def get_reqstate(repos: List[GithubRepo]):
    reqstate: dict[JobUrl, GithubJob] = {}
    for repo in repos:
        runs = gh_get(
            f"{repo.github.url}/repos/{repo.repo}/actions/runs",
            repo.github.token,
            "workflow_runs",
        )
        for run in runs:
            if run["status"] in ["queued"]:
                jobs = gh_get(
                    f"{repo.github.url}/repos/{repo.repo}/actions/runs/{run['id']}/jobs",
                    repo.github.token,
                    "jobs",
                )
                for job in jobs:
                    if job["status"] in "queued running".split():
                        gj = GithubJob(repo.github, repo.repo, run, job)
                        reqstate[gj.job_url()] = gj
    desc: str = ", ".join(s.labels() + " for " + s.job_url() for s in reqstate.values())
    log.info(f"Found {len(reqstate)} required runners to run  : {desc}")
    for s in reqstate.values():
        logging.debug(f"REQ: {s}")
    return reqstate


@dataclass
class Todo:
    tostart: List[dict] = field(default_factory=list)
    tostop: List[str] = field(default_factory=list)
    topurge: List[str] = field(default_factory=list)


def loop():
    repos = get_repos()
    reqstate: dict[JobUrl, GithubJob] = get_reqstate(repos)
    curstatearr: list[NomadJob] = get_curstate()
    #
    tostart: List[dict] = []
    tostop: List[str] = []
    topurge: List[str] = []
    #
    curstate: Dict[JobUrl, NomadJob] = {}
    for nj in curstatearr:
        joburl = nj.job_url()
        assert joburl
        if joburl not in curstate:
            curstate[joburl] = nj
        else:
            log.error(
                f"Found two nomad jobs {nj['ID']} and {curstate[joburl]['ID']} for one github job {joburl} - stopping"
            )
            tostop.append(nj["ID"])
    #
    for joburl, req in reqstate.items():
        nomadjob = curstate.get(joburl)
        if not nomadjob:
            # The job is missing - run the job.
            jobspec = find_runner_jobspec(set(req.job["labels"]))
            if not jobspec:
                logging.error(
                    f"Did not found runner matching {req.labels()} for {req.job_url()}"
                )
                continue
            # Change the ID and Name of the job with the prefix we need.
            for key in ["ID", "Name"]:
                if key in jobspec:
                    # Github runner name can be max 64 characters and no special chars.
                    # Still try to be as verbose as possible.
                    # The hash generated from job_url should be unique enough.
                    jobspec[key] = urllib.parse.quote(
                        "-".join(
                            [
                                CONFIG.nomad.jobprefix,
                                str(abs(hash(req.job_url()))),
                                req.repo.replace("/", "-"),
                                req.run["display_title"],
                            ]
                        ),
                        safe="",
                    )[:64]
            jobspec["Namespace"] = CONFIG.nomad.namespace
            jobspec["Meta"] = {
                **(jobspec["Meta"] or {}),
                **{
                    CONFIG.nomad.meta + "_" + k: str(v)
                    for k, v in dict(
                        job_url=req.job["html_url"],
                        job_id=req.job["id"],
                        run_id=req.job["run_id"],
                        labels=req.labels(),
                        repo_url=req.run["repository"]["html_url"],
                        token=req.github.access_token or req.github.token,
                    ).items()
                },
            }
            log.info(
                f"Running job {jobspec['ID']} with {req.labels()} for {req.job_url()}"
            )
            log.debug(f"Running {jobspec}")
            tostart.append(jobspec)
    for joburl, nomadjob in curstate.items():
        if joburl not in reqstate:
            if nomadjob["Status"].lower() != "dead" and not nomadjob["Stop"]:
                log.warning(
                    f"Job {nomadjob['ID']} for {joburl} is not required, stopping"
                )
                tostop.append(nomadjob["ID"])
            elif CONFIG.nomad.purge and nomadjob["Status"].lower() == "dead":
                # Check that there are no running evaluations.
                evals = mynomad.get(f"job/{nomadjob['ID']}/evaluations")
                if len(evals) == 0 or all(
                    eval["Status"] == "complete" for eval in evals
                ):
                    name = nomadjob["ID"]
                    log.info(f"Purging {name} for {joburl} with no running deployments")
                    topurge.append(name)
    #
    for nj in tostop:
        assert nj not in [x["ID"] for x in tostart], f"{nj} is tostop and tostart"
    for nj in tostart:
        assert nj not in tostop
        assert nj not in topurge
    for nj in tostop:
        assert nj not in topurge
    for nj in topurge:
        assert nj not in [x["ID"] for x in tostart]
    #
    if tostart:
        log.info(f"tostart: {' '.join(x['ID'] for x in tostart)}")
    if tostop:
        log.info(f"tostop: {' '.join(tostop)}")
    if topurge:
        log.info(f"topurge: {' '.join(topurge)}")
    if ARGS.dryrun:
        log.error("DRYRUN")
    else:
        for spec in tostart:
            jsonspec = json.dumps(spec)
            try:
                subprocess.run(
                    "nomad job run -detach -json -".split(),
                    input=jsonspec,
                    check=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                log.exception(f"Could not start: {jsonspec}")
                raise
        for name in tostop:
            subprocess.check_call("nomad job stop -detach".split() + [name])
        if CONFIG.nomad.purge:
            for name in topurge:
                subprocess.check_call("nomad job stop -detach -purge".split() + [name])


###############################################################################

DEFAULTCONFIG = """
---
nomad:
  namespace: default
github:
  - repos:
      - Kamilcuk/runnertest
"""


def get_default_runners() -> List[str]:
    arr = """
    latest
    ubuntu-focal
    ubuntu-noble
    ubuntu-jammy
    debian-buster
    debian-bookworm
    debian-sid
    """
    ret = []
    for tag in arr.split():
        for mode in ["", "docker", "hostdocker"]:
            name = f"nomadtools.{mode}{'.' if mode else ''}{tag}"
            txt = f"""
job "{name}" {{
  type = "batch"
  meta {{
    {CONFIG.nomad.meta}_INFO = <<EOF
This is a runner based on myoung34/github-runner:{tag} image.
{'It also starts a docker daemon and is running as privileged' if mode == 'docker' else ''}
{'It also mounts a docker daemon from the host it is running on' if mode == 'hostdocker' else ''}
EOF
    {CONFIG.nomad.meta}_LABELS = "{name}"
  }}
  group "{name}" {{
    reschedule {{
      attempts  = 0
      unlimited = false
    }}
    restart {{
      attempts = 0
      mode     = "fail"
    }}
    task "{name}" {{
      driver = "docker"
      config {{
        image = "myoung34/github-runner:{tag}"
        entrypoint = ["bash", "-x", "/entrypoint.sh"]
        args = ["./bin/Runner.Listener", "run", "--startuptype", "service"]
        {'privileged = true' if mode == 'docker' else ''}
        {'''
        mount {
            type   = "bind"
            target = "/var/run/docker.sock"
            source = "/var/run/docker.sock"
        }
        ''' if mode == "hostdocker" else ''}
      }}
      template {{
        destination = "local/env"
        env = true
        data = <<-EOF
        ACCESS_TOKEN = "{{{{env "NOMAD_META_NT_token"}}}}"
        REPO_URL     = "{{{{env "NOMAD_META_NT_repo_url"}}}}"
        RUNNER_NAME  = "{{{{env "NOMAD_JOB_NAME"}}}}"
        EOF
      }}
      env {{
        RUNNER_SCOPE        = "repo"
        LABELS              = "{name}"
        EPHEMERAL           = "true"
        DISABLE_AUTO_UPDATE = "true"
        {'START_DOCKER_SERVICE = "true"' if mode == 'docker' else ''}
      }}
      resources {{
        memory = 1000
      }}
    }}
  }}
}}
            """.strip()
            # Remvoe empty lines.
            txt = "\n".join(line for line in txt.splitlines() if line)
            ret.append(txt)
    return ret


@dataclass
class Runner:
    spec: dict
    labels: Set[str]


def load_runner(spec: str):
    jobspec = nomad_job_to_json(spec)
    id = jobspec["ID"]
    meta = jobspec["Meta"]
    assert meta, f"Job {id} has no meta"
    key = CONFIG.nomad.meta + "_LABELS"
    assert key in meta, f"Job {id} meta has no {key}"
    labels = set(meta[key].split(","))
    return Runner(jobspec, labels)


def load_runners(specs: List[str]):
    """Given a list of files or job specifications, convert them to JSON"""
    # handle directories
    arr: List[str] = []
    for i in specs:
        if Path(i).is_dir():
            arr.extend(list(glob.glob("*.nomad.hcl", root_dir=i, recursive=True)))
        else:
            arr.append(i)
    # Runners are loaded concurrently to speed up calling Nomad process
    with concurrent.futures.ThreadPoolExecutor() as p:
        ret: List[Runner] = list(p.map(load_runner, arr))
    #
    runnersstr: str = " ".join(" ".join(r.labels) for r in ret)
    log.info(f"Loaded {len(ret)} runners: {runnersstr}")
    for r in ret:
        log.debug(f"RUNNER: {r}")
    # check
    assert ret, "No runners found"
    # Check for duplicated runners.
    labels: List[str] = [",".join(sorted(list(x.labels))) for x in ret]
    for item, count in collections.Counter(labels).items():
        assert count == 1, f"Found {count} runners with same labels: {item}"
    return ret


def find_runner_jobspec(labels: Set[str]):
    for r in RUNNERS:
        if r.labels == labels:
            # This _has to_ return a COPY of the dictionary,
            # because it is going to be modified.
            return copy.deepcopy(r.spec)
    return None


###############################################################################

ARGS: Args
CONFIG: Config
RUNNERS: List[Runner]


@click.group("githubrunner")
@clickdc.adddc("args", Args)
def cli(args: Args):
    global ARGS
    ARGS = args
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    #
    if args.config:
        with args.config.open() as f:
            configstr = f.read()
    else:
        configstr = DEFAULTCONFIG
    global CONFIG
    tmp = yaml.safe_load(configstr)
    CONFIG = Config(tmp)
    # Load runners
    if CONFIG.load_default_runners:
        CONFIG.runners += get_default_runners()
    global RUNNERS
    RUNNERS = load_runners(CONFIG.runners)
    #
    os.environ["NOMAD_NAMESPACE"] = CONFIG.nomad.namespace


@cli.command()
def purgeall():
    curstate = get_curstate()
    for x in curstate:
        cmd = "nomad job stop --detach --purge".split() + [x["ID"]]
        print(f"+ {cmd}")
        subprocess.check_call(cmd)


@cli.command(help="Dump the configuration")
def dumpconfig():
    print(yaml.safe_dump(CONFIG.asdict()))


@cli.command(help="Dump the configuration")
def listrunners():
    arr: List[str] = [
        (
            " ".join(r.labels)
            + "  "
            + (r.spec["Meta"] or {})
            .get(CONFIG.nomad.meta + "_INFO", "")
            .replace("\n", " ")
            .strip()
        )
        for r in RUNNERS
    ]
    print("\n".join(sorted(arr)))


@cli.command(help="Execute the loop once")
def once():
    loop()


@cli.command(help="Main entrypoint - run the loop periodically")
def run():
    while True:
        loop()
        log.info(f"Sleeping for {CONFIG.loop} seconds")
        time.sleep(CONFIG.loop)


if __name__ == "__main__":
    cli()
