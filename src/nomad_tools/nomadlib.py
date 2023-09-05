import dataclasses
import json
import logging
import os
from typing import Dict, List, Optional

import requests.adapters
import requests.auth

log = logging.getLogger(__name__)


class AttrDict(dict):
    def __init__(self, *args, **kw):
        dict.__init__(self, *args, **kw)
        self.__dict__ = self
        self.__post_init__()

    def __post_init__(self):
        pass


class Job(AttrDict):
    ID: str
    Version: int
    Status: str
    Namespace: str
    ModifyIndex: int
    JobModifyIndex: int

    def description(self):
        return f"{self.ID}@{self.Namespace} v{self.Version}"


class Eval(AttrDict):
    ID: str
    JobID: str
    JobModifyIndex: int
    ModifyIndex: int


class AllocTaskStateEvent(AttrDict):
    DisplayMessage: str
    Time: int
    Type: str


class AllocTaskStates(AttrDict):
    Events: List[AllocTaskStateEvent]
    State: str

    def __post_init__(self):
        self.Events = [
            AllocTaskStateEvent(e) for e in (getattr(self, "Events", []) or [])
        ]

    def find_event(self, type_: str) -> Optional[dict]:
        """Find event in TaskStates task Events. Return empty dict if not found"""
        return next((e for e in self.Events if e.Type == type_), None)


class Alloc(AttrDict):
    ID: str
    JobID: str
    EvalID: str
    ClientStatus: str
    TaskStates: Dict[str, AllocTaskStates]
    Namespace: str
    ModifyIndex: int
    # May be missing.
    JobVersion: int

    def __post_init__(self):
        self.TaskStates = {
            k: AllocTaskStates(v)
            for k, v in (getattr(self, "TaskStates", {}) or {}).items()
        }

    def is_pending_or_running(self):
        return self.ClientStatus in ["pending", "running"]

    def is_finished(self):
        return not self.is_pending_or_running()


@dataclasses.dataclass
class Conn:
    """Represents connection to Nomad"""

    namespace: Optional[str] = None
    session: requests.Session = dataclasses.field(default_factory=requests.Session)

    def __post_init__(self):
        a = requests.adapters.HTTPAdapter(
            pool_connections=1000, pool_maxsize=1000, max_retries=3
        )
        self.session.mount("http://", a)

    def request(
        self,
        method,
        url,
        params: Optional[dict] = None,
        *args,
        **kvargs,
    ):
        params = dict(params or {})
        assert "namespace" not in params
        params["namespace"] = self.namespace or os.environ.get("NOMAD_NAMESPACE", "*")
        return self.session.request(
            method,
            os.environ.get("NOMAD_ADDR", "http://127.0.0.1:4646") + "/v1/" + url,
            *args,
            auth=(
                requests.auth.HTTPBasicAuth(*os.environ["NOMAD_HTTP_AUTH"].split(":"))
                if "NOMAD_HTTP_AUTH" in os.environ
                else None
            ),
            headers=(
                {"X-Nomad-Token": os.environ["NOMAD_TOKEN"]}
                if "NOMAD_TOKEN" in os.environ
                else None
            ),
            params=params,
            **kvargs,
        )

    def _reqjson(self, mode, *args, **kvargs):
        rr = self.request(mode, *args, **kvargs)
        rr.raise_for_status()
        return rr.json()

    def get(self, *args, **kvargs):
        return self._reqjson("GET", *args, **kvargs)

    def put(self, *args, **kvargs):
        return self._reqjson("PUT", *args, **kvargs)

    def post(self, *args, **kvargs):
        return self._reqjson("POST", *args, **kvargs)

    def delete(self, *args, **kvargs):
        return self._reqjson("DELETE", *args, **kvargs)

    def stream(self, *args, **kvargs):
        stream = self.request("GET", *args, stream=True, **kvargs)
        stream.raise_for_status()
        return stream

    def start_job(self, txt: str):
        try:
            jobjson = json.loads(txt)
        except json.JSONDecodeError:
            jobjson = self.post("jobs/parse", json={"JobHCL": txt})
        return self.post("jobs", json={"Job": jobjson, "Submission": txt})

    def stop_job(self, jobid: str, purge: bool = False):
        assert self.namespace
        if purge:
            log.info(f"Purging job {jobid}")
        else:
            log.info(f"Stopping job {jobid}")
        resp: dict = self.delete(f"job/{jobid}", params={"purge": purge})
        assert resp["EvalID"], f"Stopping {jobid} did not trigger evaluation: {resp}"
        return resp

    def find_job(self, jobprefix: str) -> str:
        jobs = self.get("jobs", params={"prefix": jobprefix})
        assert len(jobs) > 0, f"No jobs found with prefix {jobprefix}"
        jobsnames = " ".join(f"{x['ID']}@{x['Namespace']}" for x in jobs)
        assert len(jobs) < 2, f"Multiple jobs found with name {jobprefix}: {jobsnames}"
        job = jobs[0]
        self.namespace = job["Namespace"]
        return job["ID"]

    def find_last_not_stopped_job(self, jobid: str) -> dict:
        assert self.namespace
        jobinit = self.get(f"job/{jobid}")
        if jobinit["Stop"]:
            # Find last job version that is not stopped.
            versions = self.get(f"job/{jobid}/versions")
            notstopedjobs = [job for job in versions["Versions"] if not job["Stop"]]
            if notstopedjobs:
                notstopedjobs.sort(key=lambda job: -job["ModifyIndex"])
                return notstopedjobs[0]
        return jobinit
