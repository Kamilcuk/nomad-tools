import dataclasses
import datetime
import enum
import json
import logging
import os
from typing import Callable, Dict, List, Optional, TypeVar

import requests.adapters
import requests.auth

from .datadict import DataDict

log = logging.getLogger(__name__)


class JobTask(DataDict):
    Name: str


class JobTaskGroup(DataDict):
    Name: str
    Tasks: List[JobTask]


class Job(DataDict):
    ID: str
    Version: int
    Status: str
    Namespace: str
    ModifyIndex: int
    JobModifyIndex: int
    TaskGroups: List[JobTaskGroup]

    def description(self):
        return f"{self.ID}@v{self.Version}@{self.Namespace}"


class Eval(DataDict):
    ID: str
    JobID: str
    JobModifyIndex: int
    ModifyIndex: int


class AllocTaskStateEvent(DataDict):
    DisplayMessage: str
    Time: int
    Type: str


class AllocTaskStates(DataDict):
    Events: List[AllocTaskStateEvent]
    State: str

    def find_event(self, type_: str) -> Optional[AllocTaskStateEvent]:
        """Find event in TaskStates task Events. Return empty dict if not found"""
        return next((e for e in self.Events if e.Type == type_), None)


class Alloc(DataDict):
    ID: str
    NodeName: str
    JobID: str
    EvalID: str
    ClientStatus: str
    Namespace: str
    ModifyIndex: int
    TaskGroup: str
    # Also may be missing!
    TaskStates: Optional[Dict[str, AllocTaskStates]] = None
    # May be missing!
    JobVersion: int

    def taskstates(self) -> Dict[str, AllocTaskStates]:
        """The same as TaskStates but returns an empty dict in case the field is None"""
        return self.get("TaskStates") or {}

    def tasks(self):
        return [k for k in self.taskstates().keys()]

    def is_pending_or_running(self):
        return self.ClientStatus in ["pending", "running"]

    def is_running(self):
        return self.ClientStatus == "running"

    def is_finished(self):
        return not self.is_pending_or_running()


class EventTopic(enum.Enum):
    """Topic of an event from Nomad event stream"""

    ACLToken = enum.auto()
    ACLPolicy = enum.auto()
    ACLRoles = enum.auto()
    Allocation = enum.auto()
    Job = enum.auto()
    Evaluation = enum.auto()
    Deployment = enum.auto()
    Node = enum.auto()
    NodeDrain = enum.auto()
    NodePool = enum.auto()
    Service = enum.auto()


class EventType(enum.Enum):
    """Type of an event from Nomad event stream"""

    ACLTokenUpserted = enum.auto()
    ACLTokenDeleted = enum.auto()
    ACLPolicyUpserted = enum.auto()
    ACLPolicyDeleted = enum.auto()
    ACLRoleUpserted = enum.auto()
    ACLRoleDeleted = enum.auto()
    AllocationCreated = enum.auto()
    AllocationUpdated = enum.auto()
    AllocationUpdateDesiredStatus = enum.auto()
    DeploymentStatusUpdate = enum.auto()
    DeploymentPromotion = enum.auto()
    DeploymentAllocHealth = enum.auto()
    EvaluationUpdated = enum.auto()
    JobRegistered = enum.auto()
    JobDeregistered = enum.auto()
    JobBatchDeregistered = enum.auto()
    NodeRegistration = enum.auto()
    NodeDeregistration = enum.auto()
    NodeEligibility = enum.auto()
    NodeDrain = enum.auto()
    NodeEvent = enum.auto()
    NodePoolUpserted = enum.auto()
    NodePoolDeleted = enum.auto()
    PlanResult = enum.auto()
    ServiceRegistration = enum.auto()
    ServiceDeregistration = enum.auto()


R = TypeVar("R")


@dataclasses.dataclass
class Event:
    """A single Event as returned from Nomad event stream"""

    topic: EventTopic
    type: EventType
    data: dict
    time: Optional[datetime.datetime] = None
    stream: bool = False

    def __str__(self):
        status = {
            "ID": self.data.get("ID", "")[:6],
            "JobID": self.data.get("JobID"),
            "Version": self.data.get("Version"),
            "JobVersion": self.data.get("JobVersion"),
            "JobModifyIndex": self.data.get("JobModifyIndex"),
            "ModifyIndex": self.data.get("ModifyIndex"),
            "Status": self.data.get("Status"),
            "ClientStatus": self.data.get("ClientStatus"),
            "stream": 1 if self.stream else 0,
        }
        statusstr = " ".join(f"{k}={v}" for k, v in status.items() if v)
        return f"Event({self.topic.name}.{self.type.name} {statusstr})"

    def is_job(self):
        return self.topic == EventTopic.Job

    def is_eval(self):
        return self.topic == EventTopic.Evaluation

    def is_alloc(self):
        return self.topic == EventTopic.Allocation

    def get_job(self) -> Optional[Job]:
        return Job(self.data) if self.is_job() else None

    def get_eval(self) -> Optional[Eval]:
        return Eval(self.data) if self.is_eval() else None

    def get_alloc(self) -> Optional[Alloc]:
        return Alloc(self.data) if self.is_alloc() else None

    def apply(
        self,
        job: Optional[Callable[[Job], R]] = None,
        eval: Optional[Callable[[Eval], R]] = None,
        alloc: Optional[Callable[[Alloc], R]] = None,
    ) -> R:
        callbacks = {}
        if job:
            callbacks[EventTopic.Job] = lambda data: job(Job(data))
        if eval:
            callbacks[EventTopic.Evaluation] = lambda data: eval(Eval(data))
        if alloc:
            callbacks[EventTopic.Allocation] = lambda data: alloc(Alloc(data))
        assert len(callbacks), f"At least one callback has to be specified"
        return callbacks[self.topic](self.data)


###############################################################################


def _default_session():
    s = requests.Session()
    # Increase the number of connections.
    a = requests.adapters.HTTPAdapter(
        pool_connections=1000, pool_maxsize=1000, max_retries=3
    )
    s.mount("http://", a)
    s.mount("https://", a)
    return s


class PermissionDenied(Exception):
    pass


class JobNotFound(Exception):
    pass


@dataclasses.dataclass
class Nomad:
    """Represents connection to Nomad"""

    namespace: Optional[str] = None
    session: requests.Session = dataclasses.field(default_factory=_default_session)

    def request(
        self,
        method,
        url,
        params: Optional[dict] = None,
        *args,
        **kvargs,
    ):
        params = dict(params or {})
        params.setdefault(
            "namespace", self.namespace or os.environ.get("NOMAD_NAMESPACE", "*")
        )
        ret = self.session.request(
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
            verify=os.environ.get("NOMAD_CACERT"),
            cert=(
                (os.environ["NOMAD_CLIENT_CERT"], os.environ["NOMAD_CLIENT_KEY"])
                if "NOMAD_CLIENT_CERT" in os.environ
                and "NOMAD_CLIENT_KEY" in os.environ
                else None
            ),
            **kvargs,
        )
        try:
            ret.raise_for_status()
        except requests.HTTPError as e:
            resp = (ret.status_code, ret.text.lower())
            if resp == (500, "permission denied"):
                raise PermissionDenied(str(e)) from e
            if resp == (404, "job not found"):
                raise JobNotFound(str(e)) from e
        return ret

    def _reqjson(self, mode, *args, **kvargs):
        rr = self.request(mode, *args, **kvargs)
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
