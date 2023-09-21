import dataclasses
import datetime
import enum
import logging
from typing import Callable, Dict, List, Optional, TypeVar

from .datadict import DataDict

log = logging.getLogger(__name__)


class JobTaskConfig(DataDict):
    image: str
    command: str
    args: List[str]
    network_mode: str
    network_aliases: List[str]


class JobTask(DataDict):
    Name: str
    Driver: str
    User: str
    Config: JobTaskConfig


class JobTaskGroup(DataDict):
    Name: str
    Tasks: List[JobTask]


class JobStatus(enum.Enum):
    # Pending means the job is waiting on scheduling
    pending = "pending"
    # Running means the job has non-terminal allocations
    running = "running"
    # Dead means all evaluation's and allocations are terminal
    dead = "dead"


class Job(DataDict):
    ID: str
    Version: int
    Status: str
    Namespace: str
    ModifyIndex: int
    JobModifyIndex: int
    TaskGroups: List[JobTaskGroup]
    Stop: bool

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


class VariableNoItems(DataDict):
    Namespace: str
    Path: str
    CreateIndex: int
    ModifyIndex: int
    CreateTime: int
    ModifyTime: int


class Variable(VariableNoItems):
    Items: Dict[str, str]


class VariableNew(DataDict):
    Namespace: str
    Path: str
    Items: Dict[str, str]
