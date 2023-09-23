from __future__ import annotations
import dataclasses
import datetime
import enum
import logging
from typing import Callable, Dict, List, Optional, TypeVar

from .datadict import DataDict

log = logging.getLogger(__name__)


class DockerMounts(DataDict):
    type: str
    target: str
    source: str
    readonly: bool


class JobTaskConfig(DataDict):
    image: str
    command: str
    args: List[str]
    network_mode: str
    network_aliases: List[str]
    volumes: List[str]
    mounts: List[DockerMounts]
    extra_hosts: List[str]


class JobTaskLifecycle(DataDict):
    Hook: str
    Sidecar: bool

    def get_sidecar(self) -> bool:
        return self.get("Sidecar", False)


class JobTask(DataDict):
    Name: str
    Driver: str
    User: str
    Config: JobTaskConfig
    Lifecycle: Optional[JobTaskLifecycle]
    Env: Optional[Dict[str, str]]


class JobTaskGroup(DataDict):
    Name: str
    Tasks: List[JobTask]


class JobStatus(enum.Enum):
    pending = "pending"
    """Pending means the job is waiting on scheduling"""
    running = "running"
    """Running means the job has non-terminal allocations"""
    dead = "dead"
    """Dead means all evaluation's and allocations are terminal"""


class Job(DataDict):
    """Returned aby job/<id> API. DO NOT mix with JobsJob"""

    ID: str
    Version: int
    Status: str
    Namespace: str
    ModifyIndex: int
    JobModifyIndex: int
    TaskGroups: List[JobTaskGroup]
    Stop: bool
    Meta: Optional[Dict[str, str]]

    def description(self):
        return f"{self.ID}@v{self.Version}@{self.Namespace}"


class JobsJob(DataDict):
    """Returned aby jobs API. DO NOT mix with Job"""

    ID: str


class Eval(DataDict):
    ID: str
    JobID: str
    JobModifyIndex: int
    ModifyIndex: int
    Status: str


class AllocTaskStateEventType:
    TaskSetupFailure = "Setup Failure"
    """indicates that the task could not be started due to a a setup failure."""
    TaskDriverFailure = "Driver Failure"
    """indicates that the task could not be started due to a failure in the driver. TaskDriverFailure is considered Recoverable."""
    TaskReceived = "Received"
    """signals that the task has been pulled by the client at the given timestamp."""
    TaskFailedValidation = "Failed Validation"
    """indicates the task was invalid and as such was not run. TaskFailedValidation is not considered Recoverable."""
    TaskStarted = "Started"
    """signals that the task was started and its timestamp can be used to determine the running length of the task."""
    TaskTerminated = "Terminated"
    """indicates that the task was started and exited."""
    TaskKilling = "Killing"
    """indicates a kill signal has been sent to the task."""
    TaskKilled = "Killed"
    """indicates a user has killed the task."""
    TaskRestarting = "Restarting"
    """indicates that task terminated and is being restarted."""
    TaskNotRestarting = "Not Restarting"
    """indicates that the task has failed and is not being restarted because it has exceeded its restart policy."""
    TaskRestartSignal = "Restart Signaled"
    """indicates that the task has been signaled to be restarted"""
    TaskSignaling = "Signaling"
    """indicates that the task is being signalled."""
    TaskDownloadingArtifacts = "Downloading Artifacts"
    """means the task is downloading the artifacts specified in the task."""
    TaskArtifactDownloadFailed = "Failed Artifact Download"
    """indicates that downloading the artifacts failed."""
    TaskBuildingTaskDir = "Building Task Directory"
    """indicates that the task directory/chroot is being built."""
    TaskSetup = "Task Setup"
    """indicates the task runner is setting up the task environment"""
    TaskDiskExceeded = "Disk Resources Exceeded"
    """indicates that one of the tasks in a taskgroup has exceeded the requested disk resources."""
    TaskSiblingFailed = "Sibling Task Failed"
    """indicates that a sibling task in the task group has failed."""
    TaskDriverMessage = "Driver"
    """is an informational event message emitted by drivers such as when they're performing a long running action like downloading an image."""
    TaskLeaderDead = "Leader Task Dead"
    """indicates that the leader task within the has finished."""
    TaskMainDead = "Main Tasks Dead"
    """indicates that the main tasks have dead"""
    TaskHookFailed = "Task hook failed"
    """indicates that one of the hooks for a task failed."""
    TaskHookMessage = "Task hook message"
    """indicates that one of the hooks for a task emitted a message."""
    TaskRestoreFailed = "Failed Restoring Task"
    """indicates Nomad was unable to reattach to a restored task."""
    TaskPluginUnhealthy = "Plugin became unhealthy"
    """indicates that a plugin managed by Nomad became unhealthy"""
    TaskPluginHealthy = "Plugin became healthy"
    """indicates that a plugin managed by Nomad became healthy"""
    TaskClientReconnected = "Reconnected"
    """indicates that the client running the task disconnected."""
    TaskWaitingShuttingDownDelay = "Waiting for shutdown delay"
    """indicates that the task is waiting for shutdown delay before being TaskKilled"""
    TaskSkippingShutdownDelay = "Skipping shutdown delay"
    """indicates that the task operation was configured to ignore the shutdown delay value set for the tas."""


class AllocTaskStateEvent(DataDict):
    DisplayMessage: str
    Time: int
    Type: str


class AllocTaskState(DataDict):
    Events: List[AllocTaskStateEvent]
    State: str
    Failed: bool
    FinishedAt: Optional[str]
    LastRestart: Optional[str]
    Restarts: int
    StartedAt: Optional[str]

    def find_event(self, type_: str) -> Optional[AllocTaskStateEvent]:
        """Find event in TaskStates task Events. Return empty dict if not found"""
        return next((e for e in self.Events if e.Type == type_), None)

    def was_started(self):
        return self.find_event(AllocTaskStateEventType.TaskStarted) is not None


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
    TaskStates: Optional[Dict[str, AllocTaskState]] = None
    # May be missing!
    JobVersion: int

    def get_taskstates(self) -> Dict[str, AllocTaskState]:
        """The same as TaskStates but returns an empty dict in case the field is None"""
        return self.get("TaskStates") or {}

    def get_tasknames(self):
        return [k for k in self.get_taskstates().keys()]

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

class JobSummaryChildren(DataDict):
    Pending: int
    Running: int
    Dead: int

class JobSummarySummary(DataDict):
    Queued: int = 0
    Complete: int = 0
    Failed: int = 0
    Running: int = 0
    Starting: int = 0
    Lost: int = 0

    def __iadd__(self, o: JobSummarySummary) -> JobSummarySummary:
        for k in set(self.asdict()) | set(o.asdict()):
            self[k] = self.get(k, 0) + o.get(k, 0)
        return self

class JobSummary(DataDict):
    JobID: str
    Summary: Dict[str, JobSummarySummary]
    Children: JobSummaryChildren
    CreateIndex: int
    ModifyIndex: int

    def get_sum_summary(self) -> JobSummarySummary:
        """Sum all summaries into one"""
        ret = JobSummarySummary({})
        for s in self.Summary.values():
            ret += s
        return ret


