from __future__ import annotations

import dataclasses
import datetime
import enum
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from .datadict import DataDict

log = logging.getLogger(__name__)


class MyStrEnum(str, enum.Enum):
    """StrEnum for python 3.7 compatibility"""

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name


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


class LifecycleHook(MyStrEnum):
    prestart = "prestart"
    poststart = "poststart"
    poststop = "poststop"


class JobTaskLifecycle(DataDict):
    Hook: str
    Sidecar: bool

    def get_sidecar(self) -> bool:
        return self.get("Sidecar", False)


class JobTask(DataDict):
    Name: str
    Driver: str
    User: str
    Config: Union[dict, JobTaskConfig]
    Lifecycle: Optional[JobTaskLifecycle] = None
    Env: Optional[Dict[str, str]]
    Services: Optional[List[Any]]


class JobTaskGroup(DataDict):
    Name: str
    Count: int
    Tasks: List[JobTask]
    Services: Optional[List[Any]]


class JobStatus(MyStrEnum):
    pending = enum.auto()
    """Pending means the job is waiting on scheduling"""
    running = enum.auto()
    """Running means the job has non-terminal allocations"""
    dead = enum.auto()
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
    SubmitTime: int

    def is_dead(self):
        return self.Status == JobStatus.dead

    def description(self):
        return f"{self.ID}#{self.Version}@{self.Namespace}"

    def allservices(self):
        """Return all services from groups and tasks of a job"""
        out = []
        for tg in self.TaskGroups:
            if tg.Services:
                out += tg.Services
            for task in tg.Tasks:
                if task.Services:
                    out += task.Services
        return out


class JobsJob(DataDict):
    """Returned aby jobs API. DO NOT mix with Job"""

    ID: str


class NodeScoreMeta(DataDict):
    NodeID: str
    Scores: Dict[str, float]
    NormScore: float


class AllocationMetric(DataDict):
    NodesEvaluated: int
    NodesFiltered: int
    NodesAvailable: Dict[str, int]
    ClassFiltered: Optional[Dict[str, int]] = None
    ConstraintFiltered: Optional[Dict[str, int]] = None
    NodesExhausted: int
    QuotaExhausted: Optional[Dict[str, int]] = None
    ScoreMetaData: Optional[List[NodeScoreMeta]] = None
    ClassExhausted: Optional[Dict[str, int]] = None
    Scores: Optional[Dict[str, float]] = None
    DimensionExhausted: Optional[Dict[str, int]] = None

    def format(self, scores: bool = False, prefix: str = "") -> str:
        """https://github.com/hashicorp/nomad/blob/484f91b893c6f054e9339f8db6bd157429c2ef20/command/monitor.go#L345"""
        out = []
        if self.NodesEvaluated == 0:
            out += [f"No nodes were eligible for evaluation"]
        for dc, available in self.NodesAvailable.items():
            if available == 0:
                out += [f"No nodes are available in datacenter {dc}"]
        for cls, num in (self.ClassFiltered or {}).items():
            out += [f"Class {cls}: {num} nodes excluded by filter"]
        for cs, num in (self.ConstraintFiltered or {}).items():
            out += [f"Constraint {cs}: {num} nodes excluded by filter"]
        ne = self.NodesExhausted
        if ne > 0:
            out += [f"Resources exhaused on {ne} nodes"]
        for cls, num in (self.ClassExhausted or {}).items():
            out += [f"Class {cls} exhaused on {num} nodes"]
        for dim, num in (self.DimensionExhausted or {}).items():
            out += [f"Dimension {dim} exhaused on {num} nodes"]
        for _, dim in (self.QuotaExhausted or {}).items():
            out += [f"Quota limit hit {dim}"]
        if scores:
            if self.ScoreMetaData:
                allScores: List[str] = sorted(
                    list(set(k for s in self.ScoreMetaData for k in s.Scores.keys()))
                )
                out += [f"Node|{'|'.join(allScores)}|final score"]
                for scoreMeta in self.ScoreMetaData:
                    out += [
                        f"{scoreMeta.NodeID}|"
                        + "|".join(
                            f"{scoreMeta.Scores[scorerName]:.3g}"
                            for scorerName in allScores
                        )
                        + f"|{scoreMeta.NormScore:.3g}"
                    ]
            else:
                for name, score in (self.Scores or {}).items():
                    out += [f"Score {name} = {score}"]
        return "\n".join(f"{prefix}{x}" for x in out)


class EvalStatus(MyStrEnum):
    blocked = enum.auto()
    pending = enum.auto()
    complete = enum.auto()
    failed = enum.auto()
    canceled = enum.auto()


class Eval(DataDict):
    ID: str
    Namespace: str
    JobID: str
    JobModifyIndex: int
    ModifyIndex: int
    ModifyTime: int
    Status: str
    WaitUntil: str
    FailedTGAllocs: Dict[str, AllocationMetric]

    def is_pending(self):
        return self.Status == "pending"

    def getWaitUntil(self) -> Optional[datetime.datetime]:
        if not self.WaitUntil:
            return None
        return (
            datetime.datetime.fromisoformat(self.WaitUntil)
            .replace(tzinfo=datetime.timezone.utc)
            .astimezone()
        )


class AllocTaskStateEventType(MyStrEnum):
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
    Events: Optional[List[AllocTaskStateEvent]]
    State: str
    Failed: bool
    FinishedAt: Optional[str]
    LastRestart: Optional[str]
    Restarts: int
    StartedAt: Optional[str]

    def find_event(self, type_: str) -> Optional[AllocTaskStateEvent]:
        """Find event in TaskStates task Events. Return empty dict if not found"""
        return next((e for e in self.Events or [] if e.Type == type_), None)

    def was_started(self):
        return self.find_event(AllocTaskStateEventType.TaskStarted) is not None


class Alloc(DataDict):
    ID: str
    NodeName: str
    JobID: str
    EvalID: str
    ClientStatus: str
    CreateTime: int
    Namespace: str
    ModifyIndex: int
    ModifyTime: int
    TaskGroup: str
    # Also may be missing!
    TaskStates: Optional[Dict[str, AllocTaskState]] = None
    # May be missing!
    JobVersion: Optional[int]
    FollowupEvalID: Optional[str] = None

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


class DeploymentStatus(MyStrEnum):
    running = enum.auto()
    paused = enum.auto()
    failed = enum.auto()
    successful = enum.auto()
    cancelled = enum.auto()
    initializing = enum.auto()
    pending = enum.auto()
    blocked = enum.auto()
    unblocking = enum.auto()


class DeploymentStatusDescription(MyStrEnum):
    Running = "Deployment is running"
    RunningNeedsPromotion = "Deployment is running but requires manual promotion"
    RunningAutoPromotion = "Deployment is running pending automatic promotion"
    Paused = "Deployment is paused"
    Successful = "Deployment completed successfully"
    StoppedJob = "Cancelled because job is stopped"
    NewerJob = "Cancelled due to newer version of job"
    FailedAllocations = "Failed due to unhealthy allocations"
    ProgressDeadline = "Failed due to progress deadline"
    FailedByUser = "Deployment marked as failed"
    FailedByPeer = "Failed because of an error in peer region"
    Blocked = "Deployment is complete but waiting for peer region"
    Unblocking = "Deployment is unblocking remaining regions"
    PendingForPeer = "Deployment is pending, waiting for peer region"


class DeploymentTaskGroup(DataDict):
    AutoPromote: bool
    AutoRevert: bool
    DesiredCanaries: int
    DesiredTotal: int
    HealthyAllocs: int
    PlacedAllocs: int
    PlacedCanaries: Optional[List[str]] = None
    ProgressDeadline: int
    Promoted: bool
    RequireProgressBy: Optional[str]
    UnhealthyAllocs: int


class Deploy(DataDict):
    ID: str
    ModifyIndex: int
    JobCreateIndex: int
    JobModifyIndex: int
    JobVersion: int
    JobID: str
    Status: str
    StatusDescription: str
    TaskGroups: Dict[str, DeploymentTaskGroup]


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


class EventType(MyStrEnum):
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

    def is_deployment(self):
        return self.topic == EventTopic.Deployment

    def get_job(self) -> Optional[Job]:
        return Job(self.data) if self.is_job() else None

    def get_eval(self) -> Optional[Eval]:
        return Eval(self.data) if self.is_eval() else None

    def get_alloc(self) -> Optional[Alloc]:
        return Alloc(self.data) if self.is_alloc() else None

    def get_deployment(self) -> Optional[Deploy]:
        return Deploy(self.data) if self.is_deployment() else None

    def apply(
        self,
        job: Optional[Callable[[Job], R]] = None,
        eval: Optional[Callable[[Eval], R]] = None,
        alloc: Optional[Callable[[Alloc], R]] = None,
        deploy: Optional[Callable[[Deploy], R]] = None,
    ) -> R:
        callbacks = {}
        if job:
            callbacks[EventTopic.Job] = lambda data: job(Job(data))
        if eval:
            callbacks[EventTopic.Evaluation] = lambda data: eval(Eval(data))
        if alloc:
            callbacks[EventTopic.Allocation] = lambda data: alloc(Alloc(data))
        if deploy:
            callbacks[EventTopic.Deployment] = lambda data: deploy(Deploy(data))
        assert len(callbacks), f"At least one callback has to be specified"
        return callbacks[self.topic](self.data)


class EventApplier(ABC, Generic[R]):
    @abstractmethod
    def apply_job(self, job: Job) -> R:
        raise NotImplementedError()

    @abstractmethod
    def apply_eval(self, eval: Eval) -> R:
        raise NotImplementedError()

    @abstractmethod
    def apply_alloc(self, alloc: Alloc) -> R:
        raise NotImplementedError()

    @abstractmethod
    def apply_deploy(self, deploy: Deploy) -> R:
        raise NotImplementedError()

    def apply(self, e: Event) -> R:
        if e.is_job():
            return self.apply_job(Job(e.data))
        elif e.is_alloc():
            return self.apply_alloc(Alloc(e.data))
        elif e.is_eval():
            return self.apply_eval(Eval(e.data))
        elif e.is_deployment():
            return self.apply_deploy(Deploy(e.data))
        else:
            raise KeyError(f"{e.topic} not handled")


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
