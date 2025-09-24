from __future__ import annotations

import dataclasses
import datetime
import enum
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

import dateutil.parser

from .datadict import DataDict
from .tools import ns2dt

log = logging.getLogger(__name__)


def strdict(__map: Dict[str, Any] = {}, **kvargs):
    """Dictionary to var=val space separated elements"""
    __map.update(kvargs)
    return " ".join(
        f"{k}={int(v) if v is True or v is False else v}"
        for k, v in __map.items()
        if v is not None
    )


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
    volumes: List[str]
    mounts: List[DockerMounts]
    extra_hosts: List[str]
    network_mode: str
    init: bool


class LifecycleHook(MyStrEnum):
    prestart = "prestart"
    poststart = "poststart"
    poststop = "poststop"


class JobTaskLifecycle(DataDict):
    Hook: str
    Sidecar: bool

    def get_sidecar(self) -> bool:
        return bool(self.get("Sidecar", False))


class JobTaskTemplate(DataDict):
    EmbeddedTmpl: str
    LeftDelim: str
    RightDelim: str


class JobTask(DataDict):
    Name: str
    Driver: str
    User: str
    Config: Union[dict, JobTaskConfig]
    Lifecycle: Optional[JobTaskLifecycle]
    Env: Optional[Dict[str, str]]
    Services: Optional[List[Any]]
    Templates: Optional[List[JobTaskTemplate]]


class JobTaskGroup(DataDict):
    Name: str
    Count: int
    Tasks: List[JobTask]
    Services: Optional[List[Any]]


class JobStatus(MyStrEnum):
    pending = "pending"
    """Pending means the job is waiting on scheduling"""
    running = "running"
    """Running means the job has non-terminal allocations"""
    dead = "dead"
    """Dead means all evaluation's and allocations are terminal"""


class _BothJobAndJobsJob(DataDict):
    ID: str
    Status: str
    Namespace: Optional[str]
    Stop: bool
    Meta: Optional[Dict[str, str]] = None
    CreateIndex: int
    ModifyIndex: int
    JobModifyIndex: int

    def is_dead(self):
        return self.Status == JobStatus.dead.value

    def is_pending(self):
        return self.Status == JobStatus.pending.value

    def is_running(self):
        return self.Status == JobStatus.running.value


class Job(_BothJobAndJobsJob):
    """Returned aby job/<id> API. DO NOT mix with JobsJob"""

    Version: int
    ModifyIndex: int
    JobModifyIndex: int
    TaskGroups: List[JobTaskGroup]
    SubmitTime: int

    def SubmitTime_dt(self):
        return ns2dt(self.SubmitTime)

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


class JobsJob(_BothJobAndJobsJob):
    """Returned aby jobs API. DO NOT mix with Job"""

    JobSummary: JobSummary


class NodeScoreMeta(DataDict):
    NodeID: str
    Scores: Dict[str, float]
    NormScore: float


class AllocationMetric(DataDict):
    NodesEvaluated: int
    NodesFiltered: int
    NodesExhausted: int
    NodesAvailable: Optional[Dict[str, int]] = None
    ClassFiltered: Optional[Dict[str, int]] = None
    ConstraintFiltered: Optional[Dict[str, int]] = None
    QuotaExhausted: Optional[Dict[str, int]] = None
    ScoreMetaData: Optional[List[NodeScoreMeta]] = None
    ClassExhausted: Optional[Dict[str, int]] = None
    Scores: Optional[Dict[str, float]] = None
    DimensionExhausted: Optional[Dict[str, int]] = None

    def format(self, scores: bool = False, prefix: str = "") -> str:
        """https://github.com/hashicorp/nomad/blob/484f91b893c6f054e9339f8db6bd157429c2ef20/command/monitor.go#L345"""
        out = []
        if self.NodesEvaluated == 0:
            out += ["No nodes were eligible for evaluation"]
        for dc, available in (self.NodesAvailable or {}).items():
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
    blocked = "blocked"
    pending = "pending"
    complete = "complete"
    failed = "failed"
    canceled = "canceled"


def fromisoformat(txt: str) -> datetime.datetime:
    return dateutil.parser.isoparse(txt)


class Eval(DataDict):
    ID: str
    Namespace: str
    JobID: str
    ModifyIndex: int
    ModifyTime: int
    Status: str
    WaitUntil: str
    FailedTGAllocs: Dict[str, AllocationMetric]
    DeploymentID: Optional[str] = None
    """May be missing when evaluation started by user starting the job"""
    JobModifyIndex: Optional[int] = None
    """May be missing. No idea when"""
    TriggeredBy: Optional[str] = None

    def ModifyTime_dt(self):
        return ns2dt(self.ModifyTime)

    def is_pending_or_blocked(self):
        return self.Status in [EvalStatus.pending, EvalStatus.blocked]

    def is_blocked(self):
        return self.Status == EvalStatus.blocked.value

    def is_finished(self) -> bool:
        return not self.is_pending_or_blocked()

    def getWaitUntil(self) -> Optional[datetime.datetime]:
        if not self.WaitUntil:
            return None
        return (
            fromisoformat(self.WaitUntil)
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
    Message: str
    DisplayMessage: str
    Time: int
    Type: str
    ExitCode: Optional[int] = None


class AllocTaskStateType(MyStrEnum):
    pending = "pending"  # The task is waiting to be run.
    running = "running"  # The task is currently running.
    dead = "dead"  # Terminal state of task.


class AllocTaskState(DataDict):
    Events: Optional[List[AllocTaskStateEvent]]
    State: str
    Failed: bool
    FinishedAt: Optional[str]
    LastRestart: Optional[str]
    Restarts: int
    StartedAt: Optional[str]

    def find_event(
        self, type_: Union[str, AllocTaskStateEventType]
    ) -> Optional[AllocTaskStateEvent]:
        """Find event in TaskStates task Events. Return empty dict if not found"""
        return next((e for e in self.Events or [] if e.Type == type_), None)

    def was_started(self) -> bool:
        return self.find_event(AllocTaskStateEventType.TaskStarted) is not None

    def get_task_started_event(self) -> Optional[AllocTaskStateEvent]:
        return self.find_event(AllocTaskStateEventType.TaskStarted)


class AllocClientStatus(MyStrEnum):
    pending = "pending"
    running = "running"
    complete = "complete"
    failed = "failed"
    lost = "lost"
    unknown = "unknown"


class Alloc(DataDict):
    ID: str
    Name: str
    NodeName: str
    NodeID: str
    JobID: str
    EvalID: str
    ClientStatus: str
    DesiredStatus: str
    CreateTime: int
    Namespace: str
    ModifyIndex: int
    ModifyTime: int
    TaskGroup: str
    # Also may be missing!
    TaskStates: Optional[Dict[str, AllocTaskState]] = None
    # May be missing!
    JobVersion: Optional[int] = None
    FollowupEvalID: Optional[str] = None

    def strshort(self):
        return f"{self.__class__.__name__}({self.ID[:6]} {strdict(JobVersion=self.JobVersion)})"

    def ModifyTime_dt(self):
        return ns2dt(self.ModifyTime)

    def get_taskstates(self) -> Dict[str, AllocTaskState]:
        """The same as TaskStates but returns an empty dict in case the field is None"""
        return self.get("TaskStates") or {}

    def get_taskstate(self, task: str) -> Optional[AllocTaskState]:
        return self.get_taskstates()[task]

    def get_tasknames(self) -> List[str]:
        return list(self.get_taskstates().keys())

    def is_pending_or_running(self):
        return self.ClientStatus in [
            AllocClientStatus.pending,
            AllocClientStatus.running,
        ]

    def is_pending(self):
        return self.ClientStatus == AllocClientStatus.pending.value

    def is_running(self):
        return self.ClientStatus == AllocClientStatus.running.value

    def any_was_started(self):
        return any(
            taskstate.was_started() for taskstate in self.get_taskstates().values()
        )

    def is_running_started(self):
        return self.is_running() and self.any_was_started()

    def is_finished(self):
        return not self.is_pending_or_running()

    def any_oom_killed(self):
        return any(
            event.Type == "Terminated" and "OOM Killed" in (event.Message or "")
            for taskstate in self.get_taskstates().values()
            for event in (taskstate.Events or [])
        )


class DeploymentStatus(MyStrEnum):
    running = "running"
    paused = "paused"
    failed = "failed"
    successful = "successful"
    cancelled = "cancelled"
    initializing = "initializing"
    pending = "pending"
    blocked = "blocked"
    unblocking = "unblocking"


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
    ProgressDeadline: int
    Promoted: bool
    RequireProgressBy: Optional[str]
    UnhealthyAllocs: int
    PlacedCanaries: Optional[List[str]] = None


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

    def is_finished(self):
        return self.Status in [
            DeploymentStatus.cancelled,
            DeploymentStatus.failed,
            DeploymentStatus.successful,
        ]


class EventTopic(MyStrEnum):
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
    """Parsed Event from Nomad event stream"""

    index: int
    """The index received with the event"""
    topic: EventTopic
    type: EventType
    data: dict

    def __str__(self):
        status = dict(
            ID=self.data.get("ID", "")[:6],
            JobID=self.data.get("JobID"),
            Version=self.data.get("Version"),
            JobVersion=self.data.get("JobVersion"),
            JobModifyIndex=self.data.get("JobModifyIndex"),
            ModifyIndex=self.data.get("ModifyIndex"),
            Status=self.data.get("Status"),
            ClientStatus=self.data.get("ClientStatus"),
            TriggeredBy=self.data.get("TriggeredBy"),
        )
        statusstr = " ".join(f"{k}={v}" for k, v in status.items() if v is not None)
        return f"Event({self.index} {self.topic.name}.{self.type.name} {statusstr})"

    def job(self) -> Job:
        return Job(self.data)

    def eval(self) -> Eval:
        return Eval(self.data)

    def alloc(self) -> Alloc:
        return Alloc(self.data)

    def deployment(self) -> Deploy:
        return Deploy(self.data)

    def is_job(self):
        return self.topic == EventTopic.Job

    def is_eval(self):
        return self.topic == EventTopic.Evaluation

    def is_alloc(self):
        return self.topic == EventTopic.Allocation

    def is_deployment(self):
        return self.topic == EventTopic.Deployment

    def apply(
        self,
        job: Callable[[Job], R],
        eval: Callable[[Eval], R],
        alloc: Callable[[Alloc], R],
        deploy: Callable[[Deploy], R],
    ) -> R:
        callbacks: Dict[EventTopic, Callable[[dict], R]] = {
            EventTopic.Job: lambda data: job(Job(data)),
            EventTopic.Evaluation: lambda data: eval(Eval(data)),
            EventTopic.Allocation: lambda data: alloc(Alloc(data)),
            EventTopic.Deployment: lambda data: deploy(Deploy(data)),
        }
        assert len(callbacks), "At least one callback has to be specified"
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
    ModifyIndex: int


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
            self[k] = cast(int, self.get(k, 0)) + cast(int, o.get(k, 0))
        return self

    def only_completed(self):
        return (
            self.Queued == 0
            and self.Complete != 0
            and self.Failed == 0
            and self.Running == 0
            and self.Starting == 0
            and self.Lost == 0
        )


class JobSummary(DataDict):
    JobID: str
    CreateIndex: int
    ModifyIndex: int
    Summary: Optional[Dict[str, JobSummarySummary]] = None
    Children: Optional[JobSummaryChildren] = None

    def get_sum_summary(self) -> JobSummarySummary:
        """Sum all summaries into one"""
        ret = JobSummarySummary()
        for s in (self.Summary or {}).values():
            ret += s
        return ret
