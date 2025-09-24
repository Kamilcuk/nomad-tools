import logging
import queue
import threading
from typing import Callable, Dict, Iterable, List, Optional, TypeVar, Union

import requests

from . import flagdebug, nomadlib
from .common import eprint, json_loads, mynomad
from .nomadlib import Event, EventTopic, EventType
from .nomadlib.types import JobStatus

log = logging.getLogger(__name__)
T = TypeVar("T")


class NomadDbJob:
    """Represents relevant state cache from Nomad database of a single job"""

    def __init__(
        self,
        topics: List[str],
        select_event_cb: Callable[[Event], bool],
        init_cb: Callable[[], List[Event]],
        force_polling: Optional[bool] = None,
    ):
        """
        :param topic The topics to listen to, see Nomad event stream API documentation.
        :param select_event_cb: Filter only relevant events from Nomad event stream.
        :param init_cb Return a list of events to populate the database with and to poll. Has to be threadsafe!
        """
        self.topics: List[str] = topics
        self.init_cb: Callable[[], List[Event]] = init_cb
        self.select_event_cb: Callable[[Event], bool] = select_event_cb
        self.force_polling = force_polling
        #
        self.queue: queue.Queue[Optional[List[Event]]] = queue.Queue()
        """Queue where database thread puts the received events"""
        self.job_deregistered_ModifyIndex: int = -1
        """Is set to the ModifyIndex of last job deregistration evaluation"""
        self.job: Optional[nomadlib.Job] = None
        """Watched job definition. Is not None means the job was at least once received."""
        self.jobversions: Dict[int, nomadlib.Job] = {}
        """Job version to job definition"""
        self.evaluations: Dict[str, nomadlib.Eval] = {}
        """Database of evaluations"""
        self.allocations: Dict[str, nomadlib.Alloc] = {}
        """Database of allocations"""
        self.deployments: Dict[str, nomadlib.Deploy] = {}
        """Database of deployments"""
        self.initialized = threading.Event()
        """Was init_cb called to initiliaze the database"""
        self.stopevent = threading.Event()
        """If set, the database thread should exit"""
        self.thread = threading.Thread(
            target=self.__thread_entry, name=self.__class__.__name__, daemon=True
        )
        assert self.topics
        assert not any(not x for x in topics)

    ###############################################################################

    def __thread_run_stream(self):
        log.debug(f"Starting listen Nomad stream with {' '.join(self.topics)}")
        with mynomad.stream(
            "event/stream",
            params={"topic": self.topics},
        ) as stream:
            for line in stream.iter_lines(decode_unicode=True):
                if line == "event dropped from buffer":
                    # https://github.com/hashicorp/nomad/blob/main/nomad/stream/event_buffer.go#L273
                    log.error(line)
                elif line:
                    data = json_loads(line)
                    if data:
                        events: List[Event] = [
                            Event(
                                data["Index"],
                                EventTopic[event["Topic"]],
                                EventType[event["Type"]],
                                event["Payload"][event["Topic"]],
                            )
                            for event in data.get("Events", [])
                        ]
                        if flagdebug.debug("recv") == 1:
                            for e in events:
                                eprint(f"RECVEVENTS:{data['Index']} {e}")
                        elif flagdebug.debug("recv") >= 2:
                            for e in events:
                                eprint(f"RECVEVENTS:{data['Index']} {e} {e.data}")
                        self.queue.put(events)
                    else:
                        self.queue.put([])
                else:
                    self.queue.put([])
                if self.stopevent.is_set():
                    break

    def __thread_poll(self):
        """If listening to stream fails, we fallback to calling init_cb in a loop"""
        # Delay polling start up until init_cb has been called.
        self.initialized.wait()
        while not self.stopevent.wait(1):
            self.queue.put(self.init_cb())

    def __thread_entry(self):
        """Database thread entry"""
        try:
            if self.force_polling is True:
                self.__thread_poll()
            else:
                try:
                    self.__thread_run_stream()
                except nomadlib.PermissionDenied as e:
                    if self.force_polling is False:
                        raise
                    else:
                        log.warning(
                            f"Falling to polling method because stream API returned permission denied: {e}"
                        )
                self.__thread_poll()
        except requests.HTTPError:
            log.exception("http request failed")
            exit(1)
        finally:
            log.debug("Nomad database thread exiting")
            self.queue.put(None)

    ###############################################################################

    def start(self):
        """Start the database thread"""
        assert (
            mynomad.namespace
        ), "Nomad namespace has to be set before starting to listen"
        self.thread.start()

    def select_is_in_db(self, e: Event):
        """Select events that are already in the database"""
        if e.topic == EventTopic.Evaluation:
            return e.data["ID"] in self.evaluations
        elif e.topic == EventTopic.Allocation:
            return e.data["ID"] in self.allocations
        elif e.topic == EventTopic.Deployment:
            return e.data["ID"] in self.deployments
        elif e.topic == EventTopic.Job:
            # Get the first available job ID to compare.
            jobid = (
                self.job.ID
                if self.job is not None
                else next((job.ID for job in self.jobversions.values()), None)
            )
            # Explicitly filter against job ID. Also compare versions.
            return (jobid is None or e.data["ID"] == jobid) and e.data[
                "Version"
            ] in self.jobversions
        return False

    def _add_event_to_db(self, e: Event):
        """Update database state to reflect received event"""
        actionstr = ""
        if e.topic == EventTopic.Job:
            job = e.job()
            self.jobversions[job.Version] = job
            if self.job is None or job.ModifyIndex >= self.job.ModifyIndex:
                # self.job follows newest modify index.
                self.job = job
                if e.type == EventType.JobDeregistered:
                    if self.job_deregistered_ModifyIndex < job.ModifyIndex:
                        pass
                        # self.job_deregistered_ModifyIndex = job.ModifyIndex
                        # actionstr = "JobDeregistered because job"
        elif e.topic == EventTopic.Evaluation:
            eval = e.eval()
            # Handle evaluation which results in job deregistration.
            if (
                self.job
                and eval.Status == "complete"
                and (
                    e.type == EventType.JobDeregistered
                    or (
                        e.type == EventType.EvaluationUpdated
                        and eval.TriggeredBy == "job-deregister"
                    )
                )
                and self.job_deregistered_ModifyIndex < eval.ModifyIndex
                and self.job.Status in [JobStatus.dead, JobStatus.pending]
                # and not eval.DeploymentID
            ):
                self.job_deregistered_ModifyIndex = eval.ModifyIndex
                actionstr = "JobDeregister because eval "
            actionstr += f"{eval}"
            self.evaluations[e.data["ID"]] = eval
        elif e.topic == EventTopic.Allocation:
            self.allocations[e.data["ID"]] = e.alloc()
        elif e.topic == EventTopic.Deployment:
            self.deployments[e.data["ID"]] = e.deployment()
        return actionstr

    def handle_event(self, e: Event) -> bool:
        if self._select_new_event(e):
            if self.select_is_in_db(e) or self.select_event_cb(e):
                actionstr = self._add_event_to_db(e)
                if flagdebug.debug("events") == 1:
                    eprint(f"EVENT: {e} {actionstr}")
                if flagdebug.debug("events") >= 2:
                    eprint(f"EVENT: {e} {e.data} {actionstr}")
                return True
            else:
                if flagdebug.debug("db"):
                    log.debug(f"EVENTFILTERED: {e}")
                pass
        else:
            if flagdebug.debug("db"):
                log.debug(f"OLDEVENT: {e}")
            pass
        return False

    def handle_events(self, events: List[Event]) -> List[Event]:
        """From a list of events, filter out ignored and add the rest to database"""
        return [e for e in events if self.handle_event(e)]

    @staticmethod
    def apply_selects(
        e: Event,
        job_select: Callable[[nomadlib.Job], bool],
        eval_select: Callable[[nomadlib.Eval], bool],
        alloc_select: Callable[[nomadlib.Alloc], bool],
        deploy_select: Callable[[nomadlib.Deploy], bool],
    ) -> bool:
        """Apply specific selectors depending on event type"""
        return e.apply(job_select, eval_select, alloc_select, deploy_select)

    def _select_new_event(self, e: Event):
        """Select events which are newer than those in the database"""

        def job_select(_: nomadlib.Job) -> bool:
            return True

        def eval_select(eval: nomadlib.Eval) -> bool:
            return (
                eval.ID not in self.evaluations
                or eval.ModifyIndex > self.evaluations[eval.ID].ModifyIndex
            )

        def alloc_select(alloc: nomadlib.Alloc) -> bool:
            return (
                alloc.ID not in self.allocations
                or alloc.ModifyIndex > self.allocations[alloc.ID].ModifyIndex
            )

        def deploy_select(deploy: nomadlib.Deploy) -> bool:
            return (
                deploy.ID not in self.deployments
                or deploy.ModifyIndex > self.deployments[deploy.ID].ModifyIndex
            )

        return e.data["Namespace"] == mynomad.namespace and self.apply_selects(
            e, job_select, eval_select, alloc_select, deploy_select
        )

    def stop(self):
        log.debug("Stopping listen Nomad stream")
        self.initialized.set()
        self.stopevent.set()
        self.queue.put(None)

    def join(self):
        # Not joining - neither requests nor stream API allow for timeouts.
        # self.thread.join()
        pass

    def events(self) -> Iterable[List[Event]]:
        """Nomad stream returns Events array. Iterate over batches of events returned from Nomad stream"""
        assert self.thread.is_alive(), "Thread not alive"
        if not self.initialized.is_set():
            events = self.init_cb()
            events = self.handle_events(events)
            self.initialized.set()
            yield events
        log.debug("Starting getting events from thread")
        for events in iter(self.queue.get, None):
            yield self.handle_events(events)
        log.debug("db exiting")

    def get_allocation_jobmodifyindex(
        self, alloc: nomadlib.Alloc, default: T = None
    ) -> Union[T, int]:
        """Given an allocation, return associated JobModifyIndex"""
        if alloc.JobVersion is not None:
            job = self.jobversions.get(alloc.JobVersion)
            if job:
                return job.JobModifyIndex
        evaluation = self.evaluations.get(alloc.EvalID)
        if evaluation:
            if evaluation.JobModifyIndex is not None:
                return evaluation.JobModifyIndex
            if evaluation.DeploymentID is not None:
                deployment = self.deployments.get(evaluation.DeploymentID)
                if deployment:
                    return deployment.JobModifyIndex
        return default

    def get_evaluation_jobversion(
        self, eval: nomadlib.Eval, default: T = None
    ) -> Union[T, int]:
        # If we can find Job from JobModifyIndex, lets find it.
        if eval.JobModifyIndex is not None:
            version = self.find_jobversion_from_modifyindex(eval.JobModifyIndex)
            if version is not None:
                return version
        # Otherwise we may find deployment that may have jobversion.
        if eval.DeploymentID is not None:
            deployment = self.deployments.get(eval.DeploymentID)
            if deployment:
                return deployment.JobVersion
        return default

    def get_allocation_jobversion(
        self, alloc: nomadlib.Alloc, default: T = None
    ) -> Union[T, int]:
        """Given an allocation return the job version associated with that allocation"""
        if False:
            # This is unreliable.
            if alloc.JobVersion:
                return alloc.JobVersion
        evaluation = self.evaluations.get(alloc.EvalID)
        if evaluation:
            return self.get_evaluation_jobversion(evaluation, default)
        return default

    def find_job_from_modifyindex(self, jobmodifyindex: int) -> Optional[nomadlib.Job]:
        # Note that job versions may not be in JobModifyIndex order.
        # Job versions of previous job (after purge and start) may be here.
        # Sort explicitly over JobModifyIndex
        for job in sorted(
            self.jobversions.values(), key=lambda job: job.JobModifyIndex, reverse=True
        ):
            if job.JobModifyIndex <= jobmodifyindex:
                return job
        return None

    def find_jobversion_from_modifyindex(self, jobmodifyindex: int) -> Optional[int]:
        ret = self.find_job_from_modifyindex(jobmodifyindex)
        return ret.Version if ret else None

    def seen_job(self) -> bool:
        """Check if the job event type has been received at least once from the event stream"""
        return self.job is not None

    def job_purged(self) -> bool:
        """Return True if the job was purged"""
        # When a job is _stopped_, then within the same events array are received:
        # - Evaluation.EvaluationUpdated TriggeredBy="job-deregister"
        # - and Job.EvaluationUpdated Status=dead
        # In contrast, when a job is _purged_,
        # then only Evaluation.EvaluationUpdated is received.
        # For a job to be purged, the modify index of evaluation job deregister
        # has to be not equal to job modify index.
        return (
            self.job is not None
            and self.job_deregistered_ModifyIndex > self.job.ModifyIndex
        )

    def send_empty_event(self):
        """Send an empty event to trigger the loop if needed by the user"""
        self.queue.put([])
