import json
import logging
import queue
import threading
from typing import Callable, Dict, Iterable, List, Optional, TypeVar, Union

import requests

from . import nomadlib
from .common import mynomad
from .nomadlib import Event, EventTopic, EventType

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
        self.queue: queue.Queue[Optional[List[Event]]] = queue.Queue()
        """Queue where database thread puts the received events"""
        self.job: Optional[nomadlib.Job] = None
        """Watched job"""
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
            target=self.__thread_entry, name="db", daemon=True
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
            for line in stream.iter_lines():
                if line:
                    data = json.loads(line)
                    events: List[Event] = [
                        Event(
                            EventTopic[event["Topic"]],
                            EventType[event["Type"]],
                            event["Payload"][event["Topic"]],
                            stream=True,
                        )
                        for event in data.get("Events", [])
                    ]
                    # log.debug(f"RECV EVENTS: {events}")
                    self.queue.put(events)
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
            finally:
                log.debug("Nomad database thread exiting")
                self.queue.put(None)
        except requests.HTTPError as e:
            log.exception("http request failed")
            exit(1)

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
            return e.data["Version"] in self.jobversions
        return False

    def _add_event_to_db(self, e: Event):
        """Update database state to reflect received event"""
        if e.topic == EventTopic.Job:
            job = nomadlib.Job(e.data)
            self.jobversions[job.get("Version")] = job
            self.job = None if e.type == EventType.JobDeregistered else job
        elif e.topic == EventTopic.Evaluation:
            if e.type == EventType.JobDeregistered:
                self.job = None
            self.evaluations[e.data["ID"]] = nomadlib.Eval(e.data)
        elif e.topic == EventTopic.Allocation:
            # Events from event stream are missing JobVersion. Try to preserve it here by preserving keys.
            self.allocations[e.data["ID"]] = nomadlib.Alloc(
                {**self.allocations.get(e.data["ID"], {}), **e.data}
            )
        elif e.topic == EventTopic.Deployment:
            self.deployments[e.data["ID"]] = nomadlib.Deploy(e.data)

    def handle_event(self, e: Event) -> bool:
        if self._select_new_event(e):
            if self.select_is_in_db(e) or self.select_event_cb(e):
                # log.debug(f"EVENT: {e}")
                # log.debug(f"EVENT: {e} {e.data}")
                self._add_event_to_db(e)
                return True
            else:
                # log.debug(f"USER FILTERED: {e}")
                pass
        else:
            # log.debug(f"OLD EVENT: {e}")
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
        job_select: Callable[[nomadlib.Job], bool] = lambda _: True
        eval_select: Callable[[nomadlib.Eval], bool] = (
            lambda eval: eval.ID not in self.evaluations
            or eval.ModifyIndex > self.evaluations[eval.ID].ModifyIndex
        )
        alloc_select: Callable[[nomadlib.Alloc], bool] = (
            lambda alloc: alloc.ID not in self.allocations
            or alloc.ModifyIndex > self.allocations[alloc.ID].ModifyIndex
        )
        deploy_select: Callable[[nomadlib.Deploy], bool] = (
            lambda deploy: deploy.ID not in self.deployments
            or deploy.ModifyIndex > self.deployments[deploy.ID].ModifyIndex
        )
        return e.data["Namespace"] == mynomad.namespace and self.apply_selects(
            e, job_select, eval_select, alloc_select, deploy_select
        )

    def stop(self):
        log.debug("Stopping listen Nomad stream")
        self.initialized.set()
        self.stopevent.set()

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
            yield events
            self.initialized.set()
        log.debug("Starting getting events from thread")
        while not self.queue.empty() or (
            self.thread.is_alive() and not self.stopevent.is_set()
        ):
            events = self.queue.get()
            if events is None:
                break
            yield self.handle_events(events)
        log.debug("db exiting")

    def get_allocation_job_version(
        self, alloc: nomadlib.Alloc, default: T = None
    ) -> Union[T, int]:
        """Given an allocation return the job version associated with that allocation"""
        if "JobVersion" in alloc:
            return alloc.JobVersion
        evaluation = self.evaluations.get(alloc.EvalID)
        if not evaluation:
            return default
        jobversion = self.find_jobversion_from_modifyindex(evaluation.JobModifyIndex)
        if jobversion is not None:
            alloc.JobVersion = jobversion
            return jobversion
        return default

    def find_job_from_modifyindex(self, jobmodifyindex: int) -> Optional[nomadlib.Job]:
        for _, job in sorted(self.jobversions.items(), reverse=True):
            if job.JobModifyIndex <= jobmodifyindex:
                return job
        return None

    def find_jobversion_from_modifyindex(self, jobmodifyindex: int) -> Optional[int]:
        ret = self.find_job_from_modifyindex(jobmodifyindex)
        return ret.Version if ret else None
