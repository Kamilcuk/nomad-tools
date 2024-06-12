from __future__ import annotations

import os
import re
import shlex
import string
import sys
import textwrap
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from click.shell_completion import CompletionItem

from . import nomadlib
from .common_base import cached_property
from .common_nomad import mynomad, nomad_find_job


@dataclass
class NomadTaskLocation:
    """Represents splitted up parts of the input"""

    _description = textwrap.dedent(
        """\
        :ALLOCATION[:TASK][:[PATH]]
        JOB[:[GROUP][:TASK]][:[PATH]]
        task://JOB[@NAMESPACE][/PATH][?group=GROUP][&alloc=ALLOC][&task=TASK][&hostname=HOSTNAME][&node=NODE]
        """
    )

    arg: str
    path: Optional[str] = None
    arr: Optional[List[str]] = None
    alloc: Optional[str] = None
    job: Optional[str] = None
    group: Optional[str] = None
    task: Optional[str] = None
    namespace: Optional[str] = None
    hostname: Optional[str] = None
    node: Optional[str] = None

    @staticmethod
    def mk(arg: str) -> NomadTaskLocation:
        """Converts the argument into it's parts represented with object"""
        if not arg.startswith("task://"):
            # Split string on un-escaped colon, and then replaced secaped colons by colons
            arr: List[str] = [x.replace(r"\:", ":") for x in re.split(r"(?<!\\):", arg)]
            assert len(arr) <= 4, f"Could not parse argument: too many colons: {arg!r}"
            path: str = arr[-1]
            if len(arr) > 1:
                if arr[0] == "":
                    return NomadTaskLocation(
                        arg=arg,
                        arr=arr,
                        path=path,
                        alloc=arr[1],
                        task=arr[2] if len(arr) > 3 else None,
                    )
                else:
                    return NomadTaskLocation(
                        arg=arg,
                        arr=arr,
                        path=path,
                        job=arr[0],
                        group=arr[1] if len(arr) > 2 else None,
                        task=arr[2] if len(arr) > 3 else None,
                    )
            return NomadTaskLocation(arg=arg, arr=arr, path=path)
        o = urlparse(arg)
        assert o.scheme == "task", "Internal error"
        assert o.port is None, f"Port in URL doesn't make sense: {arg!r}"
        qsl: Dict[str, List[str]] = parse_qs(o.query)
        qs: Dict[str, str] = {k: v[-1] for k, v in qsl.items()}
        fields = "alloc task group hostname node".split()
        for k in qs.keys():
            assert k in fields, f"Unknown URL parameter {k}: {arg!r}"
        return NomadTaskLocation(
            arg=arg,
            path=o.path[1:] if o.path and o.path[0] == "/" else o.path,
            job=o.username if o.username else o.hostname,
            namespace=o.hostname if o.username else None,
            arr=None,
            **qs,
        )

    def __post_init__(self):
        assert (
            (not self.alloc and not self.job)
            or (self.alloc and not self.job)
            or (not self.alloc and self.job)
        ), f"Internal error: initialized in both alloc and job mode: {self.arg!r} {self.alloc} {self.job}"
        if self.alloc:
            alloweddigits = string.hexdigits + "-"
            assert all(
                c in alloweddigits for c in self.alloc
            ), f"Allocation ID can only be one of {alloweddigits!r}: {self.alloc}"

    @cached_property
    def __find_jobid(self):
        assert self.job
        return nomad_find_job(self.job)

    @cached_property
    def __allocations(self):
        """Return running allocations with ID starting with self.alloc"""
        assert self.alloc is not None
        # It is only possible to prefix using even length. Query uneven length using filter expression.
        alloc = "".join(c for c in self.alloc if c in string.hexdigits)
        params = dict(
            prefix=alloc[: len(alloc) // 2 * 2],
            filter=f'ID matches "^{self.alloc}"' if len(alloc) % 2 != 0 else None,
        )
        allocations: List[nomadlib.Alloc] = [
            nomadlib.Alloc(x) for x in mynomad.get("allocations", params=params)
        ]
        return self.__filter_alocations(allocations)

    def __filter_alocations(
        self, allocations: List[nomadlib.Alloc]
    ) -> List[nomadlib.Alloc]:
        return [
            alloc
            for alloc in allocations
            if alloc.is_running()
            and len(alloc.get_taskstates()) != 0
            and (not self.alloc or alloc.ID.startswith(self.alloc))
            and (not self.task or self.task in alloc.get_tasknames())
            and (not self.group or alloc.TaskGroup == self.group)
            and (not self.hostname or alloc.NodeName == self.hostname)
            and (not self.node or alloc.NodeID == self.node)
        ]

    @cached_property
    def __allocation(self):
        """Return the single running allocation with ID starting with self.alloc"""
        assert self.alloc
        allocs = self.__allocations
        assert len(allocs) > 0, f"Found no running allocations matching {self.arg!r}"
        assert (
            len(allocs) == 1
        ), f"Found multiple running allocations matching {self.arg!r}"
        return allocs[0]

    @staticmethod
    def __filter(
        arr: List[str], prefix: str, suffix: str = ":", addnospace: bool = True
    ) -> List[CompletionItem]:
        """Filter list of string by prefix and convert to CompletionItem. Also remove duplicates"""
        nospace = CompletionItem("", type="nospace")
        ret = [
            CompletionItem(shlex.quote(f"{x}{suffix}"))
            for x in sorted(list(set(x for x in arr if x.startswith(prefix))))
        ]
        if ret and addnospace:
            ret = [nospace] + ret
        return ret

    @staticmethod
    def debug(msg: str):
        if os.environ.get("COMP_DEBUG"):
            print(f"\n{msg}\n", file=sys.stderr)

    @classmethod
    def debugexception(cls, e: Exception):
        cls.debug(f"{traceback.format_exc()} Exception: {e}")

    def __complete_job_name(self, last: str, suffix: str = ":"):
        jobs = [nomadlib.Job(x) for x in mynomad.get("jobs", params=dict(prefix=last))]
        jobs = [
            x
            for x in jobs
            if not x.is_dead()
            and (not self.group or self.group in [g.Name for g in x.TaskGroups])
            and (
                not self.task
                or self.task in [t.Name for g in x.TaskGroups for t in g.Tasks]
            )
        ]
        return self.__filter([x.ID for x in jobs], last, suffix=suffix)
