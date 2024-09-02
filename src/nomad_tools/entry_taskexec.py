from __future__ import annotations

import enum
import json
import logging
import os
import string
import sys
from dataclasses import asdict, dataclass, fields
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import click
import clickdc
import clickforward

from . import common, common_click, nomadlib
from .common_base import cached_property
from .common_nomad import mynomad, nomad_find_job

log = logging.getLogger(__name__)


def filterquote(txt: str):
    # quote string for nomad filter expression
    # https://developer.hashicorp.com/nomad/api-docs#values
    replaces: Dict[str, str] = {"\\": "\\\\", '"': '\\"', "\n": "\\n"}
    for a, b in replaces.items():
        txt = txt.replace(a, b)
    return '"' + txt + '"'


def filterstartswith(key: str, txt: str) -> str:
    txt = "".join([f"[{x}]" for x in txt])
    return f'({key} matches "^{filterquote(txt)}")'


@dataclass
class TaskAlloc:
    task: str
    allocid: str


def findtask_completor(
    cb: Callable[[FindTask], Iterable[str]],
    ctx: click.Context,
    param: str,
    incomplete: str,
) -> List[str]:
    try:
        common_click.complete_set_namespace(ctx)
        self = FindTask(**{f.name: ctx.params.get(f.name) for f in fields(FindTask)})
        return sorted(list(x for x in cb(self) if x.startswith(incomplete)))
    except Exception as e:
        if 0:
            print(e, file=sys.stderr)
        return []


class StartsWith(enum.Enum):
    none = enum.auto()
    alloc = enum.auto()
    node = enum.auto()
    nodeid = enum.auto()
    job = enum.auto()
    group = enum.auto()
    task = enum.auto()


@dataclass
class FilterConfig:
    startswith: StartsWith
    key: str
    allocfiltereq: Callable[[nomadlib.Alloc, str], bool]
    allocfilterstartswith: Callable[[nomadlib.Alloc, str], bool]


@dataclass
class FindTask:
    alloc: Optional[str] = clickdc.option(
        "-a",
        shell_complete=lambda *a: findtask_completor(FindTask.complete_alloc, *a),
        help="Find allocation starting with this ID",
    )
    job: Optional[str] = clickdc.option(
        "-j",
        shell_complete=lambda *a: findtask_completor(FindTask.complete_job, *a),
        help="Find jobs matching this name.",
    )
    node: Optional[str] = clickdc.option(
        "-n",
        shell_complete=lambda *a: findtask_completor(FindTask.complete_node, *a),
        help="Filter allocation only running on nodes with this node name",
    )
    nodeid: Optional[str] = clickdc.option(
        shell_complete=lambda *a: findtask_completor(FindTask.complete_nodeid, *a),
        help="Filter allocation only running on nodes with this ID.",
    )
    group: Optional[str] = clickdc.option(
        "-g",
        shell_complete=lambda *a: findtask_completor(FindTask.complete_group, *a),
        help="Filter allocation only running task group named like that.",
    )
    task: Optional[str] = clickdc.option(
        "-t",
        shell_complete=lambda *a: findtask_completor(FindTask.complete_task, *a),
        help="Filter allocation only running task named like that.",
    )

    def is_none(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))

    def complete_alloc(self) -> List[str]:
        allocs = self.get_allocations(StartsWith.alloc)
        return [x.ID for x in allocs]

    def complete_job(self):
        allocs = self.get_allocations(StartsWith.job)
        return [x.JobID for x in allocs]

    def complete_node(self) -> List[str]:
        allocs = self.get_allocations(StartsWith.node)
        return [x["NodeName"] for x in allocs]

    def complete_nodeid(self) -> List[str]:
        allocs = self.get_allocations(StartsWith.nodeid)
        return [x["NodeID"] for x in allocs]

    def complete_group(self) -> List[str]:
        allocs = self.get_allocations(StartsWith.group)
        tasks = [a.TaskGroup for a in allocs]
        return tasks

    def complete_task(self) -> List[str]:
        allocs = self.get_allocations(StartsWith.task)
        tasks = [t for a in allocs for t in a.get_tasknames()]
        return tasks

    def get_allocations(
        self,
        startswith: StartsWith = StartsWith.none,
    ) -> List[nomadlib.Alloc]:
        """Return running allocations with ID starting with self.alloc"""
        # List of checks that can be done with simple comparison.
        simplechecks = [
            (StartsWith.nodeid, "NodeID", self.nodeid),
            (StartsWith.node, "NodeName", self.node),
            (StartsWith.job, "JobID", self.job),
            (StartsWith.group, "TaskGroup", self.group),
        ]
        # Construct filter expressionf or Nomad API.
        filter: str = " and ".join(
            [
                '(ClientStatus == "running")',
                "(TaskStates is not empty)",
                *([f'("{self.task}" in TaskStates)'] if self.task else []),
                *[
                    (
                        filterstartswith(key, value)
                        if startswith == mark
                        else f"({key} == {filterquote(value)})"
                    )
                    for mark, key, value in simplechecks
                    if value
                ],
            ]
        )
        # If self.alloc is set, filter on it.
        params = dict(filter=filter)
        if self.alloc:
            alloc = "".join(c for c in self.alloc if c in string.hexdigits)
            # It is only possible to prefix using even length. Query uneven length using filter expression.
            prefix = alloc[: len(alloc) // 2 * 2]
            if prefix:
                params["prefix"] = prefix
        log.debug(f"GET allocations {params}")
        # Finally query allocations.
        allocations: List[nomadlib.Alloc] = [
            nomadlib.Alloc(x) for x in mynomad.get("allocations", params=params)
        ]
        allocations = [
            alloc
            for alloc in allocations
            if alloc.is_running()
            and len(alloc.get_taskstates()) != 0
            and all(
                (
                    not value
                    or (
                        alloc[key].startswith(value)
                        if startswith == mark
                        else alloc[key] == value
                    )
                )
                for mark, key, value in simplechecks
            )
            and (not self.alloc or alloc.ID.startswith(self.alloc))
            and (
                not self.task
                or (
                    any(x.startswith(self.task) for x in alloc.get_tasknames())
                    if startswith == StartsWith.task
                    else self.task in alloc.get_tasknames()
                )
            )
        ]
        return allocations

    @cached_property
    def __find_jobid(self):
        assert self.job
        return nomad_find_job(self.job)

    def findtask(self) -> TaskAlloc:
        allocs = self.get_allocations()
        assert len(allocs) > 0, f"Found no running allocations matching {self}"
        assert len(allocs) == 1, f"Found multiple running allocations matching {self}"
        alloc = allocs[0]
        #
        tasks = alloc.get_tasknames()
        tasks = [task for task in tasks if not self.task or task == self.task]
        assert len(tasks) > 0, f"Found no tasks matching {self}"
        assert len(tasks) == 1, f"Multiple tasks found matching {self}: {tasks}"
        task = tasks[0]
        return TaskAlloc(task=task, allocid=alloc.ID)


@dataclass
class Cmd:
    json: bool = clickdc.option(help="Json output. Command is ignored")
    cp: bool = clickdc.option(
        help="For use like: 'nomadtools cp $(nomadtools findtask --cp ...):/this ...'"
    )
    stdin: Optional[bool] = clickdc.option(
        " /--nostdin",
        " /-I",
        help="Pass -i=true or -i=false to nomad alloc exec",
        default=None,
    )
    tty: Optional[bool] = clickdc.option(
        " /--notty", help="Pass -t=true or -t=false to nomad alloc exec", default=None
    )
    escape: Optional[str] = clickdc.option("-e", help="See nomad alloc exec --help")
    dryrun: bool = clickdc.option(
        help="Print nomad alloc exec command instead of executing it"
    )
    command: Tuple[str, ...] = clickdc.argument(nargs=-1, type=clickforward.FORWARD)

    def execute(self, taskalloc: TaskAlloc):
        cmd = [
            "nomad",
            "alloc",
            "exec",
            # f"-namespace={mynomad.namespace}",
            f"-task={taskalloc.task}",
            *([f"-i={str(self.stdin).lower()}"] if self.stdin is not None else []),
            *([f"-t={str(self.tty).lower()}"] if self.tty is not None else []),
            *([f"-e={self.escape}"] if self.escape else []),
            taskalloc.allocid,
            *self.command,
        ]
        if self.json:
            print(json.dumps(asdict(taskalloc)))
        if self.cp:
            print(":" + taskalloc.allocid + ":" + taskalloc.task.replace(":", "\\:"))
        if self.dryrun:
            print(common.quotearr(cmd))
        elif not self.command:
            print(taskalloc.allocid, taskalloc.task)
        else:
            print(f"+ {common.quotearr(cmd)}", file=sys.stderr)
            os.execvp(cmd[0], cmd)


@click.command(
    "taskexec",
    help="""
    Given command line aruments finds a matching running allocation and task.
    If COMMAND is given execute nomad alloc exec with that command.

    \b
    Example:
        Execute bash login shell inside system job 'promtail' running on node 'host1':
            %(prog)s -j promtail -n host1 bash -l
    """
    % dict(prog="nomadtools taskexec"),
)
@clickdc.adddc("findtask", FindTask)
@clickdc.adddc("cmd", Cmd)
@common_click.common_options()
@common_click.verbose_option()
def cli(findtask: FindTask, cmd: Cmd):
    logging.basicConfig()
    if findtask.is_none():
        opts = " ".join("--" + f.name for f in fields(findtask))
        click.get_current_context().fail(f"At least one option has to be present: {opts}")
    f = findtask.findtask()
    cmd.execute(f)
