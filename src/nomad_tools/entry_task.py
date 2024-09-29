from __future__ import annotations

import enum
import json
import logging
import os
import shlex
import string
import sys
from dataclasses import dataclass, fields
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import click
import click.shell_completion
import clickdc
import clickforward
from typing_extensions import get_args, get_origin

from . import common, common_click, nomadlib
from .aliasedgroup import AliasedGroup
from .common_nomad import mynomad, nomad_find_job
from .entry_cp import ArgPath, NomadMypath

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
    alloc: nomadlib.Alloc

    def asdict(self):  # noqa: F811
        return {"task": self.task, "alloc": self.alloc.asdict()}


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


def is_optional(field):
    return get_origin(field) is Union and type(None) in get_args(field)


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
    multiple: Optional[bool] = clickdc.option(
        "-m", help="Allow returning multiple tasks, instead of only one"
    )
    notonlyrunning: Optional[bool] = clickdc.option(
        "-N", help="Find also tasks that are not running"
    )

    @property
    def onlyrunning(self) -> bool:
        return not self.notonlyrunning

    @classmethod
    def get_filter_options(cls) -> List[str]:
        return [f.name for f in fields(cls) if f.type is not bool]

    def has_no_filters(self) -> bool:
        return all(getattr(self, name) is None for name in self.get_filter_options())

    def complete_alloc(self) -> List[str]:
        allocs = self.find_allocations(StartsWith.alloc)
        return [x.ID for x in allocs]

    def complete_job(self):
        allocs = self.find_allocations(StartsWith.job)
        return [x.JobID for x in allocs]

    def complete_node(self) -> List[str]:
        allocs = self.find_allocations(StartsWith.node)
        return [x["NodeName"] for x in allocs]

    def complete_nodeid(self) -> List[str]:
        allocs = self.find_allocations(StartsWith.nodeid)
        return [x["NodeID"] for x in allocs]

    def complete_group(self) -> List[str]:
        allocs = self.find_allocations(StartsWith.group)
        tasks = [a.TaskGroup for a in allocs]
        return tasks

    def complete_task(self) -> List[str]:
        allocs = self.find_allocations(StartsWith.task)
        tasks = [t for a in allocs for t in a.get_tasknames()]
        return tasks

    def find_allocations(
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
        # Get the initial list of matching allocations.
        if self.job and not self.alloc:
            # If job is known and allocation is not know,
            # then we should prefer jobs api, because it is hashed search over job name.
            job = nomad_find_job(self.job)
            allocationsraw = mynomad.get(f"job/{nomadlib.urlquote(job)}/allocations")
        else:
            # Construct filter expression for Nomad API.
            filter: str = " and ".join(
                [
                    *(['(ClientStatus == "running")'] if self.notonlyrunning else []),
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
            # Query allocations for the jobs.
            allocationsraw = mynomad.get("allocations", params=params)
        #
        allocations: List[nomadlib.Alloc] = [nomadlib.Alloc(x) for x in allocationsraw]
        allocations = [
            alloc
            for alloc in allocations
            # Filter only running allocations.
            if (alloc.is_running() if self.onlyrunning else True)
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

    def find_tasks(self) -> List[TaskAlloc]:
        allocs = self.find_allocations()
        assert len(allocs) > 0, f"Found no running allocations matching {self}"
        if not self.multiple:
            assert (
                len(allocs) == 1
            ), f"Found multiple running allocations matching {self}"
        #
        ret: List[TaskAlloc] = []
        for alloc in allocs:
            tasks = alloc.get_tasknames()
            tasks = [task for task in tasks if not self.task or task == self.task]
            assert len(tasks) > 0, f"Found no tasks matching {self}"
            assert len(tasks) == 1, f"Multiple tasks found matching {self}: {tasks}"
            task = tasks[0]
            ret.append(TaskAlloc(task=task, alloc=alloc))
        return ret


TASKS: List[TaskAlloc]


@click.command(
    "task",
    cls=AliasedGroup,
    help="""
Find task inside an allocation given command line arguments and execute an action in it.
""",
)
@clickdc.adddc("findtask", FindTask)
@common_click.help_h_option()
@common_click.verbose_option()
def cli(findtask: FindTask):
    logging.basicConfig()
    if findtask.has_no_filters():
        opts = " ".join("--" + name for name in findtask.get_filter_options())
        click.get_current_context().fail(
            f"At least one option has to be present: {opts}"
        )
    global FINDTASK
    FINDTASK = findtask


###############################################################################


@dataclass
class Cmd:
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
        "-n", help="Print nomad alloc exec command instead of executing it"
    )
    command: Tuple[str, ...] = clickdc.argument(
        nargs=-1, type=clickforward.FORWARD, required=True
    )

    def run(self, taskalloc: TaskAlloc):
        cmd = [
            "nomad",
            "alloc",
            "exec",
            # f"-namespace={mynomad.namespace}",
            f"-task={taskalloc.task}",
            *([f"-i={str(self.stdin).lower()}"] if self.stdin is not None else []),
            *([f"-t={str(self.tty).lower()}"] if self.tty is not None else []),
            *([f"-e={self.escape}"] if self.escape else []),
            "--",
            taskalloc.alloc.ID,
            *self.command,
        ]
        if self.dryrun:
            print(common.quotearr(cmd))
        else:
            log.info(f"+ {common.quotearr(cmd)}")
            os.execvp(cmd[0], cmd)


@cli.command(
    "exec",
    help="""
Execute a command inside the allocation.

\b
Examples:
    nomadtools task -j mail exec bash -l
""",
)
@common_click.help_h_option()
@clickdc.adddc("cmd", Cmd)
def mode_exec(cmd: Cmd):
    for t in FINDTASK.find_tasks():
        cmd.run(t)


@cli.command(
    "xargs",
    help="""
Output in the form -task <task> <allocid> that is usable with xargs nomad alloc

\b
Examples:
  nomadtools task -j mail xargs -0 logs -- -stderr | xargs -0 nomad alloc
  nomadtools task -j mail xargs logs -- -stderr | xargs nomad alloc
  nomad alloc logs $(nomadtools task -j mail xargs) -stderr
""",
)
@common_click.help_h_option()
@click.option("-0", "--zero", is_flag=True)
@click.argument("args", nargs=-1, type=clickforward.FORWARD)
def mode_xargs(zero: bool, args: Tuple[str, ...]):
    for t in FINDTASK.find_tasks():
        out = ["-task=" + t.task, t.alloc.ID, *args]
        if zero:
            print("\0".join(out))
        else:
            print(" ".join(shlex.quote(x) for x in out))


@cli.command(
    "json",
    help="Output found allocations and task names in json form",
)
@common_click.help_h_option()
def mode_json():
    for t in FINDTASK.find_tasks():
        print(json.dumps(t.asdict()))


@cli.command(
    "ls",
    help="Output found allocations and task names",
)
@common_click.help_h_option()
def mode_ls():
    for t in FINDTASK.find_tasks():
        print(t.alloc.ID, t.task)


def task_path_completor(
    ctx: click.Context, param: click.Parameter, incomplete: str
) -> List[click.shell_completion.CompletionItem]:
    try:
        assert ctx.parent
        # Problem: I need to set FINDTASK to something.
        # Solution: just call parent with parent arguments.
        cli.invoke(ctx.parent)
        t = FINDTASK.find_tasks()[0]
        path = NomadMypath(incomplete, t.alloc.ID, t.task)
        return ArgPath.compgen_nomadmypath(path)
    except Exception:
        return []


@cli.command(
    "path",
    help="""
Output in the form properly escaped for use with nomadtools cp.

\b
Examples:
    nomadtools cp "$(nomadtools task -j mail path /etc/fstab)" ./fstab
""",
)
@common_click.help_h_option()
@click.argument("path", default="", shell_complete=task_path_completor)
def mode_print(path: str):
    for t in FINDTASK.find_tasks():
        print(
            ":"
            + t.alloc.ID
            + ":"
            + t.task.replace(":", "\\:")
            + ":"
            + path.replace(":", "\\:")
        )
