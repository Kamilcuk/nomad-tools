import contextlib
import datetime
import io
import json
import logging
import multiprocessing
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from functools import partialmethod
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, TypeVar

import click
import clickdc
import clickforward
import dotenv

from . import entry_watch, exit_on_thread_exception, taskexec
from .common_base import NOMAD_NAMESPACE, dict_remove_none, quotearr
from .common_click import EPILOG, help_h_option
from .common_nomad import namespace_option

clickforward.init()
log = logging.getLogger(__name__)
T = TypeVar("T")


def trueornone(x: Optional[T]) -> Optional[T]:
    return x if x else None


@contextlib.contextmanager
def tempfile_with(txt: str):
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(txt)
        f.flush()
        yield f.name


@contextlib.contextmanager
def terminating(p: multiprocessing.Process):
    """Make Process be able to be used in with: clause"""
    try:
        p.start()
        yield p
    finally:
        p.terminate()
        p.join(15)
        if p.exitcode is None:
            p.kill()
            p.join()


def snake_to_camel_case(txt: str) -> str:
    return "".join(x.capitalize() for x in txt.split("_")) if txt.islower() else txt


def timestr_to_nanos(time_str: str) -> int:
    """Convert like 1h into 3600000000"""
    RGX = re.compile(
        r"^((?P<days>[\.\d]+?)d)?((?P<hours>[\.\d]+?)h)?((?P<minutes>[\.\d]+?)m)?((?P<seconds>[\.\d]+?)s)?$"
    )
    parts = RGX.match(time_str)
    assert (
        parts is not None
    ), "Could not parse any time information from '{}'.  Examples of valid strings: '8h', '2d8h5m20s', '2m4s'".format(
        time_str
    )
    time_params = {
        name: float(param) for name, param in parts.groupdict().items() if param
    }
    return int(datetime.timedelta(**time_params).total_seconds()) * 10**9


###############################################################################


class JsonType(click.ParamType):
    name = "JSON"

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> dict:
        """Entrypoint for click option conversion"""
        if not isinstance(value, str):
            return value
        return json.loads(value)


class KvType(click.ParamType):
    name = "K=V"


def strs_to_kv(data: Tuple[str, ...]) -> Dict[str, str]:
    """Split list of strings in the form of k=v to a dictionary"""
    return {
        k: v
        for elem in data
        for k, v in [elem.split("=", 1) if "=" in elem else (elem, "")]
    }


class JsonOrShellKvType(click.ParamType):
    """
    The click argument is a json array if it can be parsed as
    or it is a list of shell quoted values in the form var=val
    """

    name = "JSON_OR_SHELLKV"

    def __init__(
        self,
        map: Dict[str, str] = {},
        snake_to_camel: bool = False,
        transform: Optional[Callable[[dict], Any]] = None,
    ):
        super().__init__()
        self.map = map
        self.snake_to_camel = snake_to_camel
        self.transform = transform

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> dict:
        """Entrypoint for click option conversion"""
        if not isinstance(value, str):
            return value
        try:
            ret = json.loads(value)
        except json.JSONDecodeError:
            ret = {k: v for elem in shlex.split(value) for k, v in [elem.split("=", 1)]}
        ret = {self.map.get(k, k): v for k, v in ret.items()}
        if self.snake_to_camel:
            ret = {snake_to_camel_case(k): v for k, v in ret.items()}
        if self.transform:
            self.transform(ret)
        return ret


class MountNfsType(click.ParamType):
    name = "SRV:SRC:TGT:OPTS"

    def convert(
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> dict:
        """Entrypoint for click option conversion"""
        if not isinstance(value, str):
            return value
        srv, src, tgt, opt = value.split(":", 3)
        return {
            "target": tgt,
            "volume_options": {
                "driver_config": {
                    "options": {
                        "type": "nfs",
                        "device": f"{srv}:{src}",
                        "o": f"addr={srv}" + ("," if opt else "") + opt,
                    }
                }
            },
        }


###############################################################################


@dataclass
class ParsedArgs:
    job: dict
    jobid: str
    cmd: List[str]


NAME_PREFIX = "nomadtoolsgo_"


@dataclass
class Args:
    type: str = clickdc.option(
        default="batch",
        type=click.Choice("service system batch sysbatch".split()),
        help="Nomad job type. Specifies Nomad scheduler to use.",
    )
    cap_add: Tuple[str, ...] = clickdc.option(
        multiple=True, help="Add Linux capabilities"
    )
    cap_drop: Tuple[str, ...] = clickdc.option(
        multiple=True, help="Drop Linux capabilities"
    )
    entrypoint: Optional[str] = clickdc.option(
        help="Overwrite the default ENTRYPOINT of the image"
    )
    expose: Tuple[int, ...] = clickdc.option(
        type=int, multiple=True, help="Expose a port"
    )
    publish: Tuple[str, ...] = clickdc.option(
        "-p",
        multiple=True,
        metavar="SRC:DST",
        help="Publish container port to the host",
    )
    group_add: Optional[str] = clickdc.option(help="Add additional groups to join")
    env: Tuple[str, ...] = clickdc.option(
        "-e", multiple=True, help="Set environment variables"
    )
    env_file: Tuple[IO, ...] = clickdc.option(
        type=click.File(), multiple=True, help="Read in a file of environment variables"
    )
    hostname: Optional[str] = clickdc.option(help="Container host name")
    init: Optional[bool] = clickdc.option(help="Add init=true to config")
    name: Optional[str] = clickdc.option(
        help=f"Job, group and task name. Default: {NAME_PREFIX}<uuid>"
    )
    mount: Tuple[dict, ...] = clickdc.option(
        multiple=True, type=JsonOrShellKvType(), help="Add a mount block"
    )
    identity: Tuple[dict, ...] = clickdc.option(
        type=JsonOrShellKvType(
            snake_to_camel=True,
            map=dict(aud="Audience", ttl="TTL"),
            transform=lambda x: (
                x.update({"TTL": timestr_to_nanos(x["TTL"])})
                if isinstance(x.get("TTL"), str)
                else None
            ),
        ),
        multiple=True,
        help="""
            Add a identity block
            Example: --identity "env=true file=true change_mode=restart"
                --identity '{"name": "example", "aud": ["oidc.example.com"], "file": true, "ttl": 1h", "change_mode": "signal", "change_signal": "SIGHUP"}'
            """,
    )
    vault: Optional[dict] = clickdc.option(
        type=JsonOrShellKvType(snake_to_camel=True),
        help="Add a vault block",
    )
    consul: Optional[dict] = clickdc.option(
        type=JsonOrShellKvType(snake_to_camel=True),
        help="Add a consul block",
    )
    numa: Optional[dict] = clickdc.option(
        type=JsonOrShellKvType(snake_to_camel=True),
        help="""
            Add a numa block.
            Example: --numa affinity=require
            """,
    )
    device: Tuple[dict, ...] = clickdc.option(
        type=JsonType(),
        help="""
            Add a device block to resources of the task.
            Example --device '{"Name":"nvidia/gpu","Count":2,"Constraints":[{"LTarget":"${device.attr.memory}","RTarget":"2 GiB","Operand":">="}],"Affinities":[{"LTarget":"${device.attr.memory}","RTarget":"4 GiB","Operand":">=","Weight":75}]}'
            """,
    )
    cores: Optional[int] = clickdc.option(help="Sepcify cores")
    mountnfs: Tuple[dict, ...] = clickdc.option(
        multiple=True,
        type=MountNfsType(),
        help="""
            Generates a mount block that mounts NFS share from SRV:SRC to TGT using OPTS options.
            """,
    )
    memorymb: Optional[int] = clickdc.option(
        "-m", type=int, help="Memory limit in MegaBytes"
    )
    memorymaxmb: Optional[int] = clickdc.option(
        type=int, help="Memory max limit in MegaBytes"
    )
    privileged: Optional[bool] = clickdc.option(help="Add privileged=true to config")
    force_pull: Optional[bool] = clickdc.option(help="Add force_pull=true to config")
    user: Optional[str] = clickdc.option("-u", help="Specify the user to execute as.")
    tty: Optional[bool] = clickdc.option("-t", help="Add tty to config")
    workdir: Optional[str] = clickdc.option("-w", help="Add work_dir to config")
    volume: Tuple[str, ...] = clickdc.option(
        multiple=True, help="Add volumes to config"
    )
    restart: Optional[str] = clickdc.option(
        type=click.Choice(["always", "no"]),
        default="no",
        help="With restart always, will add always restarting restart policy",
    )
    driver: str = clickdc.option(
        default="docker",
        help="Nomad task driver. One of docker, raw_exec, exec, java, etc.",
    )
    datacenters: Tuple[str, ...] = clickdc.option(
        "--dc",
        multiple=True,
        help="Datacenter to run in. Can be specified multiple times.",
    )
    constraint: Tuple[str, ...] = clickdc.option(
        multiple=True,
        help="""
            List of constrains of 3 elements parsed with shlex.split(). Can be specified multiple times.
            Example: --constarint '${attr.os.name} regexp "Ubuntu"'
            """,
    )
    constraint_here: bool = clickdc.alias_option(
        aliased=dict(
            constraint="${attr.unique.hostname} == " + shlex.quote(socket.gethostname())
        )
    )
    template: Optional[Any] = clickdc.option(
        multiple=True,
        type=JsonOrShellKvType(
            snake_to_camel=True,
            map=dict(
                destination="DestPath",
                data="EmbeddedTmpl",
                source="SourcePath",
                error_on_missing_key="ErrMissingKey",
            ),
        ),
        help="""
            Add template block.
            """,
    )
    auth: Optional[Any] = clickdc.option(
        type=JsonOrShellKvType(),
        help="""
            Add config.auth block.
            Example: --auth "username=dockerhub_user password=dockerhub_password"
            """,
    )
    image_pull_timeout: Optional[str] = clickdc.option(
        help="Add image_pull_timeout to task config"
    )
    purge: Optional[bool] = clickdc.option(
        "--rm", is_flag=True, help="After the job is run, purge the job from Nomad"
    )
    output: int = clickdc.option(
        "-O",
        count=True,
        help="""
            Output the JSON Nomad job specification.
            If specified once, will not run the job, only output.
            If specified twice or more, will show the job and then run it.
            """,
    )
    detach: Optional[bool] = clickdc.option(is_flag=True, help="Run job in background")
    interactive: Optional[bool] = clickdc.option(
        "-i",
        is_flag=True,
        help="""
            Run the job, wait for the job to be started,
            then get the allocation ID of the running task and execute inside it.
            """,
    )
    interactive_foreground: str = clickdc.option(
        default="sleep infinity",
        help="""
            If using foreground, this is the command that will run in the job,
            and interactive terminal will connect using websockets.
            Default: sleep infinity
            """,
    )
    network: Optional[str] = clickdc.option(help="Connect a container to a network")
    group_network_mode: Optional[str] = clickdc.option(
        help="Group network mode configuration"
    )
    cpu: Optional[int] = clickdc.option(type=int)
    image: Optional[str] = clickdc.option(
        help=""" The image to execute. If the driver is docker, podman or containerd,
            the image is taken from the first command line argument. """
    )
    count: Optional[int] = clickdc.option(type=int, help="The group count")
    kill_timeout: Optional[str] = clickdc.option(
        help="""
        Specifies the duration to wait for an application to gracefully quit
        before force-killing. Nomad first sends a kill_signal. If the task does
        not exit before the configured timeout, SIGKILL is sent to the task. Note
        that the value set here is capped at the value set for max_kill_timeout
        on the agent running the task, which has a default value of 30 seconds.
        """
    )
    kill_signal: Optional[str] = clickdc.option(
        help="""
        Specifies a configurable kill signal for a task, where the default
        is SIGINT (or SIGTERM for docker, or CTRL_BREAK_EVENT for raw_exec on
        Windows). Note that this is only supported for drivers sending signals
        (currently docker, exec, raw_exec, and java drivers).
        """
    )

    meta: Tuple[str, ...] = clickdc.option(
        type=KvType(), multiple=True, help="Add job meta"
    )
    group_meta: Tuple[str, ...] = clickdc.option(
        type=KvType(), multiple=True, help="Add group meta"
    )
    task_meta: Tuple[str, ...] = clickdc.option(
        type=KvType(), multiple=True, help="Add task meta"
    )

    extra_config: Any = clickdc.option(
        type=JsonOrShellKvType(), default={}, help="Add extra JSON to config"
    )
    extra_task: Any = clickdc.option(
        type=JsonOrShellKvType(), default={}, help="Add extra JSON to task"
    )
    extra_group: Any = clickdc.option(
        type=JsonOrShellKvType(), default={}, help="Add extra JSON to group"
    )
    extra_job: Any = clickdc.option(
        type=JsonOrShellKvType(), default={}, help="Add extra JSON to job"
    )
    command: Tuple[str, ...] = clickdc.argument(
        required=True, nargs=-1, type=clickforward.FORWARD
    )

    def parse(self) -> ParsedArgs:
        name: str = self.name if self.name else f"{NAME_PREFIX}{uuid.uuid4()}"
        if self.image is None and self.driver in "podman docker containerd".split():
            self.image = self.command[0]
            self.command = self.command[1:]
        cmd: List[str] = []
        if self.interactive:
            assert self.command, "No command to execute"
            cmd = list(self.command)
            self.command = tuple(shlex.split(self.interactive_foreground))
        #
        job = {
            "ID": name,
            "Datacenters": self.datacenters if self.datacenters else None,
            "Type": self.type,
            "Constraints": (
                [
                    {
                        "LTarget": ll,
                        "Operand": oo,
                        "RTarget": rr,
                    }
                    for cc in self.constraint
                    for ll, oo, rr in [shlex.split(cc)]
                ]
                if self.constraint
                else None
            ),
            "Meta": strs_to_kv(self.meta) if self.meta else None,
            "TaskGroups": [
                {
                    "Name": name,
                    "Count": self.count,
                    "Meta": strs_to_kv(self.group_meta) if self.group_meta else None,
                    "Networks": (
                        [
                            {
                                "Mode": self.group_network_mode,
                                "DynamicPorts": (
                                    [
                                        {
                                            "HostNetwork": "default",
                                            "Label": f"port_{port}",
                                            "To": int(port),
                                        }
                                        for port in self.expose
                                    ]
                                    if self.expose
                                    else None
                                ),
                                "ReservedPorts": (
                                    [
                                        {
                                            "HostNetwork": "default",
                                            "Label": f"port_{src}_{dst}",
                                            "To": int(src),
                                            "Value": int(dst),
                                        }
                                        for port in self.publish
                                        for src, dst in [port.split(":", 1)]
                                    ]
                                    if self.publish
                                    else None
                                ),
                            }
                        ]
                        if self.expose or self.publish
                        else None
                    ),
                    "Tasks": [
                        {
                            "Name": name,
                            "Driver": self.driver,
                            "User": self.user,
                            "Meta": (
                                strs_to_kv(self.task_meta) if self.task_meta else None
                            ),
                            "Config": {
                                "entrypoint": (
                                    shlex.split(self.entrypoint)
                                    if self.entrypoint is not None
                                    else None
                                ),
                                "ports": (
                                    [f"port_{port}" for port in self.expose]
                                    + [
                                        f"port_{src}_{dst}"
                                        for port in self.publish
                                        for src, dst in [port.split(":", 1)]
                                    ]
                                    if self.expose or self.publish
                                    else None
                                ),
                                "command": self.command[0] if self.command else None,
                                "args": (
                                    self.command[1:] if len(self.command) > 1 else None
                                ),
                                "image": trueornone(self.image),
                                "volumes": trueornone(self.volume),
                                "init": trueornone(self.init),
                                "privileged": trueornone(self.privileged),
                                "group_add": trueornone(self.group_add),
                                "cap_add": trueornone(self.cap_add),
                                "cap_drop": trueornone(self.cap_drop),
                                "tty": (
                                    None if self.interactive else trueornone(self.tty)
                                ),
                                "force_pull": trueornone(self.force_pull),
                                "work_dir": self.workdir,
                                "image_pull_timeout": self.image_pull_timeout,
                                "auth": self.auth,
                                "hostname": self.hostname,
                                "mounts": (
                                    self.mount + self.mountnfs
                                    if self.mount or self.mountnfs
                                    else None
                                ),
                                "network_mode": self.network,
                                **self.extra_config,
                            },
                            "kill_timeout": self.kill_timeout,
                            "kill_signal": self.kill_signal,
                            "Templates": trueornone(self.template),
                            "Env": (
                                {
                                    **{
                                        k: v
                                        for e in self.env
                                        for k, v in [e.split("=", 1)]
                                    },
                                    **{
                                        k: v
                                        for e in self.env_file
                                        for k, v in dotenv.dotenv_values(
                                            stream=e
                                        ).items()
                                    },
                                }
                                if self.env or self.env_file
                                else None
                            ),
                            "Resources": (
                                {
                                    "MemoryMB": self.memorymb,
                                    "MemoryMaxMB": self.memorymaxmb,
                                    "CPU": self.cpu,
                                    "NUMA": self.numa,
                                    "Device": self.device,
                                    "Cores": self.cores,
                                }
                                if self.memorymb is not None
                                or self.cpu is not None
                                or self.numa is not None
                                or self.memorymaxmb is not None
                                or self.cores is not None
                                or self.device
                                else None
                            ),
                            "Vault": self.vault,
                            "Consul": self.consul,
                            "Identity": self.identity[0] if self.identity else None,
                            "Identities": (
                                self.identity[1:] if len(self.identity) > 1 else None
                            ),
                            **self.extra_task,
                        }
                    ],
                    "RestartPolicy": {"Attempts": 0, "Mode": "fail"},
                    "ReschedulePolicy": (
                        {"Unlimited": True}
                        if self.restart == "always"
                        else {"Attempts": 0, "Unlimited": False}
                    ),
                    **self.extra_group,
                }
            ],
            **self.extra_job,
        }
        return ParsedArgs(job, name, cmd)


class Interactive:
    """Small wrapper to handle ineractivity properly"""

    def __init__(self, par: ParsedArgs, cmd: List[str]):
        self.par = par
        self.rfd, self.wfd = os.pipe()
        threading.Thread(target=self.__thread, daemon=True).start()
        cmd.insert(0, f"--notifystarted={self.wfd}")
        self.proc = None

    def __thread(self):
        try:
            # Wait for started notification from nomad watch.
            os.read(self.rfd, 1)
            os.close(self.rfd)
            #
            at = taskexec.find_job(job=self.par.jobid)
            cmd = [
                "nomad",
                "alloc",
                "exec",
                f"-t={str(bool(ARGS.tty)).lower()}",
                "-task",
                at[1],
                at[0],
            ] + self.par.cmd
            log.info(f"+ {quotearr(cmd)}")
            with subprocess.Popen(cmd) as self.proc:
                self.proc.wait()
        except Exception as e:
            log.exception(e)
            raise
        finally:
            # Stop the main thread from itself.
            os.kill(os.getpid(), signal.SIGINT)

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
            return self.proc.returncode
        return -1


@click.command()
@clickdc.adddc("logoptions", entry_watch.LogOptions)
def default_logoptions(logoptions: entry_watch.LogOptions):
    return logoptions


@click.command(
    "go",
    help="""
Execute one show Nomad job, like docker run.

Given the arguments on the command this command constructs a JSON Nomad job specification.
Then this specification is executed using `nomadt watch run` command.

The JSON_OR_SHELLKV options argument is parsed as a JSON. If it is not a valid JSON,
then the value is split using shlex.split() and parsed as a list of values in the form
val=var. See examples below.
""",
    epilog="""
\b
Examples:
    go --rm hello-world
    go --rm -m 3000 -e NAME=you alpine sh -c 'echo hello $NAME'
    go --rm -OO --meta key=val --group-meta key2=val2 alpine sh -c 'env | grep NOMAD_'
    go --rm -OO --extra-task '{"Meta":{"foo":"bar"}}' alpine sh -c 'env | grep NOMAD_'
    go --constraint '${attr.os.name} regexp Ubuntu' --driver raw_exec sh
    go --cores=3 --identity '{"name": "example", "aud": ["oidc.example.com"], "file": true, "change_mode": "signal", "change_signal": "SIGHUP", "TTL": "1h"}' --driver raw_exec echo hello
    go --rm --template "destination=local/script.sh data=$(printf "%q" "$(cat ./script.sh)")" alpine sh -c 'sh ${NOMAD_TASK_DIR}/script.sh'
"""
    f"{EPILOG}",
)
@help_h_option()
@namespace_option()
@clickdc.adddc("args", Args)
@clickdc.adddc("logoptions", entry_watch.LogOptions)
@clickdc.adddc("notifyargs", entry_watch.NotifyOptions)
@click.option("-v", "--verbose", is_flag=True)
def cli(
    args: Args,
    notifyargs: entry_watch.NotifyOptions,
    logoptions: entry_watch.LogOptions,
    verbose: bool,
):
    global ARGS
    ARGS = args
    if args.command[0].startswith("-"):
        raise click.UsageError(f"No such option: {args.command[0]}")
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    if os.environ.get(NOMAD_NAMESPACE, "*") == "*":
        os.environ[NOMAD_NAMESPACE] = "default"
    exit_on_thread_exception.install()
    log.debug(f"{args}")
    par: ParsedArgs = args.parse()
    # log.debug(f"{par}")
    job: dict = dict_remove_none(par.job)
    log.debug(f"{job}")
    jobjson: str = json.dumps(job, indent=2, sort_keys=True)
    if args.output == 1:
        print(jobjson)
        return
    if args.output >= 2:
        print(jobjson)
    cmd: List[str] = [
        *(["--verbose"] if verbose else []),
        "--attach",
        "--json",
        "-0",
        "--lines=-1",
        *(
            clickdc.to_args(logoptions)
            if logoptions != default_logoptions([], standalone_mode=False)
            else []
        ),
        *clickdc.to_args(notifyargs),
        "--jobarg",
        "start" if args.detach else "run",
        jobjson,
    ]
    if args.purge:
        assert not args.detach, "detach conflicts with purge"
        cmd = ["--purge", *cmd]
    terminal: Optional[Interactive] = None
    if args.interactive:
        assert not args.detach, "interactive with detach doesn't make sense"
        terminal = Interactive(par, cmd)
    log.debug(f"+ nomadtools watch {quotearr(cmd)}")
    try:
        entry_watch.cli.main(args=cmd)
    finally:
        if terminal:
            exit(terminal.stop())


if __name__ == "__main__":
    cli()
