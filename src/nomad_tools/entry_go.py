import contextlib
import datetime
import io
import json
import logging
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
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

import click
import clickdc
import clickforward
import dotenv

from . import nomad_watch, taskexec
from .common_base import NOMAD_NAMESPACE, quotearr
from .common_click import EPILOG, common_options
from .common_nomad import namespace_option

clickforward.init()
log = logging.getLogger(__name__)
T = TypeVar("T")


def dict_remove_none(data: T) -> T:
    """Remove all elements that are set to None"""
    if isinstance(data, dict):
        ret = {
            k: dict_remove_none(v)
            for k, v in data.items()
            if v is not None and v is not {}
        }
    elif isinstance(data, list):
        ret = [dict_remove_none(e) for e in data if e is not None]
    else:
        ret = data
    return cast(T, ret)


@contextlib.contextmanager
def tempfile_with(txt: str):
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(txt)
        f.flush()
        yield f.name


class redirect_stdin_str(contextlib.AbstractContextManager):
    def __init__(self, txt: str):
        self.txt = txt

    def __enter__(self):
        self.old = sys.stdin
        sys.stdin = self.new = io.StringIO(self.txt)
        return self.new

    def __exit__(self, exctype, excinst, exctb):
        self.new.close()
        sys.stdin = self.old


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
        return json.loads(value) if isinstance(value, str) else value


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
class Parsed:
    job: dict
    jobid: str
    cmd: List[str]


NAME_PREFIX = "nomad_tools_go_"


@dataclass
class Args:
    name: Optional[str] = clickdc.option(help="Set the name of the job, group and task")
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
        "-p", multiple=True, help="Publish container port to the host"
    )
    group_add: Optional[str] = clickdc.option(help="Add additional groups to join")
    env: Tuple[str, ...] = clickdc.option(
        "-e", multiple=True, help="Set environment variables"
    )
    env_file: Tuple[IO, ...] = clickdc.option(
        type=click.File(), multiple=True, help="Read in a file of environment variables"
    )
    hostname: Optional[str] = clickdc.option(help="Container host name")
    init: bool = clickdc.option(is_flag=True, help="Add init=true to config")
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
            transform=lambda x: x.update({"TTL": timestr_to_nanos(x["TTL"])})
            if isinstance(x.get("TTL"), str)
            else None,
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
    privileged: bool = clickdc.option(
        is_flag=True, help="Add privileged=true to config"
    )
    pull: Optional[str] = clickdc.option(
        type=click.Choice(["always", "missing"]),
        help="When always, add force_pull=true to config",
    )
    user: Optional[str] = clickdc.option("-u", help="Specify the user to execute as.")
    tty: bool = clickdc.option("-t", is_flag=True, help="Add tty to config")
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
    purge: bool = clickdc.option(
        "--rm", is_flag=True, help="After the job is run, purge the job from Nomad"
    )
    output: bool = clickdc.option(
        "-O",
        is_flag=True,
        help="Instead of running the job, output Nomad job JSON specification.",
    )
    detach: bool = clickdc.option(is_flag=True, help="Run job in background")
    interactive: bool = clickdc.option(
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
    #
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

    def parse(self) -> Parsed:
        name: str = self.name if self.name else f"{NAME_PREFIX}{uuid.uuid4()}"
        if self.image is None and self.driver in "podman docker containerd".split():
            self.image = self.command[0]
            self.command = self.command[1:]
        cmd: List[str] = []
        if self.interactive:
            assert self.command, "No command to execute"
            cmd = list(self.command)
            self.command = tuple(shlex.split(self.interactive_foreground))
            self.init = True
        #
        job = {
            "ID": name,
            "Datacenters": self.datacenters if self.datacenters else None,
            "Type": self.type,
            "Constraints": [
                {
                    "LTarget": ll,
                    "Operand": oo,
                    "RTarget": rr,
                }
                for cc in self.constraint
                for ll, oo, rr in [shlex.split(cc)]
            ]
            if self.constraint
            else None,
            "TaskGroups": [
                {
                    "Name": name,
                    "Count": self.count,
                    "Networks": (
                        [
                            {
                                "Mode": self.group_network_mode,
                                "DynamicPorts": [
                                    {
                                        "HostNetwork": "default",
                                        "Label": f"port_{port}",
                                        "To": int(port),
                                    }
                                    for port in self.expose
                                ]
                                if self.expose
                                else None,
                                "ReservedPorts": [
                                    {
                                        "HostNetwork": "default",
                                        "Label": f"port_{src}_{dst}",
                                        "Value": int(dst),
                                    }
                                    for port in self.publish
                                    for src, dst in [port.split(":", 1)]
                                ]
                                if self.publish
                                else None,
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
                            "Config": {
                                "entrypoint": shlex.split(self.entrypoint)
                                if self.entrypoint is not None
                                else None,
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
                                "args": self.command[1:]
                                if len(self.command) > 1
                                else None,
                                "image": self.image if self.image else None,
                                "volumes": self.volume if self.volume else None,
                                "init": self.init if self.init else None,
                                "privileged": self.privileged
                                if self.privileged
                                else None,
                                "group_add": self.group_add if self.group_add else None,
                                "cap_add": self.cap_add if self.cap_add else None,
                                "cap_drop": self.cap_drop if self.cap_drop else None,
                                "tty": self.tty if self.tty else None,
                                **(
                                    {"force_pull": True}
                                    if self.pull == "always"
                                    else {}
                                ),
                                "work_dir": self.workdir,
                                "image_pull_timeout": self.image_pull_timeout,
                                "auth": self.auth,
                                "hostname": self.hostname,
                                "mounts": self.mount + self.mountnfs
                                if self.mount or self.mountnfs
                                else None,
                                "network_mode": self.network,
                                **self.extra_config,
                            },
                            "kill_timeout": self.kill_timeout,
                            "kill_signal": self.kill_signal,
                            "Templates": (self.template if self.template else None),
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
                            "RestartPolicy": {"Attempts": 0, "Mode": "fail"},
                            "Vault": self.vault,
                            "Consul": self.consul,
                            "Identity": self.identity[0] if self.identity else None,
                            "Identities": self.identity[1:]
                            if len(self.identity) > 1
                            else None,
                            **self.extra_task,
                        }
                    ],
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
        return Parsed(job, name, cmd)


class Interactive:
    """Small wrapper to handle ineractivity properly"""

    def __init__(self, par: Parsed):
        self.par = par
        self.returncode: Optional[int] = None
        """The returncode returned from interactive command"""

    def setup(self):
        self.rfd, wfd = os.pipe()
        threading.Thread(target=self.__thread).start()
        return wfd

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
            ret = subprocess.run(cmd)
            self.returncode = ret.returncode
        except Exception as e:
            log.exception(e)
            raise
        finally:
            # Stop the main thread from itself.
            os.kill(os.getpid(), signal.SIGINT)


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
        Runs hello-world docker image.
    go --rm -m 3000 -e NAME=you alpine sh -c 'echo hello $NAME'
        Runs alpine docker imge with 3G of memory and NAME environment variable
    go --constraint '${attr.os.name} regexp Ubuntu' --driver raw_exec sh
    go --cores=3 --identity '{"name": "example", "aud": ["oidc.example.com"], "file": true, "change_mode": "signal", "change_signal": "SIGHUP", "TTL": "1h"}' --driver raw_exec echo hello
    go --rm --template "destination=local/script.sh data=$(printf "%q" "$(cat ./script.sh)")" alpine sh -c 'sh ${NOMAD_TASK_DIR}/script.sh'
"""
    f"{EPILOG}",
)
@common_options()
@namespace_option()
@clickdc.adddc("args", Args)
@clickdc.adddc("notifyargs", nomad_watch.NotifyOptions)
@click.option("-v", "--verbose", is_flag=True)
def cli(args: Args, notifyargs: nomad_watch.NotifyOptions, verbose: bool):
    global ARGS
    ARGS = args
    if args.command[0].startswith("-"):
        raise click.UsageError(f"No such option: {args.command[0]}")
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    if os.environ.get(NOMAD_NAMESPACE, "*") == "*":
        os.environ[NOMAD_NAMESPACE] = "default"
    log.debug(f"{args}")
    par: Parsed = args.parse()
    log.debug(f"{par}")
    job: dict = dict_remove_none(par.job)
    log.debug(f"{job}")
    jobjson: str = json.dumps(job, indent=2, sort_keys=True)
    if args.output:
        print(jobjson)
    else:
        cmd: List[str] = [
            *(["--verbose"] if verbose else []),
            "--attach",
            "--json",
            "-0",
            "--lines=-1",
            *clickdc.to_args(notifyargs),
            "start" if args.detach else "run",
            "-",
        ]
        if args.purge:
            assert not args.detach, "detach conflicts with purge"
            cmd = ["--purge", *cmd]
        if args.interactive:
            assert not args.detach, "interactive with detach doesn't make sense"
            wfd = Interactive(par).setup()
            cmd = [f"--notifyfdstarted={wfd}", *cmd]
        with redirect_stdin_str(jobjson):
            log.debug(f"+ {quotearr(cmd)}")
            nomad_watch.cli.main(args=cmd)


if __name__ == "__main__":
    cli()
