from __future__ import annotations

import fcntl
import json
import logging
import re
import string
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import click
import clickdc
from click.shell_completion import CompletionItem
from packaging.version import Version

from .common import mynomad
from .common_click import EPILOG, help_h_option, verbose_option
from .mytabulate import mytabulate

log = logging.getLogger(__name__)


###############################################################################


@dataclass
class Context:
    """The context used to evaluate operator"""

    attribute: str
    value: str
    node: Dict[str, str]
    """Node attributes"""


def notimplemented(c: Context):
    raise NotImplementedError()


def op_version(c: Context):
    ops: Dict[str, Callable[[Version, Version], bool]] = {
        "=": lambda aa, bb: aa == bb,
        "!=": lambda aa, bb: aa != bb,
        ">": lambda aa, bb: aa > bb,
        ">=": lambda aa, bb: aa >= bb,
        "<": lambda aa, bb: aa < bb,
        "<=": lambda aa, bb: aa <= bb,
    }
    if not c.attribute:
        return False
    ver = Version(c.attribute)
    for part in c.value.split(","):
        op, val = part.split()
        if not ops[op](ver, Version(val)):
            return False
    return True


def op_semver(c: Context):
    return op_version(c)


REFERENCES: List[str] = []
"""List of referenced attributes during evaluation"""

OPERATORS: Dict[str, Callable[[Context], bool]] = {
    "=": lambda c: c.attribute == c.value,
    "!=": lambda c: c.attribute != c.value,
    ">": lambda c: c.attribute > c.value,
    ">=": lambda c: c.attribute >= c.value,
    "<": lambda c: c.attribute < c.value,
    "<=": lambda c: c.attribute <= c.value,
    "regexp": lambda c: bool(re.search(c.value, c.attribute)),
    "set_contains": lambda c: c.value in c.attribute.split(","),
    "set_contains_any": lambda c: any(
        v in c.attribute.split(",") for v in c.value.split(",")
    ),
    "version": op_version,
    "semver": op_semver,
    "is_set": lambda c: (REFERENCES.append(c.attribute), c.attribute in c.node)[1],
    "is_not_set": lambda c: (REFERENCES.append(c.attribute), c.attribute not in c.node)[
        1
    ],
}
"""List of operators"""

###############################################################################


def operator_complete(
    ctx: click.Context, param: click.Parameter, incomplete: str
) -> List[CompletionItem]:
    print(ctx.params, file=sys.stderr)
    print(ctx.args, file=sys.stderr)
    print("", file=sys.stderr)
    arr = list(OPERATORS.keys())
    arr = [x for x in arr if incomplete.startswith(x)]
    return [CompletionItem(x) for x in arr]


@dataclass
class NodeCacheArgs:
    no_cache: bool = clickdc.option(help="Set to disable cache", show_default=True)
    cache: Path = clickdc.option(
        type=click.Path(dir_okay=False, writable=True, path_type=Path),
        default=Path().home() / ".cache/nomadtools/nodes.json",
        help="Cache file location",
        show_default=True,
    )
    cachetime: float = clickdc.option(
        help="Number of seconds the cache if valid for.",
        default=600.0,
        show_default=True,
    )
    json: bool = clickdc.option(
        "-j",
        help="Output matched nodes information with attributes in json",
        show_default=True,
    )
    parallel: int = clickdc.option(
        "-P",
        default=20,
        help="When getting all nodes metadata, make this many connections in parallel.",
        show_default=True,
    )
    verbose: bool = clickdc.option(
        "-v",
        show_default=True,
    )


@dataclass
class ConstraintArgs:
    constraints: Tuple[str, ...] = clickdc.argument(
        required=True,
        nargs=-1,
    )


###############################################################################


class Interpolater(string.Template):
    pattern: str = str(  # pyright: ignore [reportIncompatibleVariableOverride]
        r"""
        \${(?P<braced>[^}]*)} |
        (?P<escaped>\x00)  |  # match nothing
        (?P<named>\x00)    |  # match nothing
        (?P<invalid>\x00)     # match nothing
        """
    )


class DictTrackDefault(Dict[str, str]):
    def __getitem__(self, key: str) -> str:
        REFERENCES.append(key)
        return self.get(key, "")


def interpolate(txt: str, attributes: Dict[str, str]) -> str:
    return Interpolater(txt).substitute(DictTrackDefault(attributes))


###############################################################################


@dataclass(frozen=True)
class Constraint:
    attribute: str = ""
    operator: str = "is_set"
    value: str = ""


def key_prefix(prefix: str, data: Dict[str, str]) -> Dict[str, str]:
    """Add a prefix to dictionary"""
    return {prefix + k: v for k, v in data.items()}


def dotflatten(
    data: Union[dict, list, str, float, int], key: str = ""
) -> Dict[str, str]:
    """Given any dictionary joins all keys with dot and returns it.
    So like {"a":{"b":[{"c": "d"}]}} becomes {"a.b[0].c": "d"}
    """
    if isinstance(data, dict):
        return {
            a: b
            for k, v in data.items()
            for a, b in dotflatten(v, key + ("." if key else "") + str(k)).items()
        }
    elif isinstance(data, list):
        return {
            a: b
            for i, v in enumerate(data)
            for a, b in dotflatten(v, key + f"[{i}]").items()
        }
    else:
        return {key: str(data)}


@dataclass(frozen=True)
class NodeAttributes:
    node: dict
    attributes: Dict[str, str]


class NodesAttributes(List[NodeAttributes]):
    @staticmethod
    def __download(args: NodeCacheArgs) -> NodesAttributes:
        nodes: List[dict] = mynomad.get("nodes")
        nodesid = [x["ID"] for x in nodes]
        with ThreadPool(args.parallel) as pool:
            nodenodes: List[dict] = pool.map(
                lambda id: mynomad.get(f"node/{id}"), nodesid
            )
        ret = NodesAttributes(
            NodeAttributes(
                node,
                {
                    "node.unique.id": node["ID"],
                    "node.region": "global",
                    "node.datacenter": node["Datacenter"],
                    "node.unique.name": node["Name"],
                    "node.class": node["NodeClass"],
                    "node.pool": node["NodePool"],
                    **key_prefix("attr.", node["Attributes"]),
                    **key_prefix("meta.", node["Meta"]),
                    **key_prefix(".", dotflatten(node)),
                },
            )
            for node in nodenodes
        )
        return ret

    def __save_to_cache(self, args: NodeCacheArgs):
        if args.no_cache:
            return
        data = {
            "timestamp": time.time(),
            "nodes": [asdict(x) for x in self],
        }
        try:
            args.cache.parent.mkdir(exist_ok=True)
            with args.cache.open("w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(data, f)
        except Exception:
            log.exception(f"Error saving nodes cache to {args.cache}")

    @staticmethod
    def __load_cache(args: NodeCacheArgs) -> Optional[NodesAttributes]:
        if args.no_cache:
            return
        try:
            try:
                with args.cache.open() as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    data = json.load(f)
            except FileNotFoundError:
                return None
            timestamp = float(data["timestamp"])
            nodes = NodesAttributes(NodeAttributes(**x) for x in data["nodes"])
            if timestamp + args.cachetime > time.time():
                log.debug(
                    f"Using cache with timestamp {timestamp} and {len(nodes)} node"
                )
                return nodes
            else:
                log.debug(
                    f"Not using cache with timestamp {timestamp} and {len(nodes)} node, too old"
                )
                return None
        except Exception:
            log.exception(f"Error loading nodes cache from {args.cache}")

    @classmethod
    def load(cls, args: NodeCacheArgs) -> NodesAttributes:
        nodes = cls.__load_cache(args)
        if nodes:
            return nodes
        nodes = cls.__download(args)
        nodes.__save_to_cache(args)
        return nodes


###############################################################################


def grouper(thelist: List[str], count: int) -> List[List[str]]:
    return [thelist[i : i + count] for i in range(0, len(thelist), count)]


@click.command(
    "constrainteval",
    help="""
    Evaluate the constaint without running a job.
    Output only matched nodes with referenced attributes.

    The CONSTRAINTS argument is grouped in groups of 3 elements. Each group
    is then assigned to 'attribute', 'operator' and 'value' of a constaint in order
    and then evaluated. Watch out for shell quoting.

    The default 'operator' is 'is_set' and default 'value' is empty ''. For operators
    that evaluate only 'attribute', the 'value' value is ignored.

    The 'regexp' is evaluted using python 're' module.

    Additional attributes with a leading dot are available.
    These are the fields from JSON Nomad API response as-is.

    \b
    Examples:
        %(prog)s attr.os.name
        %(prog)s attr.os.name is_set
        %(prog)s '${attr.os.name}' = ubuntu '${attr.kernel.name}' = linux
        %(prog)s '${attr.os.name}' '!=' ubuntu attr.os.name is_set
        %(prog)s attr.os.name is_set '' attr.kernel.name is_set
    """
    % dict(prog="nomadtools constrainteval"),
    epilog=EPILOG,
)
@clickdc.adddc("args", NodeCacheArgs)
@clickdc.adddc("constraintsargs", ConstraintArgs)
@verbose_option()
@help_h_option()
def cli(args: NodeCacheArgs, constraintsargs: ConstraintArgs):
    return main(args, constraintsargs)


def main(args: NodeCacheArgs, constraintsargs: ConstraintArgs):
    logging.basicConfig()
    nodesattributes = NodesAttributes.load(args)
    # Group attributes in groups of 3 for constraints.
    for group in grouper(list(constraintsargs.constraints), 3):
        constraint = Constraint(*group)
        # Remove nodes that do match constraint by evaluating it.
        newlist: List[NodeAttributes] = []
        for node in nodesattributes:
            context = Context(
                attribute=interpolate(constraint.attribute, node.attributes),
                value=interpolate(constraint.value, node.attributes),
                node=node.attributes,
            )
            log.debug(
                f"Evaluating {context.attribute!r} {constraint.operator!r} {context.value!r}"
                f" for {node.node['Name']!r}"
            )
            try:
                opfunc = OPERATORS[constraint.operator]
            except KeyError:
                exit(
                    f"No such operator: {constraint.operator!r}. Must be one of: {' '.join(OPERATORS.keys())}"
                )
            if opfunc(context):
                newlist.append(node)
        nodesattributes = newlist
    # Output.
    if not nodesattributes:
        exit(2)
    global REFERENCES
    REFERENCES = sorted(list(set(REFERENCES)))
    nodesattributes = sorted(
        nodesattributes, key=lambda x: x.attributes["node.unique.name"]
    )
    if args.json:
        print(json.dumps([asdict(x) for x in nodesattributes]))
    else:
        toout: List[List[str]] = [
            [
                "name",
                *[node.attributes["node.unique.name"] for node in nodesattributes],
            ],
            *[
                [
                    k,
                    *[node.attributes.get(k, "") for node in nodesattributes],
                ]
                for k in REFERENCES
            ],
        ]
        print(mytabulate(toout))


if __name__ == "__main__":
    cli()
