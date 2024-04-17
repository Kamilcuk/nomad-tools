import json
import logging
import re
import string
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import Callable, Dict, List, Optional, Tuple

import click
import clickdc
import clickforward
from packaging.version import Version

from .common import mynomad

log = logging.getLogger(__name__)


@dataclass
class Context:
    attribute: str
    value: str
    node: Dict[str, str]


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
    ver = Version(c.attribute)
    for part in c.value.split(","):
        for op, val in part.split(" ", 1):
            if not ops[op](ver, Version(val)):
                return False
    return True


def op_semver(c: Context):
    return op_version(c)


REFERENCES: List[str] = []

OPERATORS: Dict[str, Callable[[Context], bool]] = {
    "=": lambda c: c.attribute == c.value,
    "!=": lambda c: c.attribute != c.value,
    ">": lambda c: c.attribute > c.value,
    ">=": lambda c: c.attribute >= c.value,
    "<": lambda c: c.attribute < c.value,
    "<=": lambda c: c.attribute <= c.value,
    "distinct_hosts": notimplemented,
    "distinct_property": notimplemented,
    "regexp": lambda c: bool(re.search(c.attribute, c.value)),
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
    log.debug(txt)
    return Interpolater(txt).substitute(DictTrackDefault(attributes))


@dataclass
class Constraint:
    attribute: str = ""
    operator: str = "="
    value: str = ""


@dataclass
class Args:
    parallel: int = clickdc.option(
        "-P",
        default=20,
        help="When getting all nodes metadata, make this many connections in parallel",
    )
    constraints: Tuple[str, ...] = clickdc.argument(
        required=True, nargs=-1, type=clickforward.FORWARD
    )


def key_prefix(prefix: str, data: Dict[str, str]) -> Dict[str, str]:
    return {prefix + k: v for k, v in data.items()}


@dataclass
class NodesAtributes:
    data: dict
    attributes: Dict[str, str]


def get_nodes_attributes(threadpool: int) -> List[NodesAtributes]:
    nodes: List[dict] = mynomad.get("nodes")
    nodesid = [x["ID"] for x in nodes]
    with ThreadPool(threadpool) as pool:
        nodenodes: List[dict] = pool.map(lambda id: mynomad.get(f"node/{id}"), nodesid)
    ret = [
        NodesAtributes(
            node,
            {
                **key_prefix("attr.", node["Attributes"]),
                **key_prefix("meta.", node["Meta"]),
                "node.class": node["NodeClass"],
                "node.unique.name": node["Name"],
                "node.datacenter": node["Datacenter"],
                "node.region": "global",
                "node.unique.id": node["ID"],
            },
        )
        for node in nodenodes
    ]
    return ret


def grouper(thelist: List[str], count: int) -> List[List[str]]:
    return [thelist[i : i + count] for i in range(0, len(thelist), count)]


@click.command(
    "constrainteval",
    help="""
    Evaluate the constaint without running a job.
    Output matched node names with referenced attributes.

    The CONSTRAINTS argument is grouped in groups of 3 elements. Each group
    is then assigned to 'attribute', 'operator' and 'value' of a constaint in order
    and then evaluated. Watch out for shell quoting.

    The 'regexp' is evaluted using python 're' module.

    \b
    CONSTRAINTS arguments examples:
        attr.os.name is_set
        '${attr.os.name}' = ubuntu '${attr.kernel.name}' = linux
        '${attr.os.name}' '!=' ubuntu attr.os.name is_set
        attr.os.name is_set '' attr.kernel.name is_set
    """
)
@clickdc.adddc("args", Args)
def cli(args: Args):
    nodes_attributes = get_nodes_attributes(args.parallel)
    for group in grouper(list(args.constraints), 3):
        constraint = Constraint(*group)
        newlist: List[NodesAtributes] = []
        for node in nodes_attributes:
            context = Context(
                attribute=interpolate(constraint.attribute, node.attributes),
                value=interpolate(constraint.value, node.attributes),
                node=node.attributes,
            )
            log.debug(f"{context.attribute!r} {constraint.operator} {context.value!r}")
            if OPERATORS[constraint.operator](context):
                newlist.append(node)
        nodes_attributes = newlist
    for node in nodes_attributes:
        print(
            node.data["Name"],
            json.dumps({k: v for k, v in node.attributes.items() if k in REFERENCES}),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    cli()
