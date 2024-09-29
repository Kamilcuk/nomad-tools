#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from shlex import split
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union

import click
from click.shell_completion import CompletionItem
from typing_extensions import override

from . import nomadlib
from .aliasedgroup import AliasedGroup
from .common import andjoin, help_h_option, mynomad, namespace_option
from .nomadlib.connection import VariableConflict

log = logging.getLogger(__file__)

###############################################################################


@dataclasses.dataclass
class Arguments:
    maxsize: float
    test: Optional[Path]
    relative: Path
    filter: Set[Path] = dataclasses.field(default_factory=set)


def dict_keys_str(data: Dict[str, str]) -> str:
    return " ".join(data.keys())


FACTORS: list = "B K M G T P E Z Y".split()
"""Number factors"""


def human_size(size: str) -> float:
    """Convert human string to number"""
    assert len(size) > 0
    try:
        factoridx = FACTORS.index(size[-1:].upper())
    except ValueError:
        return int(size.strip())
    return 2 ** (10 * factoridx) * float(size[:-1].strip())


NOMADJOBS = "nomad/jobs/"

###############################################################################


@dataclasses.dataclass
class ParseJobFile:
    """
    From Nomad service file extract Namespace and job id
    and also extract the referenced nomad variables in templates.
    """

    path: Path
    pattern: str = (
        r'{{ with nomadVar "nomad/jobs/{job.ID}/[^"]*" }}'
        r' {{ (.(\S*)|index \. "([^"]*)") }}'
        r" {{ end }}"
    ).replace(" ", r"\s+")
    rgx: re.Pattern = re.compile(pattern, re.MULTILINE | re.DOTALL)

    @dataclasses.dataclass
    class Ret:
        namespace: Optional[str]
        ID: str
        refkeys: List[str]

    def parse(self):
        try:
            with self.path.open() as f:
                jobjson = json.load(f)
        except json.JSONDecodeError:
            jobjson = json.loads(
                subprocess.check_output(
                    "nomad job run -output".split() + [str(self.path)], text=True
                )
            )
        job: nomadlib.Job = nomadlib.Job(jobjson.get("Job", jobjson))
        refkeys: List[str] = []
        for group in job.TaskGroups:
            for task in group.Tasks:
                for tmpl in task.Templates or []:
                    txt = tmpl.EmbeddedTmpl
                    # TODO: handle rightdelim and leftdelim
                    match = self.rgx.search(txt)
                    log.debug(f"{txt} -> {match}")
                    if match:
                        groups = match.group(2, 3)
                        refkeys.append("".join(groups))
        return self.Ret(
            namespace=job.Namespace,
            ID=job.ID,
            refkeys=refkeys,
        )


###############################################################################


@dataclasses.dataclass
class NomadVariable:
    path: str

    def put_cb(self, items: Dict[str, str], cas: Optional[int] = None):
        """Update variable with items"""
        mynomad.variables.create(self.path, items, cas)

    def get_cb(self) -> Optional[nomadlib.Variable]:
        """Get the variable content"""
        try:
            return mynomad.variables.read(self.path)
        except nomadlib.VariableNotFound:
            return None

    def delete_cb(self, cas: Optional[int] = None):
        """Delete the variable from Nomad"""
        mynomad.variables.delete(self.path, cas)

    def put(self, items: Dict[str, str], cas: Optional[int] = None):
        assert items
        self.put_cb(items, cas)

    def get(self) -> Dict[str, str]:
        r = self.get_cb()
        return r.Items if r else {}

    def delete(self, cas: Optional[int] = None):
        self.delete_cb(cas)

    def get_select(self, keys: Union[List[str], List[Path]]) -> Dict[str, str]:
        """Get key:value from Nomad variable and select only given keys"""
        d = self.get()
        if not keys:
            return d
        return {str(key): d[str(key)] for key in keys}

    def desc(self):
        return f"{self.path}@{os.environ['NOMAD_NAMESPACE']}"

    def updater(
        self,
        gen_newitems_cb: Callable[[Dict[str, str]], Dict[str, str]],
        check_index: Optional[int],
        force: bool,
    ):
        """
        Retrieve the current Noamd variable value, run the callback,
        and update the Nomad variable value with the returned data.
        """
        assert sum([force, check_index is None]) <= 1, "either --force or --check-index"
        oldvar = self.get_cb()
        olditems = oldvar.Items if oldvar else {}
        log.debug(f"olditems={olditems}")
        newitems = gen_newitems_cb(dict(olditems))
        log.debug(f"newitems={newitems}")
        #
        if newitems == olditems:
            log.info(f"No changes in {VAR.desc()}")
        else:
            cas: Optional[int] = (
                check_index
                if check_index is not None
                else None if force else oldvar.ModifyIndex if oldvar else 0
            )
            if newitems:
                log.info(
                    f"Putting var {VAR.desc()} with keys: {dict_keys_str(newitems)}"
                )
                try:
                    VAR.put(newitems, cas)
                except VariableConflict as e:
                    raise Exception(
                        f"Variable update conflict. Pass --check-index={e.variable.ModifyIndex}"
                    ) from e
            else:
                log.info(f"Removing empty var {VAR.desc()}")
                VAR.delete(cas)


@dataclasses.dataclass
class MockNomadVariableDb:
    vars: Dict[str, Dict[str, str]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class MockNomadVariable(NomadVariable):
    @staticmethod
    def __load() -> Dict[str, Dict[str, str]]:
        assert ARGS.test
        try:
            if ARGS.test.stat().st_size == 0:
                return {}
            with ARGS.test.open("r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    @staticmethod
    def __save(data: Dict[str, Dict[str, str]]):
        assert ARGS.test
        with ARGS.test.open("w") as f:
            json.dump(data, f)

    @override
    def put_cb(self, items: Dict[str, str], cas: Optional[int] = None):
        assert items
        data = self.__load()
        data[self.path] = items
        self.__save(data)

    @override
    def get_cb(self) -> Optional[nomadlib.Variable]:
        return nomadlib.Variable(
            dict(Items=self.__load().get(self.path, {}), ModifyIndex=1)
        )

    @override
    def delete_cb(self, cas: Optional[int] = None):
        data = self.__load()
        del data[self.path]
        self.__save(data)


@dataclasses.dataclass
class MakeNomadVariable:
    """Convert command line arguments to path"""

    path: str
    job: Optional[bool]
    jobfile: Optional[bool]
    test: Optional[Path] = None

    def make(self) -> NomadVariable:
        assert not (self.job and self.jobfile), "--job and --jobfile conflict"
        if not self.job and self.jobfile:
            res = ParseJobFile(Path(self.path)).parse()
            if res.namespace is not None:
                os.environ["NOMAD_NAMESPACE"] = res.namespace
            path = f"{NOMADJOBS}{res.ID}"
        else:
            path: str = f"{NOMADJOBS}{self.path}" if self.job else f"{self.path}"
            if "@" in path:
                path, os.environ["NOMAD_NAMESPACE"] = path.split("@", 2)
        return (MockNomadVariable if self.test else NomadVariable)(path)


def create_tree(dir: Path, items: Dict[str, str], dryrun: bool = False):
    """Create files with paths as keys from data and content as values from data in directory dir"""
    dryrunstr = "DRY RUN: " if dryrun else ""
    if not dryrun:
        dir.mkdir(parents=True, exist_ok=True)
    for k, v in items.items():
        p = dir / k
        log.info(f"{dryrunstr}Creating {p} with {len(v)} bytes")
        if not dryrun:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w+") as f:
                f.write(v)


def recurse_directories(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            for file in path.rglob("*"):
                if file.is_file():
                    files.append(file)
        elif path.is_file():
            files.append(path)
    return files


def load_directory(paths: Iterable[Path]) -> Dict[str, str]:
    """Load all files from a directory minding conditions"""
    assert paths
    files: List[Path] = recurse_directories(paths)
    limit_mb: float = ARGS.maxsize
    if limit_mb:
        for file in files:
            filesize_mb = int(file.stat().st_size / 1024 / 1024)
            assert (
                filesize_mb < limit_mb
            ), f"{file} size is {filesize_mb} greater than {limit_mb} mb, exiting"
    log.debug(f"Found files: {files} {ARGS.filter}")
    return {str(file): file.read_text() for file in files}


def show_diff(olditems: Dict[str, str], newitems: Dict[str, str]) -> bool:
    if shutil.which("diff"):
        with tempfile.TemporaryDirectory(prefix="nomad-var-dir_") as tmpd:
            tmpd = Path(tmpd)
            create_tree(tmpd / "nomad", olditems)
            create_tree(tmpd / "local", newitems)
            cmd = "diff --color -ru nomad local"
            log.info(f"+ {cmd}")
            subprocess.run(split(cmd), cwd=tmpd)
    else:
        log.error("diff: command not found")
    return newitems != olditems


###############################################################################


def complete_vardir_path(
    ctx: click.Context, param: click.Parameter, incomplete: str
) -> Optional[Union[List[str], List[CompletionItem]]]:
    jobfile: bool = ctx.params["jobfile"]
    job: bool = ctx.params["job"]
    if jobfile:
        return click.Path(dir_okay=False, exists=True).shell_complete(
            ctx, param, incomplete
        )
    else:
        if "@" in incomplete:
            prefix, _ = incomplete.split("@", 1)
        else:
            prefix = incomplete
        if job:
            prefix = f"{NOMADJOBS}{prefix}"
        ret = mynomad.get("vars", params=dict(prefix=prefix))
        vars = [f"{x['Path']}@{x['Namespace']}" for x in ret]
        if job:
            vars = [x[len(NOMADJOBS) :] for x in vars]
            vars = [x for x in vars if x]
        vars = [x for x in vars if x.startswith(incomplete)]
        return vars
    return []


def complete_vardir_paths(
    ctx: click.Context, param: click.Parameter, incomplete: str
) -> Optional[Union[List[str], List[CompletionItem]]]:
    if not ctx.parent:
        return []
    par = ctx.parent.params
    VAR = MakeNomadVariable(
        path=par["path"], job=par["job"], jobfile=par["jobfile"]
    ).make()
    d = VAR.get()
    return [k for k in d.keys()]


def click_vardir_keys(required: bool = False, name: str = "keys", type: Any = str):
    return click.argument(
        name,
        nargs=-1,
        shell_complete=complete_vardir_paths,
        type=type,
        required=required,
    )


def click_vardir_paths(required: bool = False):
    return click_vardir_keys(
        required=required, name="paths", type=click.Path(path_type=Path, exists=False)
    )


@click.command(
    "vardir",
    cls=AliasedGroup,
    help="""
This is a solution for managing Nomad variables as directories and files.
Single Nomad variable can be represented as a directory.
Each file inside the directory represent a JSON key inside the Nomad variable.
This tool can update and edit the keys in Nomad variables as files.

\b
Typical workflow would look like the following:
- create a template to generate a file that you want to upload to nomad variables,
   - for example an `nginx.conf` configuration,
- write a makefile that will generate the `nginx.conf` from the template using consul-template,
- use this script on the directory containing generated `nginx.conf` to upload it to Nomad variables.
""",
    epilog="""
\b
Examples:
     nomad/jobs/nginx@nginx get nginx.conf
    nomad-vardir -j nginx@nginx ls
    nomad-vardir -j nginx@nginx put ./nginx.conf
    nomad-vardir -j nginx@nginx cat ./nginx.conf
    nomad-vardir -j nginx@nginx get ./nginx.conf
    nomad-vardir -j nginx@nginx diff
    nomad-vardir -j nginx@nginx rm ./nginx.conf

Written by Kamil Cukrowski 2023. All rights reserved.
""",
)
@click.option("-v", "--verbose", is_flag=True)
@namespace_option()
@help_h_option()
@click.option(
    "--test",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    hidden=True,
)
@click.option(
    "-j",
    "--job",
    is_flag=True,
    help="Prepends the path with nomad/jobs/",
)
@click.option(
    "-f",
    "--jobfile",
    is_flag=True,
    help="The path is a Nomad job file rfom which the job name and namespace is read",
)
@click.option(
    "--maxsize",
    default="1M",
    show_default=True,
    help=f"""
        Protect against uploading files greater than this size.
        Supports following units: {andjoin(FACTORS)}.
        """,
    type=human_size,
)
@click.option(
    "-C",
    "--relative",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Parse everything relative to this directory",
    default=Path("."),
)
@click.argument(
    "path",
    shell_complete=complete_vardir_path,
)
def cli(
    path: str,
    job: Optional[bool],
    jobfile: Optional[bool],
    verbose: bool,
    maxsize: float,
    relative: Path,
    test: Optional[Path],
    **kwargs,
):
    logging.basicConfig(
        format="%(levelname)s %(module)s:%(filename)s:%(lineno)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )
    global ARGS
    ARGS = Arguments(maxsize=maxsize, test=test, relative=relative)
    global VAR
    VAR = MakeNomadVariable(path=path, job=job, jobfile=jobfile, test=test).make()


@cli.command("ls")
@click.option("-1", "--one", is_flag=True, help="List keys one per line")
@click_vardir_keys()
def mode_ls(keys: List[str], one: bool):
    kv = VAR.get_select(keys)
    if not kv:
        log.error(f"Nomad variable not found at {VAR.desc()}")
        return
    if one:
        for k in kv.keys():
            print(k)
        return
    log.info(f"Listing Nomad variable at {VAR.desc()}")
    # Convert to key: length(value).
    kv = {k: str(len(v)) for k, v in kv.items()}
    # Calculate column widths.
    w = [max(len(k) for k in kv.keys()), max(len(v) for v in kv.values())]
    # Output.
    print(f"{'key':{w[0]}} {'size':{w[1]}}")
    for k, v in kv.items():
        print(f"{k:{w[0]}} {v:{w[1]}}")


@cli.command("cat")
@click_vardir_keys(required=True)
def mode_cat(keys: List[str]):
    for v in VAR.get_select(keys).values():
        print(v)


@cli.command(
    "get",
    help="Get keys stored in Nomad variable and save them as files",
)
@click.option("-n", "--dryrun", is_flag=True)
@click_vardir_paths(required=True)
def mode_get(paths: List[Path], dryrun: bool):
    create_tree(ARGS.relative, VAR.get_select(paths), dryrun)


@cli.command(
    "diff",
    help="Given files, show diff between the file content and keys values stored in Nomad variable",
)
@click_vardir_paths(required=True)
def mode_diff(paths: List[Path]):
    d = VAR.get()
    paths = paths if paths else [Path(x) for x in d.keys()]
    exit(2 * show_diff(d, load_directory(paths)))


@cli.command("rm", help="Remove keys from Nomad variable")
@click.option("--force", is_flag=True, help="Like nomad var put -force")
@click.option("--check-index", type=int, help="Like nomad var put -check-index")
@click.option(
    "-c",
    "--continue",
    "continue_",
    is_flag=True,
    help="Do not error on inexisting keys",
)
@click_vardir_keys()
def mode_rm(keys: List[str], check_index: Optional[int], force: bool, continue_: bool):
    def mode_rm_gen_newitems(newitems: Dict[str, str]) -> Dict[str, str]:
        for key in keys:
            if key in newitems:
                log.info(f"Removing {key}")
            if not continue_ or key in newitems:
                try:
                    del newitems[key]
                except KeyError:
                    raise Exception(f"Key not found: {key}")

        return newitems

    VAR.updater(mode_rm_gen_newitems, check_index, force)


@cli.command(
    "put",
    help="""
    For each file, set key as the name of the file and value to the file content.
    """,
)
@click.option("-r", "--recursive", is_flag=True)
@click.option("--force", is_flag=True, help="Like nomad var put -force")
@click.option("--check-index", type=int, help="Like nomad var put -check-index")
@click.option("--clear", is_flag=True, help="Remove keys that are not found in files")
@click.option(
    "-D",
    "--define",
    help="Add additional key and values in the form of VAR=VAL",
    multiple=True,
)
@click.argument(
    "paths",
    type=click.Path(path_type=Path, exists=True),
    nargs=-1,
    required=True,
)
def mode_put(
    recursive: bool,
    force: bool,
    check_index: Optional[int],
    clear: bool,
    define: List[str],
    paths: List[Path],
):
    def mode_put_gen_newitems(newitems: Dict[str, str]) -> Dict[str, str]:
        if clear:
            newitems = {}
        paths2 = recurse_directories(paths) if recursive else paths
        if paths2:
            for p in paths2:
                assert p.exists(), f"Does not exists: {p}"
            newitems.update(load_directory(paths2))
        newitems.update({v[0]: v[1] for e in define for v in [e.split("=", 2)]})
        return newitems

    VAR.updater(mode_put_gen_newitems, check_index, force)


@cli.command(
    "set",
    help="Set key to the value at given path",
)
@click.option("--force", is_flag=True, help="Like nomad var put -force")
@click.option("--check-index", type=int, help="Like nomad var put -check-index")
@click.argument("key")
@click.argument("value")
def mode_set(
    force: bool,
    check_index: Optional[int],
    key: str,
    value: str,
):
    def mode_set_key(newitems: Dict[str, str]) -> Dict[str, str]:
        newitems[key] = value
        return newitems

    VAR.updater(mode_set_key, check_index, force)


###############################################################################

if __name__ == "__main__":
    cli.main()
