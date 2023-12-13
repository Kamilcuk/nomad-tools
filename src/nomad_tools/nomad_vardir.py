#!/usr/bin/env python3

import dataclasses
import distutils.spawn
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from shlex import split
from typing import Dict, List, Optional

import click

from .common import andjoin, common_options, complete_job, mynomad, namespace_option
from .nomadlib.connection import VariableConflict

log = logging.getLogger(__file__)

###############################################################################


@dataclasses.dataclass
class Arguments:
    maxsize: float


def create_tree(dir: Path, items: Dict[str, str]):
    """Create files with paths as keys from data and content as values from data in directory dir"""
    dir.mkdir(parents=True, exist_ok=True)
    for k, v in items.items():
        p = dir / k
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w+") as f:
            f.write(v)


def load_directory(directory: Path) -> Dict[str, str]:
    files: List[Path] = [x for x in directory.rglob("*") if x.is_file()]
    limit_mb: float = ARGS.maxsize
    if limit_mb:
        for file in files:
            filesize_mb = int(file.stat().st_size / 1024 / 1024)
            assert (
                filesize_mb < limit_mb
            ), f"{file} size is {filesize_mb} greater than {limit_mb} mb, exiting"
    return {str(file): file.read_text() for file in files}


def show_diff(directory: Path, olditems: Dict[str, str]) -> bool:
    newitems = load_directory(directory)
    if distutils.spawn.find_executable("diff"):
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


def dict_keys_str(data: Dict[str, str]) -> str:
    return " ".join(data.keys())


###############################################################################


@dataclasses.dataclass
class NomadVariable:
    path: str

    def put(self, items: Dict[str, str], cas: Optional[int] = None):
        mynomad.variables.create(self.path, items, cas)

    def get(self) -> Dict[str, str]:
        return mynomad.variables.read(self.path).Items

    def __str__(self):
        return f"{self.path}@{os.environ['NOMAD_NAMESPACE']}"


@dataclasses.dataclass
class MockNomadVariableDb:
    vars: Dict[str, Dict[str, str]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class MockNomadVariable(NomadVariable):
    db: MockNomadVariableDb
    path: str

    def put(self, items: Dict[str, str], cas: Optional[int] = None):
        self.db.vars[self.path] = items

    def get(self) -> Dict[str, str]:
        return self.db.vars.get(self.path, {})


@dataclasses.dataclass
class MakeNomadVariable:
    """Convert command line arguments to path"""

    path: Optional[str]
    service: Optional[Path]
    job: Optional[str]

    @staticmethod
    def get_namespace_job_from_nomad_service_file(service: Path):
        try:
            with service.open() as f:
                jobjson = json.load(f)
        except json.JSONDecodeError:
            jobjson = json.loads(
                subprocess.check_output(
                    "nomad job run -output".split() + [str(service)], text=True
                )
            )
        return jobjson["Job"]["Namespace"], jobjson["Job"]["ID"]

    def make(self) -> NomadVariable:
        if self.path:
            assert not self.service, "--service can't be passed with --path"
            assert not self.job, "--job can't be passed with --path"
            if "@" in self.path:
                path, os.environ["NOMAD_NAMESPACE"] = self.path.split("@", 2)
            else:
                path = self.path
        elif self.service:
            assert not self.job, "--service can't be passed with --job"
            (
                os.environ["NOMAD_NAMESPACE"],
                job,
            ) = self.get_namespace_job_from_nomad_service_file(self.service)
            path = f"nomad/job/{job}"
        elif self.job:
            path = f"nomad/job/{self.job}"
        else:
            raise Exception("One of --path --service or --job has to be given.")
        return NomadVariable(path)


FACTORS: list = "B K M G T P E Z Y".split()


def human_size(size: str) -> float:
    assert len(size) > 0
    try:
        factoridx = FACTORS.index(size[-1:].upper())
    except ValueError:
        return int(size.strip())
    return 2 ** (10 * factoridx) * float(size[:-1].strip())


###############################################################################


@click.group(
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
    nomad-vardir -p nomad/job/nginx@nginx get nginx.conf
    nomad-vardir -j nginx@nginx get nginx.conf
    nomad-vardir nginx.nomad.hcl ls
    nomad-vardir nginx.nomad.hcl diff
    nomad-vardir nginx.nomad.hcl put
    nomad-vardir nginx.nomad.hcl get

Written by Kamil Cukrowski 2023. All rights reserved.
""",
)
@click.option("-v", "--verbose", is_flag=True)
@namespace_option()
@common_options()
@click.option("-p", "--path", help="The first argument is the full path of Nomad variable")
@click.option(
    "-j",
    "--job",
    help="Equal to --path=nomad/job/<JOB>",
    shell_complete=complete_job,
)
@click.option(
    "-f",
    "--file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Get namespace and job name from this nomad service file",
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
@click.argument("path")
def cli(
    path: Optional[str],
    file: Optional[Path],
    job: Optional[str],
    verbose: bool,
    maxsize: float,
    **kwargs,
):
    logging.basicConfig(
        format="%(module)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )
    global ARGS
    ARGS = Arguments(maxsize)
    global VAR
    VAR = MakeNomadVariable(path, service, job).make()


@cli.command("ls")
def mode_ls():
    d = VAR.get()
    w = max(len(k) for k in d.keys())
    print("{:w} {}".format("name", "size"))
    for k, v in d.items():
        print(f"{k:{w}} {len(v)}")


@cli.command("cat")
@click.argument("filename")
def mode_cat(filename):
    print(VAR.get()[filename])


@cli.command(
    "get",
    help="Get files stored in Nomad variables and store them",
)
@click.argument(
    "paths",
    type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path),
    nargs=-1,
)
def mode_get(paths: List[Path]):
    create_tree(destination, VAR.get())


@cli.command("diff", help="Show diff between directory and Nomad variable")
@click.argument(
    "paths",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    nargs=-1,
)
@common_options()
def mode_diff(directory):
    exit(2 * show_diff(directory, VAR.get()))


@cli.command(
    "put",
    help="""
    Recursively scan files in given PATHS and upload filenames as key and file content as value to nomad variable store.
    """,
)
@click.option("--force", is_flag=True, help="Like nomad var put -force")
@click.option("--check-index", type=int, help="Like nomad var put -check-index")
@click.option("--clear", is_flag=True, help="Remove keys that are not found in files")
@click.argument(
    "paths",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    nargs=-1,
)
def mode_put(force: bool, check_index: Optional[int], clear: bool, directory: Path):
    if force:
        assert check_index is None, "either --force or --check-index"
    olditems = VAR.get()
    items = {} if clear else dict(olditems)
    items.update(load_directory(directory))
    if items == olditems:
        log.info("No changes in {VAR}")
    else:
        log.info(f"Putting var {VAR} with keys: {dict_keys_str(items)}")
        cas: Optional[int] = (
            check_index if check_index is not None else 0 if not force else None
        )
        try:
            VAR.put(items, cas)
        except VariableConflict as e:
            raise Exception(
                f"Variable update conflict. Pass --check-index={e.variable.ModifyIndex}"
            ) from e


###############################################################################

if __name__ == "__main__":
    cli.main()
