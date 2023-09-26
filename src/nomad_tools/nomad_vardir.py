#!/usr/bin/env python3

import argparse
import distutils.spawn
import json
import logging
import os
import subprocess
import tempfile
from itertools import chain
from pathlib import Path
from shlex import quote, split
from typing import Dict

import click

from .common import common_options, complete_job, mynomad, namespace_option
from .nomadlib import VariableNotFound
from .nomadlib.connection import VariableConflict

log = logging.getLogger(__file__)

###############################################################################


def dryrunstr():
    return "DRYRUN: " if args.dryrun else ""


def get_namespace_job_from_nomad_service_file(file: Path):
    try:
        with file.open() as f:
            jobjson = json.load(f)
    except json.JSONDecodeError:
        jobjson = json.loads(
            subprocess.check_output(
                "nomad job run -output".split() + [str(file)], text=True
            )
        )
    return jobjson["Job"]["Namespace"], jobjson["Job"]["ID"]


def create_tree(dir: Path, data: Dict[str, str]):
    """Create files with pahts as keys from data and content as values from data in directory dir"""
    dir.mkdir(parents=True, exist_ok=True)
    for k, v in data.items():
        p = dir / k
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w+") as f:
            f.write(v)


###############################################################################


class NomadVarDirMain:
    def __init__(self, ctx):
        global args
        args = argparse.Namespace(**{**vars(args), **ctx.params})
        logging.basicConfig(
            format="%(module)s: %(message)s",
            level=logging.DEBUG if args.verbose else logging.INFO,
        )
        if args.path:
            assert not args.service, "--service can't be passed with --path"
            assert not args.job, "--job can't be passed with --path"
        elif args.service:
            assert not args.job, "--service can't be passed with --job"
            args.namespace, args.job = get_namespace_job_from_nomad_service_file(
                args.service
            )
            os.environ["NOMAD_NAMESPACE"] = args.namespace
        elif args.job:
            args.path = f"nomad/job/{args.job}"
        else:
            assert 0, "One of --path --service or --job has to be given."
        self.get_old_items()

    def get_old_items(self):
        try:
            olditems = mynomad.variables.read(args.path).Items
        except VariableNotFound:
            olditems = {}
        self.olditems: Dict[str, str] = olditems
        return olditems

    def gen_new_items(self):
        # Get all files and directories recursively passed by args.paths.
        files = []
        for path in args.paths:
            if path.is_dir():
                files += (x for x in path.rglob("*") if x.is_file())
            else:
                files += [path]
        # If args.relative, the paths are relative to the directory.
        if args.relative:
            os.chdir(args.relative)
            files = [x.relative_to(args.relative) for x in files]
        if not args.disable_size_check:
            limit_mb = 10
            for file in files:
                filesize_mb = int(file.stat().st_size / 1024 / 1024)
                assert (
                    filesize_mb < limit_mb
                ), f"{file} size is {filesize_mb} greater than {limit_mb} mb, exiting"
        self.newitems: Dict[str, str] = {str(file): file.read_text() for file in files}
        if "D" in args and args.D:
            assert all("=" in x for x in args.D), "-D options have to be var=value"
            self.newitems.update({k: v for x in args.D for k, v in x.split("=", 2)})
        if "clear" in args and not args.clear:
            self.newitems = {**self.olditems, **self.newitems}
        return self.newitems

    def show_diff_11(self):
        if not distutils.spawn.find_executable("diff"):
            log.warning("diff executable not found")
            return
        with tempfile.NamedTemporaryFile("w", prefix="nomad-var-dir_") as tmpf:
            for k in sorted(
                list(set(chain(self.newitems.keys(), self.olditems.keys())))
            ):
                old = self.olditems.get(k, "")
                tmpf.seek(0)
                tmpf.write(old)
                tmpf.flush()
                #
                newfile = (
                    (k if Path(k).exists() else "-")
                    if k in self.newitems
                    else subprocess.DEVNULL
                )
                stdin = self.newitems[k] if newfile == "-" else None
                subprocess.run(
                    split(f"diff --color -u {quote(tmpf.name)} {newfile}"),
                    input=stdin,
                    text=True,
                )

    def show_diff(self) -> bool:
        newitems = dict(self.newitems)
        olditems = dict(self.olditems)
        if distutils.spawn.find_executable("diff"):
            with tempfile.TemporaryDirectory(prefix="nomad-var-dir_") as tmpd:
                tmpd = Path(tmpd)
                create_tree(tmpd / "nomad", olditems)
                create_tree(tmpd / "local", newitems)
                cmd = "diff --color -ru nomad local"
                log.info(f"+ {cmd}")
                subprocess.run(split(cmd), cwd=tmpd)
        return newitems != olditems


###############################################################################


@click.group(
    help="Given a list of files puts the file content into a nomad variable storage.",
    epilog="Written by Kamil Cukrowski 2023. All right reserved.",
)
@click.option("-n", "--dryrun", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
@namespace_option()
@click.option("-p", "--path", help="The path of the variable to save")
@click.option(
    "-j",
    "--job",
    help="Equal to --path=nomad/job/<JOB>",
    shell_complete=complete_job,
)
@click.option(
    "-s",
    "--service",
    type=click.Path(exists=True, dir_okay=False),
    help="Get namespace and job name from this nomad service file",
)
@click.option(
    "--disable-size-check",
    is_flag=True,
    help="Disable checking if the file is smaller than 10mb",
)
@click.pass_context
@common_options()
def cli(ctx, **kwargs):
    global args
    args = argparse.Namespace(**ctx.params)


@cli.command(
    "put",
    help="Recursively scan files in given PATHS and upload filenames as key and file content as value to nomad variable store.",
)
@click.option("--force", is_flag=True, help="Like nomad var put -force")
@click.option("--check-index", type=int, help="Like nomad var put -check-index")
@click.option(
    "--relative",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Files have paths relative to this directory instead of current working directory",
)
@click.option(
    "-D",
    "D",
    type=str,
    help="Additional var=value to store in nomad variables",
    multiple=True,
)
@click.option("--clear", is_flag=True, help="Remove keys that are not found in files")
@click.argument(
    "paths",
    type=click.Path(exists=True, path_type=Path),
    nargs=-1,
)
@common_options()
@click.pass_context
def mode_put(ctx, **kvargs):
    nvd = NomadVarDirMain(ctx)
    print(ctx.params)
    if args.force:
        assert args.check_index is None, "either --force or --check-index"
    nvd.gen_new_items()
    if not nvd.show_diff():
        log.info("No difference in var {args.path}@{args.namespace}")
    else:
        keysstr = " ".join(nvd.newitems.keys())
        log.info(
            f"{dryrunstr()}Putting var {args.path}@{args.namespace} with keys: {keysstr}"
        )
        if not args.dryrun:
            try:
                mynomad.variables.create(
                    args.path,
                    nvd.newitems,
                    cas=args.check_index
                    if args.check_index is not None
                    else 0
                    if not args.force
                    else None,
                )
            except VariableConflict as e:
                raise Exception(
                    f"Variable update conflict. Pass --check-index={e.variable.ModifyIndex}"
                ) from e


@cli.command("diff", help="Show only diff")
@click.option(
    "--relative",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Files have paths relative to this directory instead of current working directory",
)
@click.argument(
    "paths",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    nargs=-1,
)
@click.pass_context
@common_options()
def mode_diff(ctx, **kwargs):
    nvd = NomadVarDirMain(ctx)
    nvd.gen_new_items()
    isdiff = nvd.show_diff()
    exit(2 if isdiff else 0)


@cli.command(
    "get",
    help="Get files stored in nomad variables adnd store them in specific directory",
)
@click.argument(
    "dest",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
)
@click.pass_context
@common_options()
def mode_get(ctx, **kwargs):
    nvd = NomadVarDirMain(ctx)
    #
    args.dest.mkdir(exist_ok=True, parents=True)
    for file, content in nvd.olditems.items():
        file = args.dest / file
        if file.exists() and file.is_file() and file.read_text() == content:
            log.info(f"nochange {file}")
        else:
            log.info(f"{dryrunstr()}writting {file}")
            if not args.dryrun:
                file.parent.mkdir(exist_ok=True, parents=True)
                with file.open("w+") as f:
                    f.write(content)


###############################################################################

if __name__ == "__main__":
    cli.main()
