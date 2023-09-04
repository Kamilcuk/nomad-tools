#!/usr/bin/env python3

import argparse
import distutils.spawn
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from shlex import quote, split
from typing import Dict, List

log = logging.getLogger(__file__)

###############################################################################


def dryrunstr():
    return "DRYRUN: " if args.dryrun else ""


def quotearr(cmd: List[str]):
    return " ".join(quote(x) for x in cmd)


def run(cmd: List[str], check=True, **kvargs):
    log.info(f"{dryrunstr()}+ {quotearr(cmd)}")
    if not args.dryrun:
        return subprocess.run(cmd, text=True, check=check, **kvargs)


def run_stdout(cmd: List[str], check=True):
    log.info(f"+ {quotearr(cmd)}")
    rr = subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE)
    assert rr is not None
    return rr.stdout


def get_namespace_job_from_nomad_service_file(file: Path):
    try:
        with file.open() as f:
            jobjson = json.load(f)
    except json.JSONDecodeError:
        jobjson = json.loads(run_stdout("nomad job run -output".split() + [str(file)]))
    return jobjson["Job"]["Namespace"], jobjson["Job"]["ID"]

###############################################################################

class NomadVarDir:
    def get_old_items(self):
        try:
            olditems: dict = json.loads(
                run_stdout(
                    split(
                        f"nomad var get -namespace={quote(args.namespace)} -out=json nomad/jobs/{quote(args.job)}"
                    ),
                    check=False,
                )
            )["Items"]
        except Exception:
            olditems = {}
        self.olditems = olditems
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
        self.newitems = {str(file): file.read_text() for file in files}
        if args.D:
            assert all("=" in x for x in args.D), "-D options have to be var=value"
            self.newitems.update({k: v for x in args.D for k, v in x.split("=", 2)})
        if not args.clear:
            self.newitems = {**self.olditems, **self.newitems}
        return self.newitems

    @staticmethod
    def create_tree(dir: Path, data: Dict[str, str]):
        for k, v in data.items():
            p = dir / k
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w+") as f:
                f.write(v)

    def show_diff(self):
        newitems = dict(self.newitems, _metadata="")
        olditems = dict(self.olditems, _metadata="")
        if distutils.spawn.find_executable("diff"):
            with tempfile.TemporaryDirectory(prefix="nomad-var-dir_") as tmpd:
                tmpd = Path(tmpd)
                self.create_tree(tmpd / "nomad", olditems)
                self.create_tree(tmpd / "local", newitems)
                cmd = "diff --color -ru nomad local"
                log.info(f"+ {cmd}")
                subprocess.run(split(cmd), cwd=tmpd)

    def mode_put(self):
        keysstr = " ".join(self.newitems.keys())
        log.info(
            f"Putting var nomad/jobs/{args.job}@{args.namespace} with keys: {keysstr}"
        )
        putargs = (
            f" -check-index={quote(args.check_index)}" if args.check_index else ""
        ) + (" -force" if args.force else "")
        run(
            split(
                f"nomad var put -namespace={quote(args.namespace)} {putargs} -in=json nomad/jobs/{quote(args.job)} -"
            ),
            input=json.dumps({"Items": self.newitems}),
        )

    def mode_get(self):
        for file, content in self.olditems.items():
            file = Path(file)
            if file.exists() and file.is_file() and file.read_text() == content:
                log.info(f"nochange {file}")
            else:
                log.info(f"{dryrunstr()}writting {file}")
                if not args.dryrun:
                    file.parent.mkdir(exist_ok=True, parents=True)
                    with file.open("w+") as f:
                        f.write(content)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Given a list of files puts the file content into a nomad variable storage.",
        epilog="Written by Kamil Cukrowski 2023. All right reserved.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-n", "--dryrun", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--namespace", default=os.environ.get("NOMAD_NAMESPACE"))
    parser.add_argument("--job", help="Job name to upload the variables to")
    parser.add_argument(
        "-s",
        "--service",
        type=Path,
        help="Get namespace and job name from this nomad service file",
    )
    parser.add_argument(
        "-D",
        type=str,
        help="Additional var=value to store in nomad variables",
        action="append",
    )
    parser.add_argument(
        "--disable-size-check",
        action="store_true",
        help="Disable checking if the file is smaller than 10mb",
    )
    parser.add_argument("--clear", help="clear keys that are not found in files")
    subparsers = parser.add_subparsers(dest="mode", help="mode to run")
    #
    parser_put = subparsers.add_parser(
        "put",
        help="recursively scan given paths and upload them with filenames as key to nomad variable store",
    )
    parser_put.add_argument(
        "--force", action="store_true", help="Pass -force to nomad var put"
    )
    parser_put.add_argument("-check-index", help="pass -check-index to nomad var put")
    parser_put.add_argument(
        "--relative",
        type=Path,
        help="Files have paths relative to this directory instead of current working directory",
    )
    parser_put.add_argument("paths", type=Path, help="List of files to put", nargs="+")
    #
    parser_diff = subparsers.add_parser(
        "diff", help="just like put, but stop after showing diff"
    )
    parser_diff.add_argument(
        "--relative",
        type=Path,
        help="Files have paths relative to this directory instead of current working directory",
    )
    parser_diff.add_argument(
        "paths", type=Path, help="List of files to put", nargs="+", default=[Path.cwd()]
    )
    #
    parser_get = subparsers.add_parser(
        "get",
        help="Get files stored in nomad variables adnd store them in specific directory",
    )
    parser_get.add_argument("dest", type=Path, help="Place to unpack the files")
    #
    args = parser.parse_args()
    logging.basicConfig(
        format="%(module)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    if args.service:
        assert not args.job, "--service can't be passed with --job"
        args.namespace, args.job = get_namespace_job_from_nomad_service_file(
            args.service
        )
    if not args.job or not args.namespace:
        exit("Either --service or both --job and --namespace has to be given")
    return args

def cli():
    global args
    args = parse_args()
    nvd = NomadVarDir()
    nvd.get_old_items()
    if args.mode == "put" or args.mode == "diff":
        nvd.gen_new_items()
        nvd.show_diff()
        if args.mode == "put":
            nvd.mode_put()
    elif args.mode == "get":
        nvd.mode_get()
    else:
        assert False, f"Internal error when parsing arguments: {args.mode}"

if __name__ == "__main__":
    cli()
