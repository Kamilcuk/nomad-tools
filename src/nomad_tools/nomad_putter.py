#!/usr/bin/env python3

import argparse
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from shlex import quote, split
from typing import List

log = logging.getLogger(__file__)

# template { data } must be exactly like this.
rgxtxt = r'{{ /\* nomad-putter \*/ }}{{ with nomadVar "(.*)" }}{{ index \. "(.*)" }}{{ end }}'.replace(
    " ", r"\s*"
)


def run(cmd: str, silent=False, **kvargs):
    if not silent:
        log.debug(f"+ {cmd}")
    try:
        return subprocess.run(split(cmd), text=True, check=True, **kvargs)
    except subprocess.CalledProcessError as e:
        if silent:
            log.error(f"+ {cmd}")
        exit(e.returncode)


class NomadPutter:
    rgx = re.compile(rgxtxt)

    def _nomad_job_file_to_json(self, file: Path):
        try:
            jobjsontmp = json.load(file.open())
        except json.JSONDecodeError:
            jobjsontmp = json.loads(
                run(
                    f"nomad job run -output {quote(str(file))}",
                    silent=True,
                    stdout=subprocess.PIPE,
                ).stdout
            )
        return jobjsontmp["Job"]

    def __init__(self, file: Path):
        # Extract files from the nomad job specification.
        self.file = file
        self.jobjson = self._nomad_job_file_to_json(file)
        self.namespace = (
            self.jobjson["Namespace"]
            if self.jobjson.get("Namespace", None)
            else os.getenv("NOMAD_NAMESPACE", "default")
        )
        self.namespacearg = f"-namespace={quote(self.namespace)}"
        self.jobid = self.jobjson.get("ID", self.jobjson["Name"])
        self.key = f"nomad/jobs/{self.jobid}"

    def _extract_files_to_upload_gen(self):
        for tg in self.jobjson["TaskGroups"]:
            for task in tg["Tasks"]:
                for tmpl in task["Templates"]:
                    data = tmpl["EmbeddedTmpl"]
                    matches = self.rgx.match(data)
                    if matches:
                        assert (
                            matches[1] == self.key
                        ), f"nomadVar key has to reference {self.key}:  {matches[1]}"
                        yield Path(matches[2])

    def extract_files_to_upload(self):
        return sorted(list(set(self._extract_files_to_upload_gen())))

    def put_var(self, files: List[Path], clear: bool):
        items = {}
        if not clear:
            try:
                item = json.dumps(
                    run(
                        f"nomad var get {self.namespacearg} -out=json {quote(self.key)}",
                        silent=True,
                        stdout=subprocess.PIPE,
                        check=False,
                    ).stdout
                )
            except Exception:
                pass
        jobid = self.jobjson["ID"]
        items = {str(file): file.open().read() for file in files}
        log.debug(
            f"Putting var {self.key}@{self.namespace} with: {' '.join(items.keys())}"
        )
        run(
            f"nomad var put {self.namespacearg} -force -in=json {quote(self.key)} -",
            input=json.dumps({"Items": items}),
            stdout=subprocess.DEVNULL,
        )

    def run_nomad_job(self, mode: str, options: List[str]):
        optionsarg = "".join(quote(x) + " " for x in args.options)
        run(
            f"nomad job {quote(mode)} {self.namespacearg} {optionsarg}{quote(str(self.file))}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"""
        Given a Nomad job checks if a task template data matches spcific
        regex in the form: '{rgxtxt}'.  If it does, the first has to match
        the job Name as a precaution. Then the next part is expected to
        be a filename relative to current working directory. All such
        extracted files are read and posted to Nomad variable storage
        with the job id as the path with filepaths as keys.
        """,
        epilog="Written by Kamil Cukrowski 2023. All right reserved.",
    )
    parser.add_argument("--clear", action="store_true")
    parser.add_argument(
        "mode",
        choices=("put", "run", "plan"),
        help="""
        'put' extracts the files from job file and puts them into Nomad
        variable. 'run' additionally follows with nomad job run call.
        """,
    )
    parser.add_argument("options", nargs="*", help="Options passed to nomad run")
    parser.add_argument("file", type=Path, help="Nomad HCL job")
    args = parser.parse_args()
    logging.basicConfig(format="%(module)s: %(message)s", level=logging.DEBUG)

    np = NomadPutter(args.file)
    files = np.extract_files_to_upload()
    if files:
        np.put_var(files, args.clear)
    if args.mode != "put":
        np.run_nomad_job(args.mode, args.options)
