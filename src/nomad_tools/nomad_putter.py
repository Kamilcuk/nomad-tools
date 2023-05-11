#!/usr/bin/env python3

import argparse
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from shlex import quote, split

log = logging.getLogger(__file__)

# template { data } must be exactly like this.
rgxtxt = r'{{ /\* nomad-putter \*/ }}{{ with nomadVar "(.*)" }}{{ index \. "(.*)" }}{{ end }}'.replace(
    " ", r"\s*"
)
rgx = re.compile(rgxtxt)


def run(cmd: str, silent=False, **kvargs):
    if not silent:
        log.debug(f"+ {cmd}")
    try:
        return subprocess.run(split(cmd), text=True, check=True, **kvargs)
    except subprocess.CalledProcessError as e:
        if silent:
            log.error(f"+ {cmd}")
        exit(e.returncode)


def extract_files_to_upload(jobjson: dict):
    jobid = jobjson["ID"]
    for tg in jobjson["TaskGroups"]:
        for task in tg["Tasks"]:
            for tmpl in task["Templates"]:
                data = tmpl["EmbeddedTmpl"]
                matches = rgx.match(data)
                if matches:
                    assert (
                        matches[1] == jobid
                    ), f"Template {data} does not reference nomadVar by job ID!"
                    yield Path(matches[2])


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
    # Extract files from the nomad job specification.
    try:
        jobjson = json.load(args.file.open())
    except json.JSONDecodeError:
        jobjson = json.loads(
            run(
                f"nomad job run -output {quote(str(args.file))}",
                silent=True,
                stdout=subprocess.PIPE,
            ).stdout
        )
    jobjson = jobjson["Job"]
    namespace = (
        jobjson["Namespace"]
        if jobjson.get("Namespace", None)
        else os.getenv("NOMAD_NAMESPACE", "default")
    )
    files = sorted(list(set(extract_files_to_upload(jobjson))))
    # Update the Nomad variable with new content if needed.
    if files:
        jobid = jobjson["ID"]
        items = {str(file): file.open().read() for file in files}
        log.debug(f"Putting var {jobid}@{namespace} with: {' '.join(items.keys())}")
        run(
            f"nomad var put -namespace={quote(namespace)} -force -in=json {quote(jobid)} -",
            input=json.dumps({"Items": items}),
            stdout=subprocess.DEVNULL,
        )
    # Optionally execute nomad job run.
    if args.mode != "put":
        options = "".join(quote(x) + " " for x in args.options)
        run(
            f"nomad job {args.mode} -namespace={quote(namespace)} {options}{quote(str(args.file))}"
        )
