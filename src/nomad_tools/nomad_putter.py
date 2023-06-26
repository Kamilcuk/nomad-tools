#!/usr/bin/env python3

import sys
import tempfile
import datetime
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
# Note - all spaces are matched with \s*
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
    rgx = re.compile(rgxtxt, re.MULTILINE)

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
                        yield matches[2]

    def extract_files_to_upload(self) -> List[str]:
        return sorted(list(set(self._extract_files_to_upload_gen())))

    def put_var(self, files: List[str], clear: bool):
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
        items = {str(file): open(file).read() for file in files}
        log.debug(
            f"Putting var {self.key}@{self.namespace} with files: {' '.join(items.keys())}"
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


def run_example(args):
    hcl = r"""
job "test_nomad_putter" {
  group "test_nomad_putter_noop" {
    task "test_nomad_putter" {
      driver = "docker"
      config {
        image = "busybox"
        args = ["sh", "-xc", "while sleep 5; do date; cat -n /local/test; echo; done"]
      }
      template {
        destination = "local/test"
        data        = "{{/*nomad-putter*/}}{{with nomadVar \"nomad/jobs/test_nomad_putter\"}}{{index . \"test_nomad_putter.txt\"}}{{end}}"
        change_mode = "noop"
      }
    }
  }
  group "test_nomad_putter_script" {
    task "test_nomad_putter_script" {
      driver = "docker"
      config {
        image = "busybox"
        args = ["sh", "-xc", "while sleep 5; do date; cat -n /local/test; cat -n /local/test2; echo; done" ]
      }
      template {
        destination = "local/test"
        data        = "{{/*nomad-putter*/}}{{with nomadVar \"nomad/jobs/test_nomad_putter\"}}{{index . \"test_nomad_putter.txt\"}}{{end}}"
        change_mode = "script"
        change_script {
          command = "cp"
          args    = ["-va", "/local/test", "/local/test2"]
        }
      }
    }
  }
  group "test_nomad_putter_signal" {
    task "test_nomad_putter_signal" {
      driver = "docker"
      config {
        image = "busybox"
        args = ["sh", "-xc", <<EOF
          refresh() { echo "RECEIVED SIGUSR1"; date; cat -n /local/test; echo; }
          refresh ; trap refresh USR1 ; while sleep 1; do sleep 1; done
          EOF
        ]
      }
      template {
        destination   = "local/test"
        data          = "{{/*nomad-putter*/}}{{with nomadVar \"nomad/jobs/test_nomad_putter\"}}{{index . \"./test_nomad_putter.txt\"}}{{end}}"
        change_mode   = "signal"
        change_signal = "SIGUSR1"
      }
    }
  }
  """ + open(args.file).read() + """
}
    """
    filecontent = f"{datetime.datetime.now()} Written by {sys.argv[0]} script for testing nomad-putter funcionality."
    with tempfile.TemporaryDirectory() as d:
        hclf = Path(d) / "test_nomad_putter.nomad.hcl"
        hclf.open('w').write(hcl)
        file = Path(d) / "test_nomad_putter.txt"
        file.open('w').write(filecontent)
        print("Running job test_nomad_putter:")
        print(hcl)
        print("With the following test_nomad_putter.txt file:")
        print(filecontent)
        subprocess.check_output([sys.argv[0], "run", str(hclf)], cwd=hclf.parent)
        print("The job is deployed. No you can re-run the example, to see that the processes inside the job received a signal.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"""
        Given a Nomad job checks if any task template data matches spcific
        regex in the form:
              {rgxtxt}
        """ r"""
        For example the job specification for a job named 'test_nomad_putter' contains the following:
            template {
              destination = "local/file"
              data = "{{/*nomad-putter*/}}{{with nomadVar \"nomad/jobs/examplejob\"}}{{index . \"./examplefile.txt\"}}{{end}}"
              #                                                                                  ^^^^^^^^^^^^^^^^^ - path to file
              #                                                        ^^^^^^^^^^ - has to be the job name
              #       ^^^^^^^^^^^^^^^^^^^^ - constant tag
              change_mode = "signal"
              change_signal = "SIGHUP"
            }

        If the nomad job specification has a template specification
        that matches such string, the path './test_nomad_putter.txt'
        is extracted.  The file 'test_nomad_putter.txt' relative to
        current workind directory is found.  The file is then uploaded
        to nomad/jobs/test_nomad_putter Nomad variable.  After that,
        the Nomad job is run.

        To run the example:
          nomad-putter example <(printf "%s\n" 'namespace="dev"' 'datacenters=["wee-dev"]')
        """,
        epilog="Written by Kamil Cukrowski 2023. All right reserved.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--clear", action="store_true")
    parser.add_argument(
        "mode",
        choices=("put", "run", "plan", "example"),
        help="""
        put     - Extracts the files from job file and puts them into Nomad variable.
        run     - Put but also follows with 'nomad job run' call.
        plan    - Runs nomad plan.
        example - Runs a job "test_nomad_putter" with a "test_nomad_putter.txt" file.
        """,
    )
    parser.add_argument("options", nargs="*", help="Options passed to nomad run")
    parser.add_argument("file", type=Path, help="Nomad HCL job")
    args = parser.parse_args()
    logging.basicConfig(format="%(module)s: %(message)s", level=logging.DEBUG)

    if args.mode in ["put", "run", "plan"]:
        np = NomadPutter(args.file)
        files = np.extract_files_to_upload()
        np.put_var(files, args.clear)
        if args.mode == "run":
            np.run_nomad_job("run", args.options)
        elif args.mode == "plan":
            np.run_nomad_job("plan", args.options)
    elif args.mode == "example":
        run_example(args)
    else:
        exit(f"Unknown mode {args.mode}")
