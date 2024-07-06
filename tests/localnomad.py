#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path
from shlex import quote, split
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log-level", default="info")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--tls", action="store_true")
parser.add_argument("version", nargs="?")
args = parser.parse_args()
os.chdir(Path(__file__).parent.parent)
if args.verbose:
    args.log_level = "debug"

assert Path("/sys/module/bridge"), "ERROR: bridge module not loaded"
assert (
    subprocess.run("docker status".split()).returncode != 0
), "ERROR: docker does not work"

if not args.version:
    cmdstr: str = "python -m nomad_tools.entrypoint downloadrelease --showversion nomad"
    cmd: List[str] = split(cmdstr)
    print(f"+ {cmdstr}")
    args.version = subprocess.check_output(cmd, text=True).strip()

exe: Path = Path(f"./build/bin/nomad{args.version}")
if not exe.exists():
    cmdstr = f"python -m nomad_tools.entrypoint downloadrelease -p {quote(args.version)} nomad {quote(str(exe))}"
    cmd = split(cmdstr)
    print(f"+ {cmdstr}")
    subprocess.check_call(cmd)

cmd: List[str] = [
    *(["sudo"] if os.getuid() != 0 else []),
    str(exe),
    *f"agent -dev -config ./tests/nomad.d/nomad.hcl -log-level {args.log_level}".split(),
    *(["-config", "./tests/nomad.d/tls.hcl"] if args.tls else []),
]
cmdstr = " ".join(quote(x) for x in cmd)
print(f"+ {cmdstr}")
os.execvp(cmd[0], cmd)
