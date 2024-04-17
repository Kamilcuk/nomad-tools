#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path
from shlex import quote, split
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log-level", default="info")
parser.add_argument("version", nargs="?")
args = parser.parse_args()
DIR: Path = Path(__file__).parent.parent

if not args.version:
    cmdstr: str = "python -m nomad_tools.entrypoint downloadrelease --showversion nomad"
    cmd: List[str] = split(cmdstr)
    print(f"+ {cmdstr}")
    args.version = subprocess.check_output(cmd, text=True).strip()

exe: Path = DIR / f"build/bin/nomad{args.version}"
if not exe.exists():
    cmdstr = f"python -m nomad_tools.entrypoint downloadrelease -p {quote(args.version)} nomad {quote(str(exe))}"
    cmd = split(cmdstr)
    print(f"+ {cmdstr}")
    subprocess.check_call(cmd)

config: Path = DIR / "tests/nomad.hcl"
assert config.exists()
cmd: List[str] = [
    *(["sudo"] if os.getuid() != 0 else []),
    str(exe),
    *f"agent -dev -config {quote(str(config))} -log-level {args.log_level}".split(),
]
cmdstr = " ".join(quote(x) for x in cmd)
print(f"+ {cmdstr}")
os.execvp(cmd[0], cmd)
