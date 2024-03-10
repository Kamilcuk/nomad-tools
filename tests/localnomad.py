#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path
from shlex import quote, split
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("version", nargs="?")
args = parser.parse_args()
DIR = Path(__file__).parent
os.chdir(DIR)

if not args.version:
    cmdstr: str = "nomad-tools downloadrelease --showversion nomad"
    cmd: List[str] = split(cmdstr)
    print(f"+ {cmdstr}")
    args.version = subprocess.check_output(cmd, text=True).strip()

exe = Path("build") / f"nomad{args.version}"
if not exe.exists():
    cmdstr = f"nomad-tools downloadrelease -p {quote(args.version)} nomad {quote(str(exe))}"
    cmd = split(cmdstr)
    print(f"+ {cmdstr}")
    subprocess.check_call(cmd)

cmd = [
    *(["sudo"] if os.getuid() != 0 else []),
    str(exe),
    *"agent -dev -config ./nomad.hcl -log-level debug".split(),
]
cmdstr = " ".join(quote(x) for x in cmd)
print(f"+ {cmdstr}")
os.execvp(cmd[0], cmd)
