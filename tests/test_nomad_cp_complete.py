#!/usr/bin/env python3

import argparse
import os

from nomad_tools.entry_cp import ArgPath

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true")
parser.add_argument("path", nargs="+")
args = parser.parse_args()
COMP_DEBUG = os.environ["COMP_DEBUG"] = "1" if args.debug else ""
COMP_POINT = os.environ["COMP_POINT"] = str(len(args.path[0]))
arg = "".join(args.path)
print(f"COMP_DEBUG={COMP_DEBUG} COMP_POINT={COMP_POINT} incomplete={arg}")
for c in ArgPath.mk(arg).gen_shell_complete():
    print(f"{c.type} {c.value}")
