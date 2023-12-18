#!/usr/bin/env python3

import dataclasses
import importlib
import stat
import string
import subprocess
from pathlib import Path
from typing import Any, List

import click
from click.testing import CliRunner

from nomad_tools.common import get_entrypoints

runner = CliRunner()


def subcommands(obj):
    if isinstance(obj, click.Group):
        return obj.commands.values()
    return []


@dataclasses.dataclass
class Command:
    obj: Any

    def key(self):
        return self.obj.__name__.split(".")[-1]

    def cmdname(self):
        return self.key().replace("_", "-")

    def gen(self, cmd=None):
        cmdarr: List[str] = [*([cmd.name] if cmd and cmd.name else []), "--help"]
        ret = runner.invoke(self.obj.cli, cmdarr, prog_name=self.cmdname())
        assert ret.exit_code == 0, f"{self.obj} {cmdarr} {ret}"
        help = ret.output
        return f"""

```
+ {self.cmdname()} {' '.join(cmdarr)}
{help}
```

"""

    def run(self):
        if self.key() == "nomadt":
            return f"""

```
+ {self.cmdname()} --help
{subprocess.check_output("nomadt --help".split(), text=True)}
```

"""
        ret = self.gen()
        for cmd in subcommands(self.obj.cli):
            ret += self.gen(cmd)
        return ret


if __name__ == "__main__":
    dir = Path(__file__).parent
    # Generate help strings
    template_parameters = {}
    for e in get_entrypoints():
        i = importlib.import_module(f"nomad_tools.{e.replace('-', '_')}")
        p = Command(i)
        template_parameters[p.key()] = p.run()
    # Generate template.
    with (dir / "template_README.md").open() as f:
        template = f.read()
    out = string.Template(template).safe_substitute(template_parameters)
    # Output
    outputfile = dir / "README.md"
    outputfile.chmod(outputfile.stat().st_mode & ~stat.S_IWRITE)
    with outputfile.open() as f:
        if out == f.read():
            print("NO DIFFERENCE")
            exit()
    outputfile.chmod(outputfile.stat().st_mode | stat.S_IWRITE)
    with outputfile.open("w") as f:
        print(out, end="", file=f)
    outputfile.chmod(outputfile.stat().st_mode & ~stat.S_IWRITE)
    print(f"Written {len(out.splitlines())} lines to {outputfile}")
