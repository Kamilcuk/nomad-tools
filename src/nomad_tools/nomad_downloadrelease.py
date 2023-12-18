import io
import logging
import platform
import re
import shutil
import stat
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional
from zipfile import ZipFile

import click
import requests

from .common_click import common_options

log = logging.getLogger(__name__)


class MyHTMLParser(HTMLParser):
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix
        self.datas: List[str] = []

    def handle_data(self, data: str):
        if data.startswith(self.prefix):
            vv = data[len(self.prefix) :]
            self.datas.append(vv)


def get_arch() -> str:
    if platform.machine() in ["AMD64", "x86_64"]:
        return "amd64"
    return ""


@click.command(
    help="""
\b
Download specific binary from releases.hashicorp
Examples:
    %(prog) -p 1.7.2 nomad ./bin/nomad
    %(prog) consul ./bin/consul
"""
)
@click.option("--verbose", is_flag=True)
@click.option(
    "-p", "--pinversion", help="Use this version instead of autodetecting latest"
)
@click.option(
    "--os",
    help="Use this operating system instead of host",
    default=platform.system().lower(),
    show_default=True,
)
@click.option(
    "-a",
    "--arch",
    help="Use this architecture instead of host",
    default=get_arch(),
    show_default=True,
)
@click.argument(
    "tool",
    shell_complete=click.Choice(
        "nomad consul consul-template vault".split()
    ).shell_complete,
)
@click.argument(
    "destination",
    type=click.Path(writable=True, path_type=Path),
    default=Path("."),
)
@click.option(
    "--suffix",
    default="",
    help="When searching for latest version, only get versions with this suffix",
)
@click.option("--ent", is_flag=True, help="Equal to --suffix=+ent")
@common_options()
def cli(
    pinversion: Optional[str],
    arch: str,
    os: str,
    tool: str,
    destination: Path,
    verbose: bool,
    suffix: str,
    ent: bool,
):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    if ent:
        assert suffix == "", "--ent and --suffix are in conflict"
        suffix = "+ent"
    #
    url = f"https://releases.hashicorp.com/{tool}"
    if pinversion is None:
        page = requests.get(url)
        parser = MyHTMLParser(f"{tool}_")
        parser.feed(page.text)
        versionsstr: str = "\n".join(parser.datas) if parser.datas else page.text
        rgx = re.compile(r"([0-9]+)\.([0-9]+)\.([0-9]+)([-+a-z1-9A-Z.]+)?")

        @dataclass(order=True)
        class Version:
            a: int
            b: int
            c: int
            suff: str

        versions: List[Version] = sorted(
            Version(a, b, c, suff or "")
            for a, b, c, suff in rgx.findall(versionsstr)
            if (suff or "") == suffix
        )
        log.debug(f"Versions = {versions}")
        vv = versions[-1]
        log.debug(f"Version = {pinversion}")
        pinversion = f"{vv.a}.{vv.b}.{vv.c}{vv.suff}"
    #
    zipurl = f"{url}/{pinversion}/{tool}_{pinversion}_{os.lower()}_{arch.lower()}.zip"
    destfile: Path = destination / tool if destination.is_dir() else destination
    destfile.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {zipurl} to {destfile}")
    with requests.get(zipurl, stream=True) as stream:
        with ZipFile(io.BytesIO(stream.content)) as zipf:
            with zipf.open(tool) as inf:
                with destfile.open("wb") as outf:
                    shutil.copyfileobj(inf, outf)
            perms = zipf.getinfo(tool).external_attr >> 16
            destfile.chmod(perms)
    #
    deststat = destfile.stat()
    mbsize = deststat.st_size / (1024 * 1024)
    perms = stat.filemode(deststat.st_mode)
    log.info(f"{zipurl} -> {perms} {mbsize:.1f}MB {destfile}")


if __name__ == "__main__":
    cli()
