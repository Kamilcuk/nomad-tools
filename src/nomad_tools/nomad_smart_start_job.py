#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import json
import logging
import shlex
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, TextIO, Tuple, Union

from .common import mynomad
from .nomadlib.connection import JobSubmission

log = logging.getLogger(__name__)

###############################################################################


def nomad_job_run(
    file: str,
    isjson: Optional[bool] = None,
    stdin: Optional[Union[TextIO, int]] = None,
    input: Optional[str] = None,
) -> str:
    """Call nomad job run to start a Nomad job from a file or from stdin or from input"""
    # When file is "-", either stdin or input has to be given.
    # log.debug(f"file={file} isjson={isjson} stdin={stdin} input={input}")
    if file == "-":
        if stdin is None:
            assert input is not None
        else:
            assert input is None
    else:
        assert stdin is None
        assert input is None
    #
    jsonarg = "-json" if isjson else ""
    cmd = shlex.split(f"nomad job run -detach -verbose {jsonarg} {shlex.quote(file)}")
    log.info(f"+ {' '.join(shlex.quote(x) for x in cmd)}")
    try:
        rr = subprocess.run(
            cmd,
            stdin=stdin
            if stdin is not None
            else None
            if input is not None
            else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            input=input,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        # nomad will print its error, we can just exit
        exit(e.returncode)
    # Extract evaluation id rfom Nomad output.
    data = rr.stdout.strip()
    for line in data.splitlines():
        log.info(line)
    evalid = next(
        x.split(" ", 3)[-1] for x in data.splitlines() if "eval" in x.lower()
    ).strip()
    return evalid


###############################################################################


@dataclasses.dataclass
class Detail(ABC):
    input: str
    """The actual inputed string from the user."""
    forcejson: bool
    """This is true, we know that content is a JSON."""

    @abstractmethod
    def seekable_read(self) -> Optional[str]:
        """Read the content of the file only if it is seekable."""
        raise NotImplementedError()

    def read(self) -> str:
        """Return the content of the file"""
        content = self.seekable_read()
        assert content is not None
        return content

    @abstractmethod
    def nomad_start(self) -> str:
        raise NotImplementedError()

    def close(self):
        """Call on finally to close potentially opened file"""
        pass

    def seekable_isjson(self):
        """If the underlying file is seekable, check if it is a json"""
        if self.forcejson:
            return True
        content = self.seekable_read()
        if content is not None:
            # Detect if its a json if we can.
            try:
                json.loads(content)
                return True
            except json.JSONDecodeError:
                return False
        return None

    def nomad_start_in(
        self,
        file: str,
        stdin: Optional[Union[TextIO, int]] = None,
        input: Optional[str] = None,
    ):
        return nomad_job_run(
            file=file,
            isjson=self.seekable_isjson(),
            stdin=stdin,
            input=input,
        )

    def read_dict(self) -> Tuple[str, str, dict]:
        content = self.read()
        # Assume input is a string with valid job specification.
        try:
            # try json
            data = json.loads(content)
            format = "json"
        except json.JSONDecodeError:
            if self.forcejson:
                raise
            data = json.loads(mynomad.jobhcl2json(content))
            format = "hcl2"
        return content, format, data

    def api_start(self):
        content, format, data = self.read_dict()
        evalid = mynomad.start_job(data, JobSubmission(Source=content, Format=format))[
            "EvalID"
        ]
        return evalid

    def run(self) -> str:
        return self.nomad_start() if shutil.which("nomad") else self.api_start()


@dataclasses.dataclass
class ModeString(Detail):
    def seekable_read(self):
        return self.input

    def nomad_start(self):
        return self.nomad_start_in(file="-", input=self.input)


class ModeFilehandle(Detail):
    @abstractmethod
    def filehandle(self):
        raise NotImplementedError()

    def seekable_read(self):
        filehandle = self.filehandle()
        if filehandle.seekable():
            content = filehandle.read()
            filehandle.seek(0)
            return content
        return None

    def read(self):
        return self.filehandle().read()


class ModeStdin(ModeFilehandle):
    def filehandle(self):
        return sys.stdin

    def nomad_start(self):
        return self.nomad_start_in(file="-", stdin=sys.stdin)


@dataclasses.dataclass
class ModeFile(ModeFilehandle):
    file: Path
    _filehandle: Optional[TextIO] = None

    def filehandle(self):
        if self._filehandle is None:
            self._filehandle = self.file.open()
        return self._filehandle

    def close(self):
        if self._filehandle is not None:
            self._filehandle.close()

    def nomad_start(self):
        return self.nomad_start_in(file=str(self.file))


def nomad_smart_start_job(input: str, forcejson: bool = False) -> str:
    """
    Be "smartly" by detecting if input is a file or not and run it depending on that.
    If input contains a newline, it means it is a JSON or HCL job specification.
    If input is the string '-', it means we run from standard input.
    If input does not have a newline nor is '-', the input is a filename.
    If input is standard input or a filename, if it is seekable and forcejson is false,
    then read the input and if it is a valid json, parse it as a JSON.
    Otherwise parse it as a HCL file.
    If Nomad executable is available, run the job with Nomad executable.
    Otherwise use the API.
    """
    mode = (
        ModeStdin(input, forcejson)
        if input == "-"
        else ModeString(input, forcejson)
        if "\n" in input or (input.count("{") > 5 and input.count("}") > 5)
        else ModeFile(input, forcejson, Path(input))
    )
    try:
        return mode.run()
    finally:
        mode.close()
