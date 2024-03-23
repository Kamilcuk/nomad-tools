#!/usr/bin/env python3
import os
import select
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional, TypeVar

T = TypeVar("T")


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.2f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.2f}Yi{suffix}"


class _TransferStatsFileno:
    def __init__(self, bufsize: int = 64 * 1024):
        self.bufsize: int = bufsize
        self.start: float = time.time()
        self.size: int = 0
        self.lock = threading.Lock()
        self.end = threading.Event()
        self.pipe = os.pipe()
        self.thread = threading.Thread(target=self.__stats)

    def __stats_in(self):
        with self.lock:
            size = self.size
        duration = time.time() - self.start
        speed: float = size / duration
        durationstr = f"{int(duration / 3600)}:{int(duration / 60 % 60):02d}:{int(duration % 60):02d}"
        print(
            f"{sizeof_fmt(size)} {durationstr} [{sizeof_fmt(speed)}/s]",
            file=sys.stderr,
            flush=True,
        )

    def __stats(self):
        while not self.end.wait(1):
            self.__stats_in()
        self.__stats_in()

    def run(self, fin: int, fout: int):
        self.thread.start()
        os.set_blocking(fin, False)
        try:
            while select.select([fin], [], [])[0] != []:
                buf = os.read(fin, self.bufsize)
                if not buf:
                    return
                os.write(fout, buf)
                with self.lock:
                    self.size += len(buf)
        finally:
            self.end.set()
        self.thread.join()


def transfer_stats(src: int, dst: int, pv: Optional[bool] = None):
    if pv is True or (pv is None and shutil.which("pv")):
        subprocess.check_call(["pv"], stdin=src, stdout=dst)
    else:
        _TransferStatsFileno().run(src, dst)
