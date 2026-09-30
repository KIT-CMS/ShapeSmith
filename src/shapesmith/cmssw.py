"""Run commands inside a CMSSW environment (combine, CombineHarvester) in a bash subshell.

The subshell starts from a clean environment: the Python environment ShapeSmith runs in (e.g. an LCG view) would
otherwise shadow CMSSW's libraries and Python packages. With a timeout, the whole process group of the subshell is
killed when it expires (a fit can iterate without end). Its output is logged line by line at DEBUG, and its last
lines at ERROR when it fails.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Sequence

from shapesmith.config import CombineConfig

logger = logging.getLogger(__name__)
KEPT_VARIABLES = ("HOME", "USER", "LOGNAME", "TMPDIR", "X509_USER_PROXY")  # the environment a CMSSW subshell starts with, besides PATH
TAIL = 40  # lines of output logged with a failure
READER_TIMEOUT = 10.0  # seconds to wait for the rest of the output after the subshell ended


def script(combine: CombineConfig, cwd: Path, commands: Sequence[str]) -> str:
    """The bash script: CMSSW environment of `combine.cmssw_dir`, then the commands in `cwd`, stopping at the first failure."""
    return "\n".join(
        [
            "set -e",
            f"export SCRAM_ARCH={combine.scram_arch}",
            "source /cvmfs/cms.cern.ch/cmsset_default.sh",
            f"cd {combine.cmssw_dir}/src",
            "eval $(scramv1 runtime -sh)",
            f"cd {Path(cwd).resolve()}",
            *commands,
        ]
    )


def clean_environment() -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", **{name: os.environ[name] for name in KEPT_VARIABLES if name in os.environ}}


def _read(stream, tail: deque) -> None:
    for line in stream:
        tail.append(line.rstrip("\n"))
        logger.debug(f"cmssw: {tail[-1]}")
    stream.close()


def _last_lines(tail: deque) -> str:
    return ", last output:\n" + "\n".join(tail) if tail else ", no output"


def run(commands: Sequence[str], combine: CombineConfig | None, cwd: Path, timeout: float | None = None) -> None:
    """Raise CalledProcessError on a failing command, TimeoutExpired after `timeout` seconds."""
    if combine is None:
        raise ValueError("the run configuration needs `combine` (cmssw_dir) to run CMSSW commands")
    logger.info(f"running {len(commands)} CMSSW commands in {cwd}")
    text = script(combine, cwd, commands)
    logger.debug(f"CMSSW script:\n{text}")
    arguments = ["bash", "-c", text]
    process = subprocess.Popen(arguments, env=clean_environment(), start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    tail: deque = deque(maxlen=TAIL)
    reader = threading.Thread(target=_read, args=(process.stdout, tail), daemon=True)
    reader.start()
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        reader.join(READER_TIMEOUT)
        logger.error(f"CMSSW commands in {cwd} killed after {timeout:g} s{_last_lines(tail)}")
        raise
    reader.join(READER_TIMEOUT)
    if code:
        logger.error(f"CMSSW commands in {cwd} failed with exit code {code}{_last_lines(tail)}")
        raise subprocess.CalledProcessError(code, arguments)
    logger.debug(f"CMSSW commands in {cwd} done")
