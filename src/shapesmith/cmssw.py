"""Run commands inside a CMSSW environment (combine, CombineHarvester) in a bash subshell.

The subshell starts from a clean environment: the Python environment ShapeSmith runs in (e.g. an LCG view) would
otherwise shadow CMSSW's libraries and Python packages. With a timeout, the whole process group of the subshell is
killed when it expires (a fit can iterate without end).
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import Sequence

from shapesmith.config import CombineConfig

logger = logging.getLogger(__name__)
KEPT_VARIABLES = ("HOME", "USER", "LOGNAME", "TMPDIR", "X509_USER_PROXY")  # the environment a CMSSW subshell starts with, besides PATH


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


def run(commands: Sequence[str], combine: CombineConfig | None, cwd: Path, timeout: float | None = None) -> None:
    """Raise CalledProcessError on a failing command, TimeoutExpired after `timeout` seconds."""
    if combine is None:
        raise ValueError("the run configuration needs `combine` (cmssw_dir) to run CMSSW commands")
    logger.info(f"running {len(commands)} CMSSW commands in {cwd}")
    arguments = ["bash", "-c", script(combine, cwd, commands)]
    process = subprocess.Popen(arguments, env=clean_environment(), start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, arguments)
