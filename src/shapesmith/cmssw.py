"""Run commands inside a CMSSW environment (combine, CombineHarvester) in a bash subshell."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Sequence

from shapesmith.config import CombineConfig

logger = logging.getLogger(__name__)


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


def run(commands: Sequence[str], combine: CombineConfig | None, cwd: Path) -> None:
    if combine is None:
        raise ValueError("the run configuration needs `combine` (cmssw_dir) to run CMSSW commands")
    logger.info(f"running {len(commands)} CMSSW commands in {cwd}")
    subprocess.run(["bash", "-c", script(combine, cwd, commands)], check=True)
