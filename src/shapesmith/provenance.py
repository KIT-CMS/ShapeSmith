"""What produced an output: versions, git hashes, the run configuration."""
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from shapesmith import __version__
from shapesmith.config import RunConfig


def git_hash(path: Path) -> str | None:
    """HEAD commit of the git checkout containing `path`, or None if there is none (or git is missing)."""
    path = Path(path)
    directory = path.parent if path.is_file() else path
    try:
        return subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def record(config: RunConfig) -> dict:
    """Versions and git hashes of core, analysis and sample database, and the resolved run configuration with its hash."""
    module = importlib.import_module(config.analysis.partition(":")[0])
    resolved = json.loads(config.model_dump_json())
    return {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shapesmith": {"version": __version__, "git": git_hash(Path(__file__))},
        "analysis": {"module": config.analysis, "git": git_hash(Path(module.__file__))},
        "sample_database": {"path": str(config.sample_database), "git": git_hash(config.sample_database)} if config.sample_database else None,
        "ntuples": config.ntuples.base,
        "config": resolved,
        "config_sha256": hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest(),
    }


def write_versions(config: RunConfig, analysis_name: str, directory: Path) -> Path:
    """`<directory>/versions/<analysis_name>.json`: one record per analysis, so analyses sharing a directory keep theirs."""
    path = Path(directory) / "versions" / f"{analysis_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record(config), indent=2))
    return path
