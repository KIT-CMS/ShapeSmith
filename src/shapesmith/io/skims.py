"""Parquet skims and their manifests (Spec §7 steps 5-6)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


class SkimMissingError(FileNotFoundError):
    pass


def skim_path(skim_dir: Path, channel: str, nick: str, basename: str) -> Path:
    return Path(skim_dir) / channel / nick / (Path(basename).stem + ".parquet")


def manifest_path(skim_dir: Path, channel: str, nick: str) -> Path:
    return Path(skim_dir) / channel / nick / "manifest.json"


def _replace(path: Path, write: Callable[[Path], None]) -> None:
    """Write through a sibling temporary file and rename it into place: readers never see a partial file,
    and an interrupted write leaves the previous file intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_skim(frame: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    _replace(path, lambda temporary: pq.write_table(table, temporary, compression="snappy"))


def read_skims(skim_dir: Path, channel: str, nicks: Iterable[str], columns: Iterable[str] | None) -> pd.DataFrame:
    """All skim rows of the given nicks in one DataFrame (only `columns` if given)."""
    frames = []
    for nick in nicks:
        directory = Path(skim_dir) / channel / nick
        files = sorted(directory.glob("*.parquet")) if directory.is_dir() else []
        if not files:
            raise SkimMissingError(f"no skims for {nick} in {directory} (run `shapesmith skim`)")
        dataset = ds.dataset([str(f) for f in files], format="parquet")
        if columns is not None:
            missing = sorted(set(columns) - set(dataset.schema.names))
            if missing:
                raise SkimMissingError(f"skims of {nick} in {directory} lack columns {missing} (re-run `shapesmith skim --force` after changing the analysis)")
        table = dataset.to_table(columns=list(columns) if columns is not None else None)
        frames.append(table.to_pandas())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_manifest(path: Path, manifest: dict) -> None:
    text = json.dumps(manifest, indent=2, sort_keys=True)
    _replace(path, lambda temporary: temporary.write_text(text))


def read_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())
