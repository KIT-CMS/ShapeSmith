"""Parquet skims and their manifests: <skim_dir>/<channel>/<nick>/{*.parquet, manifest.json}.

Repeated events: read_skims drops, in the samples it is told to, every row whose event key (e.g. run, lumi, event) an
earlier row of the same sample has, in file order (repeated_rows): an event that two n-tuple files of a sample both
hold is counted once. The skims themselves keep every row.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


SKIM_COLUMNS = ("sample_nick", "norm_weight", "is_data", "is_mc", "is_embedding")  # added to every skim

logger = logging.getLogger(__name__)


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


def _files(skim_dir: Path, channel: str, nick: str) -> list[Path]:
    directory = Path(skim_dir) / channel / nick
    files = sorted(directory.glob("*.parquet")) if directory.is_dir() else []
    if not files:
        raise SkimMissingError(f"no skims for {nick} in {directory} (run `shapesmith skim`)")
    return files


def schema(skim_dir: Path, channel: str, nicks: Iterable[str]) -> set[str]:
    """The columns every skim of the given nicks holds."""
    columns = None
    for nick in nicks:
        names = set(pq.read_schema(_files(skim_dir, channel, nick)[0]).names)
        columns = names if columns is None else columns & names
    return columns or set()


def repeated_rows(table: pa.Table, key: Iterable[str]) -> np.ndarray:
    """Per row whether an earlier row of the table has the same values of the `key` columns (the first one is kept).
    Only rows whose last key column (the event number) occurs more than once can repeat: the full key is compared among
    those (half the time of comparing every row on 19 M rows)."""
    key = list(key)
    values = {column: table.column(column).to_numpy() for column in key}
    candidate = pd.Series(values[key[-1]]).duplicated(keep=False).to_numpy()
    repeated = np.zeros(len(candidate), dtype=bool)
    if candidate.any():
        repeated[candidate] = pd.DataFrame({column: v[candidate] for column, v in values.items()}).duplicated(keep="first").to_numpy()
    return repeated


def _missing(dataset: ds.Dataset, columns: Iterable[str], skim_dir: Path, channel: str, nick: str) -> None:
    missing = sorted(set(columns) - set(dataset.schema.names))
    if missing:
        raise SkimMissingError(f"skims of {nick} in {Path(skim_dir) / channel / nick} lack columns {missing} (re-run `shapesmith skim --force` after changing the analysis)")


def read_skims(skim_dir: Path, channel: str, nicks: Iterable[str], columns: Iterable[str] | None, key: tuple[str, ...] = (), unique: Iterable[str] = ()) -> pd.DataFrame:
    """All skim rows of the given nicks in one DataFrame, in nick and file order (only `columns` if given). The nicks in
    `unique` keep only the first row of every value of the `key` columns (module docstring); without `key` none."""
    frames = []
    nicks = list(nicks)
    unique = set(unique) if key else set()
    for nick in nicks:
        dataset = ds.dataset([str(f) for f in _files(skim_dir, channel, nick)], format="parquet")
        wanted = None if columns is None else list(dict.fromkeys(columns))
        if nick in unique and wanted is not None:
            wanted += [column for column in key if column not in wanted]
        if wanted is not None:
            _missing(dataset, wanted, skim_dir, channel, nick)
        elif nick in unique:
            _missing(dataset, key, skim_dir, channel, nick)
        table = dataset.to_table(columns=wanted)
        if nick in unique:
            repeated = repeated_rows(table, key)
            if repeated.any():
                table = table.filter(pa.array(~repeated))
                logger.debug(f"{channel}/{nick}: {int(repeated.sum())} of {len(repeated)} skim rows repeat the event ({', '.join(key)}) of an earlier row, dropped")
            if columns is not None:
                table = table.select([column for column in wanted if column in set(columns)])
        frames.append(table.to_pandas())
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.debug(f"{channel}: {len(frame)} skim rows, {len(frame.columns)} columns of {len(nicks)} samples read")
    return frame


def repeated_events(skim_dir: Path, channel: str, nick: str, key: tuple[str, ...]) -> dict:
    """What read_skims drops of a sample: its skim rows, the rows that repeat the event of an earlier row and per skim
    file the repeated rows in it (only files with any), from the `key` columns alone."""
    files = _files(skim_dir, channel, nick)
    dataset = ds.dataset([str(f) for f in files], format="parquet")
    _missing(dataset, key, skim_dir, channel, nick)
    tables = [pq.read_table(f, columns=list(key)) for f in files]
    repeated = repeated_rows(pa.concat_tables(tables), key) if tables else np.zeros(0, dtype=bool)
    bounds = np.cumsum([0] + [t.num_rows for t in tables])
    per_file = {f.name: int(repeated[a:b].sum()) for f, a, b in zip(files, bounds[:-1], bounds[1:])}
    return {"rows": int(len(repeated)), "repeated": int(repeated.sum()), "files": {name: n for name, n in per_file.items() if n}}


def write_manifest(path: Path, manifest: dict) -> None:
    text = json.dumps(manifest, indent=2, sort_keys=True)
    _replace(path, lambda temporary: temporary.write_text(text))


def read_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())
