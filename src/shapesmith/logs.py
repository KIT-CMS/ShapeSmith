"""Logging: the console (rich, on stderr) and a log file per command, one level for ShapeSmith and the analysis.

The level (`log_level` of the run configuration) applies to the logger `shapesmith` and to those of the analysis
packages; every other library (matplotlib, numexpr, uproot, ...) logs from WARNING on, so that DEBUG stays readable.
Python warnings are logged as well. The console shows the messages, the log file also the tracebacks, the time, the
process and the logger. Worker processes of `parallel.run_jobs` hand their records to the main process through a
queue (`worker_init`, `forwarding`), so their messages reach the same handlers.
"""
from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from rich.console import Console
from rich.logging import RichHandler

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(processName)s %(name)s: %(message)s"
CORE = "shapesmith"
OTHERS = logging.WARNING  # the level of every other logger

_installed: list[logging.Handler] = []  # the handlers of configure(), removed by reset()
_levels: dict[str, int] = {}  # the package levels of configure(), handed to the workers


class _MessageOnly(logging.Formatter):
    """The message without the traceback: on the console, typer shows the traceback of a failing command."""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


def configure(level: str = "INFO", log_file: Path | None = None, packages: Iterable[str] = ()) -> Path | None:
    """Log to the console and, if given, to `log_file`, at `level` for `shapesmith` and `packages`; replaces the
    handlers of an earlier call."""
    reset()
    root = logging.getLogger()
    console = RichHandler(console=Console(stderr=True), show_path=False, markup=False)
    console.setFormatter(_MessageOnly())
    _installed.append(console)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file = logging.FileHandler(log_file, encoding="utf-8")
        file.setFormatter(logging.Formatter(FILE_FORMAT))
        _installed.append(file)
    for handler in _installed:
        root.addHandler(handler)
    root.setLevel(OTHERS)
    for name in dict.fromkeys((CORE, *packages)):
        logging.getLogger(name).setLevel(level)
        _levels[name] = logging.getLogger(name).level
    logging.captureWarnings(True)
    return log_file


def reset() -> None:
    """Remove and close the handlers of configure() (the tests configure more than once per process)."""
    root = logging.getLogger()
    for handler in _installed:
        root.removeHandler(handler)
        handler.close()
    _installed.clear()
    _levels.clear()
    logging.captureWarnings(False)


def active() -> bool:
    return bool(_installed)


def log_path(directory: Path, command: str) -> Path:
    """`<directory>/logs/<command>_<YYYYmmdd-HHMMSS>.log`."""
    return Path(directory) / "logs" / f"{command}_{datetime.now():%Y%m%d-%H%M%S}.log"


def levels() -> dict[str, int]:
    """The package levels for the workers: those of configure(), else the effective level of `shapesmith`."""
    return dict(_levels) or {CORE: logging.getLogger(CORE).getEffectiveLevel()}


def worker_init(queue, package_levels: dict[str, int]) -> None:
    """Initializer of a worker process: every record goes to the queue, with the package levels of the main process."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(logging.handlers.QueueHandler(queue))
    root.setLevel(OTHERS)
    for name, level in package_levels.items():
        logging.getLogger(name).setLevel(level)
    logging.captureWarnings(True)


class _ToLoggers(logging.Handler):
    """Hand a record of a worker to the logger of its name in the main process, and so to its handlers."""

    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


@contextmanager
def forwarding(context: multiprocessing.context.BaseContext) -> Iterator[tuple]:
    """The initializer and its arguments for a process pool of `context`, while a listener forwards the records."""
    queue = context.Queue()
    listener = logging.handlers.QueueListener(queue, _ToLoggers())
    listener.start()
    try:
        yield worker_init, (queue, levels())
    finally:
        listener.stop()
        queue.close()


def progress_steps(total: int, parts: int = 10) -> list[int]:
    """The numbers of finished jobs at which the progress is logged: about every tenth, and the last."""
    return sorted({max(1, round(total * k / parts)) for k in range(1, parts + 1)}) if total else []


def duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes} min {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"
