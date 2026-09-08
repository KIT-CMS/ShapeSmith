"""Run independent jobs inline (workers <= 1) or in a spawn-based process pool.

`spawn` instead of `fork`: uproot, pyarrow and XRootD start threads, and forking a multi-threaded
process can deadlock the children. With one worker everything runs in the calling process, which
keeps tracebacks and debuggers usable (`--workers 1` / `workers: 1`).
"""
from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")


def run_jobs(function: Callable[[tuple], T], jobs: Iterable[tuple], workers: int) -> Iterator[tuple[tuple, T | None, Exception | None]]:
    """Yield (job, result, error) for every job; errors are handed back instead of raised so that callers can report all of them."""
    jobs = list(jobs)
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            try:
                yield job, function(job), None
            except Exception as error:  # reported by the caller
                yield job, None, error
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(function, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                yield futures[future], future.result(), None
            except Exception as error:
                yield futures[future], None, error
