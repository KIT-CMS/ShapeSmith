"""Run independent jobs inline (workers <= 1) or in a spawn-based process pool.

`spawn` instead of `fork`: uproot, pyarrow and XRootD start threads, and forking a multi-threaded
process can deadlock the children. With one worker everything runs in the calling process, which
keeps tracebacks and debuggers usable (`--workers 1` / `workers: 1`). The workers log through the
main process (logs.forwarding); the progress is logged about every tenth of the jobs.
"""
from __future__ import annotations

import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, Iterator, TypeVar

from shapesmith import logs

T = TypeVar("T")

logger = logging.getLogger(__name__)


def run_jobs(function: Callable[[tuple], T], jobs: Iterable[tuple], workers: int, label: str = "jobs") -> Iterator[tuple[tuple, T | None, Exception | None]]:
    """Yield (job, result, error) for every job; errors are handed back instead of raised so that callers can report all of them."""
    jobs = list(jobs)
    steps, start, done = set(logs.progress_steps(len(jobs))), time.monotonic(), 0

    def progress() -> None:
        nonlocal done
        done += 1
        if done in steps:
            logger.info(f"{label}: {done}/{len(jobs)} jobs done ({100 * done / len(jobs):.0f} %), {logs.duration(time.monotonic() - start)}")

    if workers <= 1 or len(jobs) <= 1:
        logger.debug(f"{label}: {len(jobs)} jobs inline")
        for job in jobs:
            try:
                result = function(job)
            except Exception as error:  # reported by the caller
                progress()
                yield job, None, error
                continue
            progress()
            yield job, result, None
        return
    context = multiprocessing.get_context("spawn")
    logger.debug(f"{label}: {len(jobs)} jobs in {min(workers, len(jobs))} worker processes")
    with logs.forwarding(context) as (initializer, initargs):
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context, initializer=initializer, initargs=initargs) as pool:
            futures = {pool.submit(function, job): job for job in jobs}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as error:
                    progress()
                    yield futures[future], None, error
                    continue
                progress()
                yield futures[future], result, None
