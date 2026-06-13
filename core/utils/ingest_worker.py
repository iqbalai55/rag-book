"""In-process async ingest worker (ADR-0003).

The worker is a single asyncio task started in the FastAPI lifespan. It loops
forever, polling `ingested_books` for `pending` rows, claiming one via
`FOR UPDATE SKIP LOCKED`, running the pipeline, and writing back `succeeded`
or `failed`. There is no HTTP callback, no Edge Function, no separate worker
runtime — the in-process loop is the entire worker surface.

The pipeline function is injected (`ingest_book_fn`) so the worker is testable
from pytest with a fake that does not touch Docling or Qdrant.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Awaitable, Callable, Optional

from core.utils import ingest_jobs
from core.utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)


# The signature the worker expects. `chunks_count` is the return value of the
# pipeline (number of chunks written). The function MUST raise on failure; the
# worker translates the exception into a `mark_failed` write.
PipelineFn = Callable[[str, str, str], int]
"""(book_id, pdf_path, storage_url) -> chunks_count"""


async def tick(
    pool_or_dsn,
    ingest_book_fn: PipelineFn,
    cache_manager: Optional[CacheManager] = None,
) -> Optional[ingest_jobs.IngestJob]:
    """Claim the oldest `pending` row and run the pipeline once.

    Returns the claimed `IngestJob` if work was done, `None` if the queue was
    empty. Errors are logged and surfaced as `mark_failed` writes; this
    function does not raise so the lifespan loop survives a single bad row.
    """
    claim = await asyncio.to_thread(ingest_jobs.claim_next_pending, pool_or_dsn)
    if claim is None:
        return None

    # Re-check the row state. A `DELETE` while we were waiting may have moved
    # the row to `failed`; if so, leave it alone and return.
    current = await asyncio.to_thread(ingest_jobs.get_job, claim.book_id, pool_or_dsn)
    if current is None or current.status != "running":
        logger.info(
            "worker.tick: row for book_id=%s moved out of 'running' (now %s); skipping",
            claim.book_id,
            None if current is None else current.status,
        )
        return claim

    if claim.storage_url is None or claim.filename is None:
        await asyncio.to_thread(
            ingest_jobs.mark_failed,
            claim.book_id,
            "missing storage_url or filename on claimed row",
            pool_or_dsn,
        )
        return claim

    local_pdf_path: Optional[str] = None
    try:
        # Same-replica safety net (per-bookId asyncio lock). Skip-locked is the
        # cross-replica guarantee; this is belt-and-braces.
        if cache_manager is not None:
            async with cache_manager.acquire_book_lock(claim.book_id):
                local_pdf_path = await asyncio.to_thread(
                    _download_to_temp, claim.storage_url
                )
                chunks_count = await asyncio.to_thread(
                    ingest_book_fn,
                    claim.book_id,
                    local_pdf_path,
                    claim.storage_url,
                )
        else:
            local_pdf_path = await asyncio.to_thread(
                _download_to_temp, claim.storage_url
            )
            chunks_count = await asyncio.to_thread(
                ingest_book_fn,
                claim.book_id,
                local_pdf_path,
                claim.storage_url,
            )

        await asyncio.to_thread(
            ingest_jobs.mark_succeeded,
            claim.book_id,
            int(chunks_count or 0),
            pool_or_dsn,
        )
        logger.info(
            "worker.tick: book_id=%s succeeded (%d chunks)", claim.book_id, chunks_count
        )
    except Exception as exc:
        logger.exception("worker.tick: book_id=%s failed", claim.book_id)
        try:
            await asyncio.to_thread(
                ingest_jobs.mark_failed,
                claim.book_id,
                str(exc)[:1000],
                pool_or_dsn,
            )
        except Exception:
            logger.exception("worker.tick: failed to write mark_failed for %s", claim.book_id)
    finally:
        if local_pdf_path:
            try:
                os.remove(local_pdf_path)
            except OSError:
                pass

    return claim


def _download_to_temp(storage_url: str) -> str:
    """Download the PDF at `storage_url` to a local temp file and return the path.

    Uses `urllib` so we do not add a dependency on `httpx` / `requests` for the
    worker module — `requests` is already a transitive dep, but a small surface
    is easier to reason about.
    """
    import tempfile
    import urllib.request

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    urllib.request.urlretrieve(storage_url, path)
    return path


async def reconcile_stale(stale_timeout_seconds: int, pool_or_dsn) -> int:
    """Reset stale `running` rows back to `pending`. Returns the reset count.

    Called once on FastAPI lifespan startup. Idempotent — safe to call on
    every restart.
    """
    return await asyncio.to_thread(
        ingest_jobs.reconcile_stale, stale_timeout_seconds, pool_or_dsn
    )


async def run_worker_loop(
    pool_or_dsn,
    ingest_book_fn: PipelineFn,
    interval_seconds: float = 5.0,
    stale_timeout_seconds: int = 1800,
    cache_manager: Optional[CacheManager] = None,
    reconcile_on_start: bool = True,
) -> None:
    """The lifespan-spawned coroutine. Loops forever; cancellable.

    `interval_seconds` is the sleep between ticks. Errors inside `tick` are
    logged and swallowed so a single bad row does not kill the loop.
    """
    if reconcile_on_start:
        try:
            n = await reconcile_stale(stale_timeout_seconds, pool_or_dsn)
            if n:
                logger.info("worker.startup: reconciled %d stale running row(s)", n)
        except Exception:
            logger.exception("worker.startup: reconcile_stale failed; continuing")

    logger.info(
        "worker.start: loop started (interval=%.1fs, stale_timeout=%ds)",
        interval_seconds,
        stale_timeout_seconds,
    )
    try:
        while True:
            try:
                await tick(pool_or_dsn, ingest_book_fn, cache_manager)
            except Exception:
                logger.exception("worker.loop: tick raised; continuing")
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        logger.info("worker.stop: cancelled")
        raise
