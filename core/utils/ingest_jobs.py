"""Data-access module for the async-ingest lifecycle on `ingested_books`.

The HTTP handler and the in-process worker both call into this module; neither
touches `psycopg` directly. Mirrors the pattern in `core.utils.token_tracker`
and `core.utils.credit_manager` (one-shot connections, no shared pool).

Connection source: `SUPABASE_DB_URL`. Functions accept a `conn_or_dsn` argument
so tests can inject a connection / DSN without going through env vars.
"""
from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import psycopg

logger = logging.getLogger(__name__)

VALID_STATUSES = ("pending", "running", "succeeded", "failed")


@dataclass
class IngestJob:
    book_id: str
    status: str
    filename: Optional[str]
    storage_url: Optional[str]
    chunks_count: int
    error: Optional[str]
    created_at: Optional[datetime]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    row_id: Optional[uuid.UUID] = None

    def to_dict(self) -> dict:
        d = {
            "book_id": self.book_id,
            "status": self.status,
            "filename": self.filename,
            "storage_url": self.storage_url,
            "chunks_count": self.chunks_count,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }
        return d


def _resolve_dsn(conn_or_dsn) -> str:
    """Return a DSN string. Accepts a DSN string, a live connection, or None."""
    if conn_or_dsn is None:
        dsn = os.getenv("SUPABASE_DB_URL")
        if not dsn:
            raise RuntimeError("SUPABASE_DB_URL not set")
        return dsn
    if isinstance(conn_or_dsn, str):
        return conn_or_dsn
    raise TypeError("Pass a DSN string; pass-through of live connections not supported in this module")


def _row_to_job(row) -> IngestJob:
    return IngestJob(
        book_id=row[0],
        status=row[1],
        filename=row[2],
        storage_url=row[3],
        chunks_count=row[4] or 0,
        error=row[5],
        created_at=row[6],
        started_at=row[7],
        finished_at=row[8],
        row_id=row[9] if len(row) > 9 else None,
    )


# --------------------------------------------------------------------- reads

def get_job(book_id: str, conn_or_dsn=None) -> Optional[IngestJob]:
    """Return the job row for `book_id`, or None if no row exists."""
    dsn = _resolve_dsn(conn_or_dsn)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select book_id, status, filename, storage_url, chunks_count,
                       error, created_at, started_at, finished_at, id
                  from public.ingested_books
                 where book_id = %s
                """,
                (book_id,),
            )
            row = cur.fetchone()
            return _row_to_job(row) if row else None


def list_jobs(
    book_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    conn_or_dsn=None,
) -> list[IngestJob]:
    """List jobs. Filters optional. `limit` capped at 100."""
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")
    limit = max(1, min(int(limit), 100))

    where = []
    params: list = []
    if book_id is not None:
        where.append("book_id = %s")
        params.append(book_id)
    if status is not None:
        where.append("status = %s")
        params.append(status)
    where_sql = ("where " + " and ".join(where)) if where else ""

    dsn = _resolve_dsn(conn_or_dsn)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                select book_id, status, filename, storage_url, chunks_count,
                       error, created_at, started_at, finished_at, id
                  from public.ingested_books
                  {where_sql}
                 order by created_at desc
                 limit %s
                """,
                (*params, limit),
            )
            return [_row_to_job(r) for r in cur.fetchall()]


# --------------------------------------------------------------- mutations

def create_or_reset_job(
    book_id: str,
    filename: str,
    storage_url: str,
    user_id: Optional[str] = None,
    conn_or_dsn=None,
) -> IngestJob:
    """Upsert the job row to `status='pending'`. Idempotent re-enqueue.

    `user_id` is optional. The public `POST /book-qa/ingest` is a trusted-
    backend call and does not yet scope per-user (per ADR-0001 §"Tenant
    identity" and ADR-0003 §"Concurrency"); the column is informational
    until per-user scoping is added. The DB column is NULL-able per
    `scripts/migrations/003_async_ingest.sql`.
    """
    dsn = _resolve_dsn(conn_or_dsn)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.ingested_books
                    (book_id, filename, storage_url, user_id, status,
                     error, chunks_count, started_at, finished_at)
                values (%s, %s, %s, %s, 'pending', null, 0, null, null)
                on conflict (book_id) do update set
                    filename = excluded.filename,
                    storage_url = excluded.storage_url,
                    user_id = excluded.user_id,
                    status = 'pending',
                    error = null,
                    chunks_count = 0,
                    started_at = null,
                    finished_at = null
                """,
                (book_id, filename, storage_url, user_id),
            )
        conn.commit()
    return get_job(book_id, dsn)


def claim_next_pending(conn_or_dsn=None) -> Optional[IngestJob]:
    """Atomically claim the oldest `pending` row and mark it `running`.

    Uses `FOR UPDATE SKIP LOCKED` so two concurrent claimers never grab the
    same row (the cross-replica guarantee from ADR-0003 §"Concurrency").
    """
    dsn = _resolve_dsn(conn_or_dsn)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                with next as (
                    select id
                      from public.ingested_books
                     where status = 'pending'
                     order by created_at
                     for update skip locked
                     limit 1
                )
                update public.ingested_books j
                   set status = 'running',
                       started_at = now()
                  from next
                 where j.id = next.id
                returning j.book_id, j.status, j.filename, j.storage_url,
                          j.chunks_count, j.error, j.created_at,
                          j.started_at, j.finished_at, j.id
                """
            )
            row = cur.fetchone()
        conn.commit()
    return _row_to_job(row) if row else None


def mark_succeeded(book_id: str, chunks_count: int, conn_or_dsn=None) -> None:
    dsn = _resolve_dsn(conn_or_dsn)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.ingested_books
                   set status = 'succeeded',
                       chunks_count = %s,
                       finished_at = now(),
                       error = null
                 where book_id = %s
                """,
                (chunks_count, book_id),
            )
        conn.commit()


def mark_failed(book_id: str, error_message: str, conn_or_dsn=None) -> None:
    dsn = _resolve_dsn(conn_or_dsn)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.ingested_books
                   set status = 'failed',
                       error = %s,
                       finished_at = now()
                 where book_id = %s
                """,
                (error_message, book_id),
            )
        conn.commit()


def mark_deleted_by_reingest(book_id: str, conn_or_dsn=None) -> None:
    """Move any non-terminal row for `book_id` to `failed` with the
    re-ingest-delete marker. The in-process worker sees this on its next tick
    and skips the work."""
    dsn = _resolve_dsn(conn_or_dsn)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.ingested_books
                   set status = 'failed',
                       error = 're-ingest: chunks deleted',
                       finished_at = now()
                 where book_id = %s
                   and status in ('pending', 'running')
                """,
                (book_id,),
            )
        conn.commit()


def reset_for_retry(book_id: str, conn_or_dsn=None) -> bool:
    """Reset a `failed` row to `pending`. Returns True if the reset happened,
    False if the row was not in `failed` state (caller surfaces 409)."""
    dsn = _resolve_dsn(conn_or_dsn)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.ingested_books
                   set status = 'pending',
                       error = null,
                       started_at = null,
                       finished_at = null,
                       chunks_count = 0
                 where book_id = %s
                   and status = 'failed'
                """,
                (book_id,),
            )
            updated = cur.rowcount
        conn.commit()
    return updated > 0


def reconcile_stale(stale_timeout_seconds: int, conn_or_dsn=None) -> int:
    """Reset `running` rows whose `started_at` is older than the timeout back
    to `pending`. Returns the number of rows reset. Called on FastAPI lifespan
    startup so a crashed mid-ingest process does not leave a row stuck."""
    dsn = _resolve_dsn(conn_or_dsn)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.ingested_books
                   set status = 'pending',
                       started_at = null
                 where status = 'running'
                   and started_at < now() - make_interval(secs => %s)
                """,
                (stale_timeout_seconds,),
            )
            n = cur.rowcount
        conn.commit()
    return n
