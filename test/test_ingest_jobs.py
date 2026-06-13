"""Tests for `core.utils.ingest_jobs`.

Behaviour-driven: drive the functions against a real Postgres test schema when
`SUPABASE_DB_URL` is set, otherwise mock the `psycopg.connect` call to assert
the SQL is shaped correctly.

Each test uses a fresh, uniquely-named `book_id` so concurrent runs and stale
state do not collide.
"""
from __future__ import annotations

import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.utils import ingest_jobs


pytestmark = pytest.mark.usefixtures()


def _skip_if_no_db():
    if not os.getenv("SUPABASE_DB_URL"):
        pytest.skip("SUPABASE_DB_URL not set; live-DB tests skipped")


def _fresh_book_id() -> str:
    return f"test_ingest_jobs_{uuid.uuid4().hex[:10]}"


def _cleanup(book_id: str):
    import psycopg
    ds = os.environ["SUPABASE_DB_URL"]
    with psycopg.connect(ds) as conn:
        with conn.cursor() as cur:
            cur.execute("delete from public.ingested_books where book_id = %s", (book_id,))
        conn.commit()


# ---------------------------------------------------------------- live DB

class TestLiveDB:
    def test_create_then_get(self):
        _skip_if_no_db()
        book_id = _fresh_book_id()
        ds = os.environ["SUPABASE_DB_URL"]
        try:
            job = ingest_jobs.create_or_reset_job(
                book_id=book_id,
                filename="a.pdf",
                storage_url="https://example/a.pdf",
                conn_or_dsn=ds,
            )
            assert job.book_id == book_id
            assert job.status == "pending"
            assert job.error is None
            assert job.chunks_count == 0

            got = ingest_jobs.get_job(book_id, ds)
            assert got is not None
            assert got.status == "pending"
        finally:
            _cleanup(book_id)

    def test_create_or_reset_is_idempotent(self):
        _skip_if_no_db()
        book_id = _fresh_book_id()
        ds = os.environ["SUPABASE_DB_URL"]
        try:
            ingest_jobs.create_or_reset_job(book_id, "a.pdf", "u1", conn_or_dsn=ds)
            ingest_jobs.mark_failed(book_id, "boom", conn_or_dsn=ds)
            assert ingest_jobs.get_job(book_id, ds).status == "failed"
            ingest_jobs.create_or_reset_job(book_id, "b.pdf", "u2", conn_or_dsn=ds)
            j = ingest_jobs.get_job(book_id, ds)
            assert j.status == "pending"
            assert j.error is None
            assert j.filename == "b.pdf"
        finally:
            _cleanup(book_id)

    def test_claim_next_pending_skip_locked(self):
        _skip_if_no_db()
        book_id_a = _fresh_book_id()
        book_id_b = _fresh_book_id()
        ds = os.environ["SUPABASE_DB_URL"]
        try:
            ingest_jobs.create_or_reset_job(book_id_a, "a.pdf", "u1", conn_or_dsn=ds)
            ingest_jobs.create_or_reset_job(book_id_b, "b.pdf", "u2", conn_or_dsn=ds)

            c1 = ingest_jobs.claim_next_pending(ds)
            c2 = ingest_jobs.claim_next_pending(ds)
            assert c1 is not None and c2 is not None
            assert {c1.book_id, c2.book_id} == {book_id_a, book_id_b}
            assert c1.status == "running"
            assert c2.status == "running"

            assert ingest_jobs.claim_next_pending(ds) is None

            ingest_jobs.mark_succeeded(c1.book_id, 42, conn_or_dsn=ds)
            assert ingest_jobs.get_job(c1.book_id, ds).status == "succeeded"
            assert ingest_jobs.get_job(c1.book_id, ds).chunks_count == 42
        finally:
            _cleanup(book_id_a)
            _cleanup(book_id_b)

    def test_mark_failed_writes_error(self):
        _skip_if_no_db()
        book_id = _fresh_book_id()
        ds = os.environ["SUPABASE_DB_URL"]
        try:
            ingest_jobs.create_or_reset_job(book_id, "a.pdf", "u", conn_or_dsn=ds)
            ingest_jobs.claim_next_pending(ds)
            ingest_jobs.mark_failed(book_id, "docling blew up", conn_or_dsn=ds)
            j = ingest_jobs.get_job(book_id, ds)
            assert j.status == "failed"
            assert j.error == "docling blew up"
            assert j.finished_at is not None
        finally:
            _cleanup(book_id)

    def test_mark_deleted_by_reingest_only_touches_non_terminal(self):
        _skip_if_no_db()
        book_id = _fresh_book_id()
        ds = os.environ["SUPABASE_DB_URL"]
        try:
            ingest_jobs.create_or_reset_job(book_id, "a.pdf", "u", conn_or_dsn=ds)
            ingest_jobs.claim_next_pending(ds)
            ingest_jobs.mark_deleted_by_reingest(book_id, ds)
            j = ingest_jobs.get_job(book_id, ds)
            assert j.status == "failed"
            assert "re-ingest: chunks deleted" in (j.error or "")

            # already terminal — no further change
            ingest_jobs.mark_deleted_by_reingest(book_id, ds)
            j2 = ingest_jobs.get_job(book_id, ds)
            assert j2.status == "failed"
        finally:
            _cleanup(book_id)

    def test_reset_for_retry_only_on_failed(self):
        _skip_if_no_db()
        book_id = _fresh_book_id()
        ds = os.environ["SUPABASE_DB_URL"]
        try:
            ingest_jobs.create_or_reset_job(book_id, "a.pdf", "u", conn_or_dsn=ds)
            # pending row — reset_for_retry should be a no-op
            assert ingest_jobs.reset_for_retry(book_id, ds) is False
            assert ingest_jobs.get_job(book_id, ds).status == "pending"

            ingest_jobs.mark_failed(book_id, "x", conn_or_dsn=ds)
            assert ingest_jobs.reset_for_retry(book_id, ds) is True
            j = ingest_jobs.get_job(book_id, ds)
            assert j.status == "pending"
            assert j.error is None
        finally:
            _cleanup(book_id)


# -------------------------------------------------------------- mock tests

@contextmanager
def _mock_conn():
    """Build a mock psycopg connection with a cursor that records execute calls."""
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    conn.cursor.return_value.__exit__.return_value = False
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    with patch("core.utils.ingest_jobs.psycopg.connect", return_value=conn) as p:
        yield p, conn, cur


class TestMockedSQL:
    def test_claim_uses_skip_locked(self):
        with _mock_conn() as (_, _, cur):
            cur.fetchone.return_value = None
            ingest_jobs.claim_next_pending(conn_or_dsn="x")
            sql = cur.execute.call_args_list[0].args[0]
            assert "for update skip locked" in sql
            assert "status = 'pending'" in sql

    def test_mark_failed_writes_error(self):
        with _mock_conn() as (_, _, cur):
            cur.rowcount = 1
            ingest_jobs.mark_failed("b1", "boom", conn_or_dsn="x")
            sql, params = cur.execute.call_args.args
            assert "status = 'failed'" in sql
            assert params == ("boom", "b1")

    def test_reset_for_retry_only_failed(self):
        with _mock_conn() as (_, _, cur):
            cur.rowcount = 0
            assert ingest_jobs.reset_for_retry("b1", conn_or_dsn="x") is False
            cur.rowcount = 1
            assert ingest_jobs.reset_for_retry("b1", conn_or_dsn="x") is True

    def test_list_jobs_rejects_bad_status(self):
        with pytest.raises(ValueError):
            ingest_jobs.list_jobs(status="weird", conn_or_dsn="x")

    def test_list_jobs_caps_limit(self):
        with _mock_conn() as (_, _, cur):
            cur.fetchall.return_value = []
            ingest_jobs.list_jobs(limit=99999, conn_or_dsn="x")
            limit_used = cur.execute.call_args.args[1][-1]
            assert limit_used == 100
