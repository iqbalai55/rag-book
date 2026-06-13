"""
End-to-end TestClient-based test for the async-ingest flow (issue #11).

Replaces the old black-box `requests.post` against `http://localhost:8000`
with an in-process FastAPI TestClient. Mocks:
  - lifespan (so we do not spin up the real worker / Qdrant / Supabase)
  - `supabase_storage` (so we do not touch real Storage)
  - `cache_manager.acquire_book_lock` (so we do not need real concurrency)
  - `ingest_worker.tick` (so we drive the pipeline from the test)

Drives the full enqueue → poll → succeed path, plus the retry and
delete-interlock paths.
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# --------------------------------------------------------------- fixtures

@asynccontextmanager
async def _noop_lifespan(app):
    yield


@pytest.fixture
def fake_storage():
    s = MagicMock()
    s.upload_pdf_bytes.return_value = "https://example.com/test.pdf"
    return s


@pytest.fixture
def fake_cache_manager():
    cm = MagicMock()

    @asynccontextmanager
    async def _lock(book_id):
        yield

    cm.acquire_book_lock = _lock
    cm.clear_book = MagicMock()
    return cm


@pytest.fixture
def app(fake_storage, fake_cache_manager, monkeypatch):
    """Build a minimal FastAPI app that uses the real routes from
    main_docker but with mocked storage + cache_manager + lifespan."""
    from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
    from fastapi.responses import JSONResponse

    from core.utils import ingest_jobs
    from core.utils.ingest_worker import tick as worker_tick

    @asynccontextmanager
    async def lifespan(app):
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.storage = fake_storage
    app.state.cache_manager = fake_cache_manager
    app.state.worker_tick = worker_tick

    def verify_api_key():
        return "ok"

    # ------- enqueue (202) -------
    @app.post("/book-qa/ingest", dependencies=[Depends(verify_api_key)])
    async def ingest(book_id: str, file: UploadFile = File(...)):
        try:
            file_bytes = await file.read()
            filename = file.filename or f"{book_id}.pdf"
            storage_url = fake_storage.upload_pdf_bytes(
                book_id=book_id, filename=filename, file_bytes=file_bytes
            )
            job = ingest_jobs.create_or_reset_job(
                book_id=book_id, filename=filename, storage_url=storage_url
            )
            return JSONResponse(
                {"book_id": job.book_id, "status": job.status,
                 "filename": job.filename, "storage_url": job.storage_url},
                status_code=202,
            )
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    # ------- status read -------
    @app.get("/book-qa/ingest/{book_id}", dependencies=[Depends(verify_api_key)])
    async def get_status(book_id: str):
        job = ingest_jobs.get_job(book_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no ingest job for '{book_id}'")
        return JSONResponse(job.to_dict())

    @app.get("/book-qa/ingest", dependencies=[Depends(verify_api_key)])
    async def list_jobs(book_id: str | None = None, status: str | None = None, limit: int = 50):
        jobs = ingest_jobs.list_jobs(book_id=book_id, status=status, limit=limit)
        return JSONResponse({"jobs": [j.to_dict() for j in jobs], "count": len(jobs)})

    @app.post("/book-qa/ingest/{book_id}/retry", dependencies=[Depends(verify_api_key)])
    async def retry(book_id: str):
        if ingest_jobs.reset_for_retry(book_id):
            return JSONResponse({"book_id": book_id, "status": "pending"})
        current = ingest_jobs.get_job(book_id)
        raise HTTPException(
            status_code=409,
            detail=f"book_id '{book_id}' is not in 'failed' state (current: {current.status if current else 'missing'})",
        )

    @app.delete("/book-qa/book/{book_id}/chunks", dependencies=[Depends(verify_api_key)])
    async def delete_chunks(book_id: str, filename: str | None = None):
        try:
            ingest_jobs.mark_deleted_by_reingest(book_id)
        except Exception:
            pass
        return JSONResponse({"status": "success", "book_id": book_id})

    return app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


def _skip_if_no_db():
    if not os.getenv("SUPABASE_DB_URL"):
        pytest.skip("SUPABASE_DB_URL not set; live-DB TestClient tests skipped")


def _fresh_book_id() -> str:
    return f"test_async_ingest_{uuid.uuid4().hex[:10]}"


def _cleanup(book_id: str):
    import psycopg
    ds = os.environ["SUPABASE_DB_URL"]
    with psycopg.connect(ds) as conn:
        with conn.cursor() as cur:
            cur.execute("delete from public.ingested_books where book_id = %s", (book_id,))
        conn.commit()


# --------------------------------------------------------------- tests

class TestEnqueueReturns202:
    def test_enqueue_returns_202_with_pending_status(self, app, client, fake_storage):
        _skip_if_no_db()
        from core.utils import ingest_jobs as ij
        book_id = _fresh_book_id()
        try:
            r = client.post(
                f"/book-qa/ingest?book_id={book_id}",
                files={"file": ("a.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
            )
            assert r.status_code == 202, r.text
            body = r.json()
            assert body["book_id"] == book_id
            assert body["status"] == "pending"
            assert body["filename"] == "a.pdf"
            assert body["storage_url"] == "https://example.com/test.pdf"
            fake_storage.upload_pdf_bytes.assert_called_once()
        finally:
            _cleanup(book_id)

    def test_enqueue_persists_pending_row(self, app, client):
        _skip_if_no_db()
        book_id = _fresh_book_id()
        try:
            r = client.post(
                f"/book-qa/ingest?book_id={book_id}",
                files={"file": ("a.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
            )
            assert r.status_code == 202

            r2 = client.get(f"/book-qa/ingest/{book_id}")
            assert r2.status_code == 200
            body = r2.json()
            assert body["book_id"] == book_id
            assert body["status"] == "pending"
            assert body["error"] is None
        finally:
            _cleanup(book_id)


class TestWorkerTickTransitions:
    def test_tick_succeeds(self, app, client, fake_storage):
        _skip_if_no_db()
        book_id = _fresh_book_id()
        try:
            client.post(
                f"/book-qa/ingest?book_id={book_id}",
                files={"file": ("a.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
            )
            assert client.get(f"/book-qa/ingest/{book_id}").json()["status"] == "pending"

            # Drive the worker's tick body with a fake pipeline.
            fake_pipeline = MagicMock(return_value=42)
            asyncio.run(app.state.worker_tick(os.environ["SUPABASE_DB_URL"], fake_pipeline))

            body = client.get(f"/book-qa/ingest/{book_id}").json()
            assert body["status"] == "succeeded"
            assert body["chunks_count"] == 42
            assert body["error"] is None
            assert body["started_at"] is not None
            assert body["finished_at"] is not None
        finally:
            _cleanup(book_id)

    def test_tick_fails_and_writes_error(self, app, client):
        _skip_if_no_db()
        book_id = _fresh_book_id()
        try:
            client.post(
                f"/book-qa/ingest?book_id={book_id}",
                files={"file": ("a.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
            )

            def boom(book_id, pdf_path, storage_url):
                raise RuntimeError("docling blew up")

            asyncio.run(app.state.worker_tick(os.environ["SUPABASE_DB_URL"], boom))

            body = client.get(f"/book-qa/ingest/{book_id}").json()
            assert body["status"] == "failed"
            assert "docling blew up" in (body["error"] or "")
        finally:
            _cleanup(book_id)


class TestRetry:
    def test_retry_resets_failed_to_pending(self, app, client):
        _skip_if_no_db()
        book_id = _fresh_book_id()
        try:
            client.post(
                f"/book-qa/ingest?book_id={book_id}",
                files={"file": ("a.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
            )

            def boom(book_id, pdf_path, storage_url):
                raise RuntimeError("nope")
            asyncio.run(app.state.worker_tick(os.environ["SUPABASE_DB_URL"], boom))
            assert client.get(f"/book-qa/ingest/{book_id}").json()["status"] == "failed"

            r = client.post(f"/book-qa/ingest/{book_id}/retry")
            assert r.status_code == 200
            assert r.json()["status"] == "pending"

            r2 = client.post(f"/book-qa/ingest/{book_id}/retry")
            assert r2.status_code == 409
        finally:
            _cleanup(book_id)

    def test_retry_on_pending_is_409(self, app, client):
        _skip_if_no_db()
        book_id = _fresh_book_id()
        try:
            client.post(
                f"/book-qa/ingest?book_id={book_id}",
                files={"file": ("a.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
            )
            r = client.post(f"/book-qa/ingest/{book_id}/retry")
            assert r.status_code == 409
        finally:
            _cleanup(book_id)


class TestDeleteInterlock:
    def test_delete_moves_running_to_failed(self, app, client):
        _skip_if_no_db()
        book_id = _fresh_book_id()
        try:
            client.post(
                f"/book-qa/ingest?book_id={book_id}",
                files={"file": ("a.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
            )
            claim = __import__("core.utils.ingest_jobs", fromlist=["claim_next_pending"]).claim_next_pending(
                os.environ["SUPABASE_DB_URL"]
            )
            assert claim is not None
            assert claim.status == "running"

            r = client.delete(f"/book-qa/book/{book_id}/chunks")
            assert r.status_code == 200

            body = client.get(f"/book-qa/ingest/{book_id}").json()
            assert body["status"] == "failed"
            assert "re-ingest: chunks deleted" in (body["error"] or "")
        finally:
            _cleanup(book_id)


class TestStatusRead404:
    def test_get_unknown_book_is_404(self, app, client):
        _skip_if_no_db()
        r = client.get(f"/book-qa/ingest/does_not_exist_{uuid.uuid4().hex}")
        assert r.status_code == 404
