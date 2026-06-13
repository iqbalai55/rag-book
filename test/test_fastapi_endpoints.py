"""Tests for FastAPI endpoints."""
import pytest


class TestAppConfiguration:
    def test_endpoint_names_in_source(self):
        """Verify endpoint function names exist in source."""
        with open("scripts/main_fastapi.py", "r") as f:
            content = f.read()

        assert "async def book_qa_stream" in content
        assert "async def ingest_pdf" in content
        assert "async def generate_mindmap" in content
        assert "async def generate_dataset" in content
        assert "async def verify_api_key" in content

    def test_endpoint_decorators(self):
        """Verify endpoints have correct decorators."""
        with open("scripts/main_fastapi.py", "r") as f:
            content = f.read()

        assert '"/book-qa/stream"' in content
        assert '"/book-qa/ingest"' in content
        assert '"/book-qa/mindmap"' in content
        assert '"/book-qa/dataset"' in content

    def test_rate_limits(self):
        """Verify rate limits are configured."""
        with open("scripts/main_fastapi.py", "r") as f:
            content = f.read()

        assert "10/minute" in content
        assert "3/minute" in content
        assert "5/minute" in content
        assert "2/minute" in content

    def test_async_ingest_lifecycle_columns_pinned(self):
        """The four async-ingest lifecycle columns must appear in source so they
        are pinned as a contract for the data-access module + worker."""
        from pathlib import Path
        sources = [
            "scripts/main_fastapi.py",
            "scripts/main_docker.py",
            "core/utils/ingest_jobs.py",
            "core/utils/ingest_worker.py",
        ]
        all_content = ""
        for s in sources:
            all_content += Path(s).read_text() + "\n"

        for col in ("status", "started_at", "finished_at", "error"):
            assert col in all_content, f"column {col!r} not pinned in any async-ingest source"

    def test_async_ingest_routes_pinned(self):
        """The new HTTP routes (status read, list, retry, enqueue comment) must
        appear in both entry points."""
        from pathlib import Path
        for entry in ("scripts/main_fastapi.py", "scripts/main_docker.py"):
            content = Path(entry).read_text()
            assert '"/book-qa/ingest/{book_id}"' in content, f"{entry}: missing GET /book-qa/ingest/{{book_id}}"
            assert '"/book-qa/ingest"' in content, f"{entry}: missing GET /book-qa/ingest"
            assert "/book-qa/ingest/{book_id}/retry" in content, f"{entry}: missing retry route"
            assert "ingest_jobs.create_or_reset_job" in content, f"{entry}: enqueue not wired to create_or_reset_job"
            assert "mark_deleted_by_reingest" in content, f"{entry}: delete-interlock not wired"

    def test_async_ingest_migration_present(self):
        """The migration file must exist and reference the new columns."""
        from pathlib import Path
        path = Path("scripts/migrations/003_async_ingest.sql")
        assert path.exists(), f"missing {path}"
        sql = path.read_text()
        assert "ingested_books" in sql
        assert "status" in sql
        assert "started_at" in sql
        assert "finished_at" in sql
        assert "error" in sql
        assert "if not exists" in sql, "migration must be idempotent"
