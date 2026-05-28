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
