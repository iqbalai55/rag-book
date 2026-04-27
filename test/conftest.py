"""
Shared test fixtures and configuration for the RAG Book test suite.
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Any
import pytest
from unittest.mock import Mock, MagicMock

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up common environment variables for testing."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def mock_qdrant_client() -> Mock:
    """Create a mock Qdrant client."""
    client = Mock()
    client.upsert = Mock()
    client.search = Mock(return_value=[])
    client.delete = Mock()
    client.get_collection = Mock()
    client.create_collection = Mock()
    return client


@pytest.fixture
def mock_embedding_model() -> Mock:
    """Create a mock embedding model."""
    model = Mock()
    model.embed_query = Mock(return_value=[0.1, 0.2, 0.3] * 128)  # 384-dim vector
    model.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3] * 128])
    return model


@pytest.fixture
def sample_text_chunks() -> list[str]:
    """Provide sample text chunks for testing."""
    return [
        "This is the first chunk of text for testing.",
        "This is the second chunk with different content.",
        "Another chunk to test various scenarios.",
        "Final chunk in the test set."
    ]


@pytest.fixture
def sample_metadata() -> list[dict[str, Any]]:
    """Provide sample metadata for testing."""
    return [
        {"source": "test_doc_1.pdf", "page": 1, "chunk_index": 0},
        {"source": "test_doc_1.pdf", "page": 1, "chunk_index": 1},
        {"source": "test_doc_2.pdf", "page": 5, "chunk_index": 0},
        {"source": "test_doc_2.pdf", "page": 5, "chunk_index": 1}
    ]


@pytest.fixture
def mock_document_processor() -> Mock:
    """Create a mock document processor."""
    processor = Mock()
    processor.process_document = Mock(return_value=(["chunk1", "chunk2"], [{"meta": 1}, {"meta": 2}]))
    processor.chunk_text = Mock(return_value=["chunk1", "chunk2"])
    return processor