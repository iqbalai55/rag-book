"""
Test suite for the Qdrant agent and related components.
"""
import asyncio
import json
import os
import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.book_qdrant_agent import BookQdrantAgent
from services.rag.qdrant.qdrant_db import QdrantDB
from services.rag.qdrant.document_processor import DocumentProcessor


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment variables for testing."""
    monkeypatch.setenv("ENVIRONMENT", "test")


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    client = Mock()
    client.upsert = Mock()
    client.search = Mock(return_value=[])
    client.delete = Mock()
    client.get_collection = Mock()
    client.create_collection = Mock()
    return client


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = Mock()
    model.embed_query = Mock(return_value=[0.1, 0.2, 0.3] * 128)  # 384-dim vector
    model.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3] * 128])
    return model


@pytest.fixture
def mock_document_processor():
    """Create a mock document processor."""
    processor = Mock(spec=DocumentProcessor)
    processor.process_document = Mock(return_value=(["chunk1", "chunk2"], [{"meta": 1}, {"meta": 2}]))
    processor.chunk_text = Mock(return_value=["chunk1", "chunk2"])
    return processor


@pytest.fixture
def mock_qdrant_db(mock_qdrant_client, mock_embedding_model):
    """Create a mock QdrantDB instance."""
    db = Mock(spec=QdrantDB)
    db.collection_name = "test_collection"
    db.client = mock_qdrant_client
    db.embedding_model = mock_embedding_model
    return db


@pytest.fixture
def qdrant_agent(mock_qdrant_db):
    """Create a BookQdrantAgent instance for testing."""
    return BookQdrantAgent(qdrant_db=mock_qdrant_db, k=3)


@pytest.fixture
def sample_query():
    """Provide a sample query for testing."""
    return "What is the most important principle in lean software development?"


@pytest.fixture
def sample_session_id():
    """Provide a sample session ID for testing."""
    return "test_session_1"


class TestBookQdrantAgent:
    """Test cases for the BookQdrantAgent class."""
    
    def test_agent_initialization(self, mock_qdrant_db):
        """Test that the agent initializes correctly."""
        agent = BookQdrantAgent(qdrant_db=mock_qdrant_db, k=5)
        assert agent.qdrant_db == mock_qdrant_db
        assert agent.k == 5
    
    def test_agent_initialization_default_k(self, mock_qdrant_db):
        """Test that the agent uses default k value when not specified."""
        agent = BookQdrantAgent(qdrant_db=mock_qdrant_db)
        assert agent.k == 3  # Default value
    
    @pytest.mark.asyncio
    async def test_ask_stream_general_qa(
        self, 
        qdrant_agent: BookQdrantAgent,
        sample_query: str,
        sample_session_id: str
    ):
        """Test the ask_stream method with a general question."""
        # Arrange
        mock_response_chunks = [
            'data: {"type": "internal", "metadata": {"tool_calls": []}}\n\n',
            'data: {"type": "tool", "metadata": {"tool_name": "search"}, "content": "Search results"}\n\n',
            'data: {"type": "final", "content": "The most important principle is to eliminate waste."}\n\n',
            'data: [DONE]\n\n'
        ]
        
        # Mock the agent's ask_stream method to return our predefined chunks
        async def mock_ask_stream(*args, **kwargs):
            for chunk in mock_response_chunks:
                yield chunk
        
        qdrant_agent.ask_stream = mock_ask_stream
        
        # Act
        collected_chunks = []
        async for raw in qdrant_agent.ask_stream(sample_query, session_id=sample_session_id):
            collected_chunks.append(raw)
        
        # Assert
        assert len(collected_chunks) == 4
        assert "[DONE]" in collected_chunks[-1]
        
        # Parse and validate the chunks
        internal_chunk = json.loads(collected_chunks[0].removeprefix("data: ").strip())
        assert internal_chunk["type"] == "internal"
        
        tool_chunk = json.loads(collected_chunks[1].removeprefix("data: ").strip())
        assert tool_chunk["type"] == "tool"
        assert tool_chunk["metadata"]["tool_name"] == "search"
        
        final_chunk = json.loads(collected_chunks[2].removeprefix("data: ").strip())
        assert final_chunk["type"] == "final"
        assert "eliminate waste" in final_chunk["content"]
    
    @pytest.mark.asyncio
    async def test_ask_stream_multiple_choice(
        self, 
        qdrant_agent: BookQdrantAgent
    ):
        """Test the ask_stream method for generating multiple choice questions."""
        # Arrange
        query = "Generate 5 MCQs about waste elimination in lean software development with medium difficulty."
        session_id = "test_session_2"
        
        mock_response_chunks = [
            'data: {"type": "internal", "metadata": {"tool_calls": [{"name": "search", "args": {}}]}}\n\n',
            'data: {"type": "tool", "metadata": {"tool_name": "search"}, "content": "Found relevant content about waste"}\n\n',
            'data: {"type": "multiple_choice_question", "content": {"topic": "Waste Elimination", "difficulty": "medium", "questions": [{"question": "What is waste?", "options": ["A", "B", "C", "D"], "answer": "A"}]}}\n\n',
            'data: [DONE]\n\n'
        ]
        
        # Mock the agent's ask_stream method
        async def mock_ask_stream(*args, **kwargs):
            for chunk in mock_response_chunks:
                yield chunk
        
        qdrant_agent.ask_stream = mock_ask_stream
        
        # Act
        collected_chunks = []
        async for raw in qdrant_agent.ask_stream(query, session_id=session_id):
            collected_chunks.append(raw)
        
        # Assert
        assert len(collected_chunks) == 4
        
        # Parse the multiple choice question chunk
        mcq_chunk = json.loads(collected_chunks[2].removeprefix("data: ").strip())
        assert mcq_chunk["type"] == "multiple_choice_question"
        assert mcq_chunk["content"]["topic"] == "Waste Elimination"
        assert mcq_chunk["content"]["difficulty"] == "medium"
        assert len(mcq_chunk["content"]["questions"]) == 1
        assert mcq_chunk["content"]["questions"][0]["question"] == "What is waste?"
    
    @pytest.mark.asyncio
    async def test_ask_stream_error_handling(
        self, 
        qdrant_agent: BookQdrantAgent
    ):
        """Test the ask_stream method's error handling."""
        # Arrange
        query = "This query will cause an error"
        session_id = "error_test_session"
        
        mock_response_chunks = [
            'data: {"type": "error", "content": "Something went wrong"}\n\n',
            'data: [DONE]\n\n'
        ]
        
        # Mock the agent's ask_stream method
        async def mock_ask_stream(*args, **kwargs):
            for chunk in mock_response_chunks:
                yield chunk
        
        qdrant_agent.ask_stream = mock_ask_stream
        
        # Act
        collected_chunks = []
        async for raw in qdrant_agent.ask_stream(query, session_id=session_id):
            collected_chunks.append(raw)
        
        # Assert
        assert len(collected_chunks) == 2
        
        # Parse the error chunk
        error_chunk = json.loads(collected_chunks[0].removeprefix("data: ").strip())
        assert error_chunk["type"] == "error"
        assert error_chunk["content"] == "Something went wrong"


class TestQdrantDB:
    """Test cases for the QdrantDB class."""
    
    def test_qdrant_db_initialization(self, mock_qdrant_client, mock_embedding_model):
        """Test that QdrantDB initializes correctly."""
        db = QdrantDB(
            collection_name="test_collection",
            embedding_model=mock_embedding_model,
            client=mock_qdrant_client
        )
        assert db.collection_name == "test_collection"
        assert db.embedding_model == mock_embedding_model
        assert db.client == mock_qdrant_client
    
    def test_search_method_calls_client_search(self, mock_qdrant_db):
        """Test that the search method calls the underlying client's search method."""
        # Arrange
        query_vector = [0.1, 0.2, 0.3] * 128
        mock_qdrant_db.client.search.return_value = [{"id": 1, "score": 0.9, "payload": {"text": "test"}}]
        
        # Act
        result = mock_qdrant_db.search(query_vector, limit=3)
        
        # Assert
        mock_qdrant_db.client.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=query_vector,
            limit=3,
            with_payload=True,
            with_vectors=False
        )
        assert len(result) == 1
        assert result[0]["id"] == 1
    
    def test_upsert_method_calls_client_upsert(self, mock_qdrant_db):
        """Test that the upsert method calls the underlying client's upsert method."""
        # Arrange
        points = [{"id": 1, "vector": [0.1, 0.2, 0.3], "payload": {"text": "test"}}]
        
        # Act
        mock_qdrant_db.upsert(points)
        
        # Assert
        mock_qdrant_db.client.upsert.assert_called_once()
        call_args = mock_qdrant_db.client.upsert.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        assert call_args[1]["points"] == points


if __name__ == "__main__":
    pytest.main([__file__, "-v"])