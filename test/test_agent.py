"""Tests for BookQdrantAgent."""
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from langchain_core.documents import Document

from core.schemas.expertise import ExpertiseDetection


class TestBookQdrantAgentInit:
    def test_init_detects_expertise(self):
        mock_qdrant = Mock()
        mock_qdrant.get_all_by_book.return_value = [
            Document(page_content="Software engineering is...", metadata={"source": "se.pdf"}),
        ]

        with patch("agents.book_qdrant_agent.get_chat_model") as mock_llm:
            mock_structured = Mock()
            mock_structured.invoke.return_value = ExpertiseDetection(
                domain="Software Engineering",
                sub_fields=["code quality"],
                expertise_prompt="ahli SE",
                book_type="textbook",
            )
            mock_llm.return_value.with_structured_output.return_value = mock_structured

            from agents.book_qdrant_agent import BookQdrantAgent
            agent = BookQdrantAgent(qdrant_db=mock_qdrant, book_id="test_book")
            assert agent.book_id == "test_book"
            assert agent.expertise.domain == "Software Engineering"

    def test_init_fallback_on_error(self):
        mock_qdrant = Mock()
        mock_qdrant.get_all_by_book.return_value = []

        with patch("agents.book_qdrant_agent.get_chat_model") as mock_llm:
            from agents.book_qdrant_agent import BookQdrantAgent
            agent = BookQdrantAgent(qdrant_db=mock_qdrant, book_id="test_book")
            assert agent.expertise.domain == "Umum"


class TestRetrieveContext:
    def test_retrieve_context_deduplicates(self):
        mock_qdrant = Mock()
        mock_qdrant.get_all_by_book.return_value = []
        mock_qdrant.query.return_value = [
            Document(page_content="Same text", metadata={"source": "a.pdf", "pages": [1]}),
            Document(page_content="Same text", metadata={"source": "a.pdf", "pages": [1]}),
            Document(page_content="Different text", metadata={"source": "b.pdf", "pages": [2]}),
        ]

        with patch("agents.book_qdrant_agent.get_chat_model") as mock_llm:
            mock_structured = Mock()
            mock_structured.invoke.return_value = ExpertiseDetection(
                domain="Umum", sub_fields=[], expertise_prompt="tutor", book_type="unknown"
            )
            mock_llm.return_value.with_structured_output.return_value = mock_structured

            from agents.book_qdrant_agent import BookQdrantAgent
            agent = BookQdrantAgent(qdrant_db=mock_qdrant, book_id="test")
            context, docs, sources = agent._retrieve_context("query")

            assert "Same text" in context
            assert "Different text" in context
            assert len(sources) == 2

    def test_retrieve_context_formats_metadata(self):
        mock_qdrant = Mock()
        mock_qdrant.get_all_by_book.return_value = []
        mock_qdrant.query.return_value = [
            Document(
                page_content="Content here",
                metadata={"source": "book.pdf", "pages": [10, 11], "storage_url": "https://storage/file.pdf"},
            ),
        ]

        with patch("agents.book_qdrant_agent.get_chat_model") as mock_llm:
            mock_structured = Mock()
            mock_structured.invoke.return_value = ExpertiseDetection(
                domain="Umum", sub_fields=[], expertise_prompt="tutor", book_type="unknown"
            )
            mock_llm.return_value.with_structured_output.return_value = mock_structured

            from agents.book_qdrant_agent import BookQdrantAgent
            agent = BookQdrantAgent(qdrant_db=mock_qdrant, book_id="test")
            context, docs, sources = agent._retrieve_context("query")

            assert "book.pdf" in context
            assert "10, 11" in context
            assert "https://storage/file.pdf" in context


class TestAskStream:
    async def test_ask_stream_yields_sse(self):
        from langchain_core.messages import AIMessage

        mock_qdrant = Mock()
        mock_qdrant.get_all_by_book.return_value = [
            Document(page_content="test content", metadata={"source": "test.pdf"}),
        ]

        with patch("agents.book_qdrant_agent.get_chat_model") as mock_llm:
            mock_structured = Mock()
            mock_structured.invoke.return_value = ExpertiseDetection(
                domain="Umum", sub_fields=[], expertise_prompt="tutor", book_type="unknown"
            )
            mock_llm.return_value.with_structured_output.return_value = mock_structured

            from agents.book_qdrant_agent import BookQdrantAgent
            agent = BookQdrantAgent(qdrant_db=mock_qdrant, book_id="test")

            ai_msg = AIMessage(content="Test answer")
            agent.agent = AsyncMock()
            agent.agent.astream = AsyncMock()
            agent.agent.astream.return_value = iter([
                {"messages": [ai_msg]},
            ])

            chunks = []
            async for chunk in agent.ask_stream("test query"):
                chunks.append(chunk)

            assert len(chunks) > 0
            assert any("[DONE]" in c for c in chunks)
