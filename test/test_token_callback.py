"""Tests for token callback handler."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from uuid import uuid4

from core.utils.token_callback import (
    TokenUsageCallbackHandler,
    create_token_callback,
    FEATURE_MAP,
)


class TestFeatureMap:
    def test_feature_map_has_expected_keys(self):
        assert "search_book_context" in FEATURE_MAP
        assert "generate_mcq" in FEATURE_MAP
        assert "generate_essay_questions" in FEATURE_MAP
        assert "generate_complete_podcast" in FEATURE_MAP


class TestTokenUsageCallbackHandler:
    def test_init(self):
        handler = TokenUsageCallbackHandler(
            session_id="s1",
            book_id="b1",
            feature="test",
        )
        assert handler.session_id == "s1"
        assert handler.book_id == "b1"
        assert handler.feature == "test"

    def test_init_defaults(self):
        handler = TokenUsageCallbackHandler()
        assert handler.session_id is None
        assert handler.book_id is None
        assert handler.feature == "unknown"

    def test_set_context_session_id(self):
        handler = TokenUsageCallbackHandler()
        handler.set_context(session_id="new_session")
        assert handler.session_id == "new_session"

    def test_set_context_book_id(self):
        handler = TokenUsageCallbackHandler()
        handler.set_context(book_id="new_book")
        assert handler.book_id == "new_book"

    def test_set_context_feature(self):
        handler = TokenUsageCallbackHandler()
        handler.set_context(feature="new_feature")
        assert handler.feature == "new_feature"

    def test_on_llm_start_records_time(self):
        handler = TokenUsageCallbackHandler()
        run_id = uuid4()
        handler.on_llm_start(
            serialized={"name": "test"},
            prompts=["test prompt"],
            run_id=run_id,
        )
        assert run_id in handler._start_times

    def test_on_llm_end_extracts_usage(self):
        handler = TokenUsageCallbackHandler()
        run_id = uuid4()
        handler._start_times[run_id] = 1000.0

        mock_generation = Mock()
        mock_generation.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "model": "gpt-4o",
        }

        mock_response = Mock()
        mock_response.generations = [[mock_generation]]
        mock_response.llm_output = None

        handler.on_llm_end(response=mock_response, run_id=run_id)

        assert run_id not in handler._start_times
        assert handler.tracker._totals["total_tokens"] > 0

    def test_on_llm_error_logs_warning(self):
        handler = TokenUsageCallbackHandler()
        run_id = uuid4()
        handler._start_times[run_id] = 1000.0

        handler.on_llm_error(
            error=Exception("test error"),
            run_id=run_id,
        )

        assert run_id not in handler._start_times


class TestCreateTokenCallback:
    def test_factory_creates_handler(self):
        handler = create_token_callback(
            session_id="s1",
            book_id="b1",
            feature="test",
        )
        assert isinstance(handler, TokenUsageCallbackHandler)
        assert handler.session_id == "s1"
        assert handler.book_id == "b1"
        assert handler.feature == "test"
