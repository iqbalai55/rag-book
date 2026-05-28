"""Tests for LLM configuration and provider selection."""
import pytest
from unittest.mock import patch, Mock

from core.utils.llm_config import get_chat_model, validate_config


class TestGetChatModel:
    @patch("core.utils.llm_config.os.getenv")
    def test_openai_provider(self, mock_getenv):
        mock_getenv.return_value = "test-key"
        model = get_chat_model(provider="openai", model="gpt-4")
        assert model is not None

    @patch("core.utils.llm_config.os.getenv")
    def test_anthropic_provider(self, mock_getenv):
        mock_getenv.return_value = "test-key"
        model = get_chat_model(provider="anthropic", model="claude-3")
        assert model is not None

    @patch("core.utils.llm_config.os.getenv")
    def test_claude_alias(self, mock_getenv):
        mock_getenv.return_value = "test-key"
        model = get_chat_model(provider="claude", model="claude-3")
        assert model is not None

    @patch("core.utils.llm_config.os.getenv")
    def test_openrouter_provider(self, mock_getenv):
        mock_getenv.return_value = "test-key"
        model = get_chat_model(provider="openrouter", model="llama-3")
        assert model is not None

    @patch("core.utils.llm_config.os.getenv")
    def test_minimax_provider(self, mock_getenv):
        mock_getenv.return_value = "test-key"
        model = get_chat_model(provider="minimax", model="MiniMax-M2.7")
        assert model is not None

    def test_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_chat_model(provider="unknown_provider")

    @patch("core.utils.llm_config.os.getenv")
    def test_missing_openai_api_key(self, mock_getenv):
        mock_getenv.return_value = None
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            get_chat_model(provider="openai")

    @patch("core.utils.llm_config.os.getenv")
    def test_missing_anthropic_api_key(self, mock_getenv):
        mock_getenv.return_value = None
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
            get_chat_model(provider="anthropic")

    @patch("core.utils.llm_config.os.getenv")
    def test_missing_openrouter_api_key(self, mock_getenv):
        mock_getenv.return_value = None
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
            get_chat_model(provider="openrouter")

    @patch("core.utils.llm_config.os.getenv")
    def test_missing_minimax_api_key(self, mock_getenv):
        mock_getenv.return_value = None
        with pytest.raises(ValueError, match="MINIMAX_API_KEY is required"):
            get_chat_model(provider="minimax")


class TestValidateConfig:
    @patch("core.utils.llm_config.get_chat_model")
    def test_valid_config(self, mock_get):
        mock_get.return_value = Mock()
        assert validate_config(provider="openai") is True

    @patch("core.utils.llm_config.get_chat_model")
    def test_invalid_config(self, mock_get):
        mock_get.side_effect = ValueError("Missing key")
        assert validate_config(provider="openai") is False
