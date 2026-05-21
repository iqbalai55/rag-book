"""
Test suite for the TTS (Text-to-Speech) engine.
"""
import os
import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.tts.tts_engine import generate_tts, generate_tts_podcast


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment variables for testing."""
    monkeypatch.setenv("ENVIRONMENT", "test")


@pytest.fixture
def mock_torch():
    """Mock torch for TTS engine testing."""
    with patch('services.audio.tts.tts_engine.torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        yield mock_torch


@pytest.fixture
def mock_huggingface_hub():
    """Mock Hugging Face hub for model loading."""
    with patch('services.audio.tts.tts_engine.hf_hub_download') as mock_download:
        mock_download.return_value = "/fake/model/path"
        yield mock_download


@pytest.fixture
def mock_chatterbox():
    """Mock ChatterboxTTS model."""
    with patch('services.audio.tts.tts_engine.ChatterboxTTS') as mock_tts:
        mock_model = Mock()
        mock_tts.from_pretrained.return_value = mock_model
        mock_model.generate = Mock(return_value=np.zeros(1000))
        yield mock_model


class TestTTSEngine:
    """Test cases for the TTS functions."""
    
    def test_generate_tts_with_mock(self, mock_torch, mock_huggingface_hub, mock_chatterbox):
        """Test generate_tts with mocked dependencies."""
        text = "Hello, this is a test."
        result = generate_tts(text)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])