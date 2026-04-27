"""
Test suite for the TTS (Text-to-Speech) engine.
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add project root to path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.audio.tts.tts_engine import TTSEngine


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
    with patch('services.audio.tts.tts_engine.snapshot_download') as mock_download:
        mock_download.return_value = "/fake/model/path"
        yield mock_download


@pytest.fixture
def tts_engine(mock_torch, mock_huggingface_hub):
    """Create a TTSEngine instance for testing."""
    with patch('services.audio.tts.tts_engine.AutoModelForCausalLM'), \
         patch('services.audio.tts.tts_engine.AutoProcessor'):
        engine = TTSEngine(
            model_name="test/model",
            device="cpu"
        )
        return engine


class TestTTSEngine:
    """Test cases for the TTSEngine class."""
    
    def test_engine_initialization(self, tts_engine):
        """Test that the TTS engine initializes correctly."""
        assert tts_engine is not None
        assert hasattr(tts_engine, 'model_name')
        assert hasattr(tts_engine, 'device')
    
    def test_engine_initialization_defaults(self, mock_torch, mock_huggingface_hub):
        """Test that the TTS engine uses default values when not specified."""
        with patch('services.audio.tts.tts_engine.AutoModelForCausalLM'), \
             patch('services.audio.tts.tts_engine.AutoProcessor'):
            engine = TTSEngine()
            assert engine.device == "cpu"  # Default when CUDA not available
    
    def test_generate_speech_method_exists(self, tts_engine):
        """Test that the generate_speech method exists."""
        assert hasattr(tts_engine, 'generate_speech')
        assert callable(getattr(tts_engine, 'generate_speech'))
    
    def test_generate_speech_with_valid_text(self, tts_engine):
        """Test generate_speech with valid text input."""
        # Arrange
        text = "Hello, this is a test."
        # Mock the actual generation to avoid loading real models
        tts_engine.model = Mock()
        tts_engine.processor = Mock()
        tts_engine.model.generate = Mock(return_value=Mock())
        tts_engine.processor.decode = Mock(return_value=[[0.1, 0.2, 0.3] * 100])  # Fake audio
        
        # Act
        result = tts_engine.generate_speech(text)
        
        # Assert
        assert result is not None
        tts_engine.model.generate.assert_called_once()
    
    def test_generate_speech_empty_text(self, tts_engine):
        """Test generate_speech with empty text."""
        # Arrange
        text = ""
        # Mock the actual generation to avoid loading real models
        tts_engine.model = Mock()
        tts_engine.processor = Mock()
        tts_engine.model.generate = Mock(return_value=Mock())
        tts_engine.processor.decode = Mock(return_value=[[0.0] * 10])  # Minimal audio
        
        # Act
        result = tts_engine.generate_speech(text)
        
        # Assert
        assert result is not None
        # Should still call generate even with empty text
        tts_engine.model.generate.assert_called_once()
    
    def test_generate_speech_long_text(self, tts_engine):
        """Test generate_speech with long text input."""
        # Arrange
        long_text = "This is a very long text. " * 100  # Create long text
        # Mock the actual generation to avoid loading real models
        tts_engine.model = Mock()
        tts_engine.processor = Mock()
        tts_engine.model.generate = Mock(return_value=Mock())
        tts_engine.processor.decode = Mock(return_value=[[0.1, 0.2, 0.3] * 1000])  # Longer audio
        
        # Act
        result = tts_engine.generate_speech(long_text)
        
        # Assert
        assert result is not None
        tts_engine.model.generate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])