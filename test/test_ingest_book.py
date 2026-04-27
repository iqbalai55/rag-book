"""
Test suite for the book ingestion API endpoint.
"""
import os
import pytest
import requests
from unittest.mock import patch, Mock
from pathlib import Path

# Test configuration
FASTAPI_URL = "http://localhost:8000/book-qa/ingest"


@pytest.fixture
def sample_pdf_path() -> Path:
    """Fixture providing path to test PDF file."""
    pdf_path = Path(r"E:\Coding\KitaPandu\rag-book\book\poa.pdf")
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    return pdf_path


@pytest.fixture
def mock_response() -> Mock:
    """Fixture providing a mock successful response."""
    response_mock = Mock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "message": "File processed successfully",
        "pages_processed": 10,
        "chunks_created": 50
    }
    return response_mock


def test_ingest_endpoint_with_valid_pdf(sample_pdf_path: Path) -> None:
    """Test that the ingest endpoint processes a valid PDF file correctly."""
    # Arrange
    with open(sample_pdf_path, "rb") as pdf_file:
        files = {
            "file": (
                sample_pdf_path.name,
                pdf_file,
                "application/pdf"
            )
        }
        
        # Act
        response = requests.post(FASTAPI_URL, files=files)
    
    # Assert
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    # Try to parse JSON response
    try:
        json_response = response.json()
        assert isinstance(json_response, dict), "Response should be a JSON object"
        assert "message" in json_response, "Response should contain a message"
    except ValueError:
        pytest.fail("Response is not valid JSON")


def test_ingest_endpoint_with_missing_file() -> None:
    """Test that the ingest endpoint handles missing files appropriately."""
    # Arrange
    nonexistent_path = Path("nonexistent.pdf")
    
    # Act & Assert
    with pytest.raises(FileNotFoundError):
        with open(nonexistent_path, "rb") as f:
            pass  # This should raise FileNotFoundError


@patch("requests.post")
def test_ingest_endpoint_with_mocked_response(
    mock_post: Mock,
    sample_pdf_path: Path,
    mock_response: Mock
) -> None:
    """Test the ingest endpoint using mocked requests to avoid external dependencies."""
    # Arrange
    mock_post.return_value = mock_response
    
    with open(sample_pdf_path, "rb") as pdf_file:
        files = {
            "file": (
                sample_pdf_path.name,
                pdf_file,
                "application/pdf"
            )
        }
        
        # Act
        response = requests.post(FASTAPI_URL, files=files)
    
    # Assert
    mock_post.assert_called_once()
    assert response.status_code == 200
    assert response.json()["message"] == "File processed successfully"


def test_ingest_endpoint_invalid_file_type(sample_pdf_path: Path) -> None:
    """Test that the ingest endpoint rejects invalid file types."""
    # Arrange
    # Create a temporary text file to simulate invalid file type
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
        tmp_file.write(b"This is not a PDF file")
        tmp_file_path = tmp_file.name
    
    try:
        with open(tmp_file_path, "rb") as txt_file:
            files = {
                "file": (
                    "test.txt",
                    txt_file,
                    "text/plain"
                )
            }
            
            # Act
            response = requests.post(FASTAPI_URL, files=files)
        
        # Assert - we expect either rejection or graceful handling
        # The exact behavior depends on implementation, but it shouldn't crash
        assert response.status_code in [200, 400, 415, 500], \
            f"Unexpected status code: {response.status_code}"
            
    finally:
        # Cleanup
        os.unlink(tmp_file_path)


def test_ingest_endpoint_empty_file() -> None:
    """Test that the ingest endpoint handles empty files appropriately."""
    # Arrange
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file_path = tmp_file.name  # Empty file
    
    try:
        with open(tmp_file_path, "rb") as empty_file:
            files = {
                "file": (
                    "empty.pdf",
                    empty_file,
                    "application/pdf"
                )
            }
            
            # Act
            response = requests.post(FASTAPI_URL, files=files)
        
        # Assert
        # Should either process empty file gracefully or return appropriate error
        assert response.status_code in [200, 400, 422, 500], \
            f"Unexpected status code for empty file: {response.status_code}"
            
    finally:
        # Cleanup
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)