"""Tests for Supabase Storage."""
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open

from core.storage.supabase_storage import SupabaseStorage


class TestSupabaseStorageInit:
    @patch("core.storage.supabase_storage.os.getenv")
    @patch("core.storage.supabase_storage.create_client")
    def test_init_success(self, mock_create, mock_getenv):
        mock_getenv.side_effect = lambda key, *args: {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_KEY": "test-key",
            "SUPABASE_STORAGE_BUCKET": "test-bucket",
        }.get(key)
        mock_client = Mock()
        mock_client.storage.list_buckets.return_value = []
        mock_create.return_value = mock_client

        storage = SupabaseStorage()
        assert storage.bucket_name == "test-bucket"

    @patch("core.storage.supabase_storage.os.getenv")
    def test_init_missing_url_raises(self, mock_getenv):
        mock_getenv.return_value = None
        with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_SERVICE_KEY"):
            SupabaseStorage()


class TestSupabaseStorageMethods:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        with patch("core.storage.supabase_storage.os.getenv") as mock_getenv, \
             patch("core.storage.supabase_storage.create_client") as mock_create:
            mock_getenv.side_effect = lambda key, *args: {
                "SUPABASE_URL": "https://test.supabase.co",
                "SUPABASE_SERVICE_KEY": "test-key",
                "SUPABASE_STORAGE_BUCKET": "test-bucket",
            }.get(key)
            self.mock_client = Mock()
            self.mock_client.storage.list_buckets.return_value = []
            mock_create.return_value = self.mock_client
            self.storage = SupabaseStorage()

    def test_upload_pdf_calls_client(self):
        m = mock_open(read_data=b"pdf content")
        with patch("builtins.open", m):
            self.storage.upload_pdf(
                file_path="/tmp/test.pdf",
                course_id="course1",
                filename="test.pdf",
            )
        self.mock_client.storage.from_.assert_called()

    def test_get_public_url(self):
        self.mock_client.storage.from_.return_value.get_public_url.return_value = (
            "https://storage.supabase.co/test.pdf"
        )
        url = self.storage.get_public_url("course1", "test.pdf")
        assert "https://storage.supabase.co/test.pdf" in url

    def test_delete_pdf_success(self):
        self.mock_client.storage.from_.return_value.remove.return_value = True
        result = self.storage.delete_pdf("course1", "test.pdf")
        assert result is True

    def test_delete_pdf_failure(self):
        self.mock_client.storage.from_.return_value.remove.side_effect = Exception("fail")
        result = self.storage.delete_pdf("course1", "test.pdf")
        assert result is False

    def test_list_pdfs(self):
        self.mock_client.storage.from_.return_value.list.return_value = [
            {"name": "file1.pdf"},
            {"name": "file2.pdf"},
        ]
        files = self.storage.list_pdfs("course1")
        assert len(files) == 2

    def test_list_pdfs_error(self):
        self.mock_client.storage.from_.return_value.list.side_effect = Exception("fail")
        files = self.storage.list_pdfs("course1")
        assert files == []
