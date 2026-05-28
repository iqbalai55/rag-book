import os
import logging
from typing import Optional

from supabase import create_client, Client

logger = logging.getLogger(__name__)


class SupabaseStorage:
    """Supabase Storage wrapper for PDF book files."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ):
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_KEY")
        self.bucket_name = bucket_name or os.getenv("SUPABASE_STORAGE_BUCKET", "book-pdfs")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment or constructor"
            )

        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        self._ensure_bucket_exists()
        logger.info(f"SupabaseStorage initialized: bucket='{self.bucket_name}'")

    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        try:
            buckets = self.client.storage.list_buckets()
            existing = [b.name for b in buckets]
            if self.bucket_name not in existing:
                self.client.storage.create_bucket(
                    self.bucket_name,
                    options={"public": True},
                )
                logger.info(f"Created public bucket: {self.bucket_name}")
            else:
                logger.info(f"Bucket '{self.bucket_name}' already exists")
        except Exception as e:
            logger.warning(f"Bucket check/create failed: {e}")

    def upload_pdf(
        self,
        file_path: str,
        course_id: str,
        filename: Optional[str] = None,
    ) -> str:
        """
        Upload PDF to Supabase Storage.

        Args:
            file_path: Local path to the PDF file
            course_id: Course identifier (used as folder)
            filename: Custom filename (defaults to basename of file_path)

        Returns:
            Public URL of the uploaded file
        """
        if filename is None:
            filename = os.path.basename(file_path)

        storage_path = f"{course_id}/{filename}"

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        self.client.storage.from_(self.bucket_name).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": "application/pdf", "upsert": "true"},
        )

        public_url = self.get_public_url(course_id, filename)
        logger.info(f"Uploaded PDF: {storage_path} -> {public_url}")
        return public_url

    def get_public_url(self, course_id: str, filename: str) -> str:
        """
        Get public URL for a stored PDF.

        Args:
            course_id: Course identifier
            filename: PDF filename

        Returns:
            Public URL string
        """
        storage_path = f"{course_id}/{filename}"
        response = self.client.storage.from_(self.bucket_name).get_public_url(storage_path)
        return response

    def delete_pdf(self, course_id: str, filename: str) -> bool:
        """
        Delete a PDF from storage.

        Args:
            course_id: Course identifier
            filename: PDF filename

        Returns:
            True if deleted successfully
        """
        storage_path = f"{course_id}/{filename}"
        try:
            self.client.storage.from_(self.bucket_name).remove([storage_path])
            logger.info(f"Deleted PDF: {storage_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete PDF: {e}")
            return False

    def list_pdfs(self, course_id: str) -> list:
        """
        List all PDFs for a course.

        Args:
            course_id: Course identifier

        Returns:
            List of file metadata dicts
        """
        try:
            files = self.client.storage.from_(self.bucket_name).list(course_id)
            return files
        except Exception as e:
            logger.error(f"Failed to list PDFs: {e}")
            return []
