"""Tests for QdrantDB."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document

from core.rag.qdrant_db import QdrantDB


class TestQdrantDBInit:
    def test_init_with_existing_collection(self):
        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()
        mock_embedding = Mock()
        mock_embedding.embed_query.return_value = [0.1] * 384
        # Mock the embed_documents method needed by QdrantVectorStore
        mock_embedding.embed_documents.return_value = [[0.1] * 384]

        with patch("core.rag.qdrant_db.QdrantVectorStore"):
            db = QdrantDB(
                collection_name="test_collection",
                client=mock_client,
                embedding_model=mock_embedding,
            )
            assert db.collection_name == "test_collection"

    def test_init_creates_collection_if_missing(self):
        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_embedding = Mock()
        mock_embedding.embed_query.return_value = [0.1] * 384
        mock_embedding.embed_documents.return_value = [[0.1] * 384]

        with patch("core.rag.qdrant_db.QdrantVectorStore"):
            db = QdrantDB(
                collection_name="new_collection",
                client=mock_client,
                embedding_model=mock_embedding,
            )
            mock_client.create_collection.assert_called_once()


class TestQdrantDBAddDocuments:
    def test_add_documents_with_course_id(self):
        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()
        mock_embedding = Mock()
        mock_embedding.embed_query.return_value = [0.1] * 384
        mock_embedding.embed_documents.return_value = [[0.1] * 384]

        with patch("core.rag.qdrant_db.QdrantVectorStore") as mock_vs:
            db = QdrantDB(
                collection_name="test",
                client=mock_client,
                embedding_model=mock_embedding,
            )
            db.vectorstore = Mock()

            chunks = [
                {"text": "Hello world", "metadata": {"course_id": "c1", "source": "test.pdf"}},
            ]

            db.add_documents(chunks, course_id="c1")
            db.vectorstore.add_documents.assert_called_once()

    def test_add_documents_without_course_id_raises(self):
        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()
        mock_embedding = Mock()
        mock_embedding.embed_query.return_value = [0.1] * 384
        mock_embedding.embed_documents.return_value = [[0.1] * 384]

        with patch("core.rag.qdrant_db.QdrantVectorStore"):
            db = QdrantDB(
                collection_name="test",
                client=mock_client,
                embedding_model=mock_embedding,
            )

            chunks = [
                {"text": "Hello world", "metadata": {"source": "test.pdf"}},
            ]

            with pytest.raises(ValueError, match="Missing 'course_id'"):
                db.add_documents(chunks)


class TestQdrantDBQuery:
    def test_query_calls_similarity_search(self):
        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()
        mock_embedding = Mock()
        mock_embedding.embed_query.return_value = [0.1] * 384
        mock_embedding.embed_documents.return_value = [[0.1] * 384]

        with patch("core.rag.qdrant_db.QdrantVectorStore"):
            db = QdrantDB(
                collection_name="test",
                client=mock_client,
                embedding_model=mock_embedding,
            )
            db.vectorstore = Mock()
            db.vectorstore.similarity_search.return_value = [
                Document(page_content="test", metadata={"course_id": "c1"})
            ]

            results = db.query("test query", course_id="c1", k=3)
            assert len(results) == 1


class TestQdrantDBGetAllByCourse:
    def test_get_all_by_course(self):
        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()
        mock_client.scroll.return_value = (
            [
                Mock(payload={"page_content": "text1", "metadata": {"course_id": "c1"}}),
                Mock(payload={"page_content": "text2", "metadata": {"course_id": "c1"}}),
            ],
            None,
        )
        mock_embedding = Mock()
        mock_embedding.embed_query.return_value = [0.1] * 384
        mock_embedding.embed_documents.return_value = [[0.1] * 384]

        with patch("core.rag.qdrant_db.QdrantVectorStore"):
            db = QdrantDB(
                collection_name="test",
                client=mock_client,
                embedding_model=mock_embedding,
            )

            docs = db.get_all_by_course("c1")
            assert len(docs) == 2
            assert docs[0].page_content == "text1"


class TestQdrantDBDelete:
    def test_delete_by_course(self):
        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()
        mock_embedding = Mock()
        mock_embedding.embed_query.return_value = [0.1] * 384
        mock_embedding.embed_documents.return_value = [[0.1] * 384]

        with patch("core.rag.qdrant_db.QdrantVectorStore"):
            db = QdrantDB(
                collection_name="test",
                client=mock_client,
                embedding_model=mock_embedding,
            )

            db.delete_by_course("c1")
            mock_client.delete.assert_called_once()

    def test_drop_collection(self):
        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()
        mock_embedding = Mock()
        mock_embedding.embed_query.return_value = [0.1] * 384
        mock_embedding.embed_documents.return_value = [[0.1] * 384]

        with patch("core.rag.qdrant_db.QdrantVectorStore"):
            db = QdrantDB(
                collection_name="test",
                client=mock_client,
                embedding_model=mock_embedding,
            )

            db.drop_collection()
            mock_client.delete_collection.assert_called_once()
