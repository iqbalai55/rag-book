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
    def test_add_documents_with_book_id(self):
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
                {"text": "Hello world", "metadata": {"book_id": "b1", "source": "test.pdf"}},
            ]

            db.add_documents(chunks, book_id="b1")
            db.vectorstore.add_documents.assert_called_once()

    def test_add_documents_without_book_id_raises(self):
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

            with pytest.raises(ValueError, match="Missing 'book_id'"):
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
                Document(page_content="test", metadata={"book_id": "b1"})
            ]

            results = db.query("test query", book_id="b1", k=3)
            assert len(results) == 1

    def test_query_with_reranker(self):
        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()
        mock_embedding = Mock()
        mock_embedding.embed_query.return_value = [0.1] * 384
        mock_embedding.embed_documents.return_value = [[0.1] * 384]

        mock_reranker = Mock()
        mock_reranker.rerank.return_value = (
            [Document(page_content="reranked", metadata={"book_id": "b1"})],
            [0.9],
        )

        with patch("core.rag.qdrant_db.QdrantVectorStore"):
            db = QdrantDB(
                collection_name="test",
                client=mock_client,
                embedding_model=mock_embedding,
                reranker=mock_reranker,
                retrieval_k=20,
                final_k=5,
            )
            db.vectorstore = Mock()
            db.vectorstore.similarity_search.return_value = [
                Document(page_content="doc1", metadata={"book_id": "b1"}),
                Document(page_content="doc2", metadata={"book_id": "b1"}),
            ]

            results = db.query("test query", book_id="b1", k=3, use_reranker=True)
            assert len(results) == 1
            assert results[0].page_content == "reranked"
            mock_reranker.rerank.assert_called_once()

    def test_query_falls_back_when_reranker_none(self):
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
                reranker=None,
            )
            db.vectorstore = Mock()
            db.vectorstore.similarity_search.return_value = [
                Document(page_content="test", metadata={"book_id": "b1"})
            ]

            results = db.query("test query", book_id="b1", k=3, use_reranker=True)
            assert len(results) == 1
            db.vectorstore.similarity_search.assert_called_once()


class TestQdrantDBGetAllByBook:
    def test_get_all_by_book(self):
        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()
        mock_client.scroll.return_value = (
            [
                Mock(payload={"page_content": "text1", "metadata": {"book_id": "b1"}}),
                Mock(payload={"page_content": "text2", "metadata": {"book_id": "b1"}}),
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

            docs = db.get_all_by_book("b1")
            assert len(docs) == 2
            assert docs[0].page_content == "text1"


class TestQdrantDBDelete:
    def test_delete_by_book(self):
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

            db.delete_by_book("b1")
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
