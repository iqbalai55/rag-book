"""Tests for Reranker."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

import sys
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sentence_transformers'].CrossEncoder = Mock()

from core.reranker.reranker import Reranker


class TestRerankerInit:
    def test_init_with_valid_model(self):
        with patch("core.reranker.reranker.CrossEncoder") as mock_ce:
            mock_ce.return_value = Mock()
            reranker = Reranker(model_name="test-model", device="cpu")
            assert reranker.model is not None
            assert reranker.device == "cpu"

    def test_init_handles_failure(self):
        with patch("core.reranker.reranker.CrossEncoder") as mock_ce:
            mock_ce.side_effect = Exception("Model not found")
            reranker = Reranker(model_name="test-model", device="cpu")
            assert reranker.model is None


class TestRerankerRerank:
    def test_rerank_returns_original_when_model_none(self):
        reranker = Reranker()
        reranker.model = None

        docs = [
            Document(page_content="doc1", metadata={}),
            Document(page_content="doc2", metadata={}),
        ]

        ranked, scores = reranker.rerank("query", docs, top_k=2)
        assert len(ranked) == 2
        assert scores == [1.0, 1.0]

    def test_rerank_returns_empty_when_no_docs(self):
        with patch("core.reranker.reranker.CrossEncoder") as mock_ce:
            mock_ce.return_value = Mock()
            reranker = Reranker(model_name="test-model", device="cpu")

            ranked, scores = reranker.rerank("query", [], top_k=5)
            assert ranked == []
            assert scores == []

    def test_rerank_orders_by_score(self):
        with patch("core.reranker.reranker.CrossEncoder") as mock_ce:
            mock_model = Mock()
            mock_model.predict.return_value = [0.1, 0.9, 0.5]
            mock_ce.return_value = mock_model

            reranker = Reranker(model_name="test-model", device="cpu")

            docs = [
                Document(page_content="doc1", metadata={}),
                Document(page_content="doc2", metadata={}),
                Document(page_content="doc3", metadata={}),
            ]

            ranked, scores = reranker.rerank("query", docs, top_k=3)
            assert len(ranked) == 3
            assert ranked[0].page_content == "doc2"
            assert ranked[1].page_content == "doc3"
            assert ranked[2].page_content == "doc1"

    def test_rerank_respects_top_k(self):
        with patch("core.reranker.reranker.CrossEncoder") as mock_ce:
            mock_model = Mock()
            mock_model.predict.return_value = [0.1, 0.9, 0.5, 0.3, 0.7]
            mock_ce.return_value = mock_model

            reranker = Reranker(model_name="test-model", device="cpu")

            docs = [
                Document(page_content=f"doc{i}", metadata={})
                for i in range(5)
            ]

            ranked, scores = reranker.rerank("query", docs, top_k=2)
            assert len(ranked) == 2
            assert ranked[0].page_content == "doc1"
            assert ranked[1].page_content == "doc4"

    def test_rerank_handles_prediction_failure(self):
        with patch("core.reranker.reranker.CrossEncoder") as mock_ce:
            mock_model = Mock()
            mock_model.predict.side_effect = Exception("Prediction failed")
            mock_ce.return_value = mock_model

            reranker = Reranker(model_name="test-model", device="cpu")

            docs = [
                Document(page_content="doc1", metadata={}),
                Document(page_content="doc2", metadata={}),
            ]

            ranked, scores = reranker.rerank("query", docs, top_k=2)
            assert len(ranked) == 2
            assert scores == [1.0, 1.0]
