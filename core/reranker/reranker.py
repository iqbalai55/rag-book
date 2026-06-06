import logging
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        max_length: int = 512,
    ):
        try:
            self.model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=device,
            )
            self.device = device
            logger.info(f"Reranker initialized: {model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.model = None
            self.device = device

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> Tuple[List[Document], List[float]]:
        if self.model is None:
            logger.warning("Reranker model not loaded, returning original order")
            return documents[:top_k], [1.0] * min(len(documents), top_k)

        if not documents:
            return [], []

        try:
            sentence_pairs = [[query, doc.page_content] for doc in documents]
            scores = self.model.predict(sentence_pairs)

            sigmoid_scores = [1 / (1 + 2.718281828459045 ** (-s)) for s in scores]

            ranked_pairs = sorted(
                zip(documents, sigmoid_scores),
                key=lambda x: x[1],
                reverse=True,
            )

            ranked_docs = [doc for doc, _ in ranked_pairs[:top_k]]
            relevance_scores = [score for _, score in ranked_pairs[:top_k]]

            return ranked_docs, relevance_scores

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k], [1.0] * min(len(documents), top_k)