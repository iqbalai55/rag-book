import logging
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class QdrantDB:
    """Multitenant Qdrant wrapper using payload-based filtering."""

    def __init__(
        self,
        collection_name: str,
        embedding_model,
        client: Optional[QdrantClient] = None,
        reranker: Optional[object] = None,
        retrieval_k: int = 20,
        final_k: int = 5,
    ):
        self.client = client if client is not None else QdrantClient(":memory:")
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.retrieval_k = retrieval_k
        self.final_k = final_k

        vector_size = self._detect_vector_size(embedding_model)

        if not self._collection_exists(collection_name):
            logger.info(
                f"Collection '{collection_name}' not found. Creating with size {vector_size}"
            )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        else:
            logger.info(f"Collection '{collection_name}' already exists.")

        # ✅ Idempotent: ensures `metadata.book_id` filter index exists on
        # every init, so collections created by older versions (or imported
        # from a backup) get the index created on first use. Prevents
        # 400 "Index required but not found" on `metadata.book_id`.
        self._ensure_payload_indexes()

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embedding_model,
        )

        logger.info(f"QdrantDB initialized: {collection_name}")

    # -------------------------
    # INTERNAL HELPERS
    # -------------------------

    def _detect_vector_size(self, embedding_model) -> int:
        try:
            test_vector = embedding_model.embed_query("test")
            return len(test_vector)
        except Exception:
            return getattr(embedding_model, "embedding_function_output_dim", 384)

    def _collection_exists(self, name: str) -> bool:
        try:
            self.client.get_collection(name)
            return True
        except Exception:
            return False

    def _index_exists(self, field_name: str) -> bool:
        """Return True if a payload index already exists for `field_name`."""
        try:
            info = self.client.get_collection(self.collection_name)
            schema = getattr(info, "payload_schema", None) or {}
            return field_name in schema
        except Exception as e:
            logger.warning(f"Could not read payload schema: {e}")
            return False

    def _ensure_payload_indexes(self):
        """Create the `metadata.book_id` keyword index if missing.

        Safe to call on every QdrantDB init. Tolerates the "already exists"
        response from the Qdrant server (HTTP 4xx) so a concurrent caller
        or an older init cannot break startup.
        """
        field_name = "metadata.book_id"
        if self._index_exists(field_name):
            return

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            logger.info(
                f"Created payload index '{field_name}' (keyword) on '{self.collection_name}'"
            )
        except UnexpectedResponse as e:
            if getattr(e, "status_code", None) == 409 or "already exists" in str(e).lower():
                logger.info(f"Payload index '{field_name}' already exists (race-safe)")
            else:
                logger.error(f"Failed to create payload index '{field_name}': {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to create payload index '{field_name}': {e}")
            raise

    # -------------------------
    # CORE METHODS
    # -------------------------

    def add_documents(self, chunks: List[Dict], book_id: Optional[str] = None):
        """
        Add documents with enforced multitenancy metadata.

        Args:
            chunks: list of {"text": ..., "metadata": {...}}
            book_id: optional global book_id to inject
        """
        docs = []

        for c in chunks:
            metadata = c.get("metadata", {})

            # ✅ enforce book_id
            if book_id:
                metadata["book_id"] = book_id

            if "book_id" not in metadata:
                raise ValueError("Missing 'book_id' in metadata")

            docs.append(Document(page_content=c["text"], metadata=metadata))

        if docs:
            self.vectorstore.add_documents(docs)
            logger.info(f"Added {len(docs)} docs to '{self.collection_name}'")
        else:
            logger.warning("No documents to add.")

    # -------------------------
    # QUERY (MULTITENANT)
    # -------------------------

    def query(
        self,
        query_text: str,
        book_id: Optional[str] = None,
        k: int = 5,
        extra_filters: Optional[Dict[str, str]] = None,
        use_reranker: bool = False,
    ) -> List[Document]:
        """
        Multitenant query with optional filtering.

        Args:
            query_text: user query
            book_id: filter by book
            k: top-k
            extra_filters: additional metadata filters
            use_reranker: whether to use cross-encoder reranking
        """

        conditions = []

        if book_id:
            conditions.append(
                FieldCondition(
                    key="metadata.book_id",
                    match=MatchValue(value=book_id),
                )
            )

        if extra_filters:
            for key, value in extra_filters.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value),
                    )
                )

        qdrant_filter = Filter(must=conditions) if conditions else None

        if use_reranker and self.reranker is not None:
            candidates = self.vectorstore.similarity_search(
                query_text,
                k=self.retrieval_k,
                filter=qdrant_filter,
            )

            if candidates:
                ranked_docs, _ = self.reranker.rerank(
                    query_text, candidates, top_k=self.final_k
                )
                results = ranked_docs[:k]
            else:
                results = []
        else:
            results = self.vectorstore.similarity_search(
                query_text,
                k=k,
                filter=qdrant_filter,
            )

        logger.info(f"Query returned {len(results)} results")
        return results

    # -------------------------
    # RETRIEVE ALL BY COURSE
    # -------------------------

    def get_all_by_book(self, book_id: str, limit: int = 100) -> List[Document]:
        """Retrieve ALL chunks for a book using scroll API."""
        all_docs = []
        offset = None

        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.book_id",
                            match=MatchValue(value=book_id),
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=False,
                offset=offset,
                limit=limit,
            )

            for point in results:
                payload = point.payload
                all_docs.append(
                    Document(
                        page_content=payload.get("page_content", ""),
                        metadata=payload.get("metadata", {}),
                    )
                )

            if offset is None:
                break

        logger.info(f"Retrieved {len(all_docs)} chunks for book: {book_id}")
        return all_docs

    # -------------------------
    # DELETE
    # -------------------------

    def delete_by_ids(self, ids: List[str]):
        if ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={"ids": ids},
            )
            logger.info(f"Deleted {len(ids)} points")
        else:
            logger.warning("No IDs provided for deletion.")

    def delete_by_book(self, book_id: str):
        """Delete all data for a book."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.book_id",
                        match=MatchValue(value=book_id),
                    )
                ]
            ),
        )
        logger.info(f"Deleted all documents for book '{book_id}'")

    def count_by_book(self, book_id: str) -> int:
        """Return the number of points in this collection whose payload
        `metadata.book_id` matches. Uses `count` so it does not materialise
        the chunks. Used by the async-ingest worker to report `chunks_count`."""
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.book_id",
                        match=MatchValue(value=book_id),
                    )
                ]
            ),
        )
        return int(getattr(result, "count", 0) or 0)

    # -------------------------
    # COLLECTION MANAGEMENT
    # -------------------------

    def drop_collection(self):
        if self._collection_exists(self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Dropped collection '{self.collection_name}'")
        else:
            logger.warning(f"Collection '{self.collection_name}' does not exist")