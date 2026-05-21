import logging
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

logger = logging.getLogger(__name__)


class QdrantDB:
    """Multitenant Qdrant wrapper using payload-based filtering."""

    def __init__(
        self,
        collection_name: str,
        embedding_model,
        client: Optional[QdrantClient] = None,
    ):
        self.client = client if client is not None else QdrantClient(":memory:")
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        vector_size = self._detect_vector_size(embedding_model)

        if not self._collection_exists(collection_name):
            logger.info(
                f"Collection '{collection_name}' not found. Creating with size {vector_size}"
            )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

            # ✅ Create payload index for multitenancy
            self._create_payload_indexes()

        else:
            logger.info(f"Collection '{collection_name}' already exists.")

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

    def _create_payload_indexes(self):
        """Create indexes for fast filtering."""
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.course_id",
                field_schema="keyword",
            )
        except Exception as e:
            logger.warning(f"Payload index creation skipped: {e}")

    # -------------------------
    # CORE METHODS
    # -------------------------

    def add_documents(self, chunks: List[Dict], course_id: Optional[str] = None):
        """
        Add documents with enforced multitenancy metadata.

        Args:
            chunks: list of {"text": ..., "metadata": {...}}
            course_id: optional global course_id to inject
        """
        docs = []

        for c in chunks:
            metadata = c.get("metadata", {})

            # ✅ enforce course_id
            if course_id:
                metadata["course_id"] = course_id

            if "course_id" not in metadata:
                raise ValueError("Missing 'course_id' in metadata")

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
        course_id: Optional[str] = None,
        k: int = 5,
        extra_filters: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Multitenant query with optional filtering.

        Args:
            query_text: user query
            course_id: filter by course
            k: top-k
            extra_filters: additional metadata filters
        """

        conditions = []

        if course_id:
            conditions.append(
                FieldCondition(
                    key="metadata.course_id",
                    match=MatchValue(value=course_id),
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

        results = self.vectorstore.similarity_search(
            query_text,
            k=k,
            filter=qdrant_filter,
        )

        logger.info(f"Query returned {len(results)} results")
        return results

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

    def delete_by_course(self, course_id: str):
        """Delete all data for a course."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.course_id",
                        match=MatchValue(value=course_id),
                    )
                ]
            ),
        )
        logger.info(f"Deleted all documents for course '{course_id}'")

    # -------------------------
    # COLLECTION MANAGEMENT
    # -------------------------

    def drop_collection(self):
        if self._collection_exists(self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Dropped collection '{self.collection_name}'")
        else:
            logger.warning(f"Collection '{self.collection_name}' does not exist")