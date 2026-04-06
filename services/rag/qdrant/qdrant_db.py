import logging
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

logger = logging.getLogger(__name__)


class QdrantDB:
    """Wrapper for LangChain QdrantVectorStore with automatic collection creation."""

    def __init__(
        self,
        collection_name: str,
        embedding_model,
        client: Optional[QdrantClient] = None,
    ):
        """
        Initialize QdrantVectorStore and auto-create collection if missing.

        Args:
            collection_name (str): Qdrant collection name
            embedding_model: LangChain embedding model instance
            client (QdrantClient, optional): Qdrant client instance. Defaults to in-memory.
        """
        self.client = client if client is not None else QdrantClient(":memory:")
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Determine vector size safely
        vector_size = self._detect_vector_size(embedding_model)

        # Auto-create collection if missing
        if not self._collection_exists(collection_name):
            logger.info(
                f"Collection '{collection_name}' not found. Creating new collection with size {vector_size}."
            )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        else:
            logger.info(f"Collection '{collection_name}' already exists.")

        # Initialize LangChain QdrantVectorStore
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embedding_model,
        )
        logger.info(f"QdrantDB initialized for collection: {collection_name}")

    def _detect_vector_size(self, embedding_model) -> int:
        """Try to detect embedding dimension."""
        try:
            test_vector = embedding_model.embed_query("test")
            return len(test_vector)
        except Exception:
            # fallback attribute
            return getattr(embedding_model, "embedding_function_output_dim", 384)

    def _collection_exists(self, name: str) -> bool:
        """Check if a Qdrant collection exists."""
        try:
            self.client.get_collection(name)
            return True
        except ValueError:
            return False

    def add_documents(self, chunks: List[Dict]):
        """
        Add a list of chunk dicts to the collection.

        Args:
            chunks (List[Dict]): Each dict should contain 'text' and optional 'metadata'
        """
        docs = [Document(page_content=c["text"], metadata=c.get("metadata", {})) for c in chunks]
        if docs:
            self.vectorstore.add_documents(docs)
            logger.info(f"Added {len(docs)} chunks to collection '{self.collection_name}'")
        else:
            logger.warning("No documents to add.")

    def query(self, query_text: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search in the collection.

        Args:
            query_text (str): Query string
            k (int, optional): Number of results. Defaults to 5.

        Returns:
            List[Document]: Top-k documents
        """
        results = self.vectorstore.similarity_search(query_text, k=k)
        logger.info(f"Query returned {len(results)} results")
        return results

    def delete_by_ids(self, ids: List[str]):
        """Delete points from the collection by their IDs."""
        if ids:
            self.vectorstore.client.delete(
                collection_name=self.collection_name,
                points_selector={"ids": ids}
            )
            logger.info(f"Deleted {len(ids)} points from collection '{self.collection_name}'")
        else:
            logger.warning("No IDs provided for deletion.")

    def drop_collection(self):
        """Delete the entire collection."""
        if self._collection_exists(self.collection_name):
            self.vectorstore.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Dropped collection '{self.collection_name}'")
        else:
            logger.warning(f"Collection '{self.collection_name}' does not exist")
