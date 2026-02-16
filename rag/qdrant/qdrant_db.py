import logging
from typing import List, Dict
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

logger = logging.getLogger(__name__)

class QdrantDB:
    """Wrapper for LangChain QdrantVectorStore operations with auto collection creation."""

    def __init__(
        self,
        collection_name: str,
        embedding_model,
        client: QdrantClient = None
    ):
        """
        Initialize QdrantVectorStore and auto-create collection if missing.

        Args:
            collection_name (str): Qdrant collection name
            embedding_model: LangChain embedding model instance
            client (QdrantClient, optional): Qdrant client. Defaults to in-memory.
        """
        self.client = client if client is not None else QdrantClient(":memory:")  # in-memory default
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Determine vector size from embedding model
        try:
            vector_size = embedding_model.embed_query("test").__len__()
        except Exception:
            # fallback for models exposing dimension differently
            vector_size = getattr(embedding_model, "embedding_function_output_dim", 384)

        # Auto-create collection if it doesn't exist
        try:
            self.client.get_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' already exists.")
        except ValueError:
            logger.info(f"Collection '{collection_name}' not found. Creating new collection with size {vector_size}.")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

        # Initialize QdrantVectorStore
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embedding_model,
        )
        logger.info(f"QdrantDB initialized for collection: {collection_name}")

    def add_documents(self, chunks: List[Dict]):
        docs = [Document(page_content=c["text"], metadata=c.get("metadata", {})) for c in chunks]
        self.vectorstore.add_documents(docs)
        logger.info(f"Added {len(docs)} chunks to Qdrant collection '{self.collection_name}'")

    def query(self, query_text: str, k: int = 5) -> List[Document]:
        results = self.vectorstore.similarity_search(query_text, k=k)
        logger.info(f"Query returned {len(results)} results")
        return results

    def delete_by_ids(self, ids: List[str]):
        self.vectorstore.client.delete(
            collection_name=self.collection_name,
            points_selector={"ids": ids}
        )
        logger.info(f"Deleted {len(ids)} points from collection '{self.collection_name}'")

    def drop_collection(self):
        self.vectorstore.client.delete_collection(collection_name=self.collection_name)
        logger.info(f"Dropped collection '{self.collection_name}'")
