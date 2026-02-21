import asyncio
from typing import Dict

from qdrant_client import QdrantClient
from rag.qdrant.qdrant_db import QdrantDB
from agents.book_qdrant_agent import BookQdrantAgent
from langchain_community.embeddings import HuggingFaceEmbeddings


class CacheManager:
    def __init__(self, qdrant_path: str, embedding_model: HuggingFaceEmbeddings):
        self.qdrant_path = qdrant_path
        self.embedding_model = embedding_model

        self._qdrant_client: QdrantClient | None = None
        self._qdrant_dbs: Dict[str, QdrantDB] = {}
        self._agents: Dict[str, BookQdrantAgent] = {}

        self._lock = asyncio.Lock()  # lock only for initialization
        self._checkpointer = None

    async def initialize(self, checkpointer):
        async with self._lock:
            if self._qdrant_client is None:
                self._qdrant_client = QdrantClient(path=self.qdrant_path)
            self._checkpointer = checkpointer

    def get_client(self) -> QdrantClient:
        if self._qdrant_client is None:
            raise RuntimeError("CacheManager not initialized")
        return self._qdrant_client

    async def get_qdrant_db(self, collection_name: str) -> QdrantDB:
        """No lock here — safe because dict writes are atomic in CPython."""
        if collection_name not in self._qdrant_dbs:
            self._qdrant_dbs[collection_name] = QdrantDB(
                collection_name=collection_name,
                client=self._qdrant_client,
                embedding_model=self.embedding_model
            )
        return self._qdrant_dbs[collection_name]

    async def get_agent(self, collection_name: str) -> BookQdrantAgent:
        """Only lock for creating a new agent to prevent race conditions."""
        async with self._lock:
            if collection_name not in self._agents:
                qdrant_db = await self.get_qdrant_db(collection_name)
                self._agents[collection_name] = BookQdrantAgent(
                    qdrant_db=qdrant_db,
                    checkpointer=self._checkpointer
                )
            return self._agents[collection_name]

    async def clear_collection(self, collection_name: str):
        async with self._lock:
            self._qdrant_dbs.pop(collection_name, None)
            self._agents.pop(collection_name, None)