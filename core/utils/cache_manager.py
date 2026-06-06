import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Optional

from qdrant_client import QdrantClient
from core.rag.qdrant_db import QdrantDB
from agents.book_qdrant_agent import BookQdrantAgent
from langchain_huggingface import HuggingFaceEmbeddings


class CacheManager:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_model: HuggingFaceEmbeddings,
        reranker: Optional[object] = None,
    ):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.reranker = reranker

        # single shared DB layer
        self._qdrant_db: Optional[QdrantDB] = None

        # per-book agents (multi-tenant)
        self._agents: Dict[str, BookQdrantAgent] = {}

        # per-book ingest/delete locks (race protection)
        self._ingest_locks: Dict[str, asyncio.Lock] = {}
        self._ingest_locks_guard = asyncio.Lock()

        self._lock = asyncio.Lock()
        self._checkpointer = None

        self._collection_name = "lms_content"

    async def initialize(self, checkpointer):
        async with self._lock:
            if self._qdrant_db is None:
                self._qdrant_db = QdrantDB(
                    collection_name=self._collection_name,
                    client=self.qdrant_client,
                    embedding_model=self.embedding_model,
                    reranker=self.reranker,
                )

            self._checkpointer = checkpointer

    def get_client(self) -> QdrantClient:
        return self.qdrant_client

    async def get_qdrant_db(self) -> QdrantDB:
        if self._qdrant_db is None:
            raise RuntimeError("CacheManager not initialized")
        return self._qdrant_db

    async def get_agent(self, book_id: str, user_id: str = None) -> BookQdrantAgent:
        """
        One agent per book (tenant isolation layer).
        """
        async with self._lock:
            cache_key = f"{book_id}:{user_id or 'anonymous'}"

            if cache_key not in self._agents:
                qdrant_db = await self.get_qdrant_db()

                self._agents[cache_key] = BookQdrantAgent(
                    qdrant_db=qdrant_db,
                    book_id=book_id,
                    user_id=user_id,
                    checkpointer=self._checkpointer
                )

            return self._agents[cache_key]

    @asynccontextmanager
    async def acquire_book_lock(self, book_id: str):
        """
        Acquire a per-bookId lock that serialises state-mutating operations
        (ingest, delete) for the same book. Different book_ids remain parallel.

        Released in the `finally` block, including on exceptions.
        """
        async with self._ingest_locks_guard:
            lock = self._ingest_locks.get(book_id)
            if lock is None:
                lock = asyncio.Lock()
                self._ingest_locks[book_id] = lock

        await lock.acquire()
        try:
            yield
        finally:
            lock.release()

    async def clear_book(self, book_id: str):
        async with self._lock:
            keys_to_remove = [k for k in self._agents if k.startswith(f"{book_id}:")]
            for k in keys_to_remove:
                self._agents.pop(k, None)
            self._ingest_locks.pop(book_id, None)
