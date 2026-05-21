import asyncio
from typing import Dict, Optional

from qdrant_client import QdrantClient
from services.rag.qdrant.qdrant_db import QdrantDB
from agents.book_qdrant_agent import BookQdrantAgent
from langchain_huggingface import HuggingFaceEmbeddings


class CacheManager:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_model: HuggingFaceEmbeddings,
    ):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model

        # single shared DB layer
        self._qdrant_db: Optional[QdrantDB] = None

        # per-course agents (multi-tenant)
        self._agents: Dict[str, BookQdrantAgent] = {}

        self._lock = asyncio.Lock()
        self._checkpointer = None

        self._collection_name = "lms_content"

    async def initialize(self, checkpointer):
        async with self._lock:
            if self._qdrant_db is None:
                self._qdrant_db = QdrantDB(
                    collection_name=self._collection_name,
                    client=self.qdrant_client,
                    embedding_model=self.embedding_model
                )

            self._checkpointer = checkpointer

    def get_client(self) -> QdrantClient:
        return self.qdrant_client

    async def get_qdrant_db(self) -> QdrantDB:
        if self._qdrant_db is None:
            raise RuntimeError("CacheManager not initialized")
        return self._qdrant_db

    async def get_agent(self, course_id: str) -> BookQdrantAgent:
        """
        One agent per course (tenant isolation layer).
        """
        async with self._lock:
            if course_id not in self._agents:
                qdrant_db = await self.get_qdrant_db()

                self._agents[course_id] = BookQdrantAgent(
                    qdrant_db=qdrant_db,
                    course_id=course_id,
                    checkpointer=self._checkpointer
                )

            return self._agents[course_id]

    async def clear_course(self, course_id: str):
        async with self._lock:
            self._agents.pop(course_id, None)