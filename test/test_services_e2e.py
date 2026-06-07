"""
Fast e2e for downstream services: summarize, mindmap, dataset generation.

Strategy:
- No PDF ingest (uses the heavy `core.utils.ingest_book` pipeline).
- Inject a handful of synthetic chunks directly via `QdrantDB.add_documents`,
  which uses the real `langchain_qdrant` write path and the same
  `metadata.book_id` payload index.
- Mock the LLM (returns valid pydantic instances per structured schema),
  so the only network call is to Qdrant Cloud.

This exercises the same Qdrant filter path that was failing in production
(`get_all_by_book` -> `metadata.book_id` filter) end-to-end through each
service, without the cost of real embeddings generation or LLM calls.
"""
import os
import sys
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logger = logging.getLogger("test_services_e2e")

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = f"rag_book_services_e2e_{uuid.uuid4().hex[:8]}"


# ----------------------
# Fixtures
# ----------------------

@pytest.fixture(scope="module")
def qdrant_client() -> QdrantClient:
    endpoint = os.getenv("QDRANT_ENDPOINT")
    api_key = os.getenv("QDRANT_API_KEY")
    if not endpoint or not api_key:
        pytest.skip("QDRANT_ENDPOINT / QDRANT_API_KEY not set")
    return QdrantClient(url=endpoint, api_key=api_key)


@pytest.fixture(scope="module")
def embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={"device": "cpu"},
    )


@pytest.fixture(scope="module")
def qdrant_db(qdrant_client, embedding_model):
    from core.rag.qdrant_db import QdrantDB
    return QdrantDB(
        collection_name=COLLECTION_NAME,
        client=qdrant_client,
        embedding_model=embedding_model,
    )


@pytest.fixture(scope="module")
def summarize_book_id() -> str:
    return f"summarize_book_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def mindmap_book_id() -> str:
    return f"mindmap_book_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def dataset_book_id() -> str:
    return f"dataset_book_{uuid.uuid4().hex[:8]}"


def _seed_chunks(qdrant_db, book_id: str, texts: List[str]) -> None:
    chunks = [
        {
            "text": t,
            "metadata": {
                "book_id": book_id,
                "source": "synthetic.pdf",
                "pages": [1, 2],
            },
        }
        for t in texts
    ]
    qdrant_db.add_documents(chunks=chunks, book_id=book_id)


# ----------------------
# LLM mock: returns a valid pydantic instance for whatever schema is bound
# ----------------------

def _build_structured_mock():
    """A mock LLM whose `.with_structured_output(SchemaCls)` returns a
    MagicMock-shaped LLM whose `.invoke(prompt)` returns a SchemaCls instance
    with sensible defaults.
    """
    from core.schemas.chapter import ChapterIdentification
    from core.schemas.summary import ChapterSummary, BookSummaryResponse
    from core.schemas.mindmap import MindmapResponse
    from core.schemas.question import MCQResponse, EssayResponse, MCQQuestion, MCQOption, EssayQuestion

    def _payload_for(schema_cls):
        if schema_cls is ChapterIdentification:
            return ChapterIdentification(chapters=["Chapter 1", "Chapter 2"])
        if schema_cls is ChapterSummary:
            return ChapterSummary(
                chapter_title="Chapter 1",
                summary="A concise summary of the chapter contents.",
                key_points=["Key point A", "Key point B"],
            )
        if schema_cls is BookSummaryResponse:
            return BookSummaryResponse(
                book_id="ignored-set-by-service",
                title="Test Book Title",
                overview="High-level overview produced by the mocked LLM.",
                chapters=[
                    ChapterSummary(
                        chapter_title="Chapter 1",
                        summary="Chapter 1 summary.",
                        key_points=["a", "b"],
                    ),
                    ChapterSummary(
                        chapter_title="Chapter 2",
                        summary="Chapter 2 summary.",
                        key_points=["c", "d"],
                    ),
                ],
                key_themes=["theme-1", "theme-2"],
                total_chapters=2,
                sources=[],
            )
        if schema_cls is MindmapResponse:
            return MindmapResponse(
                title="Test Mindmap",
                mermaid="mindmap\n  root((Test))\n    Branch A\n    Branch B",
                sources=[],
            )
        if schema_cls is MCQResponse:
            return MCQResponse(
                topic="Chapter 1",
                difficulty="medium",
                questions=[
                    MCQQuestion(
                        question="What is the main idea of Chapter 1?",
                        options=[
                            MCQOption(label="A", text="Idea A"),
                            MCQOption(label="B", text="Idea B"),
                            MCQOption(label="C", text="Idea C"),
                            MCQOption(label="D", text="Idea D"),
                        ],
                        correct_answer="A",
                        explanation="Because it is the main idea.",
                    )
                ],
                sources=[],
            )
        if schema_cls is EssayResponse:
            return EssayResponse(
                topic="Chapter 1",
                difficulty="medium",
                questions=[
                    EssayQuestion(
                        question="Discuss the implications of Chapter 1.",
                        key_points=["implication-1", "implication-2"],
                        explanation="Tests deep understanding.",
                    )
                ],
                sources=[],
            )
        # Fallback
        try:
            return schema_cls()
        except Exception:
            return MagicMock()

    def for_schema(schema_cls):
        payload = _payload_for(schema_cls)
        bound = MagicMock()
        bound.invoke = MagicMock(return_value=payload)
        return bound

    llm = MagicMock()
    llm.with_structured_output.side_effect = for_schema
    return llm


@pytest.fixture
def mock_llm():
    return _build_structured_mock()


# ----------------------
# Tests
# ----------------------

@pytest.mark.asyncio
async def test_summarize_service_e2e(qdrant_db, qdrant_client, summarize_book_id):
    """Summarize: seed synthetic chunks, run BookSummarizer.summarize,
    assert response shape. No real LLM, no real PDF."""
    from core.services.summarize_book import BookSummarizer

    _seed_chunks(qdrant_db, summarize_book_id, [
        "Chapter 1 introduces the foundational concept of refactoring.",
        "Smells are surface indicators of deeper design problems.",
        "Techniques like Extract Method and Move Field are demonstrated.",
    ])

    llm = _build_structured_mock()
    summarizer = BookSummarizer(qdrant_db=qdrant_db, llm=llm)

    result = await summarizer.summarize(book_id=summarize_book_id)

    assert result["book_id"] == summarize_book_id
    assert isinstance(result["chapters"], list) and len(result["chapters"]) >= 1
    assert result["total_chapters"] == len(result["chapters"])
    assert result["title"], "title must be populated"
    assert result["overview"], "overview must be populated"
    assert isinstance(result["key_themes"], list)
    assert isinstance(result["sources"], list) and result["sources"], "sources from Qdrant metadata"
    assert any("synthetic.pdf" in s for s in result["sources"]), f"sources missing file ref: {result['sources']}"

    # Direct Qdrant cross-check: confirm the seeded chunks are still there.
    cnt = qdrant_client.count(
        collection_name=COLLECTION_NAME,
        count_filter=models.Filter(
            must=[models.FieldCondition(
                key="metadata.book_id",
                match=models.MatchValue(value=summarize_book_id),
            )]
        ),
        exact=True,
    )
    assert cnt.count == 3, f"expected 3 seeded chunks, got {cnt.count}"


@pytest.mark.asyncio
async def test_mindmap_service_e2e(qdrant_db, qdrant_client, mindmap_book_id):
    """Mindmap: seed synthetic chunks, run MindmapGenerator.generate,
    assert response shape. No real LLM, no real PDF."""
    from core.services.mindmap_generator import MindmapGenerator

    _seed_chunks(qdrant_db, mindmap_book_id, [
        "Refactoring is a disciplined technique for restructuring code.",
        "Design smells indicate problematic structures.",
    ])

    llm = _build_structured_mock()
    gen = MindmapGenerator(qdrant_db=qdrant_db, llm=llm)

    result = await gen.generate(book_id=mindmap_book_id)

    assert result["book_id"] == mindmap_book_id
    assert result["title"], "title must be populated"
    assert "mindmap" in result["mermaid"].lower(), f"mermaid missing 'mindmap' keyword: {result['mermaid']!r}"
    assert isinstance(result["sources"], list) and result["sources"]
    assert any("synthetic.pdf" in s for s in result["sources"])


@pytest.mark.asyncio
async def test_dataset_service_e2e(qdrant_db, qdrant_client, dataset_book_id):
    """Dataset: seed synthetic chunks, run DatasetGenerator.generate,
    assert response shape with MCQ + essay per chapter."""
    from core.services.dataset_generator import DatasetGenerator

    _seed_chunks(qdrant_db, dataset_book_id, [
        "Chapter 1: working with smells in legacy code.",
        "Catalog of smells: long method, large class, divergent change.",
        "Chapter 2: a step-by-step refactoring recipe.",
    ])

    llm = _build_structured_mock()
    gen = DatasetGenerator(qdrant_db=qdrant_db, llm=llm)

    result = await gen.generate(
        book_id=dataset_book_id,
        difficulty="medium",
        num_mcq=1,
        num_essay=1,
    )

    assert result["book_id"] == dataset_book_id
    assert result["difficulty"] == "medium"
    assert result["total_chapters"] >= 1
    assert isinstance(result["chapters"], list) and result["chapters"], "no chapters generated"

    for ch in result["chapters"]:
        assert ch["chapter_title"], f"chapter missing title: {ch}"
        # The mock returns one MCQ and one essay per chapter; the service may
        # skip a chapter if its query returns no context, so only assert for
        # the chapters we got back.
        if "mcq" in ch:
            assert isinstance(ch["mcq"]["questions"], list) and ch["mcq"]["questions"]
            q = ch["mcq"]["questions"][0]
            assert q["correct_answer"] in {"A", "B", "C", "D"}
            assert len(q["options"]) == 4
        if "essay" in ch:
            assert isinstance(ch["essay"]["questions"], list) and ch["essay"]["questions"]
            assert ch["essay"]["questions"][0]["key_points"], "essay must have key_points"


@pytest.mark.asyncio
async def test_services_return_404_when_book_id_missing(qdrant_db):
    """Regression: every service should raise HTTPException(404) when the
    book_id has no chunks — not crash on a Qdrant filter error. This guards
    the fix for the `metadata.book_id` index bug, since a missing index
    previously produced a 400 instead of a clean 404."""
    from fastapi import HTTPException
    from core.services.summarize_book import BookSummarizer
    from core.services.mindmap_generator import MindmapGenerator
    from core.services.dataset_generator import DatasetGenerator

    missing_id = f"missing_{uuid.uuid4().hex[:8]}"
    llm = _build_structured_mock()

    with pytest.raises(HTTPException) as exc_info:
        await BookSummarizer(qdrant_db=qdrant_db, llm=llm).summarize(book_id=missing_id)
    assert exc_info.value.status_code == 404

    with pytest.raises(HTTPException) as exc_info:
        await MindmapGenerator(qdrant_db=qdrant_db, llm=llm).generate(book_id=missing_id)
    assert exc_info.value.status_code == 404

    with pytest.raises(HTTPException) as exc_info:
        await DatasetGenerator(qdrant_db=qdrant_db, llm=llm).generate(book_id=missing_id)
    assert exc_info.value.status_code == 404


# ----------------------
# Teardown
# ----------------------

def test_drop_collection_teardown(qdrant_client):
    qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
    with pytest.raises(Exception):
        qdrant_client.get_collection(COLLECTION_NAME)
