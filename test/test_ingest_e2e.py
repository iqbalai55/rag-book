"""
End-to-end ingest validation: run the real `ingest_book` flow, then read
the points back directly from Qdrant (bypassing the QdrantDB wrapper) to
confirm the partition and payload index are correct.
"""
import os
import sys
import uuid
import time
import logging
from pathlib import Path

import pytest
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logger = logging.getLogger("test_ingest_e2e")

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = f"rag_book_e2e_{uuid.uuid4().hex[:8]}"


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
def book_id() -> str:
    return f"test_book_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory) -> Path:
    repo_books = PROJECT_ROOT / "book"
    candidates = sorted(repo_books.glob("*.pdf"))
    if not candidates:
        pytest.skip("No PDF in book/ to ingest")
    src = candidates[0]
    dst = tmp_path_factory.mktemp("pdf") / src.name
    dst.write_bytes(src.read_bytes())
    return dst


def test_collection_and_index_created(qdrant_client, qdrant_db):
    info = qdrant_client.get_collection(COLLECTION_NAME)
    assert info.config.params.vectors.size == 384, f"unexpected vector size: {info.config.params.vectors.size}"

    payload_schema = getattr(info, "payload_schema", None) or {}
    book_id_index = payload_schema.get("metadata.book_id")
    assert book_id_index is not None, f"no index on metadata.book_id, schema={payload_schema}"
    data_type = getattr(book_id_index, "data_type", None)
    data_type_str = getattr(data_type, "value", None) or str(data_type)
    assert data_type_str.lower() == "keyword", f"expected keyword index, got {data_type_str}"


@pytest.mark.slow
def test_ingest_writes_chunks(qdrant_client, qdrant_db, sample_pdf, book_id):
    from core.utils.ingest_book import ingest_book

    t0 = time.time()
    ingest_book(
        pdf_path=str(sample_pdf),
        qdrant_db=qdrant_db,
        book_id=book_id,
        embed_model_id=EMBED_MODEL_ID,
        extra_metadata={"source": "e2e_test"},
    )
    logger.info("ingest took %.1fs", time.time() - t0)

    count_result = qdrant_client.count(
        collection_name=COLLECTION_NAME,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.book_id",
                    match=models.MatchValue(value=book_id),
                )
            ]
        ),
        exact=True,
    )
    assert count_result.count > 0, f"no chunks written for book_id={book_id}"

    limit = min(3, count_result.count)
    points, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.book_id",
                    match=models.MatchValue(value=book_id),
                )
            ]
        ),
        with_payload=True,
        with_vectors=False,
        limit=limit,
    )
    assert len(points) >= 1, "scroll returned no points"

    for p in points:
        payload = p.payload or {}
        meta = payload.get("metadata", {})
        assert meta.get("book_id") == book_id, f"wrong book_id in payload: {meta}"
        assert meta.get("source") == "e2e_test", f"extra_metadata lost: {meta}"
        assert "page_content" in payload or "text" in payload, f"missing text field in payload keys={list(payload)}"

    sample = points[0]
    logger.info(
        "sample point id=%s book_id=%s pages=%s text_len=%d",
        sample.id,
        (sample.payload or {}).get("metadata", {}).get("book_id"),
        (sample.payload or {}).get("metadata", {}).get("pages"),
        len((sample.payload or {}).get("page_content", "") or (sample.payload or {}).get("text", "")),
    )


def test_delete_by_book_removes_only_target(qdrant_client, qdrant_db, book_id):
    other_book = f"test_book_other_{uuid.uuid4().hex[:8]}"
    qdrant_db.add_documents(
        chunks=[
            {"text": "noise chunk", "metadata": {"book_id": other_book, "source": "noise"}},
        ],
        book_id=other_book,
    )

    qdrant_db.delete_by_book(book_id)

    remaining_target = qdrant_client.count(
        collection_name=COLLECTION_NAME,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="metadata.book_id", match=models.MatchValue(value=book_id))]
        ),
        exact=True,
    )
    assert remaining_target.count == 0, f"expected 0 chunks for {book_id}, got {remaining_target.count}"

    remaining_other = qdrant_client.count(
        collection_name=COLLECTION_NAME,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="metadata.book_id", match=models.MatchValue(value=other_book))]
        ),
        exact=True,
    )
    assert remaining_other.count == 1, f"delete_by_book leaked into other book: {remaining_other.count}"

    qdrant_db.delete_by_book(other_book)


def test_drop_collection_teardown(qdrant_client):
    qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
    with pytest.raises(Exception):
        qdrant_client.get_collection(COLLECTION_NAME)


def test_ensure_payload_indexes_creates_missing_index(qdrant_client, embedding_model):
    """Regression: re-init against an existing collection that lacks the
    `metadata.book_id` payload index must create the index (idempotent),
    not silently leave it missing and break `metadata.book_id` filters.
    """
    from core.rag.qdrant_db import QdrantDB

    legacy_name = f"rag_book_legacy_{uuid.uuid4().hex[:8]}"
    # Simulate a legacy/imported collection: create it WITHOUT the payload index.
    qdrant_client.create_collection(
        collection_name=legacy_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    schema = qdrant_client.get_collection(legacy_name).payload_schema or {}
    assert "metadata.book_id" not in schema, "precondition: legacy collection should not have the index"

    try:
        QdrantDB(
            collection_name=legacy_name,
            client=qdrant_client,
            embedding_model=embedding_model,
        )

        post = qdrant_client.get_collection(legacy_name).payload_schema or {}
        assert "metadata.book_id" in post, "QdrantDB init did not create the missing payload index"
        idx = post["metadata.book_id"]
        data_type = getattr(idx, "data_type", None)
        data_type_str = getattr(data_type, "value", None) or str(data_type)
        assert data_type_str.lower() == "keyword"

        # Second init must be a no-op, not raise.
        QdrantDB(
            collection_name=legacy_name,
            client=qdrant_client,
            embedding_model=embedding_model,
        )
    finally:
        qdrant_client.delete_collection(collection_name=legacy_name)
