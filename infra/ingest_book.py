import os
import logging
from typing import Optional, Dict

from services.rag.qdrant.document_processor import DocumentProcessor
from services.rag.qdrant.qdrant_db import QdrantDB

from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


# INGEST FUNCTION (PRODUCTION READY)
def ingest_book(
    pdf_path: str,
    qdrant_db: QdrantDB,
    course_id: str,
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens: int = 256,
    extra_metadata: Optional[Dict] = None,
) -> None:

    logger.info(f"🚀 Ingesting {pdf_path} into course '{course_id}'")

    tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
    chunker = HybridChunker(tokenizer=tokenizer)

    processor = DocumentProcessor(
        chunker=chunker,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )

    docs = processor.process_document(pdf_path)
    logger.info(f"📄 Created {len(docs)} chunks")

    enriched_docs = []

    for d in docs:
        metadata = d.get("metadata", {})

        # -----------------------------
        # MULTITENANT ENFORCEMENT
        # -----------------------------
        metadata.update({
            "course_id": course_id
        })

        if extra_metadata:
            metadata.update(extra_metadata)

        enriched_docs.append({
            "text": d["text"],
            "metadata": metadata
        })

    logger.info("📦 Indexing into Qdrant Cloud...")

    # 🔥 IMPORTANT: use per-course collection (recommended)
    qdrant_db.add_documents(
        chunks=enriched_docs,
        course_id=course_id
    )

    logger.info("✅ Ingestion complete")


# =========================================================
# LOCAL TEST (CLOUD QDRANT VERSION)
# =========================================================
if __name__ == "__main__":


    qdrant_client = QdrantClient(
        url=os.environ["QDRANT_ENDPOINT"],
        api_key=os.environ["QDRANT_API_KEY"]
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    qdrant_db = QdrantDB(
        collection_name="lms_content_refactoring_book",
        embedding_model=embedding_model,
        client=qdrant_client,
    )


    pdf_path = r"book/Refactoring for Software Design Smells Managing Technical Debt.pdf"

    ingest_book(
        pdf_path=pdf_path,
        qdrant_db=qdrant_db,
        course_id="refactoring_book",
        extra_metadata={
            "module_id": "m1",
            "source": "pdf"
        }
    )