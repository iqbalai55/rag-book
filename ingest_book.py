import logging
from typing import Optional, Dict

from services.rag.qdrant.document_processor import DocumentProcessor
from services.rag.qdrant.qdrant_db import QdrantDB

from transformers import AutoTokenizer
from docling.chunking import HybridChunker

logger = logging.getLogger(__name__)


def ingest_book(
    pdf_path: str,
    qdrant_db: QdrantDB,
    course_id: str,  # ✅ REQUIRED for multitenancy
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens: int = 256,
    extra_metadata: Optional[Dict] = None,  # ✅ extensible
) -> None:
    """
    Process PDF → Chunk → Embed → Index (Multitenant)

    Args:
        pdf_path: path to PDF
        qdrant_db: QdrantDB instance
        course_id: tenant identifier (REQUIRED)
        extra_metadata: optional metadata (module_id, lesson_id, etc.)
    """

    logger.info(f"🚀 Ingesting {pdf_path} into course '{course_id}'...")

    tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
    chunker = HybridChunker(tokenizer=tokenizer)

    processor = DocumentProcessor(
        chunker=chunker,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )

    logger.info("📄 Processing PDF...")
    docs = processor.process_document(pdf_path)
    logger.info(f"✅ Created {len(docs)} chunks")

    # -------------------------
    # 🔥 MULTITENANCY INJECTION
    # -------------------------
    enriched_docs = []

    for d in docs:
        metadata = d.get("metadata", {})

        # enforce course_id
        metadata["course_id"] = course_id

        # merge extra metadata if provided
        if extra_metadata:
            metadata.update(extra_metadata)

        enriched_docs.append({
            "text": d["text"],
            "metadata": metadata
        })

    logger.info("📦 Indexing into Qdrant...")
    qdrant_db.add_documents(enriched_docs)

    logger.info("✅ Book indexed successfully!")

if __name__ == "__main__":

    from qdrant_client import QdrantClient
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    qdrant_client = QdrantClient(path="./qdrant_storage")

    qdrant_db = QdrantDB(
        collection_name="lms_content",  # ✅ single collection now
        embedding_model=embedding_model,
        client=qdrant_client,
    )

    pdf_path = r"book\Refactoring for Software Design Smells Managing Technical Debt.pdf"

    ingest_book(
        pdf_path=pdf_path,
        qdrant_db=qdrant_db,
        course_id="refactoring_book",  # 🔥 REQUIRED
        extra_metadata={
            "module_id": "m1",
            "source": "pdf"
        }
    )