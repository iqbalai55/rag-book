from rag.qdrant.document_processor import DocumentProcessor
from rag.qdrant.qdrant_db import QdrantDB
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from qdrant_client import QdrantClient

import logging

logger = logging.getLogger(__name__)


def ingest_book(
    pdf_path: str,
    qdrant_db: QdrantDB,
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens: int = 256,
) -> None:
    """
    Process PDF → Chunk → Embed → Index
    QdrantDB already contains embedding_model
    """

    logger.info(f"🚀 Ingesting {pdf_path}...")

    tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
    chunker = HybridChunker(tokenizer=tokenizer)

    processor = DocumentProcessor(
        chunker=chunker,
        tokenizer=tokenizer,
        max_tokens=max_tokens
    )

    logger.info("📄 Processing PDF...")
    docs = processor.process_document(pdf_path)
    logger.info(f"✅ Created {len(docs)} chunks")

    logger.info("📦 Indexing into Qdrant...")
    qdrant_db.add_documents(docs)

    logger.info("✅ Book indexed successfully!")