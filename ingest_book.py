from services.rag.qdrant.document_processor import DocumentProcessor
from services.rag.qdrant.qdrant_db import QdrantDB
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

if __name__ == "__main__":

    from qdrant_client import QdrantClient
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    qdrant_client = QdrantClient(path="./qdrant_storage")

    qdrant_db = QdrantDB(
        collection_name="real_books",
        embedding_model=embedding_model,
        client=qdrant_client
    )

    pdf_path = r"book\Refactoring for Software Design Smells Managing Technical Debt.pdf"

    ingest_book(pdf_path=pdf_path, qdrant_db=qdrant_client)

