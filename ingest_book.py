from rag.qdrant.document_processor import DocumentProcessor
from rag.qdrant.qdrant_db import QdrantDB
from transformers import AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from docling.chunking import HybridChunker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from qdrant_client import QdrantClient

def ingest_book(pdf_path: str,
                client: QdrantClient,
                collection_name: str = "real_books",
                embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
                max_tokens: int = 256) -> QdrantDB:
    """
    Process PDF → Chunk → Embed → Index to a provided Qdrant client
    """
    logger.info(f"🚀 Ingesting {pdf_path}...")

    # Initialize components
    tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
    chunker = HybridChunker(tokenizer=tokenizer)
    processor = DocumentProcessor(chunker=chunker, tokenizer=tokenizer, max_tokens=max_tokens)

    embedding_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={"device": "cpu"}
    )

    qdrant_db = QdrantDB(
        collection_name=collection_name,
        embedding_model=embedding_model,
        client=client  # Use the provided Qdrant client
    )

    # Process and index
    logger.info("📄 Processing PDF...")
    docs = processor.process_document(pdf_path)
    logger.info(f"✅ Created {len(docs)} chunks")

    logger.info("📦 Indexing into Qdrant...")
    qdrant_db.add_documents(docs)
    logger.info("✅ Book indexed successfully!")

    # Optional test retrieval
    #test_query = "What is on page 42?"
    #retrieved = qdrant_db.query(test_query, k=3)
    #logger.info("🧪 Sample retrieval:")
    #for i, doc in enumerate(retrieved):
    #    pages = doc.metadata.get("pages", [])
    #    logger.info(f"  Chunk {i+1}: Pages {pages}")

    return qdrant_db


if __name__ == "__main__":
    from qdrant_client import QdrantClient

    # Example: use a local persistent Qdrant path
    client = QdrantClient(path="./qdrant_storage")
    pdf_path = r"book\Lean Software Development An Agile Toolkit (Mary Poppendieck  Tom Poppendieck) (Z-Library).pdf"
    db = ingest_book(pdf_path, client=client)