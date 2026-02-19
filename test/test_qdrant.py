import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from transformers import AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from docling.chunking import HybridChunker
import mlflow


from agents.book_qdrant_agent import BookQdrantAgent
from rag.qdrant.qdrant_db import QdrantDB  
from rag.qdrant.document_processor import DocumentProcessor 


logging.basicConfig(level=logging.INFO)

mlflow.set_experiment("Test qdrant agent")
mlflow.langchain.autolog()

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256

pdf_path = r"book\Lean Software Development An Agile Toolkit (Mary Poppendieck  Tom Poppendieck) (Z-Library).pdf"


tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
chunker = HybridChunker(tokenizer=tokenizer)

processor = DocumentProcessor(
    chunker=chunker,
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS
)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    model_kwargs={"device": "cpu"}  # change to "cuda" if GPU
)

qdrant_db = QdrantDB(
    collection_name="real_books",
    embedding_model=embedding_model
)

chunks = processor.process_document(pdf_path)
qdrant_db.add_documents(chunks)

agent = BookQdrantAgent(qdrant_db=qdrant_db, k=3)

query = (
    "What is most important principle beetwen fundamental principle in lean software development?\n\n"
)

agent.ask(query=query)
