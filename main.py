import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from agents.book_qdrant_agent import BookQdrantAgent
from rag.qdrant.qdrant_db import QdrantDB  
from rag.qdrant.document_processor import DocumentProcessor 

logging.basicConfig(level=logging.INFO)

pdf_path = r"book\Lean Software Development An Agile Toolkit (Mary Poppendieck  Tom Poppendieck) (Z-Library).pdf"  

embedding_model = HuggingFaceInstructEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # or "cuda" if GPU available
)

qdrant_db = QdrantDB(collection_name="real_books", embedding_model=embedding_model)

processor = DocumentProcessor()
chunks = processor.process_document(pdf_path)

qdrant_db.add_documents(chunks)

agent = BookQdrantAgent(qdrant_db=qdrant_db, k=3)

query = (
    "What is fundamental principle in lean software development?\n\n"
    "can you eleborate more in simple."
)

agent.ask(query=query)