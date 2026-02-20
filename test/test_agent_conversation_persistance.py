import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver

from rag.qdrant.qdrant_db import QdrantDB
from agents.book_qdrant_agent import BookQdrantAgent

load_dotenv()

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

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

with PostgresSaver.from_conn_string(SUPABASE_DB_URL) as checkpointer:
    checkpointer.setup()

    agent = BookQdrantAgent(
        qdrant_db=qdrant_db,
        checkpointer=checkpointer
    )

    # ===== Test Ask =====
    agent.ask("What is fundamental principle in lean software development?")