import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver

from agents.book_podcast_agent import BookPodcastAgent
from services.rag.qdrant.qdrant_db import QdrantDB

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

    agent = BookPodcastAgent(
        qdrant_db=qdrant_db,
        checkpointer=checkpointer
    )

    for event in agent.agent.stream(
        {
            "messages": [
                {"role": "user", "content": "Buat podcast tentang code smell"}
            ]
        },
        config={
            "configurable": {
                "thread_id": "podcast-session-1"
            }
        },
        stream_mode="updates"  # 🔥 WAJIB
    ):
        print(event)