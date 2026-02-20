from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from agents.book_qdrant_agent import BookQdrantAgent
from qdrant_client import QdrantClient  
import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.qdrant.qdrant_db import QdrantDB
from schemas.chat import ChatPayload

from langgraph.checkpoint.postgres import PostgresSaver

import mlflow
import mlflow.langchain  

load_dotenv()

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
STORAGE_PATH = "./qdrant_storage"
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    model_kwargs={"device": "cpu"}
)

os.makedirs(STORAGE_PATH, exist_ok=True)
qdrant_client = QdrantClient(path=STORAGE_PATH)

qdrant_db = QdrantDB(
    collection_name="real_books",
    embedding_model=embedding_model,
    client=qdrant_client  
)

agent_instance = None
_checkpointer_ctx = None

# ------------------ LIFESPAN ------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_instance, _checkpointer_ctx

    # Postgres checkpointer
    _checkpointer_ctx = PostgresSaver.from_conn_string(SUPABASE_DB_URL)
    checkpointer = _checkpointer_ctx.__enter__()
    checkpointer.setup()

    # Book agent
    agent_instance = BookQdrantAgent(qdrant_db=qdrant_db, checkpointer=checkpointer)

    # Setup MLflow autolog
    mlflow.set_tracking_uri("file:./mlruns")  # bisa diganti ke server MLflow
    mlflow.set_experiment("book_qa_streaming")
    mlflow.langchain.autolog()  # <- autolog semua interaksi LangChain

    yield

    # Shutdown checkpointer
    _checkpointer_ctx.__exit__(None, None, None)


app = FastAPI(lifespan=lifespan)

# ------------------ ENDPOINT ------------------
@app.post("/book-qa/stream")
async def book_qa_stream(payload: ChatPayload):
    async def event_generator():
        async for chunk in agent_instance.ask_stream(
            payload.messages[-1].content,
            session_id=payload.session_id
        ):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream")