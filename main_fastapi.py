from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from agents.book_qdrant_agent import BookQdrantAgent
from qdrant_client import QdrantClient  
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.qdrant.qdrant_db import QdrantDB
from schemas.chat import ChatPayload

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256
STORAGE_PATH = "./qdrant_storage"  # Your persistent storage

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_instance
    agent_instance = BookQdrantAgent(qdrant_db)
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/book-qa/stream")
async def book_qa_stream(payload: ChatPayload):
    async def event_generator():
        async for chunk in agent_instance.ask_stream(
            payload.messages[-1].content, 
            session_id=payload.session_id
        ):
            yield chunk
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
