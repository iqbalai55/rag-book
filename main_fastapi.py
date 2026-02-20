import asyncio
import selectors

import os
from contextlib import asynccontextmanager
import uvicorn

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

from agents.book_qdrant_agent import BookQdrantAgent
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.qdrant.qdrant_db import QdrantDB
from schemas.chat import ChatPayload
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  

from ingest_book import ingest_book

import mlflow
import mlflow.langchain

# ---------------- ENV ----------------
load_dotenv()
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
STORAGE_PATH = "./qdrant_storages"
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
MLRUNS_PATH = "./mlruns"

# ---------------- QDRANT + EMBEDDINGS ----------------
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    model_kwargs={"device": "cpu"}
)
os.makedirs(STORAGE_PATH, exist_ok=True)

qdrant_client = QdrantClient(path=STORAGE_PATH)
qdrant_db = QdrantDB(
    collection_name="test",
    embedding_model=embedding_model,
    client=qdrant_client
)

agent_instance = None  # akan diinisialisasi di lifespan

# ------------------ LIFESPAN ------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_instance
    
    # Check to make sure that we bypassed the original eventloop Policy....
    # assert isinstance(asyncio.get_event_loop_policy(), winloop.EventLoopPolicy)

    # Async Postgres checkpointer
    async with AsyncPostgresSaver.from_conn_string(SUPABASE_DB_URL) as checkpointer:
        # await checkpointer.setup()  # ⚡ async setup

        # Inisialisasi Book Agent
        agent_instance = BookQdrantAgent(qdrant_db=qdrant_db, checkpointer=checkpointer)

        # Setup MLflow
        os.makedirs(MLRUNS_PATH, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{MLRUNS_PATH}")
        mlflow.set_experiment("book_qa_streaming")
        mlflow.langchain.autolog()

        yield  # FastAPI siap jalan

app = FastAPI(lifespan=lifespan)

# ------------------ STREAMING ENDPOINT ------------------
@app.post("/book-qa/stream")
async def book_qa_stream(payload: ChatPayload):
    async def event_generator():
        async for chunk in agent_instance.ask_stream(
            payload.messages[-1].content,
            session_id=payload.session_id
        ):
            yield chunk
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ------------------ INGEST BOOK ENDPOINT ------------------
@app.post("/book-qa/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        tmp_path = f"/tmp/{file.filename}"
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        # Ingest ke Qdrant
        ingest_book(pdf_path=tmp_path, client=qdrant_client, collection_name="test")

        os.remove(tmp_path)
        return JSONResponse({"status": "success", "message": f"{file.filename} ingested"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    
async def main():
    config = uvicorn.Config(app=app, host="127.0.0.1", port=8001)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main(), debug=True, loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()))