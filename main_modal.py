import os
from contextlib import asynccontextmanager

import modal
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

from agents.book_qdrant_agent import BookQdrantAgent
from rag.qdrant.qdrant_db import QdrantDB
from schemas.chat import ChatPayload
from ingest_book import ingest_book  

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  
from qdrant_client import QdrantClient

import mlflow
import mlflow.langchain

# ---------------- ENV ----------------
load_dotenv()
QDRANT_PATH = "/qdrant_storage"
MLFLOW_PATH = "/mlruns"
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# ---------------- MODAL APP ----------------
stub = modal.App("book_qa_app")

# Persistent volumes
qdrant_volume = modal.SharedVolume().persist("qdrant_storage_volume")
mlflow_volume = modal.SharedVolume().persist("mlflow_runs_volume")
embedding_cache_volume = modal.SharedVolume().persist("hf_embedding_cache")

# ---------------- LIFESPAN ----------------
@stub.function(
    timeout=2*3600,
    volumes={QDRANT_PATH: qdrant_volume, MLFLOW_PATH: mlflow_volume, "/hf_cache": embedding_cache_volume},
    image=modal.Image.debian_slim().pip_install_from_file("requirements.txt")
)
@asynccontextmanager
async def lifespan():
    global agent_instance, qdrant_client

    # Postgres async checkpointer
    async with AsyncPostgresSaver.from_conn_string(SUPABASE_DB_URL) as checkpointer:
        await checkpointer.setup()  # ⚡ async-safe

        # Persistent Qdrant client
        qdrant_client = QdrantClient(path=QDRANT_PATH)
        qdrant_db = QdrantDB(
            collection_name="real_books",
            client=qdrant_client
        )

        # Agent instance
        agent_instance = BookQdrantAgent(qdrant_db=qdrant_db, checkpointer=checkpointer)

        # MLflow autolog
        os.makedirs(MLFLOW_PATH, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{MLFLOW_PATH}")
        mlflow.set_experiment("book_qa_streaming")
        mlflow.langchain.autolog()

        yield  # FastAPI siap jalan

# ---------------- FASTAPI ----------------
@stub.function()
def create_app():
    app = FastAPI(lifespan=lifespan)

    # ---------------- Streaming QA endpoint ----------------
    @app.post("/book-qa/stream")
    async def book_qa_stream(payload: ChatPayload):
        async def event_generator():
            async for chunk in agent_instance.ask_stream(
                payload.messages[-1].content,
                session_id=payload.session_id
            ):
                yield chunk
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ---------------- Ingest Book API ----------------
    @app.post("/book-qa/ingest")
    async def ingest_pdf(file: UploadFile = File(...)):
        try:
            tmp_path = f"/tmp/{file.filename}"
            with open(tmp_path, "wb") as f:
                f.write(await file.read())

            # Ingest ke Qdrant
            ingest_book(pdf_path=tmp_path, client=qdrant_client)

            os.remove(tmp_path)
            return JSONResponse({"status": "success", "message": f"{file.filename} ingested"})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    return app

# ---------------- DEPLOY ----------------
if __name__ == "__main__":
    stub.serve(create_app)