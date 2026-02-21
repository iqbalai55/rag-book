import os
from contextlib import asynccontextmanager

import modal
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Security
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader
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
API_KEY = os.getenv("API_KEY")


# ---------------- MODAL APP ----------------
stub = modal.App("book_qa_app")

qdrant_volume = modal.SharedVolume().persist("qdrant_storage_volume")
mlflow_volume = modal.SharedVolume().persist("mlflow_runs_volume")
embedding_cache_volume = modal.SharedVolume().persist("hf_embedding_cache")


# ---------------- API KEY SECURITY ----------------
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key


# ---------------- GLOBAL OBJECTS ----------------
agent_checkpointer = None
qdrant_client = None


# ---------------- LIFESPAN ----------------
@stub.function(
    timeout=2*3600,
    volumes={
        QDRANT_PATH: qdrant_volume,
        MLFLOW_PATH: mlflow_volume,
        "/hf_cache": embedding_cache_volume
    },
    image=modal.Image.debian_slim().pip_install_from_file("requirements.txt")
)
@asynccontextmanager
async def lifespan():
    global agent_checkpointer, qdrant_client

    async with AsyncPostgresSaver.from_conn_string(SUPABASE_DB_URL) as checkpointer:
        await checkpointer.setup()

        qdrant_client = QdrantClient(path=QDRANT_PATH)

        agent_checkpointer = checkpointer

        os.makedirs(MLFLOW_PATH, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{MLFLOW_PATH}")
        mlflow.set_experiment("book_qa_streaming")
        mlflow.langchain.autolog()

        yield


# ---------------- FASTAPI ----------------
@stub.function()
def create_app():
    app = FastAPI(lifespan=lifespan)

    # ---------------- STREAMING QA ----------------
    @app.post("/book-qa/stream", dependencies=[Depends(verify_api_key)])
    async def book_qa_stream(payload: ChatPayload):

        async def event_generator():

            dynamic_qdrant_db = QdrantDB(
                collection_name=payload.collection_name,
                client=qdrant_client
            )

            dynamic_agent = BookQdrantAgent(
                qdrant_db=dynamic_qdrant_db,
                checkpointer=agent_checkpointer
            )

            async for chunk in dynamic_agent.ask_stream(
                payload.messages[-1].content,
                session_id=payload.session_id
            ):
                yield chunk

        return StreamingResponse(event_generator(), media_type="text/event-stream")


    # ---------------- INGEST ----------------
    @app.post("/book-qa/ingest", dependencies=[Depends(verify_api_key)])
    async def ingest_pdf(
        collection_name: str,
        file: UploadFile = File(...)
    ):
        try:
            tmp_path = f"/tmp/{file.filename}"
            with open(tmp_path, "wb") as f:
                f.write(await file.read())

            ingest_book(
                pdf_path=tmp_path,
                client=qdrant_client,
                collection_name=collection_name
            )

            os.remove(tmp_path)

            return JSONResponse({
                "status": "success",
                "collection": collection_name,
                "message": f"{file.filename} ingested"
            })

        except Exception as e:
            return JSONResponse(
                {"status": "error", "message": str(e)},
                status_code=500
            )

    return app


# ---------------- DEPLOY ----------------
if __name__ == "__main__":
    stub.serve(create_app)