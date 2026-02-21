import os
from contextlib import asynccontextmanager
import modal
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Security
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader

from schemas.chat import ChatPayload
from ingest_book import ingest_book
from utils.chace_manager import CacheManager
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import mlflow
import mlflow.langchain

# ---------------- MODAL SECRET ----------------
secret = modal.Secret.from_dict({
    "SUPABASE_DB_URL": os.getenv("SUPABASE_DB_URL"),
    "API_KEY": os.getenv("API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "OPENAI_MODEL": os.getenv("OPENAI_MODEL"),
})

# ---------------- CONSTANTS ----------------
QDRANT_PATH = "/qdrant_storage"
MLFLOW_PATH = "/mlruns"
HF_CACHE_PATH = "/hf_cache"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------- EMBEDDING ----------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    model_kwargs={"device": device},
    cache_folder=HF_CACHE_PATH,
)

# ---------------- CACHE MANAGER ----------------
cache_manager = CacheManager(QDRANT_PATH, embedding_model=embedding_model)

# ---------------- API KEY ----------------
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.environ.get("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key

# ---------------- MODAL APP ----------------
app = modal.App("book_qa_app", secrets=[secret])
image = (
    modal.Image.debian_slim()
    .apt_install([
        "libgl1",             # OpenCV dependency
        "libglib2.0-0",       # OpenCV dependency
        "libsm6",             # OpenCV dependency
        "libxext6",           # OpenCV dependency
        "libxrender1",        # OpenCV dependency
        "poppler-utils",      # PDF processing
        "ffmpeg",             # Video/audio if needed
    ])
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("schemas")
    .add_local_python_source("utils")
    .add_local_python_source("agents")
    .add_local_python_source("rag")
    .add_local_python_source("ingest_book")
)

qdrant_volume = modal.Volume.from_name("qdrant_storage_volume")
mlflow_volume = modal.Volume.from_name("mlflow_runs_volume")
embedding_cache_volume = modal.Volume.from_name("hf_embedding_cache")


# ---------------- FASTAPI LIFESPAN ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPostgresSaver.from_conn_string( os.environ["SUPABASE_DB_URL"]) as checkpointer:

        await cache_manager.initialize(checkpointer)

        # 3️⃣ MLflow setup
        os.makedirs(MLFLOW_PATH, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{MLFLOW_PATH}")
        mlflow.set_experiment("book_qa_streaming")
        mlflow.langchain.autolog()

        yield


# ---------------- FASTAPI LIFESPAN ----------------
@app.function(
    timeout=2*3600,
    gpu="T4",
    volumes={
        QDRANT_PATH: qdrant_volume,
        MLFLOW_PATH: mlflow_volume,
        HF_CACHE_PATH: embedding_cache_volume,
    },
    image=image
)
@modal.asgi_app(label="book-qa-fastapi")
def fastapi_app():
    
    web_app = FastAPI(lifespan=lifespan)

    # ---------------- STREAMING ----------------
    @web_app.post("/book-qa/stream", dependencies=[Depends(verify_api_key)])
    async def book_qa_stream(payload: ChatPayload):
        async def event_generator():
            agent = await cache_manager.get_agent(payload.collection_name)
            async for chunk in agent.ask_stream(
                payload.messages[-1].content,
                session_id=payload.session_id
            ):
                yield chunk
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ---------------- INGEST ----------------
    @web_app.post("/book-qa/ingest", dependencies=[Depends(verify_api_key)])
    async def ingest_pdf(collection_name: str, file: UploadFile = File(...)):
        tmp_path = f"/tmp/{file.filename}"
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        qdrant_db = await cache_manager.get_qdrant_db(collection_name)
        ingest_book(pdf_path=tmp_path, qdrant_db=qdrant_db, embed_model_id=EMBED_MODEL_ID)
        embedding_cache_volume.commit()
        os.remove(tmp_path)

        return JSONResponse({
            "status": "success",
            "collection": collection_name,
            "message": f"{file.filename} ingested"
        })

    return web_app