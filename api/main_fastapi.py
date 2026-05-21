import asyncio
import selectors

import os
from contextlib import asynccontextmanager
import uvicorn
import uuid

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Security, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader
from qdrant_client import QdrantClient

from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from core.schemas.chat import ChatPayload
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  

from core.utils.ingest_book import ingest_book
from core.utils.cache_manager import CacheManager

import torch

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import PlainTextResponse

limiter = Limiter(key_func=get_remote_address)

# ---------------- ENV ----------------
load_dotenv()
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_ENDPOINT"),
    api_key=os.getenv("QDRANT_API_KEY")
)
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
API_KEY = os.getenv("API_KEY")

# ---------------- EMBEDDINGS ----------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    model_kwargs={"device": device}
)

# ---------------- CACHE MANAGER ----------------
cache_manager = CacheManager(qdrant_client, embedding_model=embedding_model)

# ------------------ LIFESPAN ------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPostgresSaver.from_conn_string(SUPABASE_DB_URL) as checkpointer:
        await cache_manager.initialize(checkpointer)

        # Setup LangSmith observability
        os.environ.setdefault("LANGSMITH_TRACING", os.getenv("LANGSMITH_TRACING", "true"))
        os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "rag-book-production"))

        yield  # FastAPI siap jalan

app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: PlainTextResponse("Rate limit exceeded", status_code=429))

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key

# ------------------ STREAMING ENDPOINT ------------------
@app.post("/book-qa/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def book_qa_stream(request: Request, payload: ChatPayload):
    
    async def event_generator():
        # Create DB per collection
        agent = await cache_manager.get_agent(payload.course_id)
        
        async for chunk in agent.ask_stream(
            payload.messages[-1].content,
            session_id=payload.session_id
        ):
            print("Sending chunk:", chunk)
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ------------------ INGEST BOOK ENDPOINT ------------------
@app.post("/book-qa/ingest", dependencies=[Depends(verify_api_key)])
@limiter.limit("3/minute")
async def ingest_pdf(
    request: Request,
    course_id: str, 
    file: UploadFile = File(...)
):
    try:
        # safer temp filename (avoid overwrite)
        tmp_path = f"./{file.filename}"

        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        # ✅ ALWAYS use single collection
        qdrant_db = await cache_manager.get_qdrant_db()

        # ✅ pass course_id into ingestion
        ingest_book(
            pdf_path=tmp_path,
            qdrant_db=qdrant_db,
            course_id=course_id,  # 🔥 key change
            embed_model_id=EMBED_MODEL_ID,
        )

        os.remove(tmp_path)

        return JSONResponse({
            "status": "success",
            "collection": "lms_content",
            "course_id": course_id,
            "message": f"{file.filename} ingested"
        })

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )
        
async def main():
    config = uvicorn.Config(app=app, host="127.0.0.1", port=8001)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main(), loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()))