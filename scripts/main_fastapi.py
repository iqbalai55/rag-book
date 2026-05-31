import asyncio
import selectors
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import asynccontextmanager
import uvicorn
import uuid

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Security, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from qdrant_client import QdrantClient

from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from core.schemas.chat import ChatPayload

from core.utils.ingest_book import ingest_book
from core.utils.cache_manager import CacheManager
from core.storage.supabase_storage import SupabaseStorage
from core.utils.token_tracker import TokenTracker
from core.utils.llm_config import get_chat_model
from core.services.dataset_generator import DatasetGenerator
from core.services.summarize_book import BookSummarizer
from core.services.mindmap_generator import MindmapGenerator
from core.schemas.mindmap import MindmapEditRequest
from core.schemas.summary import SummaryEditRequest

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

# ---------------- SUPABASE STORAGE ----------------
supabase_storage = SupabaseStorage()

# ---------------- TOKEN TRACKER ----------------
token_tracker = TokenTracker()

# ------------------ LIFESPAN ------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpointer = None
    try:
        from supabase import create_client
        from scripts.supabase_checkpointer import SupabaseCheckpointer

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        if supabase_url and supabase_key:
            client = create_client(supabase_url, supabase_key)
            checkpointer = SupabaseCheckpointer(client)
            print("Supabase REST checkpointer initialized")
        else:
            print("Warning: SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
    except Exception as e:
        print(f"Warning: Could not initialize Supabase checkpointer: {e}")

    await cache_manager.initialize(checkpointer)

    # Setup LangSmith observability
    os.environ.setdefault("LANGSMITH_TRACING", os.getenv("LANGSMITH_TRACING", "true"))
    os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "rag-book-production"))

    yield  # FastAPI siap jalan

app = FastAPI(lifespan=lifespan)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fix: Force 422 for validation errors (override slowapi's 200 response)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

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

        # Upload PDF to Supabase Storage
        storage_url = supabase_storage.upload_pdf(
            file_path=tmp_path,
            course_id=course_id,
            filename=file.filename,
        )

        # ✅ ALWAYS use single collection
        qdrant_db = await cache_manager.get_qdrant_db()

        # ✅ pass course_id + storage_url into ingestion
        ingest_book(
            pdf_path=tmp_path,
            qdrant_db=qdrant_db,
            course_id=course_id,
            embed_model_id=EMBED_MODEL_ID,
            extra_metadata={"storage_url": storage_url},
        )

        os.remove(tmp_path)

        return JSONResponse({
            "status": "success",
            "collection": "lms_content",
            "course_id": course_id,
            "storage_url": storage_url,
            "message": f"{file.filename} ingested and stored"
        })

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

# ------------------ MINDMAP ENDPOINT ------------------
@app.post("/book-qa/mindmap", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def generate_mindmap(
    request: Request,
    course_id: str,
    user_prompt: str | None = None,
):
    try:
        qdrant_db = await cache_manager.get_qdrant_db()
        llm = get_chat_model()
        generator = MindmapGenerator(qdrant_db, llm)
        result = await generator.generate(course_id, user_prompt=user_prompt)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

# ------------------ MINDMAP EDIT ENDPOINT ------------------
@app.post("/book-qa/mindmap/edit", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def edit_mindmap(request: Request, body: MindmapEditRequest):
    try:
        qdrant_db = await cache_manager.get_qdrant_db()
        llm = get_chat_model()
        generator = MindmapGenerator(qdrant_db, llm)
        result = await generator.edit(
            course_id=body.course_id,
            mermaid=body.mermaid,
            instruction=body.instruction,
        )
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

# ------------------ DATASET ENDPOINT ------------------
@app.post("/book-qa/dataset", dependencies=[Depends(verify_api_key)])
@limiter.limit("2/minute")
async def generate_dataset(
    request: Request,
    course_id: str,
    difficulty: str = "medium",
    num_mcq: int = 3,
    num_essay: int = 2,
):
    try:
        qdrant_db = await cache_manager.get_qdrant_db()
        llm = get_chat_model()
        generator = DatasetGenerator(qdrant_db, llm)
        result = await generator.generate(course_id, difficulty, num_mcq, num_essay)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

# ------------------ SUMMARIZE BOOK ENDPOINT ------------------
@app.post("/book-qa/summarize", dependencies=[Depends(verify_api_key)])
@limiter.limit("2/minute")
async def summarize_book(
    request: Request,
    course_id: str,
    user_prompt: str | None = None,
):
    """Summarize entire book using hierarchical map-reduce approach."""
    try:
        qdrant_db = await cache_manager.get_qdrant_db()
        llm = get_chat_model()
        summarizer = BookSummarizer(qdrant_db, llm)
        result = await summarizer.summarize(course_id, user_prompt=user_prompt)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

# ------------------ SUMMARIZE EDIT ENDPOINT ------------------
@app.post("/book-qa/summarize/edit", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def edit_summary(request: Request, body: SummaryEditRequest):
    """Edit existing summary based on user instructions."""
    try:
        qdrant_db = await cache_manager.get_qdrant_db()
        llm = get_chat_model()
        summarizer = BookSummarizer(qdrant_db, llm)
        result = await summarizer.edit(
            course_id=body.course_id,
            title=body.title,
            overview=body.overview,
            key_themes=body.key_themes,
            chapters=[c.model_dump() for c in body.chapters],
            instruction=body.instruction,
        )
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

# ------------------ TOKEN USAGE ENDPOINT ------------------
@app.get("/token-usage", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def get_token_usage(
    request: Request,
    course_id: str,
    session_id: str | None = None,
    feature: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
):
    """Query token usage statistics."""
    try:
        # Get in-memory summary
        in_memory_summary = token_tracker.get_summary()

        # Get from database
        db_usage = await token_tracker.query_usage(
            course_id=course_id,
            session_id=session_id,
            feature=feature,
            from_date=from_date,
            to_date=to_date,
        )

        return JSONResponse({
            "in_memory": in_memory_summary,
            "database": db_usage,
        })
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/token-usage/daily", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def get_daily_token_usage(
    request: Request,
    course_id: str,
    days: int = 7,
):
    """Query daily token usage aggregation."""
    try:
        daily_usage = await token_tracker.query_daily_usage(
            course_id=course_id,
            days=days,
        )
        return JSONResponse({"daily_usage": daily_usage})
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.post("/token-usage/flush", dependencies=[Depends(verify_api_key)])
async def flush_token_usage():
    """Manually flush buffered token usage records to DB."""
    try:
        await token_tracker.flush()
        return JSONResponse({"status": "flushed"})
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