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
from core.reranker.reranker import Reranker
from core.storage.supabase_storage import SupabaseStorage
from core.utils.token_tracker import TokenTracker
from core.utils.llm_config import get_chat_model
from core.utils.credit_manager import CreditManager, FEATURE_COST
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

# ---------------- RERANKER ----------------
use_reranker = os.getenv("USE_RERANKER", "false").lower() == "true"
reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = None
if use_reranker:
    try:
        reranker = Reranker(model_name=reranker_model, device=device)
    except Exception as e:
        print(f"Warning: Failed to initialize reranker: {e}")

# ---------------- CACHE MANAGER ----------------
cache_manager = CacheManager(qdrant_client, embedding_model=embedding_model, reranker=reranker)

# ---------------- SUPABASE STORAGE ----------------
supabase_storage = SupabaseStorage()

# ---------------- TOKEN TRACKER ----------------
token_tracker = TokenTracker()

# ---------------- CREDIT MANAGER ----------------
credit_manager = CreditManager()

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
    # Check credits before processing
    is_sufficient, cost = credit_manager.check_sufficient_credits(payload.user_id, "agent_reasoning")
    if not is_sufficient:
        raise HTTPException(status_code=402, detail=f"Insufficient credits. Balance too low for feature 'agent_reasoning' (cost: {cost})")

    # Burn credits
    if not credit_manager.burn_credits(payload.user_id, "agent_reasoning"):
        raise HTTPException(status_code=500, detail="Failed to deduct credits")

    async def event_generator():
        try:
            agent = await cache_manager.get_agent(payload.book_id, payload.user_id)

            async for chunk in agent.ask_stream(
                payload.messages[-1].content,
                session_id=payload.session_id
            ):
                print("Sending chunk:", chunk)
                yield chunk
        except Exception as e:
            # Refund credits on failure
            credit_manager.refund_credits(payload.user_id, "agent_reasoning")
            yield f"data: {json.dumps({'id': 'chatcmpl', 'type': 'error', 'content': str(e), 'metadata': {}})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ------------------ INGEST BOOK ENDPOINT ------------------
@app.post("/book-qa/ingest", dependencies=[Depends(verify_api_key)])
@limiter.limit("3/minute")
async def ingest_pdf(
    request: Request,
    book_id: str,
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
            book_id=book_id,
            filename=file.filename,
        )

        # ✅ ALWAYS use single collection
        qdrant_db = await cache_manager.get_qdrant_db()

        async with cache_manager.acquire_book_lock(book_id):
            # ✅ pass book_id + storage_url into ingestion
            ingest_book(
                pdf_path=tmp_path,
                qdrant_db=qdrant_db,
                book_id=book_id,
                embed_model_id=EMBED_MODEL_ID,
                extra_metadata={"storage_url": storage_url},
            )

        os.remove(tmp_path)

        return JSONResponse({
            "status": "success",
            "collection": "lms_content",
            "book_id": book_id,
            "storage_url": storage_url,
            "message": f"{file.filename} ingested and stored"
        })

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

# ------------------ DELETE BOOK CHUNKS ENDPOINT ------------------
@app.delete("/book-qa/book/{book_id}/chunks", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def delete_book_chunks(
    request: Request,
    book_id: str,
    filename: str | None = None,
):
    """
    Hard-delete all chunks and the Supabase Storage file for a book.
    If `filename` is provided, only that file is removed from storage.
    If omitted, every file under `{book_id}/` is removed.
    Idempotent: returns 200 even if the book never existed.
    """
    async with cache_manager.acquire_book_lock(book_id):
        # 1) Qdrant vectors
        qdrant_db = await cache_manager.get_qdrant_db()
        qdrant_db.delete_by_book(book_id)

        # 2) Supabase Storage (best-effort)
        if filename:
            supabase_storage.delete_pdf(book_id, filename)
        else:
            existing = supabase_storage.list_pdfs(book_id) or []
            for f in existing:
                name = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
                if name:
                    supabase_storage.delete_pdf(book_id, name)

        # 3) Evict cached agent + lock for this book
        await cache_manager.clear_book(book_id)

    return JSONResponse({
        "status": "success",
        "book_id": book_id,
        "message": f"All chunks and storage for '{book_id}' removed"
    })

# ------------------ MINDMAP ENDPOINT ------------------
@app.post("/book-qa/mindmap", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def generate_mindmap(
    request: Request,
    user_id: str,
    book_id: str,
    user_prompt: str | None = None,
):
    # Check credits
    is_sufficient, cost = credit_manager.check_sufficient_credits(user_id, "mindmap")
    if not is_sufficient:
        raise HTTPException(status_code=402, detail=f"Insufficient credits (cost: {cost})")

    if not credit_manager.burn_credits(user_id, "mindmap"):
        raise HTTPException(status_code=500, detail="Failed to deduct credits")

    try:
        qdrant_db = await cache_manager.get_qdrant_db()
        llm = get_chat_model()
        generator = MindmapGenerator(qdrant_db, llm)
        result = await generator.generate(book_id, user_prompt=user_prompt)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        credit_manager.refund_credits(user_id, "mindmap")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

# ------------------ MINDMAP EDIT ENDPOINT ------------------
@app.post("/book-qa/mindmap/edit", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def edit_mindmap(request: Request, body: MindmapEditRequest):
    # Check credits
    is_sufficient, cost = credit_manager.check_sufficient_credits(body.user_id, "mindmap")
    if not is_sufficient:
        raise HTTPException(status_code=402, detail=f"Insufficient credits (cost: {cost})")

    if not credit_manager.burn_credits(body.user_id, "mindmap"):
        raise HTTPException(status_code=500, detail="Failed to deduct credits")

    try:
        qdrant_db = await cache_manager.get_qdrant_db()
        llm = get_chat_model()
        generator = MindmapGenerator(qdrant_db, llm)
        result = await generator.edit(
            book_id=body.book_id,
            mermaid=body.mermaid,
            instruction=body.instruction,
        )
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        credit_manager.refund_credits(body.user_id, "mindmap")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

# ------------------ DATASET ENDPOINT ------------------
@app.post("/book-qa/dataset", dependencies=[Depends(verify_api_key)])
@limiter.limit("2/minute")
async def generate_dataset(
    request: Request,
    user_id: str,
    book_id: str,
    difficulty: str = "medium",
    num_mcq: int = 3,
    num_essay: int = 2,
):
    # Check credits
    is_sufficient, cost = credit_manager.check_sufficient_credits(user_id, "dataset")
    if not is_sufficient:
        raise HTTPException(status_code=402, detail=f"Insufficient credits (cost: {cost})")

    if not credit_manager.burn_credits(user_id, "dataset"):
        raise HTTPException(status_code=500, detail="Failed to deduct credits")

    try:
        qdrant_db = await cache_manager.get_qdrant_db()
        llm = get_chat_model()
        generator = DatasetGenerator(qdrant_db, llm)
        result = await generator.generate(book_id, difficulty, num_mcq, num_essay)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        credit_manager.refund_credits(user_id, "dataset")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

# ------------------ SUMMARIZE BOOK ENDPOINT ------------------
@app.post("/book-qa/summarize", dependencies=[Depends(verify_api_key)])
@limiter.limit("2/minute")
async def summarize_book(
    request: Request,
    user_id: str,
    book_id: str,
    user_prompt: str | None = None,
):
    # Check credits
    is_sufficient, cost = credit_manager.check_sufficient_credits(user_id, "summarize")
    if not is_sufficient:
        raise HTTPException(status_code=402, detail=f"Insufficient credits (cost: {cost})")

    if not credit_manager.burn_credits(user_id, "summarize"):
        raise HTTPException(status_code=500, detail="Failed to deduct credits")

    try:
        qdrant_db = await cache_manager.get_qdrant_db()
        llm = get_chat_model()
        summarizer = BookSummarizer(qdrant_db, llm)
        result = await summarizer.summarize(book_id, user_prompt=user_prompt)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        credit_manager.refund_credits(user_id, "summarize")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

# ------------------ SUMMARIZE EDIT ENDPOINT ------------------
@app.post("/book-qa/summarize/edit", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def edit_summary(request: Request, body: SummaryEditRequest):
    # Check credits
    is_sufficient, cost = credit_manager.check_sufficient_credits(body.user_id, "summarize")
    if not is_sufficient:
        raise HTTPException(status_code=402, detail=f"Insufficient credits (cost: {cost})")

    if not credit_manager.burn_credits(body.user_id, "summarize"):
        raise HTTPException(status_code=500, detail="Failed to deduct credits")

    try:
        qdrant_db = await cache_manager.get_qdrant_db()
        llm = get_chat_model()
        summarizer = BookSummarizer(qdrant_db, llm)
        result = await summarizer.edit(
            book_id=body.book_id,
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
        credit_manager.refund_credits(body.user_id, "summarize")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

# ------------------ TOKEN USAGE ENDPOINT ------------------
@app.get("/token-usage", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def get_token_usage(
    request: Request,
    book_id: str,
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
            book_id=book_id,
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
    book_id: str,
    days: int = 7,
):
    """Query daily token usage aggregation."""
    try:
        daily_usage = await token_tracker.query_daily_usage(
            book_id=book_id,
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

# ------------------ CREDIT ENDPOINTS ------------------
@app.get("/credits/{user_id}", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def get_credits(request: Request, user_id: str):
    """Get user's current credit balance and recent transactions."""
    try:
        balance = credit_manager.get_balance(user_id)
        recent = await credit_manager.get_recent_transactions(user_id, limit=10)
        return JSONResponse({
            "balance": balance,
            "recent_transactions": recent,
        })
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.post("/credits/{user_id}/add", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def add_credits(
    request: Request,
    user_id: str,
    amount: float,
    transaction_type: str = "purchase",
):
    """Add credits to user account (admin only)."""
    try:
        success = credit_manager.add_credits(user_id, amount, transaction_type)
        if success:
            return JSONResponse({
                "status": "success",
                "new_balance": credit_manager.get_balance(user_id),
            })
        return JSONResponse(
            {"status": "error", "message": "Failed to add credits"},
            status_code=500
        )
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.post("/credits/{user_id}/burn", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def burn_credits(
    request: Request,
    user_id: str,
    feature: str = "unknown",
    amount: float = 0.0,
):
    """Manually burn credits for a user."""
    try:
        success = credit_manager.burn_credits(user_id, feature, estimated_cost=amount)
        if success:
            return JSONResponse({
                "status": "success",
                "new_balance": credit_manager.get_balance(user_id),
            })
        return JSONResponse(
            {"status": "error", "message": "Failed to burn credits"},
            status_code=500
        )
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
