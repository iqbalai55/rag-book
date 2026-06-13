import os
import json
import sys
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Security, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.responses import PlainTextResponse

from supabase import create_client
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from qdrant_client import QdrantClient

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Make project root importable when run as `python scripts/main_docker.py` too
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schemas.chat import ChatPayload
from core.schemas.mindmap import MindmapEditRequest
from core.schemas.summary import SummaryEditRequest
from core.utils.ingest_book import ingest_book
from core.utils.cache_manager import CacheManager
from core.storage.supabase_storage import SupabaseStorage
from core.utils.token_tracker import TokenTracker
from core.utils.llm_config import get_chat_model
from core.utils.credit_manager import CreditManager, FEATURE_COST
from core.utils import ingest_jobs
from core.utils.ingest_worker import run_worker_loop
from core.services.dataset_generator import DatasetGenerator
from core.services.summarize_book import BookSummarizer
from core.services.mindmap_generator import MindmapGenerator
from core.api.book_feature_endpoint import (
    book_feature_endpoint,
    book_feature_endpoint_streaming,
)
from scripts.supabase_checkpointer import SupabaseCheckpointer

# Modal volume handle is only usable when running inside a Modal context.
# We attempt to import modal so the image can be the same artefact for
# `docker run` (no Modal) and `modal deploy` (with Modal); commit() becomes
# a silent no-op in the non-Modal case.
try:
    import modal as _modal
    _hf_cache_volume = _modal.Volume.from_name("hf_embedding_cache")
except Exception:
    _modal = None
    _hf_cache_volume = None


def _commit_hf_cache_volume() -> None:
    if _hf_cache_volume is None:
        return
    try:
        _hf_cache_volume.commit()
    except Exception as e:
        print(f"[hf_cache] volume commit skipped: {e}")


# ---------------- CONSTANTS ----------------
HF_CACHE_PATH = os.getenv("HF_CACHE_PATH", "/hf_cache")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_ENDPOINT"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# ---------------- EMBEDDING ----------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    model_kwargs={"device": device},
    cache_folder=HF_CACHE_PATH,
)

# ---------------- SERVICES ----------------
cache_manager = CacheManager(qdrant_client=qdrant_client, embedding_model=embedding_model)
supabase_storage = SupabaseStorage()
token_tracker = TokenTracker()
credit_manager = CreditManager()

# ---------------- API KEY ----------------
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.environ.get("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key


# ---------------- LIMITER ----------------
limiter = Limiter(key_func=get_remote_address)


# ---------------- LIFESPAN ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpointer = None
    try:
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

    os.environ.setdefault("LANGSMITH_TRACING", os.getenv("LANGSMITH_TRACING", "true"))
    os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "rag-book-production"))

    # Start the in-process async-ingest worker (ADR-0003). One task per process.
    import asyncio as _asyncio
    db_url = os.getenv("SUPABASE_DB_URL")
    interval = float(os.getenv("INGEST_WORKER_INTERVAL_SECONDS", "5"))
    stale_timeout = int(os.getenv("INGEST_WORKER_STALE_TIMEOUT_SECONDS", "1800"))

    async def _pipeline(book_id: str, pdf_path: str, storage_url: str) -> int:
        qdb = await cache_manager.get_qdrant_db()
        ingest_book(
            pdf_path=pdf_path,
            qdrant_db=qdb,
            book_id=book_id,
            embed_model_id=EMBED_MODEL_ID,
            extra_metadata={"storage_url": storage_url},
        )
        return qdb.count_by_book(book_id)

    worker_task = None
    if db_url:
        try:
            worker_task = _asyncio.create_task(
                run_worker_loop(
                    pool_or_dsn=db_url,
                    ingest_book_fn=_pipeline,
                    interval_seconds=interval,
                    stale_timeout_seconds=stale_timeout,
                    cache_manager=cache_manager,
                )
            )
            print(f"Async-ingest worker started (interval={interval}s, stale_timeout={stale_timeout}s)")
        except Exception as e:
            print(f"Warning: could not start async-ingest worker: {e}")
    else:
        print("Warning: SUPABASE_DB_URL not set; async-ingest worker not started")

    try:
        yield
    finally:
        if worker_task is not None:
            worker_task.cancel()
            try:
                await worker_task
            except (_asyncio.CancelledError, Exception):
                pass


# ---------------- APP ----------------
app = FastAPI(title="book-qa", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    lambda request, exc: PlainTextResponse("Rate limit exceeded", status_code=429),
)


# ---------------- HEALTH ----------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------- STREAMING ----------------
@app.post("/book-qa/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def book_qa_stream(request: Request, payload: ChatPayload):
    async def _stream(uid: str, _payload):
        agent = await cache_manager.get_agent(payload.book_id, payload.user_id)
        async for chunk in agent.ask_stream(
            payload.messages[-1].content,
            session_id=payload.session_id,
        ):
            yield chunk

    return await book_feature_endpoint_streaming(
        feature_name="agent_reasoning",
        user_id=payload.user_id,
        credit_manager=credit_manager,
        stream_call=_stream,
    )


# ---------------- INGEST (ASYNC, 202) ----------------
# Rate limit is the per-IP *enqueue* limit. The actual Docling pipeline runs
# in the in-process worker (see lifespan + core.utils.ingest_worker).
@app.post("/book-qa/ingest", dependencies=[Depends(verify_api_key)])
@limiter.limit("3/minute")
async def ingest_pdf(
    request: Request,
    book_id: str,
    file: UploadFile = File(...),
):
    try:
        file_bytes = await file.read()
        filename = file.filename or f"{book_id}.pdf"

        storage_url = supabase_storage.upload_pdf_bytes(
            book_id=book_id,
            filename=filename,
            file_bytes=file_bytes,
        )

        job = ingest_jobs.create_or_reset_job(
            book_id=book_id,
            filename=filename,
            storage_url=storage_url,
        )

        return JSONResponse(
            {
                "book_id": job.book_id,
                "status": job.status,
                "filename": job.filename,
                "storage_url": job.storage_url,
            },
            status_code=202,
        )
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ---------------- INGEST STATUS ENDPOINTS ----------------
@app.get("/book-qa/ingest/{book_id}", dependencies=[Depends(verify_api_key)])
async def get_ingest_status(book_id: str):
    job = ingest_jobs.get_job(book_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no ingest job for book_id '{book_id}'")
    return JSONResponse(job.to_dict())


@app.get("/book-qa/ingest", dependencies=[Depends(verify_api_key)])
async def list_ingest_jobs(
    book_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
):
    jobs = ingest_jobs.list_jobs(book_id=book_id, status=status, limit=limit)
    return JSONResponse({"jobs": [j.to_dict() for j in jobs], "count": len(jobs)})


@app.post("/book-qa/ingest/{book_id}/retry", dependencies=[Depends(verify_api_key)])
async def retry_ingest(book_id: str):
    if ingest_jobs.reset_for_retry(book_id):
        return JSONResponse({"book_id": book_id, "status": "pending"})
    current = ingest_jobs.get_job(book_id)
    raise HTTPException(
        status_code=409,
        detail=f"book_id '{book_id}' is not in 'failed' state (current: {current.status if current else 'missing'})",
    )


# ---------------- DELETE BOOK CHUNKS ----------------
@app.delete("/book-qa/book/{book_id}/chunks", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def delete_book_chunks(
    request: Request,
    book_id: str,
    filename: str | None = None,
):
    """
    Hard-delete all chunks and the Supabase Storage file for a book.
    Also marks any in-flight `ingested_books` row for the BookId as
    `failed` with `error='re-ingest: chunks deleted'` so the in-process
    worker sees the row has moved out of `running` on its next tick.
    """
    async with cache_manager.acquire_book_lock(book_id):
        try:
            ingest_jobs.mark_deleted_by_reingest(book_id)
        except Exception as e:
            print(f"Warning: could not mark ingest row as deleted: {e}")

        qdrant_db = await cache_manager.get_qdrant_db()
        qdrant_db.delete_by_book(book_id)

        if filename:
            supabase_storage.delete_pdf(book_id, filename)
        else:
            existing = supabase_storage.list_pdfs(book_id) or []
            for f in existing:
                name = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
                if name:
                    supabase_storage.delete_pdf(book_id, name)

        await cache_manager.clear_book(book_id)

    return JSONResponse({
        "status": "success",
        "book_id": book_id,
        "message": f"All chunks and storage for '{book_id}' removed",
    })


# ---------------- MINDMAP ----------------
@app.post("/book-qa/mindmap", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def generate_mindmap(
    request: Request,
    user_id: str,
    book_id: str,
    user_prompt: str | None = None,
):
    async def _service(uid: str, _payload):
        qdrant_db = await cache_manager.get_qdrant_db()
        llm = get_chat_model()
        generator = MindmapGenerator(qdrant_db, llm)
        return await generator.generate(book_id, user_prompt=user_prompt)

    return await book_feature_endpoint(
        feature_name="mindmap",
        user_id=user_id,
        credit_manager=credit_manager,
        service_call=_service,
    )


# ---------------- MINDMAP EDIT ----------------
@app.post("/book-qa/mindmap/edit", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def edit_mindmap(request: Request, body: MindmapEditRequest):
    async def _service(uid: str, _payload):
        qdrant_db = await cache_manager.get_qdrant_db()
        llm = get_chat_model()
        generator = MindmapGenerator(qdrant_db, llm)
        return await generator.edit(
            book_id=body.book_id,
            mermaid=body.mermaid,
            instruction=body.instruction,
        )

    return await book_feature_endpoint(
        feature_name="mindmap",
        user_id=body.user_id,
        credit_manager=credit_manager,
        service_call=_service,
    )


# ---------------- DATASET ----------------
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
    async def _service(uid: str, _payload):
        qdrant_db = await cache_manager.get_qdrant_db()
        llm = get_chat_model()
        generator = DatasetGenerator(qdrant_db, llm)
        return await generator.generate(book_id, difficulty, num_mcq, num_essay)

    return await book_feature_endpoint(
        feature_name="dataset",
        user_id=user_id,
        credit_manager=credit_manager,
        service_call=_service,
    )


# ---------------- SUMMARIZE ----------------
@app.post("/book-qa/summarize", dependencies=[Depends(verify_api_key)])
@limiter.limit("2/minute")
async def summarize_book(
    request: Request,
    user_id: str,
    book_id: str,
    user_prompt: str | None = None,
):
    async def _service(uid: str, _payload):
        qdrant_db = await cache_manager.get_qdrant_db()
        llm = get_chat_model()
        summarizer = BookSummarizer(qdrant_db, llm)
        return await summarizer.summarize(book_id, user_prompt=user_prompt)

    return await book_feature_endpoint(
        feature_name="summarize",
        user_id=user_id,
        credit_manager=credit_manager,
        service_call=_service,
    )


# ---------------- SUMMARIZE EDIT ----------------
@app.post("/book-qa/summarize/edit", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def edit_summary(request: Request, body: SummaryEditRequest):
    async def _service(uid: str, _payload):
        qdrant_db = await cache_manager.get_qdrant_db()
        llm = get_chat_model()
        summarizer = BookSummarizer(qdrant_db, llm)
        return await summarizer.edit(
            book_id=body.book_id,
            title=body.title,
            overview=body.overview,
            key_themes=body.key_themes,
            chapters=[c.model_dump() for c in body.chapters],
            instruction=body.instruction,
        )

    return await book_feature_endpoint(
        feature_name="summarize",
        user_id=body.user_id,
        credit_manager=credit_manager,
        service_call=_service,
    )


# ---------------- TOKEN USAGE ----------------
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
    try:
        in_memory_summary = token_tracker.get_summary()
        db_usage = await token_tracker.query_usage(
            book_id=book_id,
            session_id=session_id,
            feature=feature,
            from_date=from_date,
            to_date=to_date,
        )
        return JSONResponse({"in_memory": in_memory_summary, "database": db_usage})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/token-usage/daily", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def get_daily_token_usage(
    request: Request,
    book_id: str,
    days: int = 7,
):
    try:
        daily_usage = await token_tracker.query_daily_usage(book_id=book_id, days=days)
        return JSONResponse({"daily_usage": daily_usage})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/token-usage/flush", dependencies=[Depends(verify_api_key)])
async def flush_token_usage():
    try:
        await token_tracker.flush()
        return JSONResponse({"status": "flushed"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ---------------- CREDIT ENDPOINTS ----------------
@app.get("/credits/{user_id}", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def get_credits(request: Request, user_id: str):
    try:
        balance = credit_manager.get_balance(user_id)
        recent = await credit_manager.get_recent_transactions(user_id, limit=10)
        return JSONResponse({"balance": balance, "recent_transactions": recent})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/credits/{user_id}/add", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def add_credits(
    request: Request,
    user_id: str,
    amount: float,
    transaction_type: str = "purchase",
):
    try:
        success = credit_manager.add_credits(user_id, amount, transaction_type)
        if success:
            return JSONResponse({
                "status": "success",
                "new_balance": credit_manager.get_balance(user_id),
            })
        return JSONResponse({"status": "error", "message": "Failed to add credits"}, status_code=500)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
