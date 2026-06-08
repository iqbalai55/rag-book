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
from core.services.dataset_generator import DatasetGenerator
from core.services.summarize_book import BookSummarizer
from core.services.mindmap_generator import MindmapGenerator
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

    yield


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
    is_sufficient, cost = credit_manager.check_sufficient_credits(payload.user_id, "agent_reasoning")
    if not is_sufficient:
        raise HTTPException(status_code=402, detail=f"Insufficient credits (cost: {cost})")

    if not credit_manager.burn_credits(payload.user_id, "agent_reasoning"):
        raise HTTPException(status_code=500, detail="Failed to deduct credits")

    async def event_generator():
        try:
            agent = await cache_manager.get_agent(payload.book_id, payload.user_id)
            async for chunk in agent.ask_stream(
                payload.messages[-1].content,
                session_id=payload.session_id,
            ):
                yield chunk
        except Exception as e:
            credit_manager.refund_credits(payload.user_id, "agent_reasoning")
            yield f"data: {json.dumps({'id': 'chatcmpl', 'type': 'error', 'content': str(e), 'metadata': {}})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------- INGEST ----------------
@app.post("/book-qa/ingest", dependencies=[Depends(verify_api_key)])
@limiter.limit("3/minute")
async def ingest_pdf(
    request: Request,
    book_id: str,
    file: UploadFile = File(...),
):
    tmp_path = f"./{file.filename}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        storage_url = supabase_storage.upload_pdf(
            file_path=tmp_path,
            book_id=book_id,
            filename=file.filename,
        )

        qdrant_db = await cache_manager.get_qdrant_db()

        async with cache_manager.acquire_book_lock(book_id):
            ingest_book(
                pdf_path=tmp_path,
                qdrant_db=qdrant_db,
                book_id=book_id,
                embed_model_id=EMBED_MODEL_ID,
                extra_metadata={"storage_url": storage_url},
            )

        _commit_hf_cache_volume()

        return JSONResponse({
            "status": "success",
            "collection": "lms_content",
            "book_id": book_id,
            "storage_url": storage_url,
            "message": f"{file.filename} ingested and stored",
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------- DELETE BOOK CHUNKS ----------------
@app.delete("/book-qa/book/{book_id}/chunks", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def delete_book_chunks(
    request: Request,
    book_id: str,
    filename: str | None = None,
):
    async with cache_manager.acquire_book_lock(book_id):
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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ---------------- MINDMAP EDIT ----------------
@app.post("/book-qa/mindmap/edit", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def edit_mindmap(request: Request, body: MindmapEditRequest):
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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ---------------- SUMMARIZE ----------------
@app.post("/book-qa/summarize", dependencies=[Depends(verify_api_key)])
@limiter.limit("2/minute")
async def summarize_book(
    request: Request,
    user_id: str,
    book_id: str,
    user_prompt: str | None = None,
):
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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ---------------- SUMMARIZE EDIT ----------------
@app.post("/book-qa/summarize/edit", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def edit_summary(request: Request, body: SummaryEditRequest):
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
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
