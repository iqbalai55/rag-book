import os
import json
from contextlib import asynccontextmanager
import modal
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Security, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader
import uuid

from core.schemas.chat import ChatPayload
from core.utils.ingest_book import ingest_book
from core.utils.cache_manager import CacheManager
from core.storage.supabase_storage import SupabaseStorage
from core.utils.token_tracker import TokenTracker
from core.utils.llm_config import get_chat_model
from core.utils.credit_manager import CreditManager, FEATURE_COST
from core.services.dataset_generator import DatasetGenerator
from core.services.summarize_book import BookSummarizer
from core.services.mindmap_generator import MindmapGenerator
from core.schemas.mindmap import MindmapEditRequest
from core.schemas.summary import SummaryEditRequest
from supabase import create_client
from scripts.supabase_checkpointer import SupabaseCheckpointer
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from qdrant_client import QdrantClient

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import PlainTextResponse

limiter = Limiter(key_func=get_remote_address)

# ---------------- MODAL SECRET ----------------
secret = modal.Secret.from_dict({
    "LLM_PROVIDER": os.getenv("LLM_PROVIDER"),
    "LLM_MODEL": os.getenv("LLM_MODEL"),
    "HUGGINGFACEHUB_API_TOKEN": os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
    "MINIMAX_API_KEY": os.getenv("MINIMAX_API_KEY"),
    "SUPABASE_DB_URL": os.getenv("SUPABASE_DB_URL"),
    "API_KEY": os.getenv("API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "OPENAI_MODEL": os.getenv("OPENAI_MODEL"),
    "QDRANT_ENDPOINT": os.getenv("QDRANT_ENDPOINT"),
    "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
    "SUPABASE_URL": os.getenv("SUPABASE_URL"),
    "SUPABASE_SERVICE_KEY": os.getenv("SUPABASE_SERVICE_KEY"),
    "SUPABASE_STORAGE_BUCKET": os.getenv("SUPABASE_STORAGE_BUCKET"),
    "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY"),
    "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING"),
    "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT")
})

# ---------------- CONSTANTS ----------------
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_ENDPOINT"),
    api_key=os.getenv("QDRANT_API_KEY")
)

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
cache_manager = CacheManager(
    qdrant_client=qdrant_client,
    embedding_model=embedding_model
)

# ---------------- SUPABASE STORAGE ----------------
supabase_storage = SupabaseStorage()

# ---------------- TOKEN TRACKER ----------------
token_tracker = TokenTracker()

# ---------------- CREDIT MANAGER ----------------
credit_manager = CreditManager()

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
    .pip_install_from_requirements(r"requirements\requirements_main.txt")
    .add_local_python_source("core/schemas")
    .add_local_python_source("core/utils")
    .add_local_python_source("agents")
    .add_local_python_source("core/rag")
    .add_local_python_source("core/storage")
    .add_local_python_source("core/tts")
    .add_local_python_source("core/prompts")
    .add_local_python_source("core/utils")
    .add_local_python_source("scripts/supabase_checkpointer")
)

qdrant_volume = modal.Volume.from_name("qdrant_storage_volume")
embedding_cache_volume = modal.Volume.from_name("hf_embedding_cache")


# ---------------- FASTAPI LIFESPAN ----------------
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

    # Setup LangSmith observability
    os.environ.setdefault("LANGSMITH_TRACING", os.getenv("LANGSMITH_TRACING", "true"))
    os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "rag-book-production"))

    yield


# ---------------- FASTAPI LIFESPAN ----------------
@app.function(
    timeout=2*3600,
    gpu="T4",
    volumes={
        HF_CACHE_PATH: embedding_cache_volume,
    },
    image=image
)
@modal.asgi_app(label="book-qa-fastapi")
def fastapi_app():

    web_app = FastAPI(lifespan=lifespan)
    web_app.state.limiter = limiter
    web_app.add_exception_handler(
        RateLimitExceeded,
        lambda request, exc: PlainTextResponse("Rate limit exceeded", status_code=429)
    )

    # ---------------- STREAMING ----------------
    @web_app.post("/book-qa/stream", dependencies=[Depends(verify_api_key)])
    @limiter.limit("10/minute")
    async def book_qa_stream(request: Request, payload: ChatPayload):
        # Check credits
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
                    session_id=payload.session_id
                ):
                    yield chunk
            except Exception as e:
                credit_manager.refund_credits(payload.user_id, "agent_reasoning")
                yield f"data: {json.dumps({'id': 'chatcmpl', 'type': 'error', 'content': str(e), 'metadata': {}})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ---------------- INGEST ----------------
    @web_app.post("/book-qa/ingest", dependencies=[Depends(verify_api_key)])
    @limiter.limit("3/minute")
    async def ingest_pdf(
        request: Request,
        book_id: str,
        file: UploadFile = File(...)
    ):
        tmp_path = f"./{file.filename}"

        try:
            with open(tmp_path, "wb") as f:
                f.write(await file.read())

            # Upload PDF to Supabase Storage
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

            embedding_cache_volume.commit()

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

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ---------------- DELETE BOOK CHUNKS ----------------
    @web_app.delete("/book-qa/book/{book_id}/chunks", dependencies=[Depends(verify_api_key)])
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
            "message": f"All chunks and storage for '{book_id}' removed"
        })

    # ---------------- MINDMAP ----------------
    @web_app.post("/book-qa/mindmap", dependencies=[Depends(verify_api_key)])
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

    # ---------------- MINDMAP EDIT ----------------
    @web_app.post("/book-qa/mindmap/edit", dependencies=[Depends(verify_api_key)])
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

    # ---------------- DATASET ----------------
    @web_app.post("/book-qa/dataset", dependencies=[Depends(verify_api_key)])
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

    # ---------------- SUMMARIZE BOOK ----------------
    @web_app.post("/book-qa/summarize", dependencies=[Depends(verify_api_key)])
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

    # ---------------- SUMMARIZE EDIT ----------------
    @web_app.post("/book-qa/summarize/edit", dependencies=[Depends(verify_api_key)])
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

    # ---------------- TOKEN USAGE ----------------
    @web_app.get("/token-usage", dependencies=[Depends(verify_api_key)])
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
            in_memory_summary = token_tracker.get_summary()
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

    @web_app.get("/token-usage/daily", dependencies=[Depends(verify_api_key)])
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

    @web_app.post("/token-usage/flush", dependencies=[Depends(verify_api_key)])
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

    # ---------------- CREDIT ENDPOINTS ----------------
    @web_app.get("/credits/{user_id}", dependencies=[Depends(verify_api_key)])
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

    @web_app.post("/credits/{user_id}/add", dependencies=[Depends(verify_api_key)])
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

    return web_app
