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
        async def event_generator():
            agent = await cache_manager.get_agent(payload.course_id)
            async for chunk in agent.ask_stream(
                payload.messages[-1].content,
                session_id=payload.session_id
            ):
                yield chunk
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ---------------- INGEST ----------------
    @web_app.post("/book-qa/ingest", dependencies=[Depends(verify_api_key)])
    @limiter.limit("3/minute")
    async def ingest_pdf(
        request: Request,
        course_id: str,  # 🔥 NEW: tenant identifier
        file: UploadFile = File(...)
    ):
        tmp_path = f"./{file.filename}"

        try:
            with open(tmp_path, "wb") as f:
                f.write(await file.read())

            # Upload PDF to Supabase Storage
            storage_url = supabase_storage.upload_pdf(
                file_path=tmp_path,
                course_id=course_id,
                filename=file.filename,
            )

            qdrant_db = await cache_manager.get_qdrant_db()

            ingest_book(
                pdf_path=tmp_path,
                qdrant_db=qdrant_db,
                course_id=course_id,
                embed_model_id=EMBED_MODEL_ID,
                extra_metadata={"storage_url": storage_url},
            )

            embedding_cache_volume.commit()

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

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ---------------- MINDMAP ----------------
    @web_app.post("/book-qa/mindmap", dependencies=[Depends(verify_api_key)])
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

    # ---------------- MINDMAP EDIT ----------------
    @web_app.post("/book-qa/mindmap/edit", dependencies=[Depends(verify_api_key)])
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

    # ---------------- DATASET ----------------
    @web_app.post("/book-qa/dataset", dependencies=[Depends(verify_api_key)])
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

    # ---------------- SUMMARIZE BOOK ----------------
    @web_app.post("/book-qa/summarize", dependencies=[Depends(verify_api_key)])
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

    # ---------------- SUMMARIZE EDIT ----------------
    @web_app.post("/book-qa/summarize/edit", dependencies=[Depends(verify_api_key)])
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

    # ---------------- TOKEN USAGE ----------------
    @web_app.get("/token-usage", dependencies=[Depends(verify_api_key)])
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
            in_memory_summary = token_tracker.get_summary()
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

    @web_app.get("/token-usage/daily", dependencies=[Depends(verify_api_key)])
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

    return web_app