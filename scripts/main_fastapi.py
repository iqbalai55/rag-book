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
from core.storage.supabase_storage import SupabaseStorage
from core.utils.llm_config import get_chat_model
from core.schemas.mindmap import MindmapResponse
from core.prompts.mindmap import MINDMAP_FROM_CONTENT_PROMPT
from core.schemas.chapter import ChapterIdentification
from core.schemas.question import MCQResponse, EssayResponse
from core.prompts.general_rag import (
    MCQ_PROMPT,
    ESSAY_QUESTION_PROMPT,
    CHAPTER_IDENTIFICATION_PROMPT,
)

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
async def generate_mindmap(request: Request, course_id: str):
    try:
        qdrant_db = await cache_manager.get_qdrant_db()
        docs = qdrant_db.get_all_by_course(course_id)

        if not docs:
            raise HTTPException(
                status_code=404, detail="No content found for this course"
            )

        context = "\n\n".join([d.page_content for d in docs])[:15000]

        llm = get_chat_model()
        structured_llm = llm.with_structured_output(MindmapResponse)
        prompt = MINDMAP_FROM_CONTENT_PROMPT.format(
            context=context, topik=course_id
        )
        result = structured_llm.invoke(prompt)

        return JSONResponse({
            "course_id": course_id,
            "title": result.title,
            "mermaid": result.mermaid,
            "sources": result.sources,
        })

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
        docs = qdrant_db.get_all_by_course(course_id)

        if not docs:
            raise HTTPException(
                status_code=404, detail="No content found for this course"
            )

        context = "\n\n".join([d.page_content for d in docs])[:15000]

        llm = get_chat_model()

        # Step 1: Identify chapters from content
        chapter_llm = llm.with_structured_output(ChapterIdentification)
        chapter_prompt = CHAPTER_IDENTIFICATION_PROMPT.format(
            context=context, topik=course_id
        )
        chapter_result = chapter_llm.invoke(chapter_prompt)
        chapters = chapter_result.chapters

        if not chapters:
            raise HTTPException(
                status_code=404, detail="Could not identify chapters from content"
            )

        # Step 2: Generate questions for each chapter
        dataset = {
            "course_id": course_id,
            "difficulty": difficulty,
            "total_chapters": len(chapters),
            "chapters": [],
        }

        for chapter_title in chapters:
            # Retrieve relevant context for this chapter
            chapter_docs = qdrant_db.query(chapter_title, course_id=course_id, k=3)
            chapter_context = "\n\n".join([d.page_content for d in chapter_docs])[:8000]

            if not chapter_context:
                continue

            # Generate MCQ
            mcq_llm = llm.with_structured_output(MCQResponse)
            mcq_prompt = MCQ_PROMPT.format(
                topic=chapter_title,
                difficulty=difficulty,
                num_questions=num_mcq,
                context=chapter_context,
            )
            mcq_result = mcq_llm.invoke(mcq_prompt)

            # Generate Essay
            essay_llm = llm.with_structured_output(EssayResponse)
            essay_prompt = ESSAY_QUESTION_PROMPT.format(
                topic=chapter_title,
                difficulty=difficulty,
                num_questions=num_essay,
                context=chapter_context,
            )
            essay_result = essay_llm.invoke(essay_prompt)

            # Collect sources
            sources = []
            for doc in chapter_docs:
                source = doc.metadata.get("source", "unknown")
                pages = doc.metadata.get("pages", [])
                pages_str = ", ".join(map(str, pages))
                sources.append(f"{source} (hal {pages_str})")

            dataset["chapters"].append({
                "chapter_title": chapter_title,
                "sources": list(set(sources)),
                "mcq": mcq_result.model_dump(),
                "essay": essay_result.model_dump(),
            })

        return JSONResponse(dataset)

    except HTTPException:
        raise
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