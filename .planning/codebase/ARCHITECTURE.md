# Architecture

**Analysis Date:** 2026-05-30

## Pattern Overview

**Overall:** Multi-tenant RAG (Retrieval Augmented Generation) system with LangGraph agents

**Key Characteristics:**
- Multi-tenant architecture with course-level isolation via `book_id` payload filtering in Qdrant
- LangGraph-based agents with tool execution for RAG, MCQ generation, essay generation, and podcast synthesis
- Hierarchical map-reduce pattern for book-level operations (summarize, mindmap, dataset generation)
- Streaming Server-Sent Events (SSE) for real-time agent responses
- Supabase-backed durable checkpointing for conversation state persistence

## Layers

**API Layer (FastAPI):**
- Location: `scripts/main_fastapi.py`
- Contains: REST endpoints for book ingestion, chat streaming, summarization, mindmap generation, dataset creation
- Depends on: `CacheManager`, `SupabaseStorage`, `TokenTracker`, service classes
- Used by: Frontend clients via HTTP/SSE

**Agent Layer (LangGraph):**
- Location: `agents/book_qdrant_agent.py`, `agents/book_podcast_agent.py`
- Contains: `BookQdrantAgent` (RAG Q&A, MCQ, essay tools), `BookPodcastAgent` (extends with TTS)
- Depends on: `QdrantDB`, `get_chat_model()`, prompt templates
- Used by: FastAPI via `CacheManager.get_agent()`

**Retrieval Layer (Qdrant):**
- Location: `core/rag/qdrant_db.py`
- Contains: `QdrantDB` - vector store wrapper with multitenant filtering
- Depends on: `qdrant_client`, embedding model
- Used by: Agents, services (summarize, mindmap, dataset)

**Storage Layer (Supabase):**
- Location: `core/storage/supabase_storage.py`, `scripts/supabase_checkpointer.py`
- Contains: PDF file storage, LangGraph checkpoint persistence
- Depends on: Supabase client
- Used by: FastAPI (PDF upload), `CacheManager` (checkpointing)

**Service Layer:**
- Location: `core/services/` (summarize_book.py, mindmap_generator.py, dataset_generator.py)
- Contains: Business logic for summaries, mindmaps, MCQ/essay datasets
- Depends on: `QdrantDB`, LLM, prompt templates
- Used by: FastAPI endpoints

**Document Processing Layer:**
- Location: `core/rag/document_processor.py`, `core/utils/ingest_book.py`
- Contains: PDF loading, chunking, metadata extraction
- Depends on: `docling`, `pypdf`, `HybridChunker`
- Used by: FastAPI `/ingest` endpoint

## Data Flow

**Book Ingestion Flow:**
1. Client POSTs PDF to `/book-qa/ingest`
2. `SupabaseStorage.upload_pdf()` stores PDF in Supabase Storage
3. `ingest_book()` calls `DocumentProcessor.process_document()` to chunk PDF
4. Chunks enriched with `book_id` are stored in Qdrant `lms_content` collection

**Chat/QA Flow:**
1. Client POSTs to `/book-qa/stream` with `book_id` and messages
2. `CacheManager.get_agent(book_id)` returns or creates `BookQdrantAgent`
3. Agent executes with `search_book_context` tool → `QdrantDB.query(book_id=...)`
4. Response streamed via SSE with message types: `human`, `tool`, `internal`, `final`, `mcq`, `essay`, `error`
5. `TokenUsageCallbackHandler` tracks token usage per feature/session

**Summary/Mindmap/Dataset Flow:**
1. Client POSTs to respective endpoint with `book_id`
2. Service retrieves all documents via `QdrantDB.get_all_by_book()`
3. For summaries/datasets: chapter identification via LLM → per-chapter summarization → map-reduce to final
4. For mindmaps: direct LLM generation from full context

## Key Abstractions

**CacheManager:**
- Purpose: Singleton managing shared `QdrantDB` and per-course agent instances
- Location: `core/utils/cache_manager.py`
- Pattern: Async lazy initialization with lock

**QdrantDB:**
- Purpose: Multitenant vector store with `book_id` payload filtering
- Location: `core/rag/qdrant_db.py`
- Pattern: Wrapper around `QdrantVectorStore` with collection management

**SupabaseCheckpointer:**
- Purpose: Durable LangGraph checkpoint storage via Supabase REST API
- Location: `scripts/supabase_checkpointer.py`
- Pattern: Implements `BaseCheckpointSaver` interface

**TokenTracker:**
- Purpose: Track and persist token usage per course/session/feature
- Location: `core/utils/token_tracker.py`
- Pattern: In-memory buffering with async flush to database

## Entry Points

**FastAPI Server:**
- Location: `scripts/main_fastapi.py`
- Triggers: `python scripts/main_fastapi.py` or `asyncio.run(main())`
- Responsibilities: API routing, rate limiting, CORS, lifespan management (checkpointer init)

**Podcast Agent Runner:**
- Location: `scripts/run_podcast_agent.py`
- Triggers: Direct execution
- Responsibilities: Standalone podcast generation

**Dataset Builder Runner:**
- Location: `scripts/run_dataset_builder.py`
- Triggers: Direct execution
- Responsibilities: Standalone dataset generation

## Error Handling

**Strategy:** HTTPException for expected errors, JSON error responses for unexpected

**Patterns:**
- 404: Content not found (course empty, no chapters identified)
- 422: Validation errors (handled by FastAPI)
- 429: Rate limit exceeded (via slowapi)
- 500: Unexpected errors with message in response

## Cross-Cutting Concerns

**Logging:** Python `logging` module with `__name__` loggers
**Validation:** Pydantic schemas in `core/schemas/`
**Authentication:** API key via `x-api-key` header, validated in `verify_api_key` dependency
**Rate Limiting:** slowapi limiter with per-endpoint limits

---

*Architecture analysis: 2026-05-30*