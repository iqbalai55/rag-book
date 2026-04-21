# Architecture

**Analysis Date:** 2026-04-21

## Pattern Overview

**Overall:** Modular, Layered Architecture with Microservices-inspired Separation of Concerns

**Key Characteristics:**
- Clear separation between API layer, service layer, and data access layer
- Domain-driven design with bounded contexts (agents, services, schemas)
- Event-driven data flow through LangGraph checkpointer pattern
- Plugin-like agent architecture for extensible functionality
- Multi-tenancy implemented at the data layer (QdrantDB with course_id filtering)

## Layers

**API Layer:**
- Purpose: Handle HTTP requests/responses, authentication, rate limiting
- Location: `main_fastapi.py`
- Contains: FastAPI endpoints, middleware, dependency injection
- Depends on: CacheManager, services layer
- Used by: External clients (web, mobile, other services)

**Service Layer:**
- Purpose: Business logic orchestration, external service integration
- Location: `services/`, `agents/`, `utils/`
- Contains: CacheManager, QdrantDB, BookQdrantAgent, LLM configuration
- Depends on: Data Access Layer, external APIs (Qdrant, Supabase, HuggingFace)
- Used by: API Layer

**Data Access Layer:**
- Purpose: Data persistence and retrieval abstraction
- Location: `services/rag/qdrant/`
- Contains: QdrantDB class with multitenant filtering
- Depends on: Qdrant client, embedding models
- Used by: Service Layer (agents, cache manager)

**Domain Layer:**
- Purpose: Data models, schemas, prompts
- Location: `schemas/`, `prompts/`
- Contains: Pydantic models, prompt templates, data transfer objects
- Depends on: None (pure domain objects)
- Used by: All layers

## Data Flow

**Book QA Streaming Flow:**

1. **Request Ingestion:** Client POST to `/book-qa/stream` with ChatPayload
2. **Authentication:** API key verified via dependency injection
3. **Agent Retrieval:** CacheManager returns course-specific BookQdrantAgent
4. **Context Retrieval:** Agent uses QdrantDB to query relevant documents (filtered by course_id)
5. **LLM Processing:** LangGraph agent processes query with retrieved context
6. **Tool Execution:** Agent may invoke search/book tools based on query
7. **Streaming Response:** Results streamed back as Server-Sent Events (SSE)
8. **Checkpointing:** Conversation state saved via AsyncPostgresSaver

**Book Ingestion Flow:**

1. **Request Ingestion:** Client POST to `/book-qa/ingest` with PDF and course_id
2. **Authentication:** API key verified
3. **File Processing:** PDF saved temporarily, text extracted
4. **Vector Storage:** Chunks added to Qdrant collection with course_id metadata
5. **Cache Update:** QdrantDB reference updated in CacheManager
6. **Response:** Success/error returned to client

**State Management:**
- **Conversation State:** Managed by LangGraph checkpointer (PostgreSQL via Supabase)
- **Application State:** CacheManager singleton holds shared resources (Qdrant client, DB, agents)
- **Session State:** Passed via session_id in requests, stored in checkpointer
- **Tenancy State:** course_id embedded in metadata for data isolation

## Key Abstractions

**QdrantDB:**
- Purpose: Multitenant vector database wrapper with automatic collection management
- Examples: `services/rag/qdrant/qdrant_db.py`
- Pattern: Wrapper pattern with dependency injection, automatic schema management

**CacheManager:**
- Purpose: Singleton managing shared resources and per-course agent instances
- Examples: `utils/chace_manager.py`
- Pattern: Factory pattern with lazy initialization, thread-safe singleton

**BookQdrantAgent:**
- Purpose: Encapsulates RAG logic with tool-based question generation capabilities
- Examples: `agents/book_qdrant_agent.py`
- Pattern: Strategy pattern (tools), Middleware pattern (LangGraph middleware)

**Message Payload:**
- Purpose: Standardized communication contract between client and API
- Examples: `schemas/chat.py`
- Pattern: Data Transfer Object (DTO) with validation (Pydantic)

## Entry Points

**main.py:**
- Location: `main.py`
- Triggers: Direct execution (`python main.py`)
- Responsibilities: Database schema setup (checkpointer initialization)

**main_fastapi.py:**
- Location: `main_fastapi.py`
- Triggers: Uvicorn server startup
- Responsibilities: 
  - FastAPI application creation
  - Lifespan management (resource initialization/cleanup)
  - API route registration (/book-qa/stream, /book-qa/ingest)
  - Middleware setup (rate limiting, authentication)
  - External service connections (MLflow, embedding models)

**uvicorn Entry Point:**
- Location: Implicit in main_fastapi.py line 143-148
- Triggers: `uvicorn main_fastapi:app` command
- Responsibilities: HTTP server binding and request handling

## Error Handling

**Strategy:** Centralized exception handling with hierarchical fallback

**Patterns:**
- **API Layer:** FastAPI exception handlers for rate limits (429) and general exceptions (500)
- **Service Layer:** Explicit try/catch blocks with logging and error propagation
- **Data Layer:** Qdrant operations wrapped with try/catch, fallback to in-memory client
- **Agent Layer:** Tool execution wrapped with error catchers returning user-friendly messages

**Cross-Cutting Concerns:**

**Logging:** Standard Python logging module with module-specific loggers
- Pattern: `logger = logging.getLogger(__name__)` in each module
- Levels: DEBUG for development, INFO for production

**Validation:** Pydantic models for request/response validation
- Examples: ChatPayload, Message, MCQResponse, EssayResponse
- Location: `schemas/` directory

**Authentication:** API key header verification
- Implementation: FastAPI Security dependency (`verify_api_key` function)
- Location: main_fastapi.py lines 77-81
- Header: `x-api-key`

---

*Architecture analysis: 2026-04-21*