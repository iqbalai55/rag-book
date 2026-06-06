# Codebase Concerns

**Analysis Date:** 2026-05-30

## Critical Security Issues

### Exposed Secrets in .env File

**Severity: CRITICAL**

The `.env` file is tracked in git history (not in `.gitignore`) and contains multiple live API keys and credentials:

- `OPENAI_API_KEY` - Full production key at line 9
- `OPENROUTER_API_KEY` - Full production key at line 11
- `MINIMAX_API_KEY` - Full production key at line 12
- `HUGGINGFACEHUB_API_TOKEN` - Full token at line 14
- `SUPABASE_SERVICE_KEY` - Full JWT at line 21
- `QDRANT_API_KEY` - Full JWT at line 26
- `LANGSMITH_API_KEY` - Full key at line 30

**Files affected:**
- `.env` (lines 9-32)

**Impact:** If this repo is public or access is compromised, all external service credentials are exposed. Attackers could:
- Use all LLM providers at victim's expense
- Access Supabase database (read/write)
- Access Qdrant vector database
- View all LangSmith traces

**Fix:** Remove `.env` from git tracking, use `.gitignore`, and never commit live credentials.

---

### Hardcoded API Key Validation

**Severity: HIGH**

In `scripts/main_fastapi.py` (line 121-124) and `scripts/main_modal.py` (line 86-89):

```python
async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
```

The `API_KEY` is compared using direct string equality (`!=`). This is vulnerable to timing attacks. Should use `secrets.compare_digest()` or equivalent constant-time comparison.

**Files affected:**
- `scripts/main_fastapi.py` (line 121-124)
- `scripts/main_modal.py` (line 86-89)

**Fix:** Use `hmac.compare_digest(api_key, API_KEY)` for constant-time comparison.

---

### CORS Wildcard Allowing Credentials

**Severity: MEDIUM**

In `scripts/main_fastapi.py` (line 101-107):

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

While origins are restricted to `localhost:3000`, the combination of `allow_credentials=True` with `allow_methods=["*"]` is risky. If the allowed origin list ever expands or is misconfigured, credentials could be sent to untrusted origins.

**Files affected:**
- `scripts/main_fastapi.py` (line 101-107)

---

## Technical Debt

### Agent Memory Leak in CacheManager

**Severity: HIGH**

In `core/utils/cache_manager.py`, agents are cached indefinitely:

```python
self._agents[book_id] = BookQdrantAgent(...)
```

Each `BookQdrantAgent` holds:
- LLM instance with callbacks
- Token usage handler
- Checkpointer reference
-检索 state

Under high tenant churn, this creates unbounded memory growth. No eviction policy exists.

**Files affected:**
- `core/utils/cache_manager.py` (lines 49-63)

**Fix:** Implement LRU eviction or time-based expiration for agents.

---

### Token Tracker Singleton with Global State

**Severity: MEDIUM**

`TokenTracker` in `core/utils/token_tracker.py` uses singleton pattern with `_instance` class variable. This pattern:
- Makes testing difficult (global state persists across tests)
- Creates implicit coupling between modules
- In-memory aggregation (`_totals`, `_by_feature`, `_by_model`) never expires

The in-memory state grows indefinitely during runtime. No mechanism to clear or prune old aggregations.

**Files affected:**
- `core/utils/token_tracker.py` (lines 46-84)

---

### In-Memory Checkpointer Fallback

**Severity: MEDIUM**

In `scripts/main_fastapi.py` (line 45):

```python
self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
```

If Supabase checkpointer fails to initialize (lines 75-88), the system silently falls back to `InMemorySaver`. This means:
- Checkpoints are lost on restart
- No cross-instance continuity
- Sessions reset after service restart

No warning is emitted to operators.

**Files affected:**
- `scripts/main_fastapi.py` (lines 74-90)
- `scripts/main_modal.py` (lines 123-142)

---

## Operational Concerns

### No Database Migration Management

**Severity: MEDIUM**

The `token_tracker.py` assumes a `token_usage` table exists but there's no migration management. Schema is defined inline in SQL (lines 174-194) but:
- No version control for schema
- No migration scripts in repository
- No validation that table exists before inserting

**Files affected:**
- `core/utils/token_tracker.py` (lines 174-194)

---

### Rate Limiting by IP Only

**Severity: MEDIUM**

In `scripts/main_fastapi.py` (line 43):

```python
limiter = Limiter(key_func=get_remote_address)
```

Rate limiting uses client IP only. In scenarios where:
- Multiple users behind same NAT
- API consumers behind shared proxy

Legitimate users get rate limited unfairly.

**Files affected:**
- `scripts/main_fastapi.py` (line 43)

---

### Temporary File Upload Race Condition

**Severity: LOW**

In `scripts/main_fastapi.py` (lines 153-157):

```python
tmp_path = f"./{file.filename}"
with open(tmp_path, "wb") as f:
    f.write(await file.read())
```

If two users upload files with the same filename (e.g., "book.pdf"), the second upload overwrites the first while it's still being processed. No atomic naming or upload queuing.

**Files affected:**
- `scripts/main_fastapi.py` (lines 153-178)

---

### No Health Check Endpoint

**Severity: LOW**

The FastAPI application has no `/health` or `/ready` endpoint. Kubernetes/load balancers cannot determine if the service is healthy without making a real request (that consumes rate limit).

**Files affected:**
- `scripts/main_fastapi.py`
- `scripts/main_modal.py`

---

## Performance Considerations

### Unbounded Scroll Pagination in Qdrant

**Severity: MEDIUM**

In `core/rag/qdrant_db.py` (lines 173-208):

```python
while True:
    results, offset = self.client.scroll(...)
    ...
    if offset is None:
        break
```

The `get_all_by_book` method paginates through ALL documents for a book without limit. For books with thousands of chunks, this:
- Loads entire dataset into memory
- Takes significant time
- No streaming or cursor-based pagination to client

**Files affected:**
- `core/rag/qdrant_db.py` (lines 173-208)

---

### Synchronous File Operations in Async Endpoint

**Severity: LOW**

In `scripts/main_fastapi.py` (lines 156-157):

```python
with open(tmp_path, "wb") as f:
    f.write(await file.read())
```

This uses synchronous file I/O in an async endpoint, blocking the event loop. Should use `aiofiles` or run in thread pool.

**Files affected:**
- `scripts/main_fastapi.py` (lines 156-157)

---

### Inefficient Double JSON Parsing

**Severity: LOW**

In `agents/book_qdrant_agent.py` (lines 88-94, 133-140):

```python
if isinstance(raw, str):
    parsed = json.loads(raw)
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
```

Multiple nested `json.loads()` calls when LLM already returns structured output via `with_structured_output()`. The parsing fallback is reasonable but could be optimized.

**Files affected:**
- `agents/book_qdrant_agent.py` (lines 88-104, 133-150)

---

## Missing Error Handling

### Supabase Checkpointer Silent Failures

**Severity: MEDIUM**

In `scripts/supabase_checkpointer.py` (lines 58-61, 95-98, 174-176):

```python
try:
    self.client.table("langgraph_checkpoints").upsert(row).execute()
except Exception as e:
    logger.error(f"put checkpoint error: {e}")
```

All exceptions are caught and only logged. No retry logic, no circuit breaker, no alerting. Checkpoint failures are silent.

**Files affected:**
- `scripts/supabase_checkpointer.py` (lines 36-69, 71-98, 100-176)

---

### No Schema Validation on Ingest

**Severity: LOW**

In `core/utils/ingest_book.py`, PDF ingestion assumes well-formed PDF. No validation that chunks meet minimum quality before indexing. Malformed PDFs could create garbage vector entries.

**Files affected:**
- `core/utils/ingest_book.py` (lines 37-66)

---

## Scalability Limitations

### Single Collection for All Courses

**Severity: MEDIUM**

In `core/utils/cache_manager.py` (line 28):

```python
self._collection_name = "lms_content"
```

All courses share a single Qdrant collection with payload-based filtering. As tenant count grows:
- Payload indexes grow
- Filter queries become slower
- No per-collection parallelism

**Files affected:**
- `core/utils/cache_manager.py` (line 28)

---

### No Connection Pooling for Supabase

**Severity: LOW**

In `core/storage/supabase_storage.py` and `core/utils/token_tracker.py`, new Supabase clients are created per operation. No connection pooling configured.

**Files affected:**
- `core/storage/supabase_storage.py` (line 28)
- `core/utils/token_tracker.py` (line 170)

---

*Concerns audit: 2026-05-30*