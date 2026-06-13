# KitaPandu RAG Book Service

A multi-tenant retrieval-augmented generation service over PDF books. Each book is chunked, embedded, and stored in a single Qdrant collection partitioned by `book_id`; the service exposes per-book chat, summarisation, mind-map, and quiz generation.

## Language

**Book**:
A single canonical PDF owned by one tenant, with associated chunks, retrievable as a unit. One book = one PDF.
_Avoid_: course, document, library, material

**BookId** (variable: `book_id`):
The unique identifier of a Book. The multitenant partition key, present as `metadata.book_id` in every Qdrant point and as a request/response field on every API.
_Avoid_: courseId, course_id, documentId, bookId (camelCase)

**Ingest**:
The act of processing a PDF and writing its chunks into the vector store, partitioned by BookId. Exposed as a two-phase async flow: `POST /book-qa/ingest` **enqueues** a job (returns `202` with the BookId) and an in-process asyncio worker task **executes** the Docling → chunk → embed → Qdrant pipeline. The worker is started in the FastAPI lifespan and polls every few seconds for `pending` rows on `ingested_books`; the lifecycle is recorded on the corresponding row (`status`, `started_at`, `finished_at`, `error`, `chunks_count`).
_Avoid_: upload (ambiguous with the Supabase Storage upload), import, index, "sync" (the endpoint is always async; the work is what is async), "Edge Function worker" / "pg_cron worker" (the worker is in-process, see [ADR-0003](./docs/adr/0003-async-ingest.md))

**Re-ingest**:
Replacing a Book's existing chunks in the vector store with the chunks of a new PDF, by hard-replace: delete all points where `metadata.book_id == book_id`, then re-add. Implemented as a separate `DELETE /book-qa/book/{book_id}/chunks` endpoint called before a new `POST /book-qa/ingest`.
_Avoid_: upsert (Qdrant `upsert` is per-point, not per-book), append, merge

**Vector state vs file state**:
The chunk data lives in Qdrant (payload `metadata.book_id`); the source PDF lives in Supabase Storage (path `{book_id}/{filename}`). These are two independent stores; the Supabase Storage upload uses `upsert: true` keyed on the path, but Qdrant does not — the Re-ingest flow exists to make Qdrant match the new file. The `ingested_books` row carries the per-BookId ingest lifecycle (`status`, `started_at`, `finished_at`, `error`, `chunks_count`) but is not the chunk store.
_Avoid_: "Supabase metadata rows" (no such table exists for chunks)

**IngestJob status**:
A BookId has at most one in-flight `ingested_books` row. `status ∈ {pending, running, succeeded, failed}`. The HTTP layer writes `pending`; the worker transitions `pending → running → succeeded` (or `failed` with `error`). `GET /book-qa/ingest/{book_id}` reads the row.
_Avoid_: a separate `jobs` / `ingest_jobs` table (the lifecycle lives on `ingested_books`)

**One-PDF-per-Book model**:
A BookId addresses exactly one canonical PDF. The Supabase Storage path uses `{book_id}/{filename}` but `filename` is informational, not a key. The `ingested_books.book_id` FK to `books.book_id` enforces the same one-row-per-BookId invariant at the DB layer.
