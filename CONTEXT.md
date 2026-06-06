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
The act of processing a PDF and writing its chunks into the vector store, partitioned by BookId.
_Avoid_: upload (ambiguous with the Supabase Storage upload), import, index

**Re-ingest**:
Replacing a Book's existing chunks in the vector store with the chunks of a new PDF, by hard-replace: delete all points where `metadata.book_id == book_id`, then re-add. Implemented as a separate `DELETE /book-qa/book/{book_id}/chunks` endpoint called before a new `POST /book-qa/ingest`.
_Avoid_: upsert (Qdrant `upsert` is per-point, not per-book), append, merge

**Vector state vs file state**:
The chunk data lives in Qdrant (payload `metadata.book_id`); the source PDF lives in Supabase Storage (path `{book_id}/{filename}`). These are two independent stores; the Supabase Storage upload uses `upsert: true` keyed on the path, but Qdrant does not — the Re-ingest flow exists to make Qdrant match the new file.
_Avoid_: "Supabase metadata rows" (no such table exists for chunks)

**One-PDF-per-Book model**:
A BookId addresses exactly one canonical PDF. The Supabase Storage path uses `{book_id}/{filename}` but `filename` is informational, not a key.
