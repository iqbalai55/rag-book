# 0001 — Book domain model and re-ingest flow

**Status**: accepted

## Context

The service was originally framed around `course_id`: each course had a canonical PDF, chunks were stored in a single Qdrant collection (`lms_content`) partitioned by `course_id` payload, and the same `course_id` was reused as the partition key across every API endpoint and the Qdrant payload field.

Two problems surfaced:

1. **The word "course" overloaded the domain.** A course is a curriculum concept (multiple modules, multiple books). The service actually only handles one canonical PDF per `course_id` — it is a Book service, not a Course service. The existing user guide already conflates the two (`docs/USER_GUIDE.md:55`).
2. **Re-ingest did not actually replace anything.** The `POST /book-qa/ingest` endpoint calls `QdrantDB.add_documents`, which only appends. The Supabase Storage upload uses `upsert: true`, so the *file* gets overwritten, but the *vectors* from prior PDFs persist. Re-ingesting the same `course_id` polluted retrieval with stale chunks.

The term `course_id` appears 177 times in `.py` and 76 times in docs, so a rename is a large surface-area change.

## Decision

We rename `courseId` → `BookId` across the entire stack (API params, Python variables, the Qdrant payload field `metadata.course_id` → `metadata.book_id`, the Supabase Storage path `{course_id}/{filename}` → `{book_id}/{filename}`, all docs), and we formalise the re-ingest flow as a two-endpoint cascade with an in-process lock.

The locked decisions:

- **One canonical PDF per BookId.** The model is "one book = one PDF = one tenant partition." Filenames in the storage path are informational, not keys.
- **Hard-replace on re-ingest.** New PDF for an existing `book_id` must replace all prior chunks. There is no append/merge mode.
- **Two-endpoint cascade.** `DELETE /book-qa/book/{book_id}/chunks?filename=<name>` (removes Qdrant points and the Storage object), then `POST /book-qa/ingest?book_id=<id>` (uploads new PDF and chunks it).
- **Delete scope = vectors + storage.** The delete endpoint removes both the Qdrant points where `metadata.book_id == book_id` and the Supabase Storage object. The `filename` query param is optional; if omitted, the only file under `{book_id}/` is removed.
- **Per-bookId `asyncio.Lock` in `CacheManager`.** A shared lock keyed by `book_id` is acquired by both the delete and ingest endpoints. Concurrent mutations to the same book serialise; mutations to different books remain parallel. Released in `finally`.
- **Failure mode is empty book + retry.** If ingest fails after delete, the book is empty in both stores. The frontend retries the full cascade; both endpoints are idempotent. No tombstone, no soft-delete, no staging area.

## Considered Options

**Re-ingest semantics**
- A (chosen): hard-replace by `book_id` — clean state, simple semantics.
- B: soft-replace by `book_id` + `filename` — kept chunks from sibling PDFs, but the M1 model forbids siblings.
- C: versioned ingest with query-side filter — more flexible but contradicts the "one canonical PDF" model.

**Owner of the delete**
- A1 (rejected): hidden delete inside `POST /book-qa/ingest` — failure mode harder to reason about, race window between delete and ingest.
- A2 (chosen): separate `DELETE /book-qa/book/{book_id}/chunks` endpoint — explicit state transition, locks are easy to add.

**Multi-PDF per book**
- M1 (chosen): one PDF per book — matches the current single-`file` ingest signature and the user guide's "book/course" framing.
- M2 (rejected): would require `delete_by_course` to also take `filename` and key replacement on `(book_id, filename)`, not just `book_id`.

**Rename scope**
- R1 (rejected): API surface only — permanent name mismatch between public API and internal data model.
- R2 (rejected): API + Qdrant payload + variables — left storage path and docs stale.
- R3 (chosen): full rename — touches all 250+ references, but data layer and public language match exactly.

**Concurrency control**
- L1 (rejected): no lock — torn state under concurrent re-ingest.
- L2 (chosen): per-bookId `asyncio.Lock` in `CacheManager` — in-process, no extra infra, sufficient for single-replica service.
- L3 (rejected): optimistic reject via `ingest_version` — more moving parts than L2 for the same guarantee.
- L4 (rejected): external lock (Postgres advisory / Redis) — overkill until horizontal scale.

**Failure handling**
- F1 (chosen for the re-ingest cascade): empty state, frontend retries — no new state, simple. The cascade is delete → ingest; if either fails, the book is briefly empty in both stores and the frontend retries the full sequence.
- F2 (rejected for the re-ingest cascade, **later accepted for the first-time ingest path** — see [ADR-0003](./0003-async-ingest.md)): persist a `books` status table — the original rejection held for the cascade but the first-time ingest path is now async and writes its lifecycle onto `ingested_books.status` instead of a new table.
- F3 (rejected): sentinel point in Qdrant — pollutes the collection with non-content points.
- F4 (rejected): two-phase commit with quarantine — too heavy for M-Empty (no production data).

## Consequences

- **Single-PDF invariant is now load-bearing.** Any future feature that wants multi-PDF per book must revisit this ADR; the storage path, the delete endpoint, and the lock all assume M1.
- **Frontend owns the cascade.** The frontend must call delete → ingest in sequence, handle 5xx from ingest by showing a retry button, and accept that the book may be briefly empty.
- **Lock is in-process.** If/when the service is scaled horizontally, the L2 lock no longer prevents races across replicas. (Superseded for the ingest path by [ADR-0003](./0003-async-ingest.md) §"Concurrency" — skip-locked is the cross-replica guarantee; L2 is now a same-replica safety net.)
- **Rename is atomic.** With no production data (M-Empty), the rename ships in one PR. There is no backfill script. The Qdrant collection is recreated on first ingest under the new name (`metadata.book_id`).
- **Public API breaks.** All current callers sending `course_id` will start receiving 422s. Coordinate with frontend before deploying.
