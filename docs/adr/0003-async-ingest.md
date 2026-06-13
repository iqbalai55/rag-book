# 0003 — Async ingest with `ingested_books` as the job table

**Status**: accepted

Supersedes the Failure-handling F1 ("empty state, frontend retries") of [ADR-0001](./0001-book-domain-and-reingest.md) for the ingest path. F1 still describes the **re-ingest** failure mode (delete → ingest cascade), not the first-time ingest path. Concurrency-control L2 is retained as a same-replica safety net; this ADR adds the cross-replica guarantee on top.

## Context

`POST /book-qa/ingest` runs the full Docling → HybridChunker → embed → Qdrant pipeline inline in the HTTP request. For a 200-page book this is multi-minute, blocks the client's HTTP connection, and gives the client no handle to:

- know whether the work is in progress, done, or failed (only the terminal HTTP response tells you),
- retry a failed job without re-uploading the PDF,
- run more than one ingest at a time per tenant.

The existing `ingested_books` Supabase table is the natural place to record ingest lifecycle, but it has no status / error / started_at / finished_at columns today — it is a post-success audit row. The pre-success state has no home.

ADR-0001 §"Failure handling" explicitly rejected a `books` status table (F2) on the grounds that "external dependency for a problem that retry solves." That judgment was correct for the *first* version of the service (M-Empty, no users, no production data) but no longer holds now that:

- the `ingested_books` table already exists and is the right shape to add status columns to,
- we want the client to receive a BookId and poll, not hold a TCP connection,
- we want a worker (not the request coroutine) to own the pipeline so horizontal scale and crash recovery are possible.

## Decision

**Ingest is two-phase and async.** The HTTP layer enqueues; a background worker executes; the `ingested_books` row carries the lifecycle.

### Schema changes (applied to existing `ingested_books`)

```sql
alter table public.ingested_books
  add column if not exists status      text not null default 'pending'
    check (status in ('pending','running','succeeded','failed')),
  add column if not exists started_at  timestamptz,
  add column if not exists finished_at timestamptz,
  add column if not exists error       text;

-- user_id is now NULL-able. The public ingest endpoint is a trusted-backend
-- call, not a user call, and the original NOT NULL would have forced a
-- sentinel auth.users entry.
alter table public.ingested_books
  alter column user_id drop not null;

-- The FK to books.book_id is dropped. The pre-async ingest code never
-- created a books row at ingest time, so the FK was never enforced in
-- practice. ingested_books is the lifecycle surface; books metadata is
-- owned by the calling flow.
alter table public.ingested_books
  drop constraint if exists ingested_books_book_id_fkey;

create index if not exists idx_ingested_books_pending
  on ingested_books(created_at) where status = 'pending';
```

The "at most one row per BookId" invariant is enforced by the `INSERT … ON CONFLICT (book_id) DO UPDATE` upsert in `core/utils/ingest_jobs.py`, not by the DB. `id` remains the surrogate PK; `book_id` is the natural key used by the API and the worker.

### HTTP surface

| Method | Path | Behaviour |
|---|---|---|
| `POST /book-qa/ingest?book_id=...` | multipart `file` | Upload PDF to `{book_id}/{filename}` in `book-pdfs` bucket with `upsert: true`; `insert … on conflict (book_id) do update` on `ingested_books` setting `status='pending'`, `error=null`; return `202 {"book_id": "...", "status": "pending"}`. Auth (`x-api-key`) and `3/minute` rate limit unchanged. |
| `GET /book-qa/ingest/{book_id}` | — | Returns the `ingested_books` row for the BookId: `status`, `chunks_count`, `error`, `started_at`, `finished_at`, `storage_url`. 404 if no row. |
| `GET /book-qa/ingest?book_id=...&status=...&limit=...` | — | List jobs (for the UI). |
| `POST /book-qa/ingest/{book_id}/retry` | — | If `status='failed'`, reset to `pending` and clear `error`. Idempotent. |
| `DELETE /book-qa/book/{book_id}/chunks` | unchanged from ADR-0001 | Hard-deletes Qdrant points + Storage object. New: also marks the latest `ingested_books` row `status='failed', error='re-ingest: chunks deleted'` if one is in flight, so the worker stops touching the book. |

Note: there is **no** `/internal/ingest/{book_id}` route. The amendment removed it; the in-process worker calls the pipeline directly.

### Worker

- An in-process asyncio task is started in the FastAPI lifespan. The task loops forever: `await asyncio.sleep(5)` → `await claim_next_pending()` → run the pipeline → write back status. Started once per process; cancelled on shutdown.
- The pipeline itself is CPU/IO-blocking (Docling + embedding + Qdrant). The tick wraps it in `asyncio.to_thread(...)` so the event loop stays responsive.
- Each tick: `update ingested_books set status='running', started_at=now() where id = (select id from ingested_books where status='pending' order by created_at for update skip locked limit 1) returning id, book_id, filename, storage_url;`. Skip-locked is the cross-replica claim — two replicas cannot grab the same row. With a single replica it is still useful (a stale `running` row will not be re-claimed until the reaper resets it).
- The tick writes back `status='succeeded', chunks_count=N, finished_at=now()` on success or `status='failed', error=<message>, finished_at=now()` on exception. The PDF stays in Storage so `/retry` can re-enqueue it.
- Startup reconciliation: on lifespan startup, any `status='running'` row whose `started_at` is older than the worker timeout (default 30 minutes) is reset to `status='pending'`, `started_at=null`. This recovers from a crashed mid-ingest process.
- **No Edge Functions, no `pg_cron`, no separate worker runtime.** The in-process loop is the entire worker surface.

### Concurrency

- The existing per-BookId `asyncio.Lock` in `CacheManager` (ADR-0001 L2) **remains** — it is still useful as a same-replica safety net inside the FastAPI process.
- The skip-locked claim is the cross-replica guarantee. L2 is no longer the *only* guarantee; the language in ADR-0001 L2's "if the service is scaled horizontally" caveat now applies to the worker, not the lock.
- The worker acquires the per-BookId `CacheManager` lock while running the pipeline, so a same-replica worker and a same-replica public caller cannot interleave against the same BookId.

### Frontend

- `POST /book-qa/ingest` returns `202` with the BookId. The UI polls `GET /book-qa/ingest/{book_id}` until `status ∈ {succeeded, failed}`.
- On `failed`, show the `error` message and a "Retry" button → `POST /book-qa/ingest/{book_id}/retry`.
- Re-ingest still requires the existing two-endpoint cascade (delete → ingest). The new `202` semantics apply to the ingest step in that cascade.

## Considered Options

**Job model**
- J1 (rejected): separate generic `jobs` table — over-general for a single feature, breaks the domain glossary.
- J2 (chosen): extend `ingested_books` with status columns — table is already there, FK already gives us the one-row-per-BookId invariant, no new cross-replica coordination surface.
- J3 (rejected): use Supabase Queues / `pgmq` — queue + status row are two sources of truth, more moving parts for the same outcome.

**Worker location**
- W1 (rejected): FastAPI `BackgroundTasks` (in-process, per-request) — no cross-replica guarantee; if the request returns 502 nothing is enqueued. Adequate for a single replica but the failure mode is bad.
- W2 (rejected): Modal `@modal.function()` background worker — tightest coupling to Modal, more infra to manage, and we still need the row as a queue.
- W3 (rejected, later superseded by amendment): Supabase Edge Function scheduled by `pg_cron` calling back into FastAPI — added a Deno runtime, a pg_cron config, and two new secrets for a problem the in-process loop solves.
- W4 (chosen, amendment): in-process asyncio task started in the FastAPI lifespan, looping every 5s and calling the same `claim_next_pending` + pipeline + write-back flow directly. Zero new infra, zero new secrets, single source of truth for the SQL.

**Worker → pipeline hand-off**
- H1 (chosen, amendment): the lifespan-spawned task calls the pipeline directly via `asyncio.to_thread(ingest_book, ...)` — no HTTP round-trip, no separate callback route. The previously-planned `POST /internal/ingest/{book_id}` route is removed; the pipeline runs in the same process that holds the per-BookId lock.
- H2 (rejected): Edge Function runs the Docling pipeline directly in Deno — Docling is not a Deno library; would mean a second Python runtime.
- H3 (rejected): a separate "executor" process — two runtimes, two failure modes, two deploys.

**Failure handling** (re-litigated from ADR-0001 §F)
- F1' (rejected for first-time ingest, retained for re-ingest cascade): empty state, frontend retries — still describes the delete → ingest cascade, not the first-time path.
- F2' (chosen): persist status on `ingested_books` — same as F2 in ADR-0001, but on an existing table rather than a new one.
- F5 (rejected): automatic retry inside the worker on failure — runaway loops on permanently broken PDFs; `/retry` is explicit and human-visible.

**Idempotency of re-enqueue**
- I1 (chosen): `insert … on conflict (book_id) do update` — `pending` always wins; the latest upload's `filename`/`storage_url` overwrite. Race window between two concurrent POSTs is closed by the same per-BookId lock the public route already takes.
- I2 (rejected): new row per upload, plus a view for the "current" one — history is nice but not a v1 requirement.

## Consequences

- The HTTP client is decoupled from the pipeline. A 5 MB PDF and a 500 MB PDF take the same time on the client (the 202).
- Crashes are recoverable: a `running` row older than the worker timeout can be reset to `pending` on the next tick (handled in the worker's "stale-claim reaper"; out of scope for this ADR but enabled by the schema).
- ADR-0001 §"Failure handling" needs a one-line patch: keep F1 wording for the *re-ingest cascade* (delete + ingest of a new PDF), point at this ADR for the *first-time ingest path* (enqueue + execute).
- L2 lock is no longer the *only* concurrency guard; the new ADR's "skip-locked" is the primary one. ADR-0001's "horizontal scale will need L4" caveat becomes "horizontal scale is now safe; L2 is belt-and-braces."
- `ingested_books` is no longer just an audit row — it is the queue and the status surface. Frontend reads from it; worker writes to it. Schema is now a public contract; any column change is an API change.
- The `3/minute` rate limit on `POST /book-qa/ingest` is now a *enqueue* rate limit, not a *pipeline* rate limit. That is the right place to put it (it is the per-IP limit the public API can enforce), but the wording in code comments should change.
