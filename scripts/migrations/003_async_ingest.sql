-- 003_async_ingest.sql
-- Adds async-ingest lifecycle columns to public.ingested_books, relaxes
-- `user_id` to NULL-able, and drops the FK to `books.book_id`. The public
-- POST /book-qa/ingest is a trusted-backend call, not a user call, and the
-- pre-async code never created a `books` row at ingest time. See
-- docs/adr/0003-async-ingest.md for the full design.

-- 1) Lifecycle columns
alter table public.ingested_books
  add column if not exists status      text not null default 'pending'
    check (status in ('pending','running','succeeded','failed')),
  add column if not exists started_at  timestamptz,
  add column if not exists finished_at timestamptz,
  add column if not exists error       text;

create index if not exists idx_ingested_books_pending
  on ingested_books(created_at)
  where status = 'pending';

create index if not exists idx_ingested_books_running
  on ingested_books(started_at)
  where status = 'running';

-- 2) Drop NOT NULL on user_id. The original schema's NOT NULL would force a
--    sentinel row + a real auth.users entry, which is out of scope for the
--    async-ingest slice. Per-user scoping is a follow-up.
alter table public.ingested_books
  alter column user_id drop not null;

-- 3) Drop the FK to books.book_id. ingested_books is the lifecycle surface;
--    books metadata is owned by the calling flow. The pre-async ingest code
--    never created a books row, so the FK was never enforced in practice.
alter table public.ingested_books
  drop constraint if exists ingested_books_book_id_fkey;
