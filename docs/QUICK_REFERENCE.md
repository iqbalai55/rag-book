# Quick Reference Card

## Essential Commands

### Start Server
```bash
python scripts/main_fastapi.py
```

### Upload a Book
```bash
curl -X POST "http://localhost:8001/book-qa/ingest?course_id=YOUR_COURSE_ID" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "file=@path/to/book.pdf"
```

### Ask a Question
```bash
curl -N http://localhost:8001/book-qa/stream \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "session_id": "s1",
    "course_id": "YOUR_COURSE_ID",
    "messages": [{"role": "user", "content": "Your question?"}]
  }'
```

---

## All Endpoints

| Action | Command |
|--------|---------|
| **Ingest PDF** | `POST /book-qa/ingest?course_id=X` + file |
| **Chat** | `POST /book-qa/stream` + ChatPayload |
| **Summarize** | `POST /book-qa/summarize?course_id=X` |
| **Edit Summary** | `POST /book-qa/summarize/edit` + SummaryEditRequest |
| **Mind Map** | `POST /book-qa/mindmap?course_id=X` |
| **Edit Mind Map** | `POST /book-qa/mindmap/edit` + MindmapEditRequest |
| **Generate Quiz** | `POST /book-qa/dataset?course_id=X&difficulty=medium` |
| **Token Usage** | `GET /token-usage?course_id=X` |
| **Daily Usage** | `GET /token-usage/daily?course_id=X&days=7` |
| **Flush Tokens** | `POST /token-usage/flush` |

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/book-qa/stream` | 10/min |
| `/book-qa/ingest` | 3/min |
| `/book-qa/summarize` | 2/min |
| `/book-qa/summarize/edit` | 10/min |
| `/book-qa/mindmap` | 5/min |
| `/book-qa/mindmap/edit` | 10/min |
| `/book-qa/dataset` | 2/min |
| `/token-usage*` | 30/min |

---

## JSON Schemas

### ChatPayload
```json
{
  "session_id": "string",
  "course_id": "string",
  "messages": [
    {"role": "user", "content": "string"}
  ]
}
```

### MindmapEditRequest
```json
{
  "course_id": "string",
  "mermaid": "string (existing mermaid code)",
  "instruction": "string"
}
```

### SummaryEditRequest
```json
{
  "course_id": "string",
  "title": "string",
  "overview": "string",
  "key_themes": ["string"],
  "chapters": [{"chapter_title": "string", "summary": "string", "key_points": ["string"]}],
  "instruction": "string"
}
```

---

## Message Types (SSE Response)

| Type | Meaning |
|------|---------|
| `final` | Final AI answer |
| `tool` | Tool execution result |
| `internal` | AI internal reasoning |
| `multiple_choice_question` | MCQ generated |
| `essay_question` | Essay question generated |
| `error` | Error occurred |

---

## Environment Variables Checklist

- [ ] `API_KEY` - Your API key for authentication
- [ ] `LLM_PROVIDER` - openai/anthropic/openrouter/minimax
- [ ] `LLM_MODEL` - Model name (e.g., gpt-4o-mini)
- [ ] `OPENAI_API_KEY` (or your provider's key)
- [ ] `QDRANT_ENDPOINT` - Qdrant Cloud URL
- [ ] `QDRANT_API_KEY` - Qdrant API key
- [ ] `SUPABASE_URL` - Supabase project URL
- [ ] `SUPABASE_SERVICE_KEY` - Supabase service key
- [ ] `SUPABASE_DB_URL` - PostgreSQL connection string

---

## Common Course IDs by Example

| Book | course_id |
|------|-----------|
| Refactoring Book | `refactoring_book` |
| Clean Architecture | `clean_arch` |
| Design Patterns | `design_patterns` |

---

*Keep this card handy for quick reference!*
