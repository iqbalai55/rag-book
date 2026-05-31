# RAG Book Agent - User Guide

## What Is This System?

**RAG Book Agent** is an intelligent system that lets you upload books (PDFs) and interact with them using AI. You can ask questions, generate summaries, create mind maps, and produce quiz questions - all powered by the content in your books.

Think of it as having a personal AI tutor for each book who has read the entire book and can help you learn from it.

---

## Quick Start

### 1. Start the Server

```bash
cd rag-book
python scripts/main_fastapi.py
```

The server will start at `http://127.0.0.1:8001`

### 2. Upload a Book

Use the ingest endpoint to upload a PDF:

```bash
curl -X POST "http://localhost:8001/book-qa/ingest?course_id=my_book" \
  -H "x-api-key: your_api_key" \
  -F "file=@path/to/your/book.pdf"
```

Replace:
- `my_book` - A unique ID for your book/course
- `your_api_key` - Your API key (set in .env)
- `path/to/your/book.pdf` - Path to your PDF file

### 3. Ask Questions

```bash
curl -N http://localhost:8001/book-qa/stream \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_api_key" \
  -d '{
    "session_id": "session_1",
    "course_id": "my_book",
    "messages": [{"role": "user", "content": "What is the main topic of this book?"}]
  }'
```

---

## Key Concepts

### Course ID
A unique identifier for each book/course. All content from one book shares the same `course_id`. This ensures:
- Data isolation between different books
- You can chat with specific books by specifying their course_id

### Session ID
A identifier for a conversation thread. The system remembers your conversation within the same session.

### Token
API calls to AI models (like GPT-4) are measured in "tokens" (words/characters). The system tracks how many tokens each operation uses.

---

## Features Explained

### 1. Chat with Book (Q&A)

**Endpoint:** `POST /book-qa/stream`

Ask any question about the book and get answers with source citations.

**Example Question:**
> "What are the key principles of software design?"

**What you get:**
- AI-generated answer based on the book content
- Links to the specific pages in the PDF where the answer came from
- Streaming response (answers appear in real-time)

### 2. Book Summary

**Endpoint:** `POST /book-qa/summarize`

Generate a comprehensive summary of the entire book.

**Returns:**
- Book title
- Overview (brief description)
- Chapter-by-chapter summaries
- Key themes across the book
- Source references

**Example:**
```bash
curl -X POST "http://localhost:8001/book-qa/summarize?course_id=my_book" \
  -H "x-api-key: your_api_key"
```

**With custom instructions:**
```bash
curl -X POST "http://localhost:8001/book-qa/summarize?course_id=my_book&user_prompt=Fokus ke aspek teknis" \
  -H "x-api-key: your_api_key"
```

### 3. Mind Map Generation

**Endpoint:** `POST /book-qa/mindmap`

Generate a visual mind map diagram showing the book's structure and main topics.

**Returns:**
- Mind map title
- Mermaid diagram code (a text format that can be rendered as a diagram)

**Example:**
```bash
curl -X POST "http://localhost:8001/book-qa/mindmap?course_id=my_book" \
  -H "x-api-key: your_api_key"
```

**With custom focus:**
```bash
curl -X POST "http://localhost:8001/book-qa/mindmap?course_id=my_book&user_prompt=Fokus ke design patterns" \
  -H "x-api-key: your_api_key"
```

### 4. Generate Quiz Questions

**Endpoint:** `POST /book-qa/dataset`

Generate multiple-choice (MCQ) and essay questions from the book content.

**Use cases:**
- Create exams
- Create study guides
- Test your understanding

**Parameters:**
- `difficulty` - easy, medium, or hard
- `num_mcq` - Number of MCQ per chapter (default: 3)
- `num_essay` - Number of essay questions per chapter (default: 2)

**Example:**
```bash
curl -X POST "http://localhost:8001/book-qa/dataset?course_id=my_book&difficulty=medium&num_mcq=5&num_essay=3" \
  -H "x-api-key: your_api_key"
```

### 5. Edit Summary/Mindmap

**Endpoints:**
- `POST /book-qa/summarize/edit` - Modify existing summary
- `POST /book-qa/mindmap/edit` - Modify existing mindmap

**Use cases:**
- Add more details
- Simplify content
- Focus on specific aspects
- Add new sections

**Example - Edit Summary:**
```bash
curl -X POST "http://localhost:8001/book-qa/summarize/edit" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_api_key" \
  -d '{
    "course_id": "my_book",
    "title": "Current Title",
    "overview": "Current overview text...",
    "key_themes": ["Theme 1", "Theme 2"],
    "chapters": [{"chapter_title": "Ch 1", "summary": "...", "key_points": ["..."]}],
    "instruction": "Add more practical examples"
  }'
```

---

## How It Works (Simple Explanation)

### 1. Book Upload Flow

```
[Your PDF] → [Supabase Storage] → [Docling Parser] → [Qdrant Vector DB]
                                                    ↓
                                          Chunks + Embeddings
```

1. **Upload**: PDF is saved to Supabase Storage (cloud file storage)
2. **Parse**: Docling reads the PDF and extracts text with structure
3. **Chunk**: Long text is split into smaller pieces (~256 tokens each)
4. **Index**: Each chunk is converted to a vector embedding and stored in Qdrant

### 2. Question Answering Flow

```
[Your Question] → [Embed Question] → [Search Qdrant] → [Find Relevant Chunks]
                                                              ↓
                                                      [Send to AI Model]
                                                              ↓
                                                      [Generate Answer]
                                                              ↓
                                                      [Stream Response]
```

1. **Embed**: Your question is converted to a vector
2. **Search**: Qdrant finds the most relevant chunks from the book
3. **Generate**: AI model reads the relevant chunks + your question
4. **Respond**: Answer is streamed back with source citations

### 3. Multi-Tenant Architecture

Each book/course is completely isolated:

```
┌─────────────────────────────────────────────────────┐
│                  Qdrant Vector DB                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Course A   │  │  Course B   │  │  Course C   │ │
│  │  (Book A)   │  │  (Book B)   │  │  (Book C)   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────┘
```

- One collection (`lms_content`) stores all books
- Each chunk has a `course_id` metadata
- Queries filter by `course_id` to ensure isolation

---

## API Reference

### Common Headers

All endpoints require:
```
x-api-key: your_api_key
Content-Type: application/json (for POST with body)
```

### Endpoints Summary

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| POST | `/book-qa/stream` | Chat with book (SSE) | 10/min |
| POST | `/book-qa/ingest` | Upload PDF book | 3/min |
| POST | `/book-qa/summarize` | Generate book summary | 2/min |
| POST | `/book-qa/summarize/edit` | Edit summary | 10/min |
| POST | `/book-qa/mindmap` | Generate mind map | 5/min |
| POST | `/book-qa/mindmap/edit` | Edit mind map | 10/min |
| POST | `/book-qa/dataset` | Generate quiz dataset | 2/min |
| GET | `/token-usage` | Get token usage stats | 30/min |
| GET | `/token-usage/daily` | Get daily usage | 30/min |
| POST | `/token-usage/flush` | Flush tokens to DB | - |

### Request/Response Examples

#### Chat Payload
```json
{
  "session_id": "unique_session_id",
  "course_id": "my_book",
  "messages": [
    {"role": "user", "content": "Your question here"}
  ]
}
```

#### Chat Response (SSE Stream)
```json
{"id": "chatcmpl", "type": "final", "content": "The answer...", "metadata": {}}
{"id": "chatcmpl", "type": "tool", "content": "Search results...", "metadata": {"tool_name": "search_book_context"}}
{"id": "chatcmpl", "type": "multiple_choice_question", "content": {...}, "metadata": {...}}
[DONE]
```

**Message Types:**
- `final` - Final AI response
- `tool` - Tool execution results (search, etc.)
- `internal` - AI internal reasoning
- `multiple_choice_question` - MCQ generated
- `essay_question` - Essay question generated
- `error` - Error occurred

---

## Environment Setup

### Required Variables (.env)

```bash
# LLM Provider Configuration
LLM_PROVIDER=openai              # openai, anthropic, openrouter, minimax
LLM_MODEL=gpt-4o-mini             # Model to use

# API Keys (depending on provider)
OPENAI_API_KEY=sk-...            # If using OpenAI
ANTHROPIC_API_KEY=sk-ant-...     # If using Anthropic

# Vector Database (Qdrant)
QDRANT_ENDPOINT=https://xxx.qdrant.tech
QDRANT_API_KEY=your_qdrant_key

# Supabase (Storage + Database)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=your_service_key
SUPABASE_DB_URL=postgresql://user:pass@host:5432/db
SUPABASE_STORAGE_BUCKET=book-pdfs

# Security
API_KEY=your_secret_api_key

# Optional: Observability
LANGSMITH_API_KEY=your_langsmith_key
```

---

## Troubleshooting

### Common Issues

**1. "Invalid or missing API Key"**
- Ensure `API_KEY` is set in .env
- Include header `-H "x-api-key: your_api_key"` in requests

**2. "No content found for this course"**
- First upload a book with the `course_id` using `/book-qa/ingest`
- Check the `course_id` matches exactly (case-sensitive)

**3. Rate limit exceeded (429)**
- Wait and retry after a minute
- Each endpoint has different rate limits (see API Reference)

**4. CUDA/GPU errors**
- The system automatically falls back to CPU if CUDA is unavailable
- Embeddings will be slower but work correctly

### Debug Mode

Check server logs for detailed operation info:
```bash
# Server console shows:
# - Chunking progress
# - Search results
# - Token usage
# - Error details
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend/Client                          │
└────────────────────────────┬────────────────────────────────────┘
                           │ HTTP/SSE
┌───────────────────────────▼────────────────────────────────────┐
│                     FastAPI Server                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Rate Limit  │  │   CORS      │  │ Auth (API)  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└────────────────────────────┬────────────────────────────────────┘
                           │
┌───────────────────────────▼────────────────────────────────────┐
│                    Service Layer                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐│
│  │ BookSummarizer   │  │ MindmapGenerator │  │DatasetGenerator││
│  └──────────────────┘  └──────────────────┘  └────────────────┘│
└────────────────────────────┬────────────────────────────────────┘
                           │
┌───────────────────────────▼────────────────────────────────────┐
│                     Agent Layer                                  │
│  ┌────────────────────────────────────────────────────────────┐│
│  │              BookQdrantAgent (LangGraph)                     ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐││
│  │  │search_book   │  │generate_mcq │  │generate_essay_questions│││
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘││
│  └────────────────────────────────────────────────────────────┘│
└────────────────────────────┬────────────────────────────────────┘
                           │
┌───────────────────────────▼────────────────────────────────────┐
│                   Retrieval Layer                               │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐ │
│  │   CacheManager           │  │   QdrantDB                   │ │
│  │   (Agent + DB Caching)   │  │   (Vector Store)             │ │
│  └─────────────────────────┘  └──────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                           │
┌───────────────────────────▼────────────────────────────────────┐
│                    Storage Layer                                 │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐ │
│  │   Supabase Storage       │  │   Supabase PostgreSQL        │ │
│  │   (PDF Files)            │  │   (Token Usage, Checkpoints)  │ │
│  └─────────────────────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## For Developers

### Project Structure

```
rag-book/
├── scripts/
│   ├── main_fastapi.py          # API server entry point
│   └── main_modal.py            # Cloud deployment
├── agents/
│   └── book_qdrant_agent.py     # Main AI agent
├── core/
│   ├── rag/
│   │   ├── qdrant_db.py         # Vector database wrapper
│   │   └── document_processor.py # PDF processing
│   ├── services/
│   │   ├── summarize_book.py     # Summary generation
│   │   ├── mindmap_generator.py  # Mind map generation
│   │   └── dataset_generator.py  # Quiz generation
│   ├── storage/
│   │   └── supabase_storage.py   # PDF storage
│   └── utils/
│       ├── cache_manager.py     # Agent caching
│       └── token_tracker.py     # Usage tracking
└── test/                         # Test suite
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest test/test_qdrant.py

# Run with coverage
pytest --cov=./ --cov-report=html
```

---

## Glossary

| Term | Definition |
|------|------------|
| **RAG** | Retrieval Augmented Generation - AI technique combining document retrieval with text generation |
| **Vector Embedding** | Numerical representation of text that captures meaning |
| **Vector Database** | Database optimized for storing and searching vector embeddings |
| **Chunk** | Small piece of text split from a larger document |
| **SSE** | Server-Sent Events - Technology for streaming responses in real-time |
| **Multitenant** | Architecture where one system serves multiple customers/books in isolation |
| **Checkpoint** | Saved state of a conversation for resume capability |
| **Token** | Unit of text measurement for AI model pricing |

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review server logs for error details
3. Verify all environment variables are set correctly
