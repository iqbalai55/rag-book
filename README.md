# RAG Book Agent

Multi-tenant RAG system for querying book content with source citations. Ask questions, generate MCQs, essay questions, podcasts, and summaries from your book knowledge base.

## Features

- **LLM Capabilities**
  - Answer questions with book sources and page citations (links to PDF in Supabase Storage)
  - Generate multiple choice and essay questions from book content
  - Generate podcast audio from book topics (Indonesian language)
  - Generate mindmap from book content (Mermaid format) with **custom user prompts**
  - Generate dataset (MCQ + Essay) per chapter
  - **Summarize entire book** with hierarchical map-reduce approach (per-chapter + overview)
  - **Edit/revise summaries and mindmaps** based on user instructions

- **User Customization**
  - Guide mindmap generation with custom prompts (e.g., "Focus on design patterns")
  - Guide summary generation with custom prompts (e.g., "More detail on technical aspects")
  - Edit existing mindmaps (add/remove branches, restructure)
  - Edit existing summaries (revise, detail, simplify, add chapters)

- **Dynamic Expertise Detection**
  - AI automatically adapts persona based on book domain
  - Detects domain (Software Engineering, History, Biology, etc.)
  - Identifies sub-fields (code quality, refactoring, clean architecture, etc.)
  - Prompts use appropriate terminology for each domain

- **Persistence & Caching**
  - Save conversation threads to Supabase/PostgreSQL
  - Cache QdrantDB, QdrantClient, and Agents

- **Supabase Integration**
  - PDF storage in Supabase Storage (book-pdfs bucket)
  - Source citations link directly to PDF pages
  - Token usage tracking with cost estimation
  - PostgreSQL-based conversation checkpointing

- **Observability**
  - LangSmith integration for production tracing
  - MLflow for benchmarking (embedding, LLM models)
  - Token usage analytics per feature, model, and course

- **Deployment**
  - FastAPI for local development
  - Modal.com for cloud deployment
  - API key security
  - Rate limiter
  - CORS middleware for frontend integration

## Tech Stack

- **LangChain/LangGraph** - Agent framework
- **Qdrant** - Vector database
- **Supabase** - Storage, PostgreSQL, Auth
- **LangSmith** - Production observability
- **MLflow** - Benchmarking
- **FastAPI** - Local API server
- **Modal.com** - Cloud deployment
- **Chatterbox TTS** - Text-to-speech
- **Docling** - PDF parsing and chunking

## Project Structure

```
rag-book/
├── scripts/                   # Entry points
│   ├── main_fastapi.py        # FastAPI server (local)
│   ├── main_modal.py          # Modal deployment (cloud)
│   ├── main_checkpointer.py   # Postgres checkpointer setup
│   ├── migrations/            # SQL migrations
│   ├── run_podcast_agent.py   # Podcast agent CLI
│   └── run_dataset_builder.py # Dataset builder CLI
├── agents/                     # AI agents
│   ├── book_qdrant_agent.py   # Main RAG agent
│   └── book_podcast_agent.py  # Podcast generation agent
├── core/                       # Core domain
│   ├── dataset/              # Dataset utilities
│   ├── evaluator/            # Evaluators & scorers
│   ├── prompts/              # Prompt templates
│   │   ├── general_rag.py    # RAG prompts (MCQ, Essay, Book QA)
│   │   ├── summary.py        # Book summarization prompts
│   │   ├── mindmap.py        # Mindmap prompts
│   │   └── podcast.py        # Podcast prompts
│   ├── rag/                  # RAG services (Qdrant)
│   ├── schemas/              # Pydantic models
│   │   ├── summary.py        # BookSummaryResponse, SummaryEditRequest
│   │   ├── mindmap.py        # MindmapResponse, MindmapEditRequest
│   │   ├── chat.py           # ChatPayload
│   │   └── question.py       # MCQResponse, EssayResponse
│   ├── services/             # Business logic (decomposed)
│   │   ├── dataset_generator.py   # DatasetGenerator class
│   │   ├── summarize_book.py      # BookSummarizer class
│   │   └── mindmap_generator.py   # MindmapGenerator class
│   ├── storage/              # Supabase Storage
│   ├── tts/                  # Text-to-speech
│   └── utils/                # Utilities
│       ├── token_tracker.py  # Token usage tracking
│       ├── token_callback.py # LangChain callback for tokens
│       └── cache_manager.py  # Agent & DB caching
├── benchmarking/              # Performance benchmarks
├── test/                      # Test suite
├── book/                      # PDF sources
└── requirements/              # Dependencies
```

## Installation

### 1️⃣ Clone Repository

```bash
git clone <repository-url>
cd rag-book
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements/requirements_main.txt
```

### 4️⃣ Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required variables:
- `LLM_PROVIDER` - LLM provider (openai, anthropic, openrouter, minimax)
- `LLM_MODEL` - Model name
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `OPENROUTER_API_KEY` - OpenRouter API key
- `MINIMAX_API_KEY` - MiniMax API key (optional)
- `QDRANT_ENDPOINT` - Qdrant Cloud endpoint
- `QDRANT_API_KEY` - Qdrant API key
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_ANON_KEY` - Supabase anonymous key
- `SUPABASE_SERVICE_KEY` - Supabase service role key
- `SUPABASE_DB_URL` - Supabase PostgreSQL connection string
- `SUPABASE_STORAGE_BUCKET` - Storage bucket name (default: book-pdfs)
- `LANGSMITH_API_KEY` - LangSmith API key (for production tracing)

## Usage

### Ingest Book to Qdrant

```bash
python scripts/main_fastapi.py
# Then use the ingest endpoint (see API section below)
```

Books are automatically:
1. Uploaded to Supabase Storage (`book-pdfs` bucket)
2. Chunked and indexed in Qdrant
3. Metadata includes `storage_url` for source citations

### Run FastAPI Server (Local)

```bash
python scripts/main_fastapi.py
```

Server runs on `http://127.0.0.1:8001`

> **Note:** On Windows, ProactorEventLoop has issues with AsyncPostgresSaver, so SelectorEventLoop is used instead.

### Deploy to Modal Cloud

```bash
modal deploy scripts.main_modal
```

### Setup Checkpointer + Token Usage Table

```bash
python scripts/main_checkpointer.py
```

### Run Podcast Agent (CLI)

```bash
python scripts/run_podcast_agent.py
```

### Build Q&A Dataset from PDF

```bash
python scripts/run_dataset_builder.py --pdf "book/your_book.pdf" --output dataset.json
```

Arguments:
- `--pdf` - Path to PDF file (required)
- `--collection` - Qdrant collection name (default: "lms_content")
- `--difficulty` - Difficulty level (default: "medium")
- `--num-mcq` - Number of MCQ per section (default: 3)
- `--num-essay` - Number of essay questions per section (default: 2)
- `--output` - Output JSON file (default: "dataset.json")

## API Endpoints

### Chat with Agent (SSE Streaming)

```bash
curl -N http://localhost:8001/book-qa/stream \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_api_key" \
  -d '{"session_id":"session_1","course_id":"software_design","messages":[{"role":"user","content":"What is the basic principle of clean architecture?"}]}'
```

### Ingest Book via API

```bash
curl -X POST "http://localhost:8001/book-qa/ingest?course_id=ai_basics" \
  -H "x-api-key: your_api_key" \
  -F "file=@book.pdf"
```

### Summarize Book

```bash
# Basic summary
curl -X POST "http://localhost:8001/book-qa/summarize?course_id=software_design" \
  -H "x-api-key: your_api_key"

# With custom user prompt
curl -X POST "http://localhost:8001/book-qa/summarize?course_id=software_design&user_prompt=Fokus+ke+aspek+teknikal" \
  -H "x-api-key: your_api_key"
```

Parameters:
- `course_id` - Course identifier (required)
- `user_prompt` - Custom instructions for summary (optional)

Returns hierarchical book summary with:
- `title` - Book summary title
- `overview` - Book overview (5-6 sentences)
- `chapters` - Per-chapter summaries with key points
- `key_themes` - Main themes across the book
- `sources` - Source files with page ranges

### Edit Summary (Revise/Detail/Simplify)

```bash
curl -X POST "http://localhost:8001/book-qa/summarize/edit" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_api_key" \
  -d '{
    "course_id": "software_design",
    "title": "Ringkasan: Clean Architecture",
    "overview": "Buku ini membahas tentang...",
    "key_themes": ["Clean Code", "SOLID"],
    "chapters": [{"chapter_title": "Chapter 1", "summary": "...", "key_points": ["..."]}],
    "instruction": "Perdetail overview, tambahkan contoh praktis"
  }'
```

Parameters:
- `course_id` - Course identifier (required)
- `title` - Current summary title (required)
- `overview` - Current overview (required)
- `key_themes` - Current key themes (required)
- `chapters` - Current chapter summaries (required)
- `instruction` - Edit instructions (required)

Example instructions:
- `"Perdetail overview, tambahkan contoh aplikasi di dunia nyata"`
- `"Kurangi jadi 3 paragraph saja, lebih ringkas"`
- `"Tambah chapter baru tentang Testing"`
- `"Fokus ke aspek business, kurangi teknikal"`

### Generate Mindmap

```bash
# Basic mindmap
curl -X POST "http://localhost:8001/book-qa/mindmap?course_id=software_design" \
  -H "x-api-key: your_api_key"

# With custom user prompt
curl -X POST "http://localhost:8001/book-qa/mindmap?course_id=software_design&user_prompt=Fokus+ke+design+patterns" \
  -H "x-api-key: your_api_key"
```

Parameters:
- `course_id` - Course identifier (required)
- `user_prompt` - Custom instructions for mindmap (optional)

Returns Mermaid mindmap syntax from all ingested content.

### Edit Mindmap

```bash
curl -X POST "http://localhost:8001/book-qa/mindmap/edit" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_api_key" \
  -d '{
    "course_id": "software_design",
    "mermaid": "mindmap\n  root((Clean Architecture))\n    Controller\n    Service",
    "instruction": "Tambah sub-branch untuk Repository pattern"
  }'
```

Parameters:
- `course_id` - Course identifier (required)
- `mermaid` - Existing mermaid mindmap to edit (required)
- `instruction` - Edit instructions (required)

Example instructions:
- `"Tambah sub-branch untuk Repository pattern di Service"`
- `"Hapus branch Controller, terlalu detail"`
- `"Tambah level kedalaman untuk Business Logic"`
- `"Rename root jadi: Arsitektur Software Modern"`

### Generate Dataset (MCQ + Essay per Chapter)

```bash
curl -X POST "http://localhost:8001/book-qa/dataset?course_id=software_design&difficulty=medium&num_mcq=3&num_essay=2" \
  -H "x-api-key: your_api_key"
```

Parameters:
- `course_id` - Course identifier (required)
- `difficulty` - easy, medium, hard (default: medium)
- `num_mcq` - MCQ per chapter (default: 3)
- `num_essay` - Essay per chapter (default: 2)

Returns structured dataset with MCQ and Essay questions for each identified chapter.

### Token Usage Tracking

```bash
# Query token usage (course_id required)
curl "http://localhost:8001/token-usage?course_id=software_design" \
  -H "x-api-key: your_api_key"

# Daily aggregation
curl "http://localhost:8001/token-usage/daily?course_id=software_design&days=30" \
  -H "x-api-key: your_api_key"

# Manual flush buffer to DB
curl -X POST "http://localhost:8001/token-usage/flush" \
  -H "x-api-key: your_api_key"
```

## Testing

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest test/test_qdrant.py

# Run tests with coverage
pytest --cov=./ --cov-report=html
```

### Test Structure

- `test/conftest.py` - Shared fixtures and configuration
- `test/test_qdrant.py` - Tests for Qdrant agent
- `test/test_ingest_book.py` - Tests for book ingestion
- `test/test_tts_engine.py` - Tests for TTS engine

## Observability

### Token Usage Tracking

Token usage is automatically tracked and stored in PostgreSQL (`token_usage` table):
- **Features tracked**: agent_reasoning, search, generate_mcq, generate_essay, podcast, summarization
- **Metrics**: input/output tokens, estimated cost (USD), latency
- **Aggregation**: In-memory by feature, model, and course
- **Persistence**: Batched writes to PostgreSQL (buffer size: 10)

Query usage data via API or `TokenTracker`:
```python
from core.utils.token_tracker import TokenTracker

tracker = TokenTracker()

# Get in-memory summary
summary = tracker.get_summary()

# Query database
usage = await tracker.query_usage(course_id="my_course")
daily = await tracker.query_daily_usage(days=7)
```

### Production (LangSmith)

LangSmith is automatically configured when environment variables are set:
- `LANGSMITH_API_KEY`
- `LANGSMITH_TRACING=true`
- `LANGSMITH_PROJECT=rag-book-production`

### Benchmarking (MLflow)

MLflow is used for performance benchmarking in `benchmarking/`:
- LLM model comparison
- Embedding model comparison

## License

MIT
