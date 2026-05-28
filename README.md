# RAG Book Agent

Multi-tenant RAG system for querying book content with source citations. Ask questions, generate MCQs, essay questions, and podcasts from your book knowledge base.

## Features

- **LLM Capabilities**
  - Answer questions with book sources and page citations
  - Generate multiple choice and essay questions from book content
  - Generate podcast audio from book topics (Indonesian language)

- **Persistence & Caching**
  - Save conversation threads to Supabase/PostgreSQL
  - Cache QdrantDB, QdrantClient, and Agents

- **Supabase Integration**
  - PDF storage in Supabase Storage (book-pdfs bucket)
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

## Tech Stack

- **LangChain/LangGraph** - Agent framework
- **Qdrant** - Vector database
- **Supabase** - Storage, PostgreSQL, Auth
- **LangSmith** - Production observability
- **MLflow** - Benchmarking
- **FastAPI** - Local API server
- **Modal.com** - Cloud deployment
- **Chatterbox TTS** - Text-to-speech

## Project Structure

```
rag-book/
├── scripts/                   # Entry points
│   ├── main_fastapi.py        # FastAPI server (local)
│   ├── main_modal.py          # Modal deployment (cloud)
│   ├── main_checkpointer.py   # Postgres checkpointer setup
│   ├── run_podcast_agent.py   # Podcast agent CLI
│   └── run_dataset_builder.py # Dataset builder CLI
├── agents/                     # AI agents
├── core/                       # Core domain
│   ├── dataset/              # Dataset utilities
│   ├── evaluator/            # Evaluators & scorers
│   ├── prompts/              # Prompt templates
│   ├── rag/                  # RAG services (Qdrant)
│   ├── schemas/              # Pydantic models
│   ├── storage/              # Supabase Storage
│   ├── tts/                  # Text-to-speech
│   └── utils/                # Utilities (token tracking, etc.)
├── benchmarking/              # Performance benchmarks
├── test/                      # Test suite
├── book/                      # PDF sources
├── scripts/migrations/        # SQL migrations
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
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `OPENROUTER_API_KEY` - OpenRouter API key
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

### Setup Checkpointer

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

### Chat with Agent

```bash
curl -N http://localhost:8001/book-qa/stream \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_api_key" \
  -d '{"session_id":"session_1","course_id":"software_design","messages":[{"role":"user","content":"What is the basic principle of clean architecture?"}]}'
```

### Ingest Book via API

```bash
# Course A
curl -X POST "http://localhost:8001/book-qa/ingest?course_id=ai_basics" \
  -H "x-api-key: your_api_key" \
  -F "file=@book.pdf"
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
- **Features tracked**: agent_reasoning, search, generate_mcq, generate_essay, podcast
- **Metrics**: input/output tokens, estimated cost (USD), latency
- **Aggregation**: In-memory by feature, model, and course
- **Persistence**: Batched writes to PostgreSQL (buffer size: 10)

Query usage data via `TokenTracker`:
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