# External Integrations

**Analysis Date:** 2026-04-21

## APIs & External Services

**LLM Providers:**
- OpenRouter - Primary LLM provider for chat completions
  - SDK/Client: langchain-openai (via OpenRouter compatibility)
  - Auth: OPENROUTER_API_KEY environment variable
- OpenAI - Alternative LLM provider
  - SDK/Client: langchain-openai
  - Auth: OPENAI_API_KEY environment variable
- Anthropic - Alternative LLM provider
  - SDK/Client: langchain-anthropic
  - Auth: ANTHROPIC_API_KEY environment variable
- HuggingFace - Embedding models and potentially LLMs
  - SDK/Client: langchain-huggingface, transformers, sentence_transformers
  - Auth: HUGGINGFACEHUB_API_TOKEN environment variable

**Data Processing & OCR:**
- Docling - Document parsing and conversion
  - SDK/Client: docling, docling-core
  - Auth: None required (local processing)
- EasyOCR - Optical Character Recognition for image text extraction
  - SDK/Client: easyocr
  - Auth: None required (local processing)

## Data Storage

**Databases:**
- Supabase PostgreSQL
  - Connection: SUPABASE_DB_URL environment variable
  - Client: psycopg[binary], langgraph-checkpoint-postgres
  - Usage: LangGraph checkpointer for persistent agent state
- Qdrant Vector Database
  - Connection: Local file path (STORAGE_PATH="./qdrant_storage") or remote via QdrantClient
  - Client: qdrant-client, langchain-qdrant
  - Usage: Vector storage for document embeddings and similarity search
  - Instance: Local (development) or remote (production via modal volumes)

**File Storage:**
- Local filesystem only - Uses ./qdrant_storage for vector database and ./tmp_* for temporary file processing
- Modal Volumes (in cloud deployment):
  - qdrant_storage_volume - Persistent Qdrant storage
  - mlflow_runs_volume - MLflow tracking data
  - hf_embedding_cache - HuggingFace embedding cache

**Caching:**
- MLflow - Experiment tracking and model management
  - Tracking URI: file:./mlruns (local) or file:/mlruns (modal)
  - Usage: Tracking LLM experiments, model parameters, and metrics
  - Client: mlflow, mlflow.langchain
- In-memory caching - CacheManager utility for Qdrant DB and agent instances
- Embedding cache - Optional HuggingFace cache via modal volumes

## Authentication & Identity

**Auth Provider:**
- Custom API Key authentication
  - Implementation: APIKeyHeader dependency in FastAPI endpoints
  - Validation: Compares provided x-api-key header with API_KEY environment variable
  - Protected endpoints: /book-qa/stream and /book-qa/ingest

## Monitoring & Observability

**Error Tracking:**
- None detected - Basic exception handling with JSON error responses

**Logs:**
- Python logging module - Standard logging throughout codebase
- MLflow autologging - Automatic tracking of LangChain/LangGraph operations
- Manual logging - Logger instances in various modules (ingest_book.py, services, etc.)

**Metrics & Tracing:**
- None detected beyond MLflow experiment tracking

## CI/CD & Deployment

**Hosting:**
- Local development - Direct Python execution
- Cloud deployment - Modal platform (optional, via main_modal.py)
- Containerization - Not detected (no Dockerfile)

**CI Pipeline:**
- None detected - No CI configuration files (.github/, .gitlab-ci.yml, jenkins, etc.)

## Environment Configuration

**Required env vars:**
- SUPABASE_DB_URL - PostgreSQL connection string for checkpointer (required)
- API_KEY - Custom API key for endpoint protection (required)
- OPENROUTER_API_KEY - Primary LLM API key (required for LLM functionality)
- HUGGINGFACEHUB_API_TOKEN - HuggingFace API key (required for embeddings/models)
- OPENAI_API_KEY - Optional OpenAI API key (alternative LLM provider)
- ANTHROPIC_API_KEY - Optional Anthropic API key (alternative LLM provider)

**Secrets location:**
- .env file - Contains all environment variables including API keys
- Modal Secrets - In cloud deployment, secrets passed via modal.Secret.from_dict()
- Never committed: .env is listed in .gitignore

## Webhooks & Callbacks

**Incoming:**
- None detected - No webhook endpoints for receiving callbacks

**Outgoing:**
- None detected - No outbound webhooks or callback registrations

---

*Integration audit: 2026-04-21*