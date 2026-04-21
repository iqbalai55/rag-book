# Technology Stack

**Analysis Date:** 2026-04-21

## Languages

**Primary:**
- Python 3.x - Primary language for backend and data processing

**Secondary:**
- Not detected - Python-only project

## Runtime

**Environment:**
- CPython 3.x

**Package Manager:**
- pip [Version not specified]
- Lockfile: missing (requirements files exist but not lockfiles)

## Frameworks

**Core:**
- FastAPI [Version not specified] - Web API framework for building REST endpoints
- LangChain [Version not specified] - Framework for LLM application development
- LangGraph [Version not specified] - For stateful LLM applications with checkpointers
- Uvicorn [Version not specified] - ASGI server for FastAPI
- SlowAPI [Version not specified] - Rate limiting for FastAPI
- Qdrant Client [Version not specified] - Vector database client
- MLflow [Version not specified] - Experiment tracking and model management
- Supabase (via langgraph-checkpoint-postgres) [Version not specified] - PostgreSQL-based checkpointer

**Testing:**
- Not explicitly detected - No test framework configured in requirements

**Build/Dev:**
- python-dotenv [Version not specified] - Environment variable management
- Jupyter notebook [Version not specified] - Interactive development
- Rich [Version not specified] - Enhanced terminal output
- Tqdm [Version not specified] - Progress bars

## Key Dependencies

**Critical:**
- langchain - Core LLM framework enabling chatbot functionality
- langchain-huggingface - HuggingFace model integration for embeddings
- langchain-openai/anthropic - LLM provider integrations
- qdrant-client - Vector storage for document embeddings
- langgraph-checkpoint-postgres - Persistent state management for LLM agents
- fastapi - High-performance web API framework
- sentence-transformers - Text embedding models
- torch - PyTorch for ML model operations
- transformers - HuggingFace transformers library
- docling - Document parsing for various formats
- easyocr - OCR capabilities for image/text extraction
- mlflow - Experiment tracking and model registry
- modal - Cloud deployment platform (optional)
- psycopg[binary] - PostgreSQL adapter
- python-dotenv - Environment configuration

**Infrastructure:**
- Supabase/PostgreSQL - Database for LangGraph checkpointer
- Qdrant - Vector database for document storage
- MLflow tracking server - Experiment tracking
- Modal (optional) - Cloud function deployment

## Configuration

**Environment:**
- Loaded via python-dotenv from .env file
- Key configs required:
  - SUPABASE_DB_URL - PostgreSQL connection string for checkpointer
  - API_KEY - Authentication key for API endpoints
  - MLRUNS_PATH - Path for MLflow tracking data
  - STORAGE_PATH - Path for Qdrant storage (defaults to ./qdrant_storage)

**Build:**
- No build system detected - Pure Python project
- Requirements managed via requirements/Main.txt and requirements/requirements_audio.txt

## Platform Requirements

**Development:**
- Python 3.8+
- pip package manager
- GPU optional (CUDA support checked for torch)
- PostgreSQL instance (via Supabase or local)
- Qdrant instance (local or remote)

**Production:**
- Can run on any Linux/Windows/macOS with Python 3.8+
- Deployment options: Direct server, Docker, Modal cloud functions
- Requires accessible PostgreSQL and Qdrant instances

---

*Stack analysis: 2026-04-21*