# Codebase Structure

**Analysis Date:** 2026-04-21

## Directory Layout

```
rag-book/
├── .planning/                 # GSD planning outputs
│   └── codebase/              # Architecture analysis documents
├── agents/                    # AI agent implementations (BookQdrantAgent, etc.)
├── audio/                     # Audio processing utilities
├── benchmarking/              # Performance benchmarking scripts
├── book/                      # Book processing and storage
├── dataset/                   # Dataset generation and management
├── evaluator/                 # Model and response evaluation tools
├── prompts/                   # Prompt templates for LLMs
├── qdrant_storage/            # Qdrant vector database storage
├── requirements/              # Dependency requirements files
├── schemas/                   # Pydantic data models and schemas
├── services/                  # Service layer implementations
│   ├── audio/                 # Audio-related services (TTS)
│   └── rag/                   # RAG-specific services
│       └── qdrant/            # Qdrant database wrapper
├── test/                      # Test files
├── utils/                     # Utility functions and helpers
├── .env                       # Environment variables (not committed)
├── env.example                # Example environment template
├── ingest_book.py             # Book ingestion script
├── main.py                    # Database setup script
├── main_fastapi.py            # Main FastAPI application
└── main_modal.py              # Modal.com deployment script
```

## Directory Purposes

**agents/:**
- Purpose: Contains AI agent implementations that orchestrate LLMs, tools, and memory
- Contains: BookQdrantAgent (main RAG agent), book_podcast_agent (audio generation)
- Key files: `book_qdrant_agent.py` (primary RAG agent with tool usage)

**audio/:**
- Purpose: Audio processing and text-to-speech functionality
- Contains: TTS engine implementations
- Key files: `services/audio/tts/tts_engine.py` (TTS engine)

**benchmarking/:**
- Purpose: Performance testing and model benchmarking utilities
- Contains: LLM and embedding benchmark scripts
- Key files: `llm_model_benchmark.py`, `qdrant_embedding_benchmark.py`

**book/:**
- Purpose: Book-related processing and storage (appears to be minimally used)
- Contains: Likely book metadata or processing utilities

**dataset/:**
- Purpose: Dataset generation and management for training/evaluation
- Contains: Scripts to build datasets from various sources
- Key files: `dataset_builder.py`

**evaluator/:**
- Purpose: Model response evaluation and grading utilities
- Contains: Relevance and toxicity evaluators
- Key files: `relevance_grade_document_evaluator.py`, `toxicity_evaluator.py**

**prompts/:**
- Purpose: Centralized prompt templates for LLM interactions
- Contains: General RAG prompts, podcast generation prompts, question templates
- Key files: `general_rag.py` (main QA prompt), `podcast.py`, `MCQ_PROMPT`, `ESSAY_QUESTION_PROMPT`

**qdrant_storage/:**
- Purpose: Persistent storage for Qdrant vector database
- Contains: Actual vector collections and metadata
- Structure: `collection/real_books/` (main content collection)

**requirements/:**
- Purpose: Dependency specification files
- Contains: Various requirements.txt files for different environments

**schemas/:**
- Purpose: Pydantic models for data validation and serialization
- Contains: Request/response models, question schemas
- Key files: `chat.py` (ChatPayload, Message), `question.py` (MCQResponse, EssayResponse)

**services/:**
- Purpose: Business logic and external service integrations
- Contains: 
  - `audio/`: TTS services
  - `rag/`: RAG-specific services including Qdrant database wrapper
- Key files: 
  - `services/rag/qdrant/qdrant_db.py` (multitenant vector store)
  - `services/audio/tts/tts_engine.py` (TTS implementation)

**test/:**
- Purpose: Test suites for various components
- Contains: Unit and integration tests
- Pattern: Test files typically mirror the structure of source code

**utils/:**
- Purpose: Cross-cutting utility functions and helpers
- Contains: CacheManager, LLM configuration, helper functions
- Key files: `chace_manager.py` (resource caching), `llm_config.py` (model configuration)

## Key File Locations

**Entry Points:**
- `main.py`: Database setup script (checkpointer initialization)
- `main_fastapi.py`: Main FastAPI application with API endpoints
- `main_modal.py`: Modal.com deployment script

**Configuration:**
- `.env`: Environment variables (API keys, database URLs)
- `env.example`: Template showing required environment variables
- `requirements/*.txt`: Dependency specifications

**Core Logic:**
- `agents/book_qdrant_agent.py`: Main RAG agent with tool usage
- `services/rag/qdrant/qdrant_db.py`: Multitenant vector database wrapper
- `utils/chace_manager.py`: Resource caching and agent factory
- `utils/llm_config.py`: LLM model configuration and initialization

**Testing:**
- `test/`: Contains test files for various modules
- Pattern: Tests organized by module (e.g., test_agents/, test_services/)

## Naming Conventions

**Files:**
- Snake_case: Used for Python files (`book_qdrant_agent.py`, `chace_manager.py`)
- Descriptive names: Files clearly indicate their purpose
- Consistent prefixes: Test files follow `test_*` or `*_test.py` pattern

**Directories:**
- Snake_case: Directory names use lowercase with underscores
- Plural for collections: Directories containing multiple similar items are plural (agents, services, schemas)
- Singular for singular concepts: Directories like `dataset`, `book` for singular purposes

**Classes:**
- PascalCase: Class names follow PascalCase convention (`BookQdrantAgent`, `QdrantDB`, `CacheManager`)
- Descriptive: Class names clearly indicate their responsibility

**Variables and Functions:**
- Snake_case: Variables and functions use snake_case (`course_id`, `get_agent`, `query`)
- Descriptive: Names clearly indicate purpose and usage

**Constants:**
- UPPER_CASE: Constants use uppercase with underscores (`COLLECTION_NAME`, `EMBED_MODEL_ID`)

## Where to Add New Code

**New Feature (Primary Business Logic):**
- Primary code: `agents/` or `services/` depending on whether it's agent-like or service-like
- Tests: `test/` directory with corresponding test file
- Configuration: Update `.env` and `env.example` if new env vars needed

**New Component/Module:**
- Implementation: Create new directory under appropriate parent (`agents/`, `services/`, `utils/`)
- Interface: Define clear public interface in `__init__.py` if needed
- Documentation: Add docstrings and comments following existing patterns

**Utilities:**
- Shared helpers: `utils/` directory for cross-cutting concerns
- Service-specific helpers: Place in relevant service directory (`services/rag/utils/` if needed)
- Follow existing naming and structure patterns

## Special Directories

**.planning/:**
- Purpose: Contains GSD-generated planning documents and analysis
- Generated: Yes (by GSD commands)
- Committed: Yes (for team sharing of analysis)

**qdrant_storage/:**
- Purpose: Persistent vector database storage
- Generated: Partially (collections created dynamically, but directory structure committed)
- Committed: Yes (directory structure, but actual data files may be large)

**__pycache__/:**
- Purpose: Python bytecode cache
- Generated: Yes (by Python interpreter)
- Committed: No (explicitly ignored in .gitignore)

**test/:**
- Purpose: Test files
- Generated: No (manually created)
- Committed: Yes

---

*Structure analysis: 2026-04-21*