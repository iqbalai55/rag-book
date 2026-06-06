# Coding Conventions

**Analysis Date:** 2026-05-30

## Naming Patterns

**Files:**
- Python modules: `snake_case.py` (e.g., `toc_extractor.py`, `token_tracker.py`, `qdrant_db.py`)
- Test files: `test_*.py` (e.g., `test_toc_extractor.py`, `test_agent.py`)
- Special files: `__init__.py`, `conftest.py`

**Classes:**
- PascalCase (e.g., `BookQdrantAgent`, `QdrantDB`, `TokenUsageCallbackHandler`)

**Functions/Methods:**
- snake_case (e.g., `extract_pages`, `llm_detect_toc`, `get_chat_model`, `with_structured_output`)

**Variables:**
- snake_case (e.g., `book_id`, `embedding_model`, `page_content`)
- Private/internal: underscore prefix (e.g., `_invoke_with_retry`, `_detect_vector_size`)

**Constants:**
- UPPER_SNAKE_CASE (e.g., `MAX_RETRIES`, `RETRY_DELAY`, `DEFAULT_PROVIDER`)

**Type Variables:**
- PascalCase or descriptive (e.g., `List[str]`, `Optional[QdrantClient]`)

## Code Style

**Formatting:**
- Tool: Not explicitly configured; follows Python PEP 8 defaults
- Indentation: 4 spaces
- Line length: Not enforced

**Linting:**
- Configured in `pyproject.toml` dev dependencies: `black`, `isort`, `flake8`, `mypy`
- No explicit linting config files found

**Import Organization:**
1. Standard library imports (`os`, `sys`, `json`, `logging`, `pathlib`)
2. Third-party imports (`pytest`, `fastapi`, `langchain`, `pydantic`, `qdrant_client`)
3. Local/relative imports (`from core.utils...`, `from core.schemas...`, `from agents...`)

**Path Aliases:**
- Project root added to `sys.path` for absolute imports from `core.*`, `agents.*`
- Import pattern: `from core.utils.toc_extractor import ...`

## Pydantic Schemas

**Location:** `core/schemas/` directory

**Pattern:**
```python
from pydantic import BaseModel, Field
from typing import Optional, List

class TOCDetection(BaseModel):
    is_toc: bool = Field(description="True if the page contains a Table of Contents")
    thinking: str = Field(default="", description="Reasoning about detection")
```

**Conventions:**
- Use `BaseModel` from Pydantic
- Use `Field()` with `description` for documentation
- Use `Optional[]` with default values for optional fields
- Nested models for complex structures

## Logging

**Pattern:**
```python
import logging
logger = logging.getLogger(__name__)

# Usage
logger.info(f"Found {len(toc_indices)} TOC pages: {toc_indices}")
logger.warning(f"Payload index creation skipped: {e}")
```

**Levels used:** `logger.info`, `logger.warning`, `logger.debug`, `logger.error`

## Error Handling

**Patterns:**
- Return `None` on failure with fallback values (e.g., `_invoke_with_retry`)
- Raise `ValueError` with descriptive messages for configuration errors
- Use `try/except` blocks with specific exception types
- Propagate errors with context using f-strings

```python
# Configuration error
if not api_key:
    raise ValueError("OPENAI_API_KEY is required for OpenAI provider")

# Fallback pattern
result = _invoke_with_retry(structured_llm, prompt)
return result.toc_text if result else toc_raw  # fallback to raw on failure
```

## Docstrings

**Style:** Google-style docstrings with description, args, returns

```python
def extract_pages(doc) -> List[str]:
    """Extract text per page using Docling provenance data."""
    ...
```

## Function Design

**Size:** Functions tend to be focused with single responsibility

**Parameters:**
- Type hints on all parameters
- Default values where appropriate
- `Optional[]` for nullable parameters

**Return Values:**
- Explicit return types
- Return `List[]`, `Tuple[]`, `bool`, or custom objects
- Return empty collections (`[]`, `{}`, `""`) rather than `None` when appropriate

## Async Code

**Pattern:**
```python
import asyncio

# Async functions use async def
async def ask_stream(self, query: str, session_id: str):
    async for chunk in self.agent.astream(messages):
        yield chunk
```

**Testing:**
- `@pytest.mark.asyncio` decorator for async test methods
- `async for ... in` pattern for consuming async generators

## Module Design

**Exports:**
- Explicit imports: `from core.utils.toc_extractor import extract_pages, llm_detect_toc`
- No `__all__` exports detected

**Barrel Files:**
- `__init__.py` files exist in packages but minimal content
- Direct imports from specific modules preferred

## Architecture Patterns

**Layer Separation:**
- `core/schemas/` - Pydantic models for data validation
- `core/utils/` - Utility functions and helpers
- `core/services/` - Business logic services
- `core/rag/` - RAG-specific implementations (Qdrant, document processing)
- `core/storage/` - Storage integrations (Supabase)
- `core/prompts/` - Prompt templates
- `agents/` - Agent implementations

**Dependency Direction:**
- `agents` depend on `core`
- `core` is framework-agnostic where possible
- `scripts/` contains entry points (FastAPI, CLI)

---

*Convention analysis: 2026-05-30*