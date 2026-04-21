# Coding Conventions

**Analysis Date:** 2026-04-21

## Naming Patterns

**Files:**
- Snake_case for Python files (e.g., `book_qdrant_agent.py`, `document_processor.py`, `llm_config.py`)
- PascalCase for class names (e.g., `BookQdrantAgent`, `QdrantDB`, `DocumentProcessor`)
- Snake_case for function and variable names (e.g., `get_chat_model`, `setup_checkpointer`, `retrieved_docs`)
- UPPER_CASE for constants (e.g., `EMBED_MODEL_ID`, `STORAGE_PATH`, `MAX_TOKENS`)
- CamelCase for TypeScript/JavaScript files when present (not detected in this Python codebase)

**Classes:**
- Descriptive names ending with their purpose: `Agent`, `DB`, `Processor`, `Config`
- Docstrings used for class descriptions (e.g., `"""Book Agent that uses QdrantDB for RAG retrieval."""`)

**Functions:**
- Verb-first naming for actions: `get_chat_model`, `setup_checkpointer`, `_retrieve_context`, `add_documents`
- Private functions prefixed with underscore: `_detect_vector_size`, `_collection_exists`, `_create_payload_indexes`
- Clear parameter names with type hints when used

**Variables:**
- Descriptive names: `qdrant_db`, `agent`, `processor`, `embedding_model`
- Collection/plural names for lists: `retrieved_docs`, `conditions`, `docs`
- Boolean-like names for flags: `enable_caching`, `tool_question_generated`

## Code Style

**Formatting:**
- Tool used: Not explicitly configured (no .prettierrc, .eslintrc, or similar config files found)
- Observed style: 4-space indentation, PEP 8 compliant
- Line length: Generally under 88 characters, some longer lines in docstrings and prompts
- Blank lines: Used to separate logical sections (imports, class definitions, method groups)

**Linting:**
- No linting configuration detected (.flake8, pylint, pylintrc, etc. not found)
- Code appears to follow PEP 8 conventions organically

## Import Organization

**Order:**
1. Standard library imports (e.g., `import asyncio`, `import os`, `import json`, `import logging`)
2. Third-party imports (e.g., `from dotenv import load_dotenv`, `from langchain.tools import tool`)
3. Local/application imports (e.g., `from utils.llm_config import get_chat_model`, `from services.rag.qdrant.qdrant_db import QdrantDB`)

**Path Aliases:**
- None detected - relative imports use explicit paths from project root
- Local imports use `from directory.module import item` pattern
- No `__init__.py` files used for package exports in some directories (empty files present)

## Error Handling

**Patterns:**
- Explicit value checking with informative error messages:
  ```python
  if not api_key:
      raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
  ```
- Try/except blocks for external service calls:
  ```python
  try:
      self.client.get_collection(name)
      return True
  except Exception:
      return False
  ```
- Logging warnings for non-fatal issues:
  ```python
  logger.warning(f"Payload index creation skipped: {e}")
  logger.warning("No documents to add.")
  ```
- Exception chaining with `logger.exception()` for unexpected errors:
  ```python
  except Exception as e:
      logger.exception("Error in ask_stream: %s", e)
  ```

## Logging

**Framework:** Python standard library `logging`

**Patterns:**
- Logger initialization: `logger = logging.getLogger(__name__)`
- Level-specific logging:
  - `logger.info()` for operational information
  - `logger.debug()` for detailed debugging info
  - `logger.warning()` for recoverable issues
  - `logger.exception()` for error stack traces
- Basic configuration in test files:
  ```python
  logging.basicConfig(level=logging.INFO)
  ```
- Structured logging with placeholders:
  ```python
  logger.debug("TYPE of retrieved_docs: %s", type(retrieved_docs))
  ```

## Comments

**When to Comment:**
- Section headers in test files (e.g., `# =============================` for visual separation)
- Explaining complex logic blocks (e.g., `# Shared retrieval logic used by all tools (multitenant-safe).`)
- Marking TODO/FIXME areas (not frequently observed)
- Explaining non-obvious business logic

**Docstrings:**
- Used for classes and public methods
- Format: Triple quotes with descriptive text
- Examples:
  ```python
  """Book Agent that uses QdrantDB for RAG retrieval."""
  ```
  ```python
  """Multitenant Qdrant wrapper using payload-based filtering."""
  ```
  ```python
  """
  Add documents with enforced multitenancy metadata.
  
  Args:
      chunks: list of {"text": ..., "metadata": {...}}
      course_id: optional global course_id to inject
  """
  ```

## Function Design

**Size:** Functions vary in size:
- Small helper functions: 5-15 lines (e.g., `_detect_vector_size`, `_collection_exists`)
- Medium functions: 20-40 lines (e.g., `get_chat_model`, `query`)
- Larger functions: 50+ lines for complex orchestration (e.g., `ask_stream` with 87 lines)

**Parameters:**
- Type hints used inconsistently but present in key functions
- Default parameters for configuration (e.g., `provider: str = DEFAULT_PROVIDER`)
- Optional parameters marked with `Optional[T]` from typing module
- Configuration objects passed as parameters rather than hardcoded

**Return Values:**
- Explicit return type hints when used (`-> bool`, `-> List[Document]`)
- Clear documentation of return values in docstrings
- Consistent use of early returns for error conditions
- Tuple returns for multiple related values (e.g., `Tuple[str, List[Document], List[str]]`)

## Module Design

**Exports:**
- `__init__.py` files present but mostly empty in packages
- No explicit `__all__` lists for controlling exports
- Classes and functions imported directly from modules when needed

**Barrel Files:**
- No barrel files (index.py) detected for re-exporting module contents
- Direct imports from specific module paths preferred

---
*Convention analysis: 2026-04-21*