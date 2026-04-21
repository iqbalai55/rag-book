# Testing Patterns

**Analysis Date:** 2026-04-21

## Test Framework

**Runner:**
- Framework: pytest (inferred from test file naming and structure)
- Config: No explicit pytest configuration file detected (pytest.ini, pyproject.toml, setup.cfg not found)
- Test discovery: Files matching `test_*.py` pattern in `test/` directory

**Assertion Library:**
- Built-in Python `assert` statements (no external assertion library like pytest-assert detected)

**Run Commands:**
```bash
python -m pytest                 # Run all tests
python -m pytest test_file.py    # Run specific test file
python -m pytest -v              # Verbose output
```

## Test File Organization

**Location:**
- Separate `test/` directory at project root
- Tests organized by component/functionality being tested

**Naming:**
- Pattern: `test_[component].py` (e.g., `test_qdrant.py`, `test_podcast_agent.py`)
- Test classes: Not used (functions only)
- Test functions: `test_[description]` pattern not strictly followed; some use descriptive function names

**Structure:**
```
test/
├── test_podcast_agent.py
├── test_psql.py
├── test_podcast_tts.py
├── test_ingest_book.py
├── test_benchmark_dataset_builder.py
├── test_agent_conversation_persistance.py
├── test_faiss.py
├── test_qdrant.py
├── test_chace_manager.py
├── test_llm_utils.py
├── test_mlflow_scorer.py
└── __init__.py
```

## Test Structure

**Suite Organization:**
- No test classes - uses procedural style with functions
- Setup/teardown via module-level code and function-scoped setup
- Test functions contain both setup and assertions

**Patterns:**
- Module-level imports and configuration:
  ```python
  import sys
  import os
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
  
  import asyncio
  import json
  import logging
  from dotenv import load_dotenv
  # ... other imports
  
  # =============================
  # ENV & LOGGING
  # =============================
  
  load_dotenv()
  logging.basicConfig(level=logging.INFO)
  ```
- Configuration constants defined at module level
- Async test functions using `asyncio.run()` or pytest-asyncio (inferred)
- Test execution via `if __name__ == "__main__":` block calling `asyncio.run(main())`

## Mocking

**Framework:** 
- No explicit mocking framework detected (no unittest.mock or pytest.mock usage observed)
- Direct instantiation of dependencies with test configurations

**Patterns:**
- Manual dependency injection for testability:
  ```python
  # In test_qdrant.py
  client = QdrantClient(path=STORAGE_PATH)
  
  qdrant_db = QdrantDB(
      collection_name="course_lean_software",
      embedding_model=embedding_model,
      client=client,
  )
  
  agent = BookQdrantAgent(qdrant_db=qdrant_db, k=3)
  ```
- Test doubles created by configuring real implementations with test parameters
- External services mocked via configuration (e.g., using `:memory:` database, local file paths)

**What to Mock:**
- External API calls (inferred from patterns)
- Database connections (using test/in-memory instances)
- File system operations (using temporary directories)

**What NOT to Mock:**
- Core business logic (tested directly)
- Pure functions with deterministic outputs
- Configuration validation

## Fixtures and Factories

**Test Data:**
- Hardcoded test values in test files:
  ```python
  EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
  STORAGE_PATH   = "./qdrant_storage"
  MAX_TOKENS     = 256
  ```
- Test queries and scenarios defined in test functions:
  ```python
  query="What is the most important principle in lean software development?"
  ```

**Location:**
- Test data defined directly in test files
- No separate fixtures directory or factory modules observed

## Coverage

**Requirements:** 
- No coverage requirements detected (no coverage configuration files)

**View Coverage:**
- No coverage commands observed in codebase
- Would need to install coverage tool separately:
  ```bash
  pip install coverage
  coverage run -m pytest
  coverage report
  ```

## Test Types

**Unit Tests:**
- Focus on individual components and functions
- Test both success and error conditions
- Examples: testing LLM configuration, document processing, database operations

**Integration Tests:**
- Test component interactions (e.g., agent with database)
- Use real implementations with test configurations
- Examples: Qdrant agent integration tests with actual database operations

**E2E Tests:**
- Not explicitly separated; some tests cover full workflows
- Examples: end-to-end agent query processing in `test_qdrant.py`

## Common Patterns

**Async Testing:**
```python
async def run_stream(label: str, query: str, session_id: str):
    print("\n" + "=" * 70)
    print(f"TEST: {label}")
    print("=" * 70)

    async for raw in agent.ask_stream(query, session_id=session_id):
        # Process stream chunks
        # ...

if __name__ == "__main__":
    asyncio.run(main())
```

**Error Testing:**
- Tests validate error handling by checking for specific error conditions
- Error scenarios tested via invalid inputs or missing configurations
- Example: testing agent behavior when context is not found:
  ```python
  if not context:
      return {"error": "Tidak ditemukan konteks relevan dari buku."}
  ```

**Resource Cleanup:**
- Limited explicit cleanup observed
- Some tests rely on process termination for resource cleanup
- Database cleanup seen in some implementations:
  ```python
  def delete_by_course(self, course_id: str):
      """Delete all data for a course."""
      # ...
  ```

---
*Testing analysis: 2026-04-21*