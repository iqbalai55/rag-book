# Code Conventions Documentation

## Overview

This document outlines the coding conventions, patterns, and practices used in the RAG Book project.

## Language & Version

**Primary Language**: Python 3.x

**Version Requirements**: 
- Python >= 3.8 (inferred from usage of walrus operator, f-strings, and other modern features)
- Specific versions managed through requirements files

## File Organization

### Directory Structure
```
rag-book/
├─ agents/              # AI agent implementations
├─ audio/               # Audio processing components
├─ benchmarking/        # Performance benchmarking tools
├─ book/                # Source books/documents
├─ dataset/             # Dataset generation utilities
├─ evaluator/           # Model/output evaluation components
├─ ingest_book.py       # Main ingestion script
├─ main.py              # Original main entry point
├─ main_fastapi.py      # FastAPI server implementation
├─ main_modal.py        # Modal.com deployment version
├─ prompts/             # Prompt templates for LLMs
├─ requirements/        # Dependency specification files
├─ schemas/             # Data models and validation schemas
├─ services/            # Core business logic and integrations
├─ test/                # Test suite
├─ utils/               # Utility functions and helpers
└─ qdrant_storage/      # Vector database storage (generated)
```

### Naming Conventions
- **Modules & Packages**: snake_case (e.g., `book_qdrant_agent.py`)
- **Classes**: PascalCase (e.g., `BookQdrantAgent`, `QdrantDB`)
- **Functions & Methods**: snake_case (e.g., `process_document`, `generate_questions`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_TOKENS`, `EMBED_MODEL_ID`)
- **Variables**: snake_case (e.g., `client`, `embedding_model`)
- **Files**: snake_case with descriptive names (e.g., `test_ingest_book.py`)

## Import Conventions

### Standard Library First
```python
import sys
import os
import json
import asyncio
import logging
```

### Third-Party Libraries
```python
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer
```

### Local Application Imports
```python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.book_qdrant_agent import BookQdrantAgent
from services.rag.qdrant.qdrant_db import QdrantDB
from services.rag.qdrant.document_processor import DocumentProcessor
```

## Code Formatting

### Indentation
- 4 spaces per indentation level
- No tabs used for indentation

### Line Length
- Maximum 88 characters (Black formatter default)
- Maximum 100 characters allowed in some cases for readability

### Blank Lines
- 2 blank lines between top-level function and class definitions
- 1 blank line between method definitions within a class
- 1 blank line between logical sections within a function

### Whitespace
- No trailing whitespace
- Single space after commas in function calls and lists
- Spaces around operators (`=`, `+`, `-`, etc.)
- No spaces immediately inside parentheses, brackets, or braces

## Type Hints

Type hints are used selectively in the codebase:

```python
from typing import List, Dict, Optional, Any

def process_document(text: str, max_tokens: int = 256) -> List[str]:
    """Process document into chunks."""
    pass

class BookQdrantAgent:
    def __init__(self, qdrant_db: QdrantDB, k: int = 3) -> None:
        self.qdrant_db = qdrant_db
        self.k = k
    
    async def ask_stream(self, query: str, session_id: str) -> AsyncGenerator[str, None]:
        pass
```

## Documentation Style

### Docstrings
- Triple double quotes (`"""`)
- Follow Google or NumPy style conventions
- Include parameter types and return types
- Describe exceptions when applicable

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    Example function demonstrating docstring conventions.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter (default: 10)
        
    Returns:
        bool: Description of return value
        
    Raises:
        ValueError: When param1 is invalid
    """
    pass
```

### Comments
- Use complete sentences
- Start with capital letter
- End with period
- Explain why, not what
- Keep comments up-to-date with code changes

## Error Handling Patterns

### Exception Handling
```python
try:
    # Risky operation
    result = risky_operation()
except SpecificException as e:
    # Handle specific exception
    logger.error(f"Specific error occurred: {e}")
    raise  # Re-raise if needed
except Exception as e:
    # Handle unexpected exceptions
    logger.error(f"Unexpected error: {e}")
    raise CustomError("Operation failed") from e
else:
    # Execute if no exception
    return result
finally:
    # Always execute
    cleanup_resources()
```

### Custom Exceptions
Custom exceptions are defined when needed:
```python
class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
    pass
```

### Logging
- Uses Python's built-in `logging` module
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Configured with `logging.basicConfig(level=logging.INFO)`
- Logger names typically match module names

## Configuration Management

### Environment Variables
- Loaded using `python-dotenv` (`load_dotenv()`)
- Stored in `.env` file (not committed)
- Example variables: API keys, database URLs, model paths

### Configuration Files
- YAML or JSON files for complex configurations
- Located in `config/` directory (not currently present, but pattern observed)
- Hardcoded constants moved to configuration where appropriate

## Testing Conventions

### Test File Naming
- Prefix: `test_`
- Suffix: `_test.py` or descriptive name
- Matches module being tested (e.g., `test_qdrant.py` tests Qdrant functionality)

### Test Function Naming
- Descriptive names starting with `test_`
- Use underscores to separate words
- Clearly indicate what is being tested

### Test Organization
- Arrange-Act-Assign (AAA) pattern
- Clear separation of setup, execution, and verification
- Use of fixtures for repeated setup (when using pytest)

## Security Practices

### Secret Management
- No hardcoded secrets or API keys
- All sensitive data via environment variables
- `.env` file included in `.gitignore`

### Input Validation
- Validation of file paths and user inputs
- Use of schemas for data validation (seen in `schemas/` directory)
- Sanitization of inputs where applicable

## Dependency Management

### Requirements Files
- Separate requirements files for different components:
  - `requirements/requirements_main.txt` - Core dependencies
  - `requirements/requirements_audio.txt` - Audio-specific dependencies
- Pinned versions where critical for compatibility
- Comments explaining purpose of dependency groups

### Virtual Environments
- Recommended use of virtual environments (`venv`, `conda`)
- Environment specification via `requirements.txt` or `environment.yml`

## Performance Considerations

### Lazy Loading
- Heavy imports inside functions when not always needed
- Conditional imports for optional dependencies

### Caching
- Use of `@lru_cache` for expensive computations
- MLflow caching for model artifacts
- Qdrant connection reuse

### Async/Await
- Used for I/O-bound operations (API calls, database queries)
- Proper use of `asyncio.gather()` for concurrent operations
- Avoid blocking calls in async functions

## Code Patterns Observed

### Factory Pattern
- Classes with factory methods for object creation
- Dependency injection through constructors

### Strategy Pattern
- Different algorithms selected at runtime
- Configuration-driven behavior selection

### Observer Pattern
- Event-driven architectures in agent systems
- Callback mechanisms for streaming responses

### Repository Pattern
- Data access abstraction (seen in QdrantDB)
- Separation of concerns between business logic and data access

## Tools & Formatting

### Recommended Formatters
- **Black**: For code formatting
- **isort**: For import sorting
- **flake8**: For linting
- **mypy**: For type checking

### Pre-commit Hooks
- Not currently implemented but recommended
- Would run formatting and linting before commits

## Documentation Generation

### Docstring Standards
- Consistent docstring style across codebase
- Tools like Sphinx or MkDocs could be used for documentation generation
- API documentation extractable from docstrings

## Areas for Improvement

1. Add explicit configuration management (pydantic-settings, dynaconf)
2. Implement more comprehensive type hinting
3. Add formal code style configuration (pyproject.toml with black/isort settings)
4. Add pre-commit hooks for automated code quality checks
5. Implement more robust error handling and logging standardization
6. Add module-level docstrings for all files
7. Standardize on a single docstring style (Google vs NumPy)
8. Add more constants and configuration files to reduce hardcoded values