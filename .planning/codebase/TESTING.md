# Testing Documentation

## Overview

This document outlines the testing practices, frameworks, and patterns used in the RAG Book project.

## Test Organization

Tests are located in the `test/` directory at the project root:
```
test/
├─ test_agent_conversation_persistance.py
├─ test_benchmark_dataset_builder.py
├─ test_chace_manager.py
├─ test_faiss.py
├─ test_ingest_book.py
├─ test_llm_utils.py
├─ test_mlflow_scorer.py
├─ test_podcast_agent.py
├─ test_podcast_tts.py
├─ test_psql.py
├─ test_qdrant.py
├─ test_tts.json
└─ __init__.py
```

## Testing Frameworks & Libraries

### Core Testing
- **pytest**: Primary testing framework (inferred from test file naming and structure)
- **unittest**: Used implicitly through pytest compatibility

### API Testing
- **requests**: For HTTP API endpoint testing
- **FastAPI TestClient**: For testing FastAPI endpoints (potential usage)

### Async Testing
- **asyncio**: For asynchronous test execution
- **pytest-asyncio**: For async test support (implied)

### ML/AI Testing
- **mlflow**: Experiment tracking and model logging
- **transformers**: For testing Hugging Face model integrations
- **langchain**: For testing LLM and agent interactions
- **sentence-transformers**: For embedding model testing

### Database Testing
- **qdrant-client**: For vector database testing
- **faiss-cpu**: For FAISS vector library testing
- **psycopg[binary]**: For PostgreSQL database testing

### Document Processing
- **docling**: For document parsing and chunking testing
- **pdfplumber**: For PDF processing testing

## Test Patterns & Practices

### API Endpoint Testing
As seen in `test_ingest_book.py`:
- Direct HTTP requests to local API endpoints
- File upload testing with multipart/form-data
- Response validation (status codes, JSON structure)
- Error handling verification

### Vector Database Testing
As seen in `test_qdrant.py`:
- Setup of embedding models and document processors
- Database connection and collection configuration
- Agent integration testing with streaming responses
- Async test patterns for long-running operations
- Metadata validation for tool calls and responses

### Benchmark Dataset Testing
As seen in `test_faiss.py` and `test_benchmark_dataset_builder.py`:
- Dataset generation from source documents
- Question/answer generation validation (MCQ, essay)
- Output format verification (JSON structure)
- File persistence testing

### Environment & Configuration
- Use of `python-dotenv` for environment variable management
- MLflow experiment tracking setup in test files
- Path manipulation for module imports (`sys.path.append`)

## Test Execution

Tests can be executed using:
```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_qdrant.py

# Run tests with verbose output
pytest -v
```

## Mocking & External Services

While not explicitly shown in current test files, recommended patterns for external service testing include:
- Using `unittest.mock` or `pytest-mock` for mocking external APIs
- Using `responses` library for HTTP request mocking
- Using `vcrpy` for recording/replaying HTTP interactions
- Using `moto` for AWS service mocking (if applicable)

## Test Data Management

- Test data is typically embedded within test files or generated dynamically
- For file-based tests, sample data should be kept in a `test_data/` directory (not currently present)
- Environment-specific configuration should use `.test.env` or similar

## Coverage & Quality

While specific coverage tools aren't evident in current tests, recommended practices include:
- Using `pytest-cov` for coverage reporting
- Setting minimum coverage thresholds
- Regular test review and maintenance
- Testing both positive and negative cases
- Validating error conditions and edge cases

## CI/CD Integration

Tests are designed to be compatible with CI/CD pipelines through:
- Standard pytest execution
- Environment variable configuration
- MLflow experiment tracking for reproducibility
- Minimal external dependencies for portability

## Recommendations for Improvement

1. Add explicit pytest configuration in `pyproject.toml` or `pytest.ini`
2. Implement test fixtures for common setup/teardown operations
3. Add property-based testing with `hypothesis` for certain components
4. Implement snapshot testing for complex output validation
5. Add performance/benchmark tests for critical paths
6. Implement test parallelization for faster execution
7. Add security scanning for test dependencies