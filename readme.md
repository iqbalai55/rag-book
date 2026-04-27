 
# RAG Book Agent Exploration

This project explores how to build a **good RAG (Retrieval-Augmented Generation) book agent**, allowing us to ask questions using our **book knowledge database**. The goal is to create an agent that can answer queries **accurately and concisely**, while always citing the source and page from the book metadata.

Currently, we have implemented **two approaches**:

- **Qdrant-based RAG agent**
- **Qdrant-based RAG Podcast Agent**

Both approaches allow retrieval from a book vector database and generation of answers using a language model.

# Feature 

- LLM Capabillity:
  - response question with book source and its pages
  - generate multiple choice and essay question base on book knowledge database 
- Save thread/conversation to Supabase/PostgresSQL
- Vector database (Qdrant)
- Tracing LLM response with MLflow 
- Evaluator to evaluate the result (on-going)
- Benchmarking with Mflow:
    - Embedding
    - LLM model (on-going)
- Fast API and Modal.com support for deployment:
  - API key security
  - Chaching QdrantDB, QdrantClient, and Agents
  - Rate limiter

## Additional Feature 

- Agent that create podcast audio from book knowledge using indonesia languange -> still need improvement

# Tech Stack

- Supabase to store Agent state
- Langchain to create Agent
- Qdrant to store vectorize book data
- MLFlow for observability and benchmarking
- Fast API dan Modal.com for deployment
- Chatterbox for tts

## Installation

### 1️⃣ Clone Repository

```bash
git clone <repository-url>
cd ad-applicability-extractor
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements/requirements_main.txt
```

## Usage

### Ingest Book

Ingest book to Qdrant db, to do that modify and use:

```bash
python ingest_book.py
```

### Fast API deployment

Deploy using fast api:

```bash
python main_fastapi.py
```

note: there issue when use ProactorEventLoop in windows with AsyncPostgresSaver because psycopg, so i force async loop to use SelectorEventLoop.

#### Example curl

#### Chat with agent

```bash
curl -N http://localhost:8001/book-qa/stream \
  -H "Content-Type: application/json" \
  -H "x-api-key: supersecretkey123" \
  -d '{
    "session_id": "session_1",
    "collection_name": "course_lean_software",
    "messages": [
      {
        "role": "user",
        "content": "What is fundamental principle of lean development in software engineering?"
      }
    ]
  }'
```

#### Ingest book via API

```bash
# Course A
curl -X POST "http://localhost:8001/book-qa/ingest?course_id=ai_basics" \
  -H "x-api-key: supersecretkey123" \
  -F "file=@ai.pdf"

# Course B
curl -X POST "http://localhost:8001/book-qa/ingest?course_id=software_design" \
  -H "x-api-key: supersecretkey123" \
  -F "file=@design.pdf"
```

## Testing

This project uses pytest for testing. To run the tests:

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest test/test_ingest_book.py

# Run Qdrant-related tests
pytest test/test_qdrant.py

# Run tests with coverage
pytest --cov=./ --cov-report=html
```

### Test Structure

- `test/conftest.py`: Shared fixtures and configuration
- `test/test_ingest_book.py`: Tests for book ingestion API endpoint
- `test/test_qdrant.py`: Tests for Qdrant agent and related components
- Additional test files for other components

### Writing Tests

When adding new tests:
1. Follow the existing test structure and naming conventions
2. Use fixtures from `conftest.py` for common setup
3. Mock external dependencies when appropriate
4. Include both positive and negative test cases
5. Use descriptive test names that clearly indicate what is being tested

## Imporvement Plan

### General 
- More stuctured test file.

### Audio
- currently on testing tts engine using https://huggingface.co/grandhigh/Chatterbox-TTS-Indonesian, i think need more robust tts engine.
- Make more natural in bahasa indonesia.