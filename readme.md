
# RAG Book Agent Exploration

This project explores how to build a **good RAG (Retrieval-Augmented Generation) book agent**, allowing us to ask questions using our **book knowledge database**. The goal is to create an agent that can answer queries **accurately and concisely**, while always citing the source and page from the book metadata.

Currently, we have implemented **two approaches**:

- **Qdrant-based RAG agent**
- **FAISS-based RAG agent**

Both approaches allow retrieval from a book vector database and generation of answers using a language model.

# Feature 

- LLM Capabillity:
  - response question with book source and its pages
  - generate multiple choice and essay question base on book knowledge database 
- Save thread/conversation to Supabase/PostgresSQL
- Vector database (Qdrant and FAISS)
- Tracing LLM response with MLflow 
- Evaluator to evaluate the result (on-going)
- Benchmarking with Mflow:
    - Embedding
    - LLM model (on-going)
- Fast API and Modal.com support for deployment:
  - API key security
  - Chaching QdrantDB, QdrantClient, and Agents

# Tech Stack

- Supabase to store Agent state
- Langchain to create Agent
- Qdrant and Faiss to store vectorize book data
- MLFlow for observability and benchmarking
- Fast API dan Modal.com for deployment

## Installation

### 1️⃣ Clone Repository

```bash
git clone <repository-url>
cd ad-applicability-extractor
````

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
pip install -r requirements.txt
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
curl -X POST "http://localhost:8001/book-qa/ingest?collection_name=course_lean_software" \
  -H "x-api-key: supersecretkey123" \
  -F "file=@poa.pdf"
```