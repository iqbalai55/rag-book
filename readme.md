
# RAG Book Agent Exploration

This project explores how to build a **good RAG (Retrieval-Augmented Generation) book agent**, allowing us to ask questions using our **book knowledge database**. The goal is to create an agent that can answer queries **accurately and concisely**, while always citing the source and page from the book metadata.

Currently, we have implemented **two approaches**:

- **Qdrant-based RAG agent**
- **FAISS-based RAG agent**

Both approaches allow retrieval from a book vector database and generation of answers using a language model.

# Feature 

- Save thread/conversation to Supabase/PostgresSQL
- Vector database (Qdrant and FAISS)
- Tracing LLM response with MLflow 
- Evaluator to evaluate the result (on-going)
- Benchmarking with Mflow:
    - Embedding
    - LLM model (on-going)
- Fast API support for deployment
- Modal.com support for deployment (on-going)

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
uvicorn main_fastapi:app --reload
```

Example curl:

Chat with agent: 

```bash
curl -N http://localhost:8000/book-qa/stream -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What is fundamental principle of lean developent on software engineering?\"}],\"session_id\":\"session_1\"}"
```

Ingest book via API:

```bash
curl -X POST "http://localhost:8000/book-qa/ingest" -F "file=@\"poa.pdf\""
```

