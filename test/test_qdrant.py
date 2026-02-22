import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import json
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from docling.chunking import HybridChunker
from qdrant_client import QdrantClient
import mlflow

from agents.book_qdrant_agent import BookQdrantAgent
from rag.qdrant.qdrant_db import QdrantDB
from rag.qdrant.document_processor import DocumentProcessor

load_dotenv()
logging.basicConfig(level=logging.INFO)

mlflow.set_experiment("Test qdrant agent")
mlflow.langchain.autolog()

# CONFIG
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
STORAGE_PATH   = "./qdrant_storage"
MAX_TOKENS     = 256


tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
chunker   = HybridChunker(tokenizer=tokenizer)

processor = DocumentProcessor(
    chunker=chunker,
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS
)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    model_kwargs={"device": "cpu"}
)

client = QdrantClient(path=STORAGE_PATH)

qdrant_db = QdrantDB(
    collection_name="course_lean_software",
    embedding_model=embedding_model,
    client=client,
)


agent = BookQdrantAgent(qdrant_db=qdrant_db, k=3)


async def run_stream(label: str, query: str, session_id: str):
    print("\n" + "="*60)
    print(f"TEST: {label}")
    print("="*60)

    async for raw in agent.ask_stream(query, session_id=session_id):
        if raw.strip() == "data: [DONE]":
            print("[DONE]")
            break

        data_str = raw.removeprefix("data: ").strip()
        try:
            chunk = json.loads(data_str)
        except Exception:
            print("RAW:", raw)
            continue

        t = chunk.get("type")

        if t == "internal":
            for tc in chunk.get("metadata", {}).get("tool_calls", []):
                print(f"  [internal] tool={tc.get('name')} | args={tc.get('args')}")

        elif t == "tool":
            tool_name = chunk.get("metadata", {}).get("tool_name")
            print(f"  [tool] {tool_name}: {str(chunk.get('content', ''))[:200]}")

        elif t == "mcq":
            content = chunk.get("content", {})
            print(f"  [MCQ] topic={content.get('topic')} | {len(content.get('questions', []))} questions")
            print(json.dumps(content, indent=2, ensure_ascii=False))

        elif t == "essay":
            content = chunk.get("content", {})
            print(f"  [Essay] topic={content.get('topic')} | {len(content.get('questions', []))} questions")
            print(json.dumps(content, indent=2, ensure_ascii=False))

        elif t == "final":
            print(f"  [final] {chunk.get('content', '')}")

        elif t == "error":
            print(f"  [ERROR] {chunk.get('content', '')}")


async def main():

    # TEST 1: General QA
    await run_stream(
        label="General QA",
        query="What is the most important principle in lean software development?",
        session_id="test_session_1"
    )

    # TEST 2: Generate MCQ
    await run_stream(
        label="Generate MCQ",
        query="Generate 5 MCQs about waste elimination in lean software development with medium difficulty.",
        session_id="test_session_2"
    )

    # TEST 3: Generate Essay
    await run_stream(
        label="Generate Essay Questions",
        query="Generate 3 essay questions about code smells with medium difficulty.",
        session_id="test_session_3"
    )

    # TEST 4: Generate MCQ Bahasa Indonesia
    await run_stream(
        label="Generate MCQ - Bahasa Indonesia",
        query="Buatkan 5 soal pilihan ganda tentang agile toolkit dari buku ini dengan tingkat kesulitan hard.",
        session_id="test_session_4"
    )


asyncio.run(main())