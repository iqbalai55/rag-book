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
import mlflow.langchain

from agents.book_qdrant_agent import BookQdrantAgent
from rag.qdrant.qdrant_db import QdrantDB
from rag.qdrant.document_processor import DocumentProcessor

# =============================
# ENV & LOGGING
# =============================

load_dotenv()
logging.basicConfig(level=logging.INFO)

mlflow.set_experiment("Test qdrant agent")
mlflow.langchain.autolog()

# =============================
# CONFIG
# =============================

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
STORAGE_PATH   = "./qdrant_storage"
MAX_TOKENS     = 256

# =============================
# TOKENIZER & CHUNKER
# =============================

tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
chunker   = HybridChunker(tokenizer=tokenizer)

processor = DocumentProcessor(
    chunker=chunker,
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS
)

# =============================
# EMBEDDING MODEL
# =============================

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    model_kwargs={"device": "cpu"}
)

# =============================
# QDRANT
# =============================

client = QdrantClient(path=STORAGE_PATH)

qdrant_db = QdrantDB(
    collection_name="course_lean_software",
    embedding_model=embedding_model,
    client=client,
)

# =============================
# AGENT
# =============================

agent = BookQdrantAgent(qdrant_db=qdrant_db, k=3)


# =============================
# STREAM PARSER
# =============================

async def run_stream(label: str, query: str, session_id: str):
    print("\n" + "=" * 70)
    print(f"TEST: {label}")
    print("=" * 70)

    async for raw in agent.ask_stream(query, session_id=session_id):

        # End of stream
        if raw.strip() == "data: [DONE]":
            print("\n[DONE]")
            break

        # Remove prefix
        data_str = raw.removeprefix("data: ").strip()

        try:
            chunk = json.loads(data_str)
        except Exception:
            print("RAW:", raw)
            continue

        msg_type = chunk.get("type")

        # =============================
        # INTERNAL TOOL PLANNING
        # =============================
        if msg_type == "internal":
            print("\n[INTERNAL TOOL CALL]")
            for tc in chunk.get("metadata", {}).get("tool_calls", []):
                print(f"  tool={tc.get('name')} | args={tc.get('args')}")

        # =============================
        # GENERIC TOOL OUTPUT
        # =============================
        elif msg_type == "tool":
            tool_name = chunk.get("metadata", {}).get("tool_name")
            print(f"\n[TOOL OUTPUT] {tool_name}")
            print(str(chunk.get("content", ""))[:300])

        # =============================
        # MULTIPLE CHOICE
        # =============================
        elif msg_type == "multiple_choice_question":
            content = chunk.get("content", {})

            print("\n[MULTIPLE CHOICE QUESTIONS]")
            print(f"Topic      : {content.get('topic')}")
            print(f"Difficulty : {content.get('difficulty')}")
            print(f"Total      : {len(content.get('questions', []))}")
            print("-" * 50)

            print(json.dumps(content, indent=2, ensure_ascii=False))

        # =============================
        # ESSAY QUESTIONS
        # =============================
        elif msg_type == "essay_question":
            content = chunk.get("content", {})

            print("\n[ESSAY QUESTIONS]")
            print(f"Topic      : {content.get('topic')}")
            print(f"Difficulty : {content.get('difficulty')}")
            print(f"Total      : {len(content.get('questions', []))}")
            print("-" * 50)

            print(json.dumps(content, indent=2, ensure_ascii=False))

        # =============================
        # FINAL ANSWER
        # =============================
        elif msg_type == "final":
            print("\n[FINAL ANSWER]")
            print(chunk.get("content", ""))

        # =============================
        # ERROR
        # =============================
        elif msg_type == "error":
            print("\n[ERROR]")
            print(chunk.get("content", ""))


# =============================
# MAIN TEST
# =============================

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


if __name__ == "__main__":
    asyncio.run(main())