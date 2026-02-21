import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.chace_manager import CacheManager
from langgraph.checkpoint.memory import InMemorySaver

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
STORAGE_PATH = "./qdrant_storage"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    model_kwargs={"device": "cpu"}
)

checkpointer = InMemorySaver()

async def test_cache_manager_and_stream():
    cache = CacheManager(STORAGE_PATH, embedding_model=embedding_model)
    await cache.initialize(checkpointer)
    
    print("waiting get agent ....")

    agent = await cache.get_agent("test_collection")

    print("Agent ready:", agent)

    query = "What is lean software development?"
    print(f"Querying agent: {query}")

    async for chunk in agent.ask_stream(query, session_id="test_session"):
        print("Received chunk:", chunk)

# Run the test
asyncio.run(test_cache_manager_and_stream())