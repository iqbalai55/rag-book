"""Run Podcast Agent - Interactive CLI"""
import asyncio
import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from core.rag.qdrant.qdrant_db import QdrantDB
from agents.book_podcast_agent import BookPodcastAgent

load_dotenv()


def main():
    print("=== Podcast Agent ===")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if __import__("torch").cuda.is_available() else "cpu"}
    )

    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_ENDPOINT"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    qdrant_db = QdrantDB(
        collection_name="lms_content",
        embedding_model=embedding,
        client=qdrant_client
    )

    agent = BookPodcastAgent(
        qdrant_db=qdrant_db,
        course_id="demo_course"
    )

    print("Ready! Type 'exit' to quit.\n")

    while True:
        query = input("\nQuery: ")
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        agent.ask(query)


if __name__ == "__main__":
    main()