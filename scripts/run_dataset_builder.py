"""Run Dataset Builder - Generate Q&A dataset from PDF"""
import argparse
import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from core.rag.qdrant.qdrant_db import QdrantDB
from core.dataset.dataset_builder import BenchmarkDatasetBuilder

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Build Q&A dataset from book PDF")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--collection", default="lms_content", help="Qdrant collection name")
    parser.add_argument("--difficulty", default="medium", help="Difficulty level")
    parser.add_argument("--num-mcq", type=int, default=3, help="Number of MCQ per section")
    parser.add_argument("--num-essay", type=int, default=2, help="Number of essay questions per section")
    parser.add_argument("--k", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument("--output", default="dataset.json", help="Output JSON file")
    args = parser.parse_args()

    print(f"=== Dataset Builder ===")
    print(f"PDF: {args.pdf}")
    print(f"Collection: {args.collection}")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if __import__("torch").cuda.is_available() else "cpu"}
    )

    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_ENDPOINT"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    qdrant_db = QdrantDB(
        collection_name=args.collection,
        embedding_model=embedding,
        client=qdrant_client
    )

    builder = BenchmarkDatasetBuilder(qdrant_db=qdrant_db, k=args.k)

    dataset = builder.build_dataset_from_book(
        pdf_path=args.pdf,
        difficulty=args.difficulty,
        num_mcq=args.num_mcq,
        num_essay=args.num_essay
    )

    import json
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDataset saved to {args.output}")
    print(f"Total sections: {dataset['total_sections']}")


if __name__ == "__main__":
    main()