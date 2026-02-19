import sys
import os
import numpy as np
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from docling.chunking import HybridChunker
import mlflow

from rag.qdrant.qdrant_db import QdrantDB  
from rag.qdrant.document_processor import DocumentProcessor 
from evaluator.relevance_grade_document_evaluator import relevance_grading_scorer

# Load environment variables from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

mlflow.set_experiment("Embedding Model Comparison")
mlflow.langchain.autolog()

DEFAULT_MAX_TOKENS = 256

DOCUMENT_PATHS = [
    r"book\Lean Software Development An Agile Toolkit (Mary Poppendieck  Tom Poppendieck) (Z-Library).pdf",
    r"book\EffectivePython.pdf",
    r"book\Sebastian Buczyński - Implementing the Clean Architecture (2020, Sebastian Buczyński) - libgen.li.pdf"
]

# Multiple questions for benchmarking
QUESTIONS = [
    "What is the most important fundamental principle in lean software development?",
    "What are the core practices of effective Python programming?",
    "How should one implement clean architecture in a software project?"
]

def run_embedding_experiment(embedding_model_name: str, max_tokens: int = DEFAULT_MAX_TOKENS):
    """
    Run a retrieval experiment using a specific embedding model and its tokenizer,
    then evaluate relevance for multiple questions using MLflow GenAI.
    """

    with mlflow.start_run(run_name=f"embedding_{embedding_model_name}"):
        mlflow.log_param("embedding_model", embedding_model_name)
        mlflow.log_param("chunk_size", max_tokens)

        # Tokenizer and chunker
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        chunker = HybridChunker(tokenizer=tokenizer)

        processor = DocumentProcessor(
            chunker=chunker,
            tokenizer=tokenizer,
            max_tokens=max_tokens
        )

        # Embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"}
        )

        # Qdrant collection
        collection_name = f"real_books_{embedding_model_name.replace('/', '_')}"
        qdrant_db = QdrantDB(
            collection_name=collection_name,
            embedding_model=embedding_model
        )

        # Process documents and add to Qdrant
        chunks = processor.process_documents(DOCUMENT_PATHS)
        qdrant_db.add_documents(chunks)

        # Build evaluation dataset
        eval_dataset = []
        for question in QUESTIONS:
            retrieved_docs = qdrant_db.query(question, k=3)
            for doc in retrieved_docs:
                eval_dataset.append({
                    "inputs": {"question": question},
                    "outputs": doc.page_content
                })

        # Evaluate all retrieved docs at once
        evaluation_result = mlflow.genai.evaluate(
            data=eval_dataset,
            scorers=[relevance_grading_scorer],
        )

        # Log all aggregated metrics from the evaluation
        print("Aggregated metrics:")
        for key, value in evaluation_result.metrics.items():
            print(f"{key}: {value:.3f}")
            mlflow.log_metric(key, value)

        # Log overall stats
        mlflow.log_metric("chunks_processed", len(chunks))
        mlflow.log_metric("retrieved_docs_total", len(eval_dataset))


if __name__ == "__main__":
    embedding_models_to_test = [
        {"name": "sentence-transformers/all-MiniLM-L6-v2", "max_tokens": 256},
        {"name": "sentence-transformers/all-MiniLM-L12-v2", "max_tokens": 256},
        {"name": "sentence-transformers/paraphrase-MiniLM-L6-v2", "max_tokens": 256},
    ]

    for emb in embedding_models_to_test:
        run_embedding_experiment(emb["name"], max_tokens=emb["max_tokens"])
