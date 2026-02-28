import sys
import os
from dotenv import load_dotenv
import mlflow

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.qdrant.qdrant_db import QdrantDB
from rag.qdrant.document_processor import DocumentProcessor
from utils.llm_config import get_chat_model

from transformers import AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from docling.chunking import HybridChunker

from mlflow.genai.scorers import CorrectnessScorer


load_dotenv()
mlflow.set_experiment("LLM Model Correctness Comparison")
mlflow.langchain.autolog()


DOCUMENT_PATHS = [
    r"book\Lean Software Development An Agile Toolkit (Mary Poppendieck  Tom Poppendieck) (Z-Library).pdf",
    r"book\EffectivePython.pdf",
    r"book\Sebastian Buczyński - Implementing the Clean Architecture (2020, Sebastian Buczyński) - libgen.li.pdf"
]

QUESTIONS = [
    "What is the most important fundamental principle in lean software development?",
    "What are the core practices of effective Python programming?",
    "How should one implement clean architecture in a software project?"
]

# 🔥 Add reference answers manually (gold standard)
REFERENCE_ANSWERS = [
    "The most fundamental principle in lean software development is eliminating waste and maximizing value to the customer.",
    "Effective Python emphasizes readability, simplicity, Pythonic idioms, proper use of standard libraries, and writing maintainable code.",
    "Clean architecture should be implemented by separating concerns into layers, keeping business logic independent from frameworks, and enforcing dependency inversion."
]

DEFAULT_MAX_TOKENS = 256


def prepare_retriever():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    chunker = HybridChunker(tokenizer=tokenizer)

    processor = DocumentProcessor(
        chunker=chunker,
        tokenizer=tokenizer,
        max_tokens=DEFAULT_MAX_TOKENS
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    qdrant_db = QdrantDB(
        collection_name="llm_benchmark_collection",
        embedding_model=embedding_model
    )

    chunks = processor.process_documents(DOCUMENT_PATHS)
    qdrant_db.add_documents(chunks)

    return qdrant_db


def run_llm_experiment(provider: str, model_name: str):
    with mlflow.start_run(run_name=f"{provider}_{model_name}"):

        mlflow.log_param("provider", provider)
        mlflow.log_param("model", model_name)

        llm = get_chat_model(provider=provider, model=model_name)
        qdrant_db = prepare_retriever()

        eval_dataset = []

        for question, reference in zip(QUESTIONS, REFERENCE_ANSWERS):
            retrieved_docs = qdrant_db.query(question, k=3)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            prompt = f"""
            Answer the question using ONLY the context below.

            Context:
            {context}

            Question:
            {question}
            """

            response = llm.invoke(prompt)

            eval_dataset.append({
                "inputs": {"question": question},
                "outputs": response.content,
                "targets": reference,   # 🔥 Needed for correctness
            })

        evaluation_result = mlflow.genai.evaluate(
            data=eval_dataset,
            scorers=[CorrectnessScorer()],
        )

        print("\nAggregated metrics:")
        for key, value in evaluation_result.metrics.items():
            print(f"{key}: {value:.3f}")
            mlflow.log_metric(key, value)

        mlflow.log_metric("total_questions", len(QUESTIONS))



if __name__ == "__main__":

    llm_models_to_test = [
        {"provider": "openai", "model": "gpt-5-nano"},
        {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
        {"provider": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct"},
        {"provider": "openrouter", "model": "mistralai/mistral-7b-instruct"},
        {"provider": "openrouter", "model": "google/gemma-2-9b-it"},
    ]

    for config in llm_models_to_test:
        run_llm_experiment(
            provider=config["provider"],
            model_name=config["model"]
        )