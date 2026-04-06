import sys
import os
import json
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from docling.chunking import HybridChunker

from services.rag.qdrant.qdrant_db import QdrantDB
from services.rag.qdrant.document_processor import DocumentProcessor
from dataset.dataset_builder import BenchmarkDatasetBuilder


# =========================================================
# ENV & LOGGING
# =========================================================
load_dotenv()
logging.basicConfig(level=logging.INFO)

mlflow.set_experiment("Test Benchmark Dataset Builder - Full Pipeline")
mlflow.langchain.autolog()


# =========================================================
# CONFIG
# =========================================================
PDF_PATH       = r"book\Object-Oriented-Analysis-and-Design-with-Applications-3rd-Edition.pdf"   # <-- change
COLLECTION     = "books_collection"
STORAGE_PATH   = "./qdrant_storage"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS     = 256


# =========================================================
# TOKENIZER & CHUNKER
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

chunker = HybridChunker(tokenizer=tokenizer)

processor = DocumentProcessor(
    chunker=chunker,
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS
)


# =========================================================
# EMBEDDING MODEL
# =========================================================
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    model_kwargs={"device": "cpu"}
)


# =========================================================
# QDRANT INIT (FRESH COLLECTION)
# =========================================================
client = QdrantClient(path=STORAGE_PATH)

vector_size = len(embedding_model.embed_query("test"))

# Drop collection if exists
collections = [c.name for c in client.get_collections().collections]
if COLLECTION in collections:
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(
        size=vector_size,
        distance=Distance.COSINE,
    )
)

qdrant_db = QdrantDB(
    collection_name=COLLECTION,
    embedding_model=embedding_model,
    client=client,
)



# =========================================================
# DATASET BUILDER
# =========================================================
builder = BenchmarkDatasetBuilder(
    qdrant_db=qdrant_db,
    k=3,
)


# =========================================================
# RUN DATASET GENERATION
# =========================================================
print("\n================ BUILDING DATASET ================")

dataset = builder.build_dataset_from_book(
    pdf_path=PDF_PATH,
    difficulty="medium",
    num_mcq=2,
    num_essay=1,
)


# =========================================================
# VISUAL OUTPUT
# =========================================================
print("\n========== DATASET SUMMARY ==========")
print("Book:", dataset["book"])
print("Total Sections:", dataset["total_sections"])
print("Generated Sections:", len(dataset["sections"]))

for section in dataset["sections"]:
    print("\n--------------------------------------")
    print("Section:", section["section_title"])
    print("Sources:", section["sources"])

    print("\nMCQ:")
    print(json.dumps(section["generated"]["mcq"], indent=2, ensure_ascii=False))

    print("\nEssay:")
    print(json.dumps(section["generated"]["essay"], indent=2, ensure_ascii=False))


# =========================================================
# SAVE FILE
# =========================================================
with open("generated_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("\nDataset saved to generated_dataset.json")