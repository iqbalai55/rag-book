import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow

from rag.qdrant.qdrant_db import QdrantDB
from agents.benchmark_dataset_builder import BenchmarkDatasetBuilder


mlflow.set_experiment("Test Benchmark Dataset Builder")
mlflow.langchain.autolog()


qdrant_db = QdrantDB(
    collection_name="books_collection",  # adjust if needed
)


builder = BenchmarkDatasetBuilder(
    qdrant_db=qdrant_db,
    k=3,
)


pdf_path = "data/sample_book.pdf"  

dataset = builder.build_dataset_from_book(
    pdf_path=pdf_path,
    difficulty="medium",
    num_mcq=2,
    num_essay=1,
)


print("\n========== DATASET SUMMARY ==========")
print("Book:", dataset["book"])
print("Total Sections:", dataset["total_sections"])
print("Generated Sections:", len(dataset["sections"]))

for section in dataset["sections"]:
    print("\n--------------------------------------")
    print("Section:", section["section_title"])
    print("Sources:", section["sources"])

    print("\nMCQ:")
    print(json.dumps(section["generated"]["mcq"], indent=2))

    print("\nEssay:")
    print(json.dumps(section["generated"]["essay"], indent=2))


# Optional: save to file
with open("generated_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("\nDataset saved to generated_dataset.json")