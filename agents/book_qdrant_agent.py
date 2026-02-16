import logging
from typing import Tuple, List
from langchain_core.prompts import PromptTemplate
from utils.llm_config import get_chat_model
from langchain_core.documents import Document
from rag.qdrant.qdrant_db import QdrantDB  

logger = logging.getLogger(__name__)

BOOK_QA_PROMPT = PromptTemplate(
    template="""
You are an expert assistant answering questions based ONLY on the provided book context and you use english.

Rules:
1. Do NOT make up answers.
2. If the answer is not in the context, say: "I don't know based on the provided context."
3. Answer in the same language as the question.
4. Be concise and accurate.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)


class BookQdrantAgent:
    """Book Agent that uses QdrantDB for RAG retrieval."""

    def __init__(self, qdrant_db: QdrantDB, k: int = 3):
        """
        Args:
            qdrant_db (QdrantDB): Your QdrantDB wrapper instance
            k (int): Number of top chunks to retrieve
        """
        self.llm = get_chat_model()
        self.qdrant_db = qdrant_db
        self.k = k
        self.chain = BOOK_QA_PROMPT | self.llm

    def retrieve_context(self, question: str) -> Tuple[str, List[dict]]:
        """Retrieve top-k chunks from QdrantDB and combine into context string."""
        
        results: List[Document] = self.qdrant_db.query(question, k=self.k)

        context_texts = []
        all_metadata = []

        for doc in results:
            context_texts.append(doc.page_content)
            all_metadata.append(doc.metadata)  # ambil seluruh metadata

        combined_context = "\n\n".join(context_texts)

        print("Retrieved metadata:", all_metadata)
        print("Combined context:", combined_context)

        return combined_context, all_metadata


    def ask(self, question: str) -> dict:
        """Perform RAG + LLM QA using QdrantDB."""
        try:
            context, metadata = self.retrieve_context(question)

            if not context.strip():
                return {
                    "answer": "I don't know based on the provided context.",
                    "metadata": []
                }

            response = self.chain.invoke({
                "context": context,
                "question": question
            })

            return {
                "answer": response.content,
                "metadata": metadata
            }

        except Exception as e:
            raise RuntimeError(f"BookQdrantAgent failed: {str(e)}") from e

