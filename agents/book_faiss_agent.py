# agents/book_agent.py

from langchain_core.prompts import PromptTemplate
from utils.llm_config import get_chat_model
from rag.retrieve_faiss import retrieve_context, load_vector_db


BOOK_QA_PROMPT = PromptTemplate(
    template="""
You are an expert assistant answering questions based ONLY on the provided book context.

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


class BookFaissAgent:
    def __init__(self, faiss_path: str, k: int = 3):
        self.llm = get_chat_model()
        self.vectordb = load_vector_db(faiss_path)
        self.k = k

        self.chain = BOOK_QA_PROMPT | self.llm

    def ask(self, question: str) -> dict:
        """
        Perform RAG + LLM generation
        """
        try:
            # 🔹 Retrieve context using rag.py
            context, pages = retrieve_context(
                self.vectordb,
                question,
                k=self.k
            )
            
            #print(pages)
            #print(context)

            # 🔹 Call LLM
            response = self.chain.invoke({
                "context": context,
                "question": question
            })

            return {
                "answer": response.content,
                "pages": pages
            }

        except Exception as e:
            raise RuntimeError(f"BookAgent failed: {str(e)}") from e
