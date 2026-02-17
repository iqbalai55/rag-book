from utils.llm_config import get_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import SystemMessage
from rag.faiss.retrieve_faiss import load_vector_db
from typing import List, Tuple

BOOK_QA_PROMPT = SystemMessage(content="""
You are an expert assistant answering questions based ONLY on the provided book context retrieved using the search_book tool. 

Rules:
1. Do NOT make up answers.
2. If the answer is not in the context, say: "I don't know based on the provided context."
3. Always include the source and page numbers from the chunk metadata of the retrieved content. Use the 'source' field as the book title and the 'pages' field as the page numbers.
4. Answer in the same language as the question.
5. Be concise and accurate.

Format example:
Answer: <your answer here>
Source: <source from metadata>, Pages <pages from metadata>
""")

class BookFaissAgent:
    def __init__(self, faiss_path: str, k: int = 3):
        self.llm = get_chat_model()
        self.vectordb = load_vector_db(faiss_path)
        self.k = k
        self.checkpointer = InMemorySaver()

        # Define tool as nested function to capture self via closure
        @tool("search_book", description="Search relevant book", response_format="content_and_artifact")
        def search_book(query: str):
            retriever = self.vectordb.as_retriever(search_kwargs={"k": self.k})
            docs: List[Document] = retriever.get_relevant_documents(query)  # standard FAISS retriever call

            merged_context = ""
            seen_texts = set()

            for doc in docs:
                # Clean text
                text = "\n".join([line.strip() for line in doc.page_content.splitlines() if line.strip()])
                if text not in seen_texts:
                    # Include metadata inline for LLM
                    source = doc.metadata.get("source", "unknown")
                    pages = doc.metadata.get("pages", [])
                    pages_str = ", ".join(map(str, pages))
                    merged_context += f"(Source: {source}, Pages: {pages_str})\nContent:\n{text}\n\n"
                    
                    seen_texts.add(text)

            return merged_context.strip(), docs

        self.agent = create_agent(
            model=self.llm,
            system_prompt=BOOK_QA_PROMPT,
            checkpointer=self.checkpointer,
            tools=[search_book],
            middleware=[
                SummarizationMiddleware(
                    model=self.llm,  # Use self.llm instead of hardcoded string
                    max_tokens_before_summary=2000,
                    messages_to_keep=20,
                    summary_prompt="Summarize previous context briefly.",
                ),
                ToolCallLimitMiddleware(
                    tool_name="search",
                    run_limit=3,
                ),
            ],
        )

    def get_agent(self):
        return self.agent