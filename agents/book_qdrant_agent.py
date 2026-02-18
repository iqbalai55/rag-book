# agents/book_agent.py

import logging
from typing import Tuple, List
from langchain.tools import tool

from utils.llm_config import get_chat_model
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from rag.qdrant.qdrant_db import QdrantDB  

from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

logger = logging.getLogger(__name__)

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
        
        self.checkpointer = InMemorySaver()

        # Define tool as nested function to capture self via closure
        @tool("search_book", description="Search relevant book", response_format="content_and_artifact")
        def search_book(question: str) -> Tuple[str, List[dict]]:
            """Retrieve top-k chunks from QdrantDB and combine into context string with metadata for citations."""
            
            # Retrieve documents from vector DB
            retrieved_docs: List[Document] = self.qdrant_db.query(question, k=self.k)

            merged_context = ""
            all_metadata = []

            seen_texts = set()
            for doc in retrieved_docs:
                text = "\n".join([line.strip() for line in doc.page_content.splitlines() if line.strip()])
                if text not in seen_texts:
                    # Include metadata inline for LLM (source and pages)
                    source = doc.metadata.get("source", "unknown")
                    pages = doc.metadata.get("pages", [])
                    pages_str = ", ".join(map(str, pages))
                    merged_context += f"(Source: {source}, Pages: {pages_str})\nContent:\n{text}\n\n"

                    seen_texts.add(text)
                
                # Keep metadata for structured output
                all_metadata.append(doc.metadata)

            logger.debug("Retrieved metadata: %s", all_metadata)
            logger.debug("Merged context for LLM:\n%s", merged_context)

            return merged_context.strip(), retrieved_docs
        
        self.agent = create_agent(
            model=self.llm,
            system_prompt=BOOK_QA_PROMPT,
            checkpointer=self.checkpointer,
            tools=[search_book], 
            middleware=[
                SummarizationMiddleware(
                    model=self.llm, 
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
    
    def ask(self, query: str):
        for event in self.agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            config={
                "configurable": {"thread_id": "1"},
                "callbacks": [LangfuseCallbackHandler()]
            },
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()
                

    def get_agent(self):
        return self.agent