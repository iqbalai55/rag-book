from utils.llm_config import get_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage
from rag.faiss.retrieve_faiss import load_vector_db
from typing import List, AsyncGenerator
import json

from prompts.general_rag import BOOK_QA_SYSTEM_PROMPT


class BookFaissAgent:
    def __init__(self, faiss_path: str, checkpointer = None, k: int = 3):
        self.llm = get_chat_model()
        self.vectordb = load_vector_db(faiss_path)
        self.k = k
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()

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
            system_prompt=BOOK_QA_SYSTEM_PROMPT,
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
                    run_limit=5,
                ),
            ],
        )
    
    def ask(self, query: str):
        for event in self.agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            config={
                "configurable": {"thread_id": "1"},
            },
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()
    
    async def ask_stream(self, query: str, session_id: str = "book_thread"):
        """
        Stream response dari agent dengan detail lengkap:
        - type: human / tool / internal / final
        - content: isi pesan
        - tool_name: nama tool yang digunakan (jika ada)
        - metadata: info tambahan seperti source, page, atau args tool
        """
        config = {"configurable": {"thread_id": session_id}}

        try:
            async for event in self.agent.astream(
                {"messages": [("user", query)]},
                config=config,
                stream_mode="values",  # atau "deltas" kalau mau
            ):
                messages = event.get("messages", [])
                if not messages:
                    continue

                last_msg = messages[-1]

                chunk = {
                    "id": "chatcmpl",
                    "type": None,
                    "content": None,
                    "tool_name": None,
                    "metadata": {}
                }

                # HumanMessage → optional dikirim
                if isinstance(last_msg, HumanMessage):
                    chunk["type"] = "human"
                    chunk["content"] = getattr(last_msg, "content", "")
                    yield f"data: {json.dumps(chunk)}\n\n"
                    continue

                # ToolMessage → internal tool output
                elif isinstance(last_msg, ToolMessage):
                    chunk["type"] = "tool"
                    chunk["content"] = getattr(last_msg, "content", "")
                    # Ambil metadata tambahan jika ada
                    chunk["metadata"] = getattr(last_msg, "metadata", {})
                    yield f"data: {json.dumps(chunk)}\n\n"
                    continue

                # AIMessage → reasoning internal atau final answer
                elif isinstance(last_msg, AIMessage):
                    tool_calls = getattr(last_msg, "tool_calls", None)
                    if tool_calls:
                        # reasoning internal
                        chunk["type"] = "internal"
                        chunk["content"] = getattr(last_msg, "content", "")
                        # bisa sertakan args atau output sementara dari tiap tool
                        chunk["metadata"] = {"tool_calls": tool_calls}
                        yield f"data: {json.dumps(chunk)}\n\n"
                        continue

                    chunk["type"] = "final"
                    chunk["content"] = getattr(last_msg, "content", "")
                    # sertakan metadata konteks jika ada
                    chunk["metadata"] = getattr(last_msg, "metadata", {})
                    yield f"data: {json.dumps(chunk)}\n\n"

            # pastikan DONE selalu dikirim
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_chunk = {
                "id": "chatcmpl",
                "type": "error",
                "content": str(e),
                "tool_name": None,
                "metadata": {}
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
    def get_agent(self):
        return self.agent