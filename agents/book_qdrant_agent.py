import logging
import json
from typing import Tuple, List, AsyncGenerator
from langchain.tools import tool

from utils.llm_config import get_chat_model
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import SystemMessage
from rag.qdrant.qdrant_db import QdrantDB  

import mlflow

logger = logging.getLogger(__name__)

BOOK_QA_PROMPT = SystemMessage(content="""
Anda adalah asisten ahli yang menjawab pertanyaan HANYA berdasarkan konteks buku yang diberikan melalui tool `search_book`.

Aturan:
1. Jangan membuat jawaban di luar konteks yang tersedia.
2. Jangan terlalu cepat menyimpulkan bahwa jawaban tidak ada.
   - Analisis pertanyaan secara mendalam.
   - Pertimbangkan kemungkinan parafrase, sinonim, atau penjelasan tidak langsung dalam konteks.
   - Jika konsepnya ada tetapi dengan istilah berbeda, gunakan informasi tersebut.
3. HANYA jika setelah analisis menyeluruh informasi memang tidak ditemukan dalam konteks,
   katakan: "Saya tidak tahu berdasarkan konteks yang diberikan."
4. Selalu sertakan sumber buku dan nomor halaman dari metadata chunk.
   - Gunakan field 'source' sebagai judul buku.
   - Gunakan field 'pages' sebagai nomor halaman.
5. Jawaban harus dalam bahasa Indonesia.
6. Fokus jawaban untuk keperluan coding atau pembelajaran.
7. Jawaban harus singkat, jelas, dan tepat.

Format jawaban:
<jawaban Anda di sini>

Sumber: <source dari metadata>, Halaman <pages dari metadata>
""")

class BookQdrantAgent:
    """Book Agent that uses QdrantDB for RAG retrieval."""

    def __init__(self, qdrant_db: QdrantDB, checkpointer = None, k: int = 3):
        self.llm = get_chat_model()
        self.qdrant_db = qdrant_db
        self.k = k
        
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()

        @tool("search_book", description="Search relevant book", response_format="content_and_artifact")
        def search_book(question: str) -> Tuple[str, List[dict]]:
            retrieved_docs: List[Document] = self.qdrant_db.query(question, k=self.k)

            merged_context = ""
            all_metadata = []
            seen_texts = set()

            for doc in retrieved_docs:
                text = "\n".join(
                    [line.strip() for line in doc.page_content.splitlines() if line.strip()]
                )

                if text not in seen_texts:
                    source = doc.metadata.get("source", "unknown")
                    pages = doc.metadata.get("pages", [])
                    pages_str = ", ".join(map(str, pages))

                    merged_context += (
                        f"(Source: {source}, Pages: {pages_str})\n"
                        f"Content:\n{text}\n\n"
                    )
                    seen_texts.add(text)

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
                    tool_name="search_book",
                    run_limit=5,
                ),
            ],
        )
    
    def ask(self, query: str):
        
        config = {
                "configurable": {"thread_id": "1"},
            }
        
        for event in self.agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()
    
    async def ask_stream(self, query: str, session_id: str = "book_thread") -> AsyncGenerator[str, None]:
        """SSE generator for streaming"""
        config = {"configurable": {"thread_id": session_id}}
        
        async for event in self.agent.astream(
            {"messages": [("user", query)]},
            config=config,
            stream_mode="values",  # Streams full state; use "updates" for deltas
        ):
            # Grab the latest message content (your agent state has "messages")
            if event["messages"]:
                last_msg = event["messages"][-1]
                content = getattr(last_msg, "content", "") or ""
                if content:
                    chunk = {
                        "id": "chatcmpl",
                        "object": "chat.completion.chunk",
                        "choices": [{"delta": {"content": content}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
        
        yield "data: [DONE]\n\n"

    def get_agent(self):
        return self.agent
