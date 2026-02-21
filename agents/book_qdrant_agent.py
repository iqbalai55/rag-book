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
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage
from rag.qdrant.qdrant_db import QdrantDB  

import mlflow

logger = logging.getLogger(__name__)

BOOK_QA_PROMPT = SystemMessage(content="""
Anda adalah tutor ahli yang menguasai materi dalam course ini.

Tugas Anda adalah menjawab pertanyaan menggunakan pengetahuan dari materi yang tersedia melalui tool `search_book`. 
Jawablah seperti seorang pengajar profesional yang benar-benar memahami materi — bukan seperti sistem yang sedang membaca konteks.

Aturan:
1. Gunakan materi dari course sebagai dasar utama jawaban.
2. Anda boleh menjelaskan ulang dengan bahasa yang lebih mudah dipahami (parafrase) selama tetap setia pada isi materi.
3. Jangan menyebutkan frasa seperti "berdasarkan konteks", "pada potongan teks", atau istilah teknis sistem lainnya.
4. Jangan terlalu cepat menyimpulkan jawaban tidak ada.
   - Pahami pertanyaan secara konseptual.
   - Cocokkan dengan konsep yang relevan meskipun istilahnya berbeda.
5. Anda boleh sedikit mengembangkan penjelasan agar lebih edukatif, selama tidak bertentangan dengan materi course.
6. Jika memang setelah analisis menyeluruh topik tersebut benar-benar tidak ada dalam materi,
   jawab hanya dengan:
   "Topik tersebut tidak dibahas pada course ini."
   (Jangan sertakan sumber dalam kondisi ini.)
7. Jika jawaban ada, sertakan sumber dan nomor halaman dari metadata:
   - Gunakan field 'source' sebagai judul.
   - Gunakan field 'pages' sebagai nomor halaman.
8. Jawaban harus dalam bahasa Indonesia.
9. Fokus pada keperluan coding atau pembelajaran.
10. Jawaban harus jelas, mengalir, dan terasa seperti penjelasan tutor.

Format jika jawaban ADA:
<penjelasan Anda>

Sumber: <source dari metadata>, Halaman <pages dari metadata>

Format jika TIDAK ADA:
Topik tersebut tidak dibahas pada course ini.
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
    
    async def ask_stream(self, query: str, session_id: str = "book_thread"):

        config = {"configurable": {"thread_id": session_id}}

        try:
            async for event in self.agent.astream(
                {"messages": [("user", query)]},
                config=config,
                stream_mode="values",  # lengkap
            ):
                messages = event.get("messages", [])
                if not messages:
                    continue

                last_msg = messages[-1]

                chunk = {"id": "chatcmpl", "type": None, "content": None}

                # HumanMessage → bisa skip atau tetap dikirim kalau mau
                if isinstance(last_msg, HumanMessage):
                    chunk["type"] = "human"
                    chunk["content"] = getattr(last_msg, "content", "")
                    yield f"data: {json.dumps(chunk)}\n\n"
                    continue

                # ToolMessage → internal tool output
                elif isinstance(last_msg, ToolMessage):
                    chunk["type"] = "tool"
                    chunk["content"] = getattr(last_msg, "content", "")
                    yield f"data: {json.dumps(chunk)}\n\n"
                    continue

                # AIMessage → bisa final atau reasoning internal
                elif isinstance(last_msg, AIMessage):

                    if getattr(last_msg, "tool_calls", None):
                        # masih reasoning → internal
                        chunk["type"] = "internal"
                        chunk["content"] = getattr(last_msg, "content", "")
                        yield f"data: {json.dumps(chunk)}\n\n"
                        continue

                    # ✅ final answer
                    chunk["type"] = "final"
                    chunk["content"] = getattr(last_msg, "content", "")
                    yield f"data: {json.dumps(chunk)}\n\n"

            # pastikan DONE selalu dikirim
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_chunk = {"id": "chatcmpl", "type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    def get_agent(self):
        return self.agent
