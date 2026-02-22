import logging
import json
from typing import Tuple, List
from langchain.tools import tool

from utils.llm_config import get_chat_model
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from rag.qdrant.qdrant_db import QdrantDB

from prompts.general_rag import BOOK_QA_SYSTEM_PROMPT, MCQ_PROMPT, ESSAY_QUESTION_PROMPT
from schemas.question import MCQResponse, EssayResponse

logger = logging.getLogger(__name__)


class BookQdrantAgent:
    """Book Agent that uses QdrantDB for RAG retrieval."""

    def __init__(self, qdrant_db: QdrantDB, checkpointer=None, k: int = 3):
        self.llm = get_chat_model()
        self.qdrant_db = qdrant_db
        self.k = k

        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()

        @tool("search_book_context", description="Search relevant book context", response_format="content_and_artifact")
        def search_book_context(question: str) -> Tuple[str, List[Document]]:
            merged_context, retrieved_docs, _ = self._retrieve_context(question)
            return merged_context, retrieved_docs

        @tool("generate_mcq", description="Generate MCQs from book based on topic")
        def generate_mcq(
            topic: str,
            num_questions: int = 5,
            difficulty: str = "medium"
        ) -> dict:
            context, _, unique_sources = self._retrieve_context(topic)

            if not context:
                return {"error": "Tidak ditemukan konteks relevan dari buku."}

            prompt = MCQ_PROMPT.format(
                topic=topic,
                num_questions=num_questions,
                difficulty=difficulty,
                context=context[:4000]
            )

            structured_llm = self.llm.with_structured_output(MCQResponse)
            result: MCQResponse = structured_llm.invoke(prompt)
            result.topic = topic
            result.difficulty = difficulty
            result.sources = unique_sources[:5]
            return result.dict()

        @tool("generate_essay_questions", description="Generate Essay Questions from book based on topic")
        def generate_essay_questions(
            topic: str,
            num_questions: int = 3,
            difficulty: str = "medium"
        ) -> dict:
            context, _, unique_sources = self._retrieve_context(topic)

            if not context:
                return {"error": "Tidak ditemukan konteks relevan dari buku."}

            prompt = ESSAY_QUESTION_PROMPT.format(
                topic=topic,
                num_questions=num_questions,
                difficulty=difficulty,
                context=context[:4000]
            )

            structured_llm = self.llm.with_structured_output(EssayResponse)
            result: EssayResponse = structured_llm.invoke(prompt)
            result.topic = topic
            result.difficulty = difficulty
            result.sources = unique_sources[:5]
            return result.dict()

        self.agent = create_agent(
            model=self.llm,
            system_prompt=BOOK_QA_SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
            tools=[search_book_context, generate_mcq, generate_essay_questions],
            middleware=[
                SummarizationMiddleware(
                    model=self.llm,
                    max_tokens_before_summary=6000,
                    messages_to_keep=40,
                    summary_prompt="Summarize previous context briefly.",
                ),
                ToolCallLimitMiddleware(
                    tool_name="search_book_context",
                    run_limit=5,
                ),
            ],
        )

    def _retrieve_context(self, topic: str) -> Tuple[str, List[Document], List[str]]:
        """Shared retrieval logic used by all tools."""
        retrieved_docs: List[Document] = self.qdrant_db.query(topic, k=self.k)

        # 🔍 Debug: confirm return type from qdrant_db.query()
        logger.debug("TYPE of retrieved_docs: %s", type(retrieved_docs))
        if retrieved_docs:
            logger.debug("FIRST ITEM type: %s | value: %s", type(retrieved_docs[0]), retrieved_docs[0])

        merged_context = ""
        sources = []
        seen_texts = set()

        for doc in retrieved_docs:
            # Guard: ensure doc is a proper Document object
            if not isinstance(doc, Document):
                logger.warning("Unexpected doc type: %s | value: %s", type(doc), doc)
                continue

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

            source = doc.metadata.get("source", "unknown")
            pages = doc.metadata.get("pages", [])
            pages_str = ", ".join(map(str, pages))
            sources.append(f"{source} (hal {pages_str})")

        unique_sources = list(set(sources))
        return merged_context.strip(), retrieved_docs, unique_sources

    def ask(self, query: str):
        config = {"configurable": {"thread_id": "1"}}

        for event in self.agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()

    async def ask_stream(self, query: str, session_id: str = "book_thread"):
        """
        Stream response dari agent dengan detail lengkap:
        - type: human / tool / internal / final / mcq / essay / error
        - content: isi pesan
        - metadata:
            - tool_name (jika ada)
            - tool_calls (jika internal reasoning)
            - metadata bawaan message
        """
        config = {"configurable": {"thread_id": session_id}}

        try:
            async for event in self.agent.astream(
                {"messages": [("user", query)]},
                config=config,
                stream_mode="values",
            ):
                messages = event.get("messages")
                if not messages:
                    continue

                last_msg = messages[-1]

                chunk = {
                    "id": "chatcmpl",
                    "type": None,
                    "content": None,
                    "metadata": {},
                }

                # HUMAN MESSAGE — skip, no need to send to frontend
                if isinstance(last_msg, HumanMessage):
                    continue

                # TOOL MESSAGE
                if isinstance(last_msg, ToolMessage):
                    tool_name = getattr(last_msg, "name", None)
                    raw_content = getattr(last_msg, "content", "")
                    tool_call_id = getattr(last_msg, "tool_call_id", None)

                    message_metadata = dict(getattr(last_msg, "metadata", {}) or {})
                    message_metadata["tool_name"] = tool_name
                    if tool_call_id:
                        message_metadata["tool_call_id"] = tool_call_id

                    chunk["metadata"] = message_metadata

                    if tool_name == "generate_mcq":
                        chunk["type"] = "multiple_choice_question"
                        try:
                            chunk["content"] = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
                        except Exception:
                            chunk["content"] = raw_content

                    elif tool_name == "generate_essay_questions":
                        chunk["type"] = "essay_question"
                        try:
                            chunk["content"] = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
                        except Exception:
                            chunk["content"] = raw_content

                    else:
                        chunk["type"] = "tool"
                        chunk["content"] = raw_content

                    yield f"data: {json.dumps(chunk)}\n\n"
                    continue

                # AI MESSAGE
                if isinstance(last_msg, AIMessage):
                    tool_calls = getattr(last_msg, "tool_calls", None)

                    # INTERNAL REASONING (tool call planning)
                    if tool_calls:
                        chunk["type"] = "internal"
                        chunk["content"] = getattr(last_msg, "content", "")
                        chunk["metadata"] = {"tool_calls": tool_calls}
                        yield f"data: {json.dumps(chunk)}\n\n"
                        continue

                    # FINAL ANSWER
                    chunk["type"] = "final"
                    chunk["content"] = getattr(last_msg, "content", "")
                    chunk["metadata"] = dict(getattr(last_msg, "metadata", {}) or {})
                    yield f"data: {json.dumps(chunk)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception("Error in ask_stream: %s", e)
            yield f"data: {json.dumps({'id': 'chatcmpl', 'type': 'error', 'content': str(e), 'metadata': {}})}\n\n"
            yield "data: [DONE]\n\n"

    def get_agent(self):
        return self.agent