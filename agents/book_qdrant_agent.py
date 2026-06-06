import logging
import json
from typing import Tuple, List
from langchain.tools import tool

from core.utils.llm_config import get_chat_model
from core.utils.token_callback import TokenUsageCallbackHandler
from core.utils.token_tracker import TokenTracker
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from core.rag.qdrant_db import QdrantDB

from core.schemas.expertise import ExpertiseDetection
from core.schemas.question import MCQResponse, EssayResponse
from core.prompts.expertise import (
    EXPERTISE_DETECTION_PROMPT,
    get_system_prompt,
    get_mcq_prompt,
    get_essay_prompt,
)

logger = logging.getLogger(__name__)


def _split_sources_by_page(sources_with_pages: List[tuple]) -> List[str]:
    """
    Format sources with page ranges, splitting non-consecutive pages into separate entries.
    Input: [("Docker in Practice", [275, 301], "https://...pdf")]
    Output: ["[Docker in Practice (halaman 275)](https://...pdf#page=275)",
             "[Docker in Practice (halaman 301)](https://...pdf#page=301)"]

    Input: [("Docker in Practice", [275, 276, 301], "https://...pdf")]
    Output: ["[Docker in Practice (halaman 275-276)](https://...pdf#page=275)",
             "[Docker in Practice (halaman 301)](https://...pdf#page=301)"]
    """
    result = []
    for item in sources_with_pages:
        source = item[0]
        pages = item[1] if len(item) > 1 else []
        url = item[2] if len(item) > 2 else ""

        if not pages:
            result.append(source)
            continue

        sorted_pages = sorted(set(pages))
        ranges = []
        start = sorted_pages[0]
        end = sorted_pages[0]

        for page in sorted_pages[1:]:
            if page == end + 1:
                end = page
            else:
                if start == end:
                    ranges.append((start, start))
                else:
                    ranges.append((start, end))
                start = end = page

        if start == end:
            ranges.append((start, start))
        else:
            ranges.append((start, end))

        for range_start, range_end in ranges:
            if range_start == range_end:
                page_ref = f"{source} (halaman {range_start})"
            else:
                page_ref = f"{source} (halaman {range_start}-{range_end})"

            if url:
                # Fix common typo in stored URLs: supbase.co -> supabase.co
                fixed_url = url.replace("supbase.co", "supabase.co")
                result.append(f"[{page_ref}]({fixed_url}#page={range_start})")
            else:
                result.append(page_ref)

    return result


class BookQdrantAgent:
    """Book Agent that uses QdrantDB for RAG retrieval."""

    def __init__(
        self,
        qdrant_db: QdrantDB,
        book_id: str,
        user_id: str = None,
        checkpointer=None,
        k: int = 3,
    ):
        self.qdrant_db = qdrant_db
        self.k = k
        self.book_id = book_id
        self.user_id = user_id

        # Token tracking callback
        self.token_callback = TokenUsageCallbackHandler(
            user_id=user_id,
            book_id=book_id,
            feature="agent_reasoning",
        )
        self.token_tracker = TokenTracker()

        self.llm = get_chat_model(callbacks=[self.token_callback])
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()

        # Detect expertise from book content
        self.expertise = self._detect_expertise()
        logger.info(
            f"Detected expertise for '{book_id}': {self.expertise.domain} - "
            f"{self.expertise.sub_fields}"
        )

        # Generate dynamic prompts based on expertise
        self.system_prompt = get_system_prompt(self.expertise)
        self.mcq_prompt_template = get_mcq_prompt(self.expertise)
        self.essay_prompt_template = get_essay_prompt(self.expertise)

        @tool("search_book_context", description="Search relevant book context", response_format="content_and_artifact")
        def search_book_context(question: str) -> Tuple[str, List[Document]]:
            self.token_callback.set_context(feature="search")
            merged_context, retrieved_docs, _ = self._retrieve_context(question)
            return merged_context, retrieved_docs

        @tool("generate_mcq", description="Generate MCQs from book based on topic")
        def generate_mcq(
            topic: str,
            num_questions: int = 5,
            difficulty: str = "medium"
        ) -> dict:
            self.token_callback.set_context(feature="generate_mcq")
            context, _, unique_sources = self._retrieve_context(topic)

            if not context:
                return {"error": "Tidak ditemukan konteks relevan dari buku."}

            prompt = self.mcq_prompt_template.format(
                topic=topic,
                num_questions=num_questions,
                difficulty=difficulty,
                context=context[:4000]
            )

            structured_llm = self.llm.with_structured_output(MCQResponse)
            raw = structured_llm.invoke(prompt)

            try:
                if isinstance(raw, str):
                    parsed = json.loads(raw)
                    if isinstance(parsed, str):
                        parsed = json.loads(parsed)
                    if isinstance(parsed, dict) and isinstance(parsed.get("questions"), str):
                        parsed["questions"] = json.loads(parsed["questions"])
                    result = MCQResponse(**parsed)
                elif isinstance(raw, MCQResponse):
                    result = raw
                elif isinstance(raw, dict):
                    if isinstance(raw.get("questions"), str):
                        raw["questions"] = json.loads(raw["questions"])
                    result = MCQResponse(**raw)
                else:
                    return {"error": f"Tipe response tidak dikenal: {type(raw)}"}
            except Exception as e:
                return {"error": f"Gagal parse response MCQ: {e}"}

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
            self.token_callback.set_context(feature="generate_essay")
            context, _, unique_sources = self._retrieve_context(topic)

            if not context:
                return {"error": "Tidak ditemukan konteks relevan dari buku."}

            prompt = self.essay_prompt_template.format(
                topic=topic,
                num_questions=num_questions,
                difficulty=difficulty,
                context=context[:4000]
            )

            structured_llm = self.llm.with_structured_output(EssayResponse)
            raw = structured_llm.invoke(prompt)

            try:
                if isinstance(raw, str):
                    parsed = json.loads(raw)
                    if isinstance(parsed, str):
                        parsed = json.loads(parsed)
                    if isinstance(parsed, dict) and isinstance(parsed.get("questions"), str):
                        parsed["questions"] = json.loads(parsed["questions"])
                    result = EssayResponse(**parsed)
                elif isinstance(raw, EssayResponse):
                    result = raw
                elif isinstance(raw, dict):
                    if isinstance(raw.get("questions"), str):
                        raw["questions"] = json.loads(raw["questions"])
                    result = EssayResponse(**raw)
                else:
                    return {"error": f"Tipe response tidak dikenal: {type(raw)}"}
            except Exception as e:
                return {"error": f"Gagal parse response Essay: {e}"}

            result.topic = topic
            result.difficulty = difficulty
            result.sources = unique_sources[:5]
            return result.dict()

        self.agent = create_agent(
            model=self.llm,
            system_prompt=self.system_prompt,
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

    def _detect_expertise(self) -> ExpertiseDetection:
        """Detect book domain from content samples."""
        docs = self.qdrant_db.get_all_by_book(self.book_id, limit=5)

        if not docs:
            return ExpertiseDetection(
                domain="Umum",
                sub_fields=[],
                expertise_prompt="tutor ahli yang menguasai materi dalam buku ini",
                book_type="unknown",
            )

        context = "\n".join([d.page_content[:500] for d in docs])[:3000]

        llm = get_chat_model()
        structured_llm = llm.with_structured_output(ExpertiseDetection)
        prompt = EXPERTISE_DETECTION_PROMPT.format(
            context=context, topik=self.book_id
        )

        try:
            result = structured_llm.invoke(prompt)
            return result
        except Exception as e:
            logger.warning(f"Expertise detection failed: {e}, using fallback")
            return ExpertiseDetection(
                domain="Umum",
                sub_fields=[],
                expertise_prompt="tutor ahli yang menguasai materi dalam buku ini",
                book_type="unknown",
            )

    def _retrieve_context(self, topic: str) -> Tuple[str, List[Document], List[str]]:
        """Shared retrieval logic used by all tools (multitenant-safe)."""

        retrieved_docs: List[Document] = self.qdrant_db.query(
            topic,
            book_id=self.book_id,
            k=self.k
        )

        logger.debug("TYPE of retrieved_docs: %s", type(retrieved_docs))

        if retrieved_docs:
            logger.debug(
                "FIRST ITEM type: %s | value: %s",
                type(retrieved_docs[0]),
                retrieved_docs[0]
            )

        merged_context = []
        sources_raw = []
        seen_texts = set()

        for doc in retrieved_docs:
            if not isinstance(doc, Document):
                logger.warning("Unexpected doc type: %s | value: %s", type(doc), doc)
                continue

            text = "\n".join(
                line.strip()
                for line in doc.page_content.splitlines()
                if line.strip()
            )

            if not text or text in seen_texts:
                continue

            source = doc.metadata.get("source", "unknown")
            pages = doc.metadata.get("pages", [])
            storage_url = doc.metadata.get("storage_url", "")
            pages_str = ", ".join(map(str, pages)) if pages else "-"

            page_url = ""
            if storage_url and pages:
                page_url = f"{storage_url}#page={pages[0]}"

            merged_context.append(
                f"(Source: {source}, Pages: {pages_str}, URL: {page_url})\n"
                f"Content:\n{text}"
            )

            sources_raw.append((source, pages, storage_url))
            seen_texts.add(text)

        sources = _split_sources_by_page(sources_raw)
        return "\n\n".join(merged_context), retrieved_docs, sources

    def ask(self, query: str, session_id: str = "book_thread"):
        config = {"configurable": {"thread_id": session_id}}
        
        # Update token callback with session context
        self.token_callback.set_context(session_id=session_id, feature="agent_reasoning")
        
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
        
        # Update token callback with session context
        self.token_callback.set_context(session_id=session_id, feature="agent_reasoning")
        
        tool_question_generated = False  

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
                        tool_question_generated = True
                        chunk["type"] = "multiple_choice_question"
                        try:
                            chunk["content"] = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
                        except Exception:
                            chunk["content"] = raw_content

                    elif tool_name == "generate_essay_questions":
                        tool_question_generated = True
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
                    
                    if tool_question_generated:
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