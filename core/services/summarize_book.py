import json
import logging
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from fastapi import HTTPException

from core.rag.qdrant_db import QdrantDB
from core.schemas.chapter import ChapterIdentification
from core.schemas.summary import BookSummaryResponse, ChapterSummary
from core.prompts.general_rag import CHAPTER_IDENTIFICATION_PROMPT
from core.prompts.summary import CHAPTER_SUMMARY_PROMPT, BOOK_SUMMARY_PROMPT, SUMMARY_EDIT_PROMPT

logger = logging.getLogger(__name__)


class BookSummarizer:
    """Summarize entire book using hierarchical map-reduce approach."""

    def __init__(self, qdrant_db: QdrantDB, llm: BaseChatModel):
        self.qdrant_db = qdrant_db
        self.llm = llm

    def _build_user_prompt_section(self, user_prompt: Optional[str]) -> str:
        """Build user prompt section for the summary prompt."""
        if user_prompt:
            return f"\n7. Instruksi tambahan dari user: {user_prompt}"
        return ""

    def _identify_chapters(self, context: str, book_id: str) -> List[str]:
        """Identify chapters from book content."""
        chapter_llm = self.llm.with_structured_output(ChapterIdentification)
        chapter_prompt = CHAPTER_IDENTIFICATION_PROMPT.format(
            context=context, topik=book_id
        )
        chapter_result = chapter_llm.invoke(chapter_prompt)
        return chapter_result.chapters

    def _summarize_chapter(
        self,
        chapter_title: str,
        context: str,
        user_prompt_section: str = "",
    ) -> ChapterSummary | None:
        """Summarize a single chapter."""
        try:
            summary_llm = self.llm.with_structured_output(ChapterSummary)
            summary_prompt = CHAPTER_SUMMARY_PROMPT.format(
                chapter_title=chapter_title,
                context=context,
                user_prompt_section=user_prompt_section,
            )
            return summary_llm.invoke(summary_prompt)
        except Exception as e:
            logger.error(f"Chapter summary error: {e}")
            return None

    def _generate_book_summary(
        self,
        chapter_summaries_text: str,
        book_id: str,
        user_prompt_section: str = "",
    ) -> BookSummaryResponse | None:
        """Generate final book summary from chapter summaries."""
        try:
            book_summary_llm = self.llm.with_structured_output(BookSummaryResponse)
            book_summary_prompt = BOOK_SUMMARY_PROMPT.format(
                chapter_summaries=chapter_summaries_text[:10000],
                topic=book_id,
                user_prompt_section=user_prompt_section,
            )
            return book_summary_llm.invoke(book_summary_prompt)
        except Exception as e:
            logger.error(f"Book summary error: {e}")
            return None

    async def summarize(
        self,
        book_id: str,
        user_prompt: Optional[str] = None,
    ) -> dict:
        """Summarize entire book."""
        docs = self.qdrant_db.get_all_by_book(book_id)

        if not docs:
            raise HTTPException(
                status_code=404, detail="No content found for this book"
            )

        # Collect all sources
        all_sources = set()
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            pages = doc.metadata.get("pages", [])
            if pages:
                all_sources.add(f"{source} (hal {min(pages)}-{max(pages)})")
            else:
                all_sources.add(source)

        # Build user prompt section
        user_prompt_section = self._build_user_prompt_section(user_prompt)

        # Step 1: Identify chapters
        context = "\n\n".join([d.page_content for d in docs])[:15000]
        chapters = self._identify_chapters(context, book_id)

        if not chapters:
            raise HTTPException(
                status_code=404, detail="Could not identify chapters from content"
            )

        # Step 2: Summarize each chapter
        chapter_summaries = []
        for chapter_title in chapters:
            chapter_docs = self.qdrant_db.query(chapter_title, book_id=book_id, k=5)
            chapter_context = "\n\n".join([d.page_content for d in chapter_docs])[:8000]

            if not chapter_context:
                continue

            chapter_summary = self._summarize_chapter(
                chapter_title,
                chapter_context,
                user_prompt_section,
            )
            if chapter_summary:
                chapter_summaries.append(chapter_summary)

        if not chapter_summaries:
            raise HTTPException(
                status_code=404, detail="Could not generate chapter summaries"
            )

        # Step 3: Generate final book summary
        chapter_summaries_text = "\n\n".join([
            f"Chapter: {cs.chapter_title}\nSummary: {cs.summary}\nKey Points: {', '.join(cs.key_points)}"
            for cs in chapter_summaries
        ])

        book_summary = self._generate_book_summary(
            chapter_summaries_text,
            book_id,
            user_prompt_section,
        )

        # Build final response
        response = {
            "book_id": book_id,
            "title": book_summary.title if book_summary else "Ringkasan Buku",
            "overview": book_summary.overview if book_summary else "",
            "chapters": [cs.model_dump() for cs in chapter_summaries],
            "key_themes": book_summary.key_themes if book_summary else [],
            "total_chapters": len(chapter_summaries),
            "sources": list(all_sources),
        }

        return response

    async def edit(
        self,
        book_id: str,
        title: str,
        overview: str,
        key_themes: List[str],
        chapters: List[dict],
        instruction: str,
    ) -> dict:
        """Edit existing summary based on user instructions."""
        # Format chapter summaries for prompt
        chapter_summaries_text = "\n\n".join([
            f"Chapter: {c.get('chapter_title', '')}\nSummary: {c.get('summary', '')}\nKey Points: {', '.join(c.get('key_points', []))}"
            for c in chapters
        ])

        prompt = SUMMARY_EDIT_PROMPT.format(
            title=title,
            overview=overview,
            key_themes=", ".join(key_themes),
            chapter_summaries=chapter_summaries_text[:10000],
            instruction=instruction,
        )

        try:
            llm = self.llm
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Parse JSON response
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            parsed = json.loads(content)

            return {
                "book_id": book_id,
                "title": parsed.get("title", title),
                "overview": parsed.get("overview", overview),
                "key_themes": parsed.get("key_themes", key_themes),
                "chapters": parsed.get("chapters", chapters),
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse edited summary: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to parse edited summary"
            )
        except Exception as e:
            logger.error(f"Summary edit error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to edit summary: {e}"
            )
