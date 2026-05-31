import json
import logging
from typing import List

from langchain_core.language_models import BaseChatModel
from fastapi import HTTPException

from core.rag.qdrant_db import QdrantDB
from core.schemas.chapter import ChapterIdentification
from core.schemas.question import MCQResponse, EssayResponse
from core.prompts.general_rag import (
    MCQ_PROMPT,
    ESSAY_QUESTION_PROMPT,
    CHAPTER_IDENTIFICATION_PROMPT,
)

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generate MCQ and Essay dataset from book content."""

    def __init__(self, qdrant_db: QdrantDB, llm: BaseChatModel):
        self.qdrant_db = qdrant_db
        self.llm = llm

    def _parse_structured_output(self, raw, schema_class):
        """Parse LLM output with robust handling for string/dict responses."""
        if isinstance(raw, str):
            parsed = json.loads(raw)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if isinstance(parsed, dict) and isinstance(parsed.get("questions"), str):
                parsed["questions"] = json.loads(parsed["questions"])
            return schema_class(**parsed)
        elif isinstance(raw, dict):
            if isinstance(raw.get("questions"), str):
                raw["questions"] = json.loads(raw["questions"])
            return schema_class(**raw)
        elif raw is not None:
            return raw
        return None

    def _generate_mcq(self, chapter_title: str, difficulty: str, num_mcq: int, context: str) -> MCQResponse | None:
        """Generate MCQ for a chapter."""
        try:
            mcq_llm = self.llm.with_structured_output(MCQResponse)
            mcq_prompt = MCQ_PROMPT.format(
                topic=chapter_title,
                difficulty=difficulty,
                num_questions=num_mcq,
                context=context,
            )
            raw = mcq_llm.invoke(mcq_prompt)
            return self._parse_structured_output(raw, MCQResponse)
        except Exception as e:
            logger.error(f"MCQ generation error: {e}")
            return None

    def _generate_essay(self, chapter_title: str, difficulty: str, num_essay: int, context: str) -> EssayResponse | None:
        """Generate Essay for a chapter."""
        try:
            essay_llm = self.llm.with_structured_output(EssayResponse)
            essay_prompt = ESSAY_QUESTION_PROMPT.format(
                topic=chapter_title,
                difficulty=difficulty,
                num_questions=num_essay,
                context=context,
            )
            raw = essay_llm.invoke(essay_prompt)
            return self._parse_structured_output(raw, EssayResponse)
        except Exception as e:
            logger.error(f"Essay generation error: {e}")
            return None

    async def generate(
        self,
        course_id: str,
        difficulty: str = "medium",
        num_mcq: int = 3,
        num_essay: int = 2,
    ) -> dict:
        """Generate dataset with MCQ and Essay for each chapter."""
        docs = self.qdrant_db.get_all_by_course(course_id)

        if not docs:
            raise HTTPException(
                status_code=404, detail="No content found for this course"
            )

        context = "\n\n".join([d.page_content for d in docs])[:15000]

        # Step 1: Identify chapters
        chapter_llm = self.llm.with_structured_output(ChapterIdentification)
        chapter_prompt = CHAPTER_IDENTIFICATION_PROMPT.format(
            context=context, topik=course_id
        )
        chapter_result = chapter_llm.invoke(chapter_prompt)
        chapters = chapter_result.chapters

        if not chapters:
            raise HTTPException(
                status_code=404, detail="Could not identify chapters from content"
            )

        # Step 2: Generate questions for each chapter
        dataset = {
            "course_id": course_id,
            "difficulty": difficulty,
            "total_chapters": len(chapters),
            "chapters": [],
        }

        for chapter_title in chapters:
            chapter_docs = self.qdrant_db.query(chapter_title, course_id=course_id, k=3)
            chapter_context = "\n\n".join([d.page_content for d in chapter_docs])[:8000]

            if not chapter_context:
                continue

            mcq_result = self._generate_mcq(chapter_title, difficulty, num_mcq, chapter_context)
            essay_result = self._generate_essay(chapter_title, difficulty, num_essay, chapter_context)

            # Collect sources
            sources = []
            for doc in chapter_docs:
                source = doc.metadata.get("source", "unknown")
                pages = doc.metadata.get("pages", [])
                pages_str = ", ".join(map(str, pages))
                sources.append(f"{source} (hal {pages_str})")

            chapter_data = {
                "chapter_title": chapter_title,
                "sources": list(set(sources)),
            }
            if mcq_result:
                chapter_data["mcq"] = mcq_result.model_dump()
            if essay_result:
                chapter_data["essay"] = essay_result.model_dump()

            dataset["chapters"].append(chapter_data)

        return dataset
