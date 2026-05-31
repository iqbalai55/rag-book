import json
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from fastapi import HTTPException

from core.rag.qdrant_db import QdrantDB
from core.schemas.mindmap import MindmapResponse
from core.prompts.mindmap import MINDMAP_FROM_CONTENT_PROMPT, MINDMAP_EDIT_PROMPT

logger = logging.getLogger(__name__)


class MindmapGenerator:
    """Generate mindmap from book content."""

    def __init__(self, qdrant_db: QdrantDB, llm: BaseChatModel):
        self.qdrant_db = qdrant_db
        self.llm = llm

    def _build_user_prompt_section(self, user_prompt: Optional[str]) -> str:
        """Build user prompt section for the mindmap prompt."""
        if user_prompt:
            return f"\n7. Instruksi tambahan dari user: {user_prompt}"
        return ""

    async def generate(
        self,
        course_id: str,
        user_prompt: Optional[str] = None,
    ) -> dict:
        """Generate mindmap from all book content."""
        docs = self.qdrant_db.get_all_by_course(course_id)

        if not docs:
            raise HTTPException(
                status_code=404, detail="No content found for this course"
            )

        context = "\n\n".join([d.page_content for d in docs])[:15000]

        structured_llm = self.llm.with_structured_output(MindmapResponse)
        user_prompt_section = self._build_user_prompt_section(user_prompt)
        prompt = MINDMAP_FROM_CONTENT_PROMPT.format(
            context=context,
            topik=course_id,
            user_prompt_section=user_prompt_section,
        )

        try:
            result = structured_llm.invoke(prompt)

            # Collect sources from document metadata
            all_sources = set()
            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                pages = doc.metadata.get("pages", [])
                if pages:
                    all_sources.add(f"{source} (hal {min(pages)}-{max(pages)})")
                else:
                    all_sources.add(source)

            # Handle various response formats
            if isinstance(result, str):
                result = json.loads(result)
            if isinstance(result, dict):
                result = MindmapResponse(**result)

        except Exception as e:
            logger.error(f"Mindmap generation error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate mindmap: {e}"
            )

        return {
            "course_id": course_id,
            "title": result.title,
            "mermaid": result.mermaid,
            "sources": list(all_sources),
        }

    async def edit(
        self,
        course_id: str,
        mermaid: str,
        instruction: str,
    ) -> dict:
        """Edit existing mindmap based on user instructions."""
        prompt = MINDMAP_EDIT_PROMPT.format(
            mermaid=mermaid,
            instruction=instruction,
        )

        try:
            llm = self.llm
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Clean up the response - extract only mermaid syntax
            content = content.strip()
            if content.startswith("```mermaid"):
                content = content[10:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            return {
                "course_id": course_id,
                "mermaid": content,
            }

        except Exception as e:
            logger.error(f"Mindmap edit error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to edit mindmap: {e}"
            )
