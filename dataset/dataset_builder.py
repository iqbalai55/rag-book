import logging
import re
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document

from utils.llm_config import get_chat_model
from schemas.question import MCQResponse, EssayResponse
from prompts.general_rag import MCQ_PROMPT, ESSAY_QUESTION_PROMPT

from utils.toc_extractor import (
    load_document,
    extract_pages,
    find_toc_pages,
    extract_toc_content,
)

logger = logging.getLogger(__name__)


class BenchmarkDatasetBuilder:
    """
    Dtaset builder to evaluate accuracy of RAG agent
    """

    def __init__(self, qdrant_db, k: int = 3):
        self.llm = get_chat_model()
        self.qdrant_db = qdrant_db
        self.k = k


    def _generate_section_questions(
        self,
        section_title: str,
        section_context: str,
        difficulty: str,
        num_mcq: int,
        num_essay: int,
    ) -> Dict[str, Any]:

        structured_mcq_llm = self.llm.with_structured_output(MCQResponse)

        mcq_prompt = MCQ_PROMPT.format(
            topic=section_title,
            difficulty=difficulty,
            num_questions=num_mcq,
            context=section_context,
        )

        mcq_result: MCQResponse = structured_mcq_llm.invoke(mcq_prompt)

        structured_essay_llm = self.llm.with_structured_output(EssayResponse)

        essay_prompt = ESSAY_QUESTION_PROMPT.format(
            topic=section_title,
            difficulty=difficulty,
            num_questions=num_essay,
            context=section_context,
        )

        essay_result: EssayResponse = structured_essay_llm.invoke(essay_prompt)

        return {
            "section_title": section_title,
            "difficulty": difficulty,
            "mcq": mcq_result.model_dump(),
            "essay": essay_result.model_dump(),
        }


    def _retrieve_context(
        self, topic: str
    ) -> Tuple[str, List[Document], List[str]]:

        retrieved_docs: List[Document] = self.qdrant_db.query(topic, k=self.k)

        merged_context = ""
        sources = []
        seen_texts = set()

        for doc in retrieved_docs:
            if not isinstance(doc, Document):
                logger.warning("Unexpected doc type: %s", type(doc))
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
                    f"{text}\n\n"
                )

                seen_texts.add(text)

            source = doc.metadata.get("source", "unknown")
            pages = doc.metadata.get("pages", [])
            pages_str = ", ".join(map(str, pages))
            sources.append(f"{source} (hal {pages_str})")

        return merged_context.strip(), retrieved_docs, list(set(sources))

    def _parse_sections_from_toc(self, toc_text: str) -> List[str]:

        lines = toc_text.split("\n")
        sections = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # remove trailing page numbers
            line = re.sub(r"\s*\d+\s*$", "", line)

            if len(line) < 5:
                continue

            sections.append(line)

        return sections

    def build_dataset_from_book(
        self,
        pdf_path: str,
        difficulty: str = "medium",
        num_mcq: int = 3,
        num_essay: int = 2,
    ) -> Dict[str, Any]:

        logger.info("Processing book: %s", pdf_path)

        document = load_document(pdf_path)
        pages = extract_pages(document)
        print(pages)
        toc_pages = find_toc_pages(pages, self.llm)

        if not toc_pages:
            raise ValueError("No TOC detected.")

        toc_text = extract_toc_content(pages, self.llm, toc_pages)
        sections = self._parse_sections_from_toc(toc_text)

        dataset = {
            "book": pdf_path,
            "total_sections": len(sections),
            "sections": [],
        }

        for section_title in sections:

            logger.info("Retrieving context for: %s", section_title)

            section_context, _, sources = self._retrieve_context(section_title)

            if not section_context:
                logger.warning("No context found for section: %s", section_title)
                continue

            generated = self._generate_section_questions(
                section_title=section_title,
                section_context=section_context,
                difficulty=difficulty,
                num_mcq=num_mcq,
                num_essay=num_essay,
            )

            dataset["sections"].append({
                "section_title": section_title,
                "sources": sources,
                "generated": generated,
            })

        return dataset

    def build_dataset_from_multiple_books(
        self,
        pdf_paths: List[str],
        difficulty: str = "medium",
        num_mcq: int = 3,
        num_essay: int = 2,
    ) -> Dict[str, Any]:

        logger.info("Processing multiple books...")

        full_dataset = {
            "total_books": len(pdf_paths),
            "books": [],
        }

        for pdf_path in pdf_paths:
            try:
                book_dataset = self.build_dataset_from_book(
                    pdf_path=pdf_path,
                    difficulty=difficulty,
                    num_mcq=num_mcq,
                    num_essay=num_essay,
                )
                full_dataset["books"].append(book_dataset)

            except Exception as e:
                logger.error("Failed processing %s: %s", pdf_path, str(e))

        return full_dataset