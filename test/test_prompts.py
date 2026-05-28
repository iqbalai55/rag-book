"""Tests for prompt templates and generators."""
import pytest

from core.schemas.expertise import ExpertiseDetection
from core.prompts.expertise import (
    EXPERTISE_DETECTION_PROMPT,
    get_system_prompt,
    get_mcq_prompt,
    get_essay_prompt,
    get_chapter_prompt,
)
from core.prompts.general_rag import (
    BOOK_QA_SYSTEM_PROMPT,
    MCQ_PROMPT,
    ESSAY_QUESTION_PROMPT,
    CHAPTER_IDENTIFICATION_PROMPT,
)
from core.prompts.mindmap import MINDMAP_PROMPT, MINDMAP_FROM_CONTENT_PROMPT


@pytest.fixture
def sample_expertise():
    return ExpertiseDetection(
        domain="Software Engineering",
        sub_fields=["code quality", "refactoring", "design patterns"],
        expertise_prompt="ahli Software Engineering spesialisasi code quality",
        book_type="textbook",
    )


# ------------------ Static Prompts ------------------
class TestStaticPrompts:
    def test_system_prompt_exists(self):
        assert "tutor ahli" in BOOK_QA_SYSTEM_PROMPT

    def test_mcq_prompt_has_placeholders(self):
        assert "{num_questions}" in MCQ_PROMPT
        assert "{topic}" in MCQ_PROMPT
        assert "{context}" in MCQ_PROMPT

    def test_essay_prompt_has_placeholders(self):
        assert "{num_questions}" in ESSAY_QUESTION_PROMPT
        assert "{topic}" in ESSAY_QUESTION_PROMPT
        assert "{context}" in ESSAY_QUESTION_PROMPT

    def test_chapter_prompt_has_placeholders(self):
        assert "{context}" in CHAPTER_IDENTIFICATION_PROMPT
        assert "{topik}" in CHAPTER_IDENTIFICATION_PROMPT


# ------------------ Mindmap Prompts ------------------
class TestMindmapPrompts:
    def test_mindmap_prompt_has_placeholders(self):
        assert "{toc_struktur}" in MINDMAP_PROMPT
        assert "{topik}" in MINDMAP_PROMPT

    def test_mindmap_from_content_has_placeholders(self):
        assert "{context}" in MINDMAP_FROM_CONTENT_PROMPT
        assert "{topik}" in MINDMAP_FROM_CONTENT_PROMPT


# ------------------ Expertise Detection Prompt ------------------
class TestExpertiseDetectionPrompt:
    def test_has_context_placeholder(self):
        assert "{context}" in EXPERTISE_DETECTION_PROMPT

    def test_has_topik_placeholder(self):
        assert "{topik}" in EXPERTISE_DETECTION_PROMPT


# ------------------ Dynamic Prompt Generators ------------------
class TestGetSystemPrompt:
    def test_contains_domain(self, sample_expertise):
        prompt = get_system_prompt(sample_expertise)
        assert "Software Engineering" in prompt

    def test_contains_expertise_prompt(self, sample_expertise):
        prompt = get_system_prompt(sample_expertise)
        assert "ahli Software Engineering spesialisasi code quality" in prompt

    def test_contains_sub_fields(self, sample_expertise):
        prompt = get_system_prompt(sample_expertise)
        assert "code quality" in prompt
        assert "refactoring" in prompt
        assert "design patterns" in prompt

    def test_contains_book_type(self, sample_expertise):
        prompt = get_system_prompt(sample_expertise)
        assert "textbook" in prompt

    def test_has_tool_references(self, sample_expertise):
        prompt = get_system_prompt(sample_expertise)
        assert "search_book_context" in prompt
        assert "generate_mcq" in prompt
        assert "generate_essay_questions" in prompt


class TestGetMcqPrompt:
    def test_has_placeholders(self, sample_expertise):
        prompt = get_mcq_prompt(sample_expertise)
        assert "{num_questions}" in prompt
        assert "{topic}" in prompt
        assert "{difficulty}" in prompt
        assert "{context}" in prompt

    def test_contains_domain(self, sample_expertise):
        prompt = get_mcq_prompt(sample_expertise)
        assert "Software Engineering" in prompt


class TestGetEssayPrompt:
    def test_has_placeholders(self, sample_expertise):
        prompt = get_essay_prompt(sample_expertise)
        assert "{num_questions}" in prompt
        assert "{topic}" in prompt
        assert "{difficulty}" in prompt
        assert "{context}" in prompt

    def test_contains_domain(self, sample_expertise):
        prompt = get_essay_prompt(sample_expertise)
        assert "Software Engineering" in prompt


class TestGetChapterPrompt:
    def test_has_placeholders(self, sample_expertise):
        prompt = get_chapter_prompt(sample_expertise)
        assert "{context}" in prompt
        assert "{topik}" in prompt

    def test_contains_domain(self, sample_expertise):
        prompt = get_chapter_prompt(sample_expertise)
        assert "Software Engineering" in prompt
